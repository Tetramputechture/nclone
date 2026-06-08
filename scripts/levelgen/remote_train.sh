#!/usr/bin/env bash
###############################################################################
# remote_train.sh — drive the synthetic level-generator pipeline on a remote
# GH200 (arm64) instance, mirroring the npp-rl remote PPO training convention
# (tarball code sync, screen-detached long jobs, Lambda-Stack venv on aarch64).
#
# Stages:
#   setup     install nclone + levelgen deps on the remote (arm64 venv aware)
#   data      build the HDF5 datasets (tiles_full.h5 + tiles_subset.h5)
#   train     full training: tile pretrain -> tile fine-tune -> entity
#   pipeline  data + train in a single detached session (the full pipeline)
#
#   status | logs [stage] | stop [session] | download | shell | help
#
# Quick start:
#   export INSTANCE_IP=1.2.3.4
#   ./scripts/levelgen/remote_train.sh setup
#   ./scripts/levelgen/remote_train.sh pipeline      # data + train end to end
#   ./scripts/levelgen/remote_train.sh logs pipeline # follow progress
#   ./scripts/levelgen/remote_train.sh download      # pull checkpoints back
#
# Everything below is overridable via environment variables (see CONFIG).
###############################################################################
set -euo pipefail

# ============================================================================
# CONFIG (env-overridable)
# ============================================================================
INSTANCE_IP="${INSTANCE_IP:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/npp-rl.pem}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_OPTS=(-o ServerAliveInterval=60 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes
          -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

# Remote layout (relative to the remote $HOME). ~ on the wire, $HOME in runners.
NCLONE_SUB="nclone-levelgen/nclone"   # remote ~/nclone-levelgen/nclone
DATA_SUB="datasets/levelgen"          # remote ~/datasets/levelgen
CKPT_SUB="levelgen_ckpt"              # remote ~/levelgen_ckpt
EXP_SUB="experiments"                 # remote ~/experiments (logs land here)
TB_SUB="experiments/tb"              # remote ~/experiments/tb (TensorBoard event files)
TB_PORT="${TB_PORT:-6006}"            # remote TensorBoard server port (SSH-tunnel to view)
VENV="levelgen-venv"                  # remote ~/levelgen-venv (arm64 only)

# Data-generation knobs
WORKERS="${WORKERS:-16}"              # attract decode workers (--from-attracts)
LABEL_FULL="${LABEL_FULL:-0}"         # 1 = run label_corpus on the full corpus too
OFFICIAL="${OFFICIAL:-1}"            # 1 = add official maps/ geometry to the tile stages

# Tile-loss token-imbalance controls (Stage B: corpus is ~84% empty tile 0).
# These push the tile stages toward human density (~47%) so generations aren't sparse.
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
EMPTY_WEIGHT="${EMPTY_WEIGHT:-0.5}"
MASK_NONEMPTY_BIAS="${MASK_NONEMPTY_BIAS:-1.5}"
IMB_FLAGS="--focal-gamma $FOCAL_GAMMA --empty-weight $EMPTY_WEIGHT --mask-nonempty-bias $MASK_NONEMPTY_BIAS"

# Tile-stage hyperparameters (doc-target 37.9M model: d512 / L12)
TILE_MODE="${TILE_MODE:-maskgit}"     # maskgit | gpt
TILE_DMODEL="${TILE_DMODEL:-512}"
TILE_LAYERS="${TILE_LAYERS:-12}"
TILE_HEADS="${TILE_HEADS:-8}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
# d512/L12 over 966 tokens: batch 256 peaks ~59GB; 512 OOMs the 96GB GH200
# (full-attention activations are batch-scaled). Keep 256 for the tile stages.
PRETRAIN_BATCH="${PRETRAIN_BATCH:-256}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
FINETUNE_BATCH="${FINETUNE_BATCH:-256}"

# Entity-stage hyperparameters (d256/L6 is small -> 512 fits comfortably)
ENTITY_DMODEL="${ENTITY_DMODEL:-256}"
ENTITY_LAYERS="${ENTITY_LAYERS:-6}"
ENTITY_EPOCHS="${ENTITY_EPOCHS:-20}"
ENTITY_BATCH="${ENTITY_BATCH:-512}"
LR="${LR:-3e-4}"                       # proven peak LR; cosine warmup/decay on top

# Efficiency + EMA (per-stage; see train_tile/train_entity)
COMPILE="${COMPILE:-1}"              # 1 = torch.compile the forward (Hopper)
LR_SCHEDULE="${LR_SCHEDULE:-cosine}" # constant | cosine (warmup+cosine decay)
WARMUP_FRAC="${WARMUP_FRAC:-0.03}"
EMA_DECAY="${EMA_DECAY:-0.999}"      # 0 = off; sampling uses EMA weights
EFF_FLAGS="--lr-schedule $LR_SCHEDULE --warmup-frac $WARMUP_FRAC --ema-decay $EMA_DECAY"
[ "$COMPILE" = "1" ] && EFF_FLAGS="$EFF_FLAGS --compile"

# Periodic generation visualization to TensorBoard (sample/grids, sample/fill,
# sample/diversity for tiles; sample/placements for entities).
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"   # epochs between TB sample logs (0 = off)
TILE_VIZ_FLAGS="--sample-every $SAMPLE_EVERY --sample-cfg 2.0"
ENTITY_VIZ_FLAGS="--sample-every $SAMPLE_EVERY"

AMP="${AMP:-1}"                       # 1 = pass --amp (bf16 autocast)
AMP_FLAG=""; [ "$AMP" = "1" ] && AMP_FLAG="--amp"

DL_WORKERS="${DL_WORKERS:-8}"         # torch DataLoader workers (feeds the GPU)
DL_WORKERS_FLAG="--num-workers $DL_WORKERS"

# Local repo root (this script lives in <root>/scripts/levelgen/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_DATA="$REPO_ROOT/datasets/levelgen"
LOCAL_CKPT="$REPO_ROOT/ckpt"
LOCAL_ATTRACTS="${LOCAL_ATTRACTS:-$REPO_ROOT/../attracts}"
TS="$(date +%Y%m%d_%H%M%S)"

# ============================================================================
# Logging
# ============================================================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[levelgen-remote]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

require_ip() {
  [ -n "$INSTANCE_IP" ] || die "INSTANCE_IP is not set. Use: INSTANCE_IP=<ip> $0 <cmd>  (or --ip <ip>)"
  [ -f "$SSH_KEY" ] || die "SSH key not found: $SSH_KEY (override with SSH_KEY=...)"
}

# ============================================================================
# SSH / transfer helpers
# ============================================================================
rmt()    { ssh -i "$SSH_KEY" "${SSH_OPTS[@]}" "${SSH_USER}@${INSTANCE_IP}" "$@"; }
scp_to() { scp -i "$SSH_KEY" "${SSH_OPTS[@]}" -r "$1" "${SSH_USER}@${INSTANCE_IP}:$2"; }
scp_from() { scp -i "$SSH_KEY" "${SSH_OPTS[@]}" -r "${SSH_USER}@${INSTANCE_IP}:$1" "$2"; }
rsync_to() {
  rsync -az --info=progress2 -e "ssh -i $SSH_KEY ${SSH_OPTS[*]}" "$@"
}

# ============================================================================
# Remote runner generation
#
# Each long stage is shipped as a self-contained bash script (avoids ssh/screen
# quoting hell) and launched inside a detached `screen` session. The runner tees
# all output to ~/experiments/<stage>-<ts>.log so `logs` can follow it.
#
# In the heredoc: \$VAR / \$(...) stay literal for the remote shell; $VAR expands
# here. Command fragments are built with single quotes so $D/$C reach the remote
# verbatim and resolve to ~/datasets/levelgen and ~/levelgen_ckpt there.
# ============================================================================
build_runner() {   # $1=stage label, $2=command block (newline separated)
  local stage="$1" cmds="$2"
  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
D="\$HOME/$DATA_SUB"
C="\$HOME/$CKPT_SUB"
T="\$HOME/$TB_SUB"
mkdir -p "\$HOME/$EXP_SUB" "\$C" "\$D" "\$T"
LOG="\$HOME/$EXP_SUB/${stage}-${TS}.log"
exec > >(tee -a "\$LOG") 2>&1
cd "\$HOME/$NCLONE_SUB"
[ -f "\$HOME/$VENV/bin/activate" ] && source "\$HOME/$VENV/bin/activate"
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOMs
echo "[levelgen-remote] stage=$stage host=\$(hostname) start \$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
set -x
$cmds
set +x
echo "[levelgen-remote] stage=$stage DONE \$(date)"
EOF
}

launch_stage() {   # $1=screen session, $2=stage label, $3=command block
  local session="$1" stage="$2" cmds="$3"
  local tmp; tmp="$(mktemp)"
  build_runner "$stage" "$cmds" > "$tmp"
  scp_to "$tmp" "~/.levelgen_runner_${stage}.sh" >/dev/null
  rm -f "$tmp"
  rmt "chmod +x ~/.levelgen_runner_${stage}.sh && \
       screen -S $session -X quit >/dev/null 2>&1 || true; \
       screen -dmS $session bash -lc '~/.levelgen_runner_${stage}.sh; echo; echo [stage $stage exited rc=\$?]; exec bash'"
  ok "Launched detached session '$session' (stage=$stage)."
  log "Follow it with:  $0 logs $stage"
  log "Attach with:     ssh -i $SSH_KEY ${SSH_USER}@${INSTANCE_IP} -t screen -r $session"
}

# ============================================================================
# Command-fragment builders (single-quoted $D/$C -> resolved on the remote)
# ============================================================================
# Official-maps geometry corpus: included in the TILE stages (pretrain + finetune)
# when OFFICIAL=1. 3.5k canonical human geometries -> macro-structure diversity.
# $D kept literal here so it resolves on the remote.
OFFICIAL_GEOM=""; [ "$OFFICIAL" = "1" ] && OFFICIAL_GEOM=' "$D/corpus_official_geom"'

cmd_ingest_official() {
  printf '%s\n' 'python -m scripts.levelgen.ingest_corpus --official nclone/maps/official --official-all-geometry --out "$D/corpus_official_geom"'
}
cmd_build_full() {   # tile pretrain set (+ official geometry)
  printf '%s\n' 'python -m scripts.levelgen.build_dataset --corpus "$D/corpus_full"'"$OFFICIAL_GEOM"' --out "$D/tiles_full.h5"'
}
cmd_build_subset() {  # ENTITY stage set: subset-valid only, NOT polluted with official entities
  printf '%s\n' 'python -m scripts.levelgen.build_dataset --corpus "$D/corpus_subset" --out "$D/tiles_subset.h5" --subset-only --mirror'
}
cmd_build_finetune() {  # tile FINETUNE set: subset geometry + official geometry (mirrored), no subset filter
  printf '%s\n' 'python -m scripts.levelgen.build_dataset --corpus "$D/corpus_subset"'"$OFFICIAL_GEOM"' --out "$D/tiles_finetune.h5" --mirror'
}
cmd_label_full() {
  printf '%s\n' 'python -m scripts.levelgen.label_corpus --corpus "$D/corpus_full"'
}
cmd_attracts_pipeline() {
  cat <<FRAG
python -m nclone.map_augmentation.attract.filter_attracts --source "\$HOME/attracts" --output "\$HOME/attracts_subset" --workers WORKERS_TOKEN --mine-cap 64 --gold-cap 40
python -m scripts.levelgen.ingest_corpus --attracts "\$HOME/attracts_subset" --workers WORKERS_TOKEN --out "\$D/corpus_subset"
python -m scripts.levelgen.label_corpus --corpus "\$D/corpus_subset"
python -m scripts.levelgen.ingest_corpus --attracts "\$HOME/attracts" --workers WORKERS_TOKEN --out "\$D/corpus_full"
$( [ "$OFFICIAL" = "1" ] && echo 'python -m scripts.levelgen.ingest_corpus --official nclone/maps/official --official-all-geometry --out "$D/corpus_official_geom"' )
python -m scripts.levelgen.build_dataset --corpus "\$D/corpus_subset" --out "\$D/tiles_subset.h5" --subset-only --mirror
python -m scripts.levelgen.build_dataset --corpus "\$D/corpus_subset"$OFFICIAL_GEOM --out "\$D/tiles_finetune.h5" --mirror
python -m scripts.levelgen.build_dataset --corpus "\$D/corpus_full"$OFFICIAL_GEOM --out "\$D/tiles_full.h5"
FRAG
}
cmd_tile_pretrain() {
  printf '%s\n' 'python -m scripts.levelgen.train_tile --data "$D/tiles_full.h5" --mode '"$TILE_MODE"' --d-model '"$TILE_DMODEL"' --n-layers '"$TILE_LAYERS"' --n-heads '"$TILE_HEADS"' --epochs '"$PRETRAIN_EPOCHS"' --batch-size '"$PRETRAIN_BATCH"' '"$DL_WORKERS_FLAG"' --lr '"$LR"' '"$AMP_FLAG"' '"$IMB_FLAGS"' '"$EFF_FLAGS"' '"$TILE_VIZ_FLAGS"' --tb-logdir "$T" --run-name tile_pretrain --out "$C/tile_pretrain.pt"'
}
cmd_tile_finetune() {
  printf '%s\n' 'python -m scripts.levelgen.train_tile --data "$D/tiles_finetune.h5" --mode '"$TILE_MODE"' --d-model '"$TILE_DMODEL"' --n-layers '"$TILE_LAYERS"' --n-heads '"$TILE_HEADS"' --init-from "$C/tile_pretrain.pt" --epochs '"$FINETUNE_EPOCHS"' --batch-size '"$FINETUNE_BATCH"' '"$DL_WORKERS_FLAG"' --lr '"$LR"' '"$AMP_FLAG"' '"$IMB_FLAGS"' '"$EFF_FLAGS"' '"$TILE_VIZ_FLAGS"' --tb-logdir "$T" --run-name tile_finetune --out "$C/tile.pt"'
}
cmd_entity() {
  printf '%s\n' 'python -m scripts.levelgen.train_entity --data "$D/tiles_subset.h5" --tile-ckpt "$C/tile.pt" --d-model '"$ENTITY_DMODEL"' --n-layers '"$ENTITY_LAYERS"' --epochs '"$ENTITY_EPOCHS"' --batch-size '"$ENTITY_BATCH"' '"$DL_WORKERS_FLAG"' --lr '"$LR"' '"$AMP_FLAG"' '"$EFF_FLAGS"' '"$ENTITY_VIZ_FLAGS"' --tb-logdir "$T" --run-name entity --out "$C/entity.pt"'
}

data_cmds() {        # default (corpus-shard) data block
  [ "$OFFICIAL" = "1" ] && cmd_ingest_official
  cmd_build_full
  [ "$LABEL_FULL" = "1" ] && cmd_label_full
  cmd_build_subset
  cmd_build_finetune
}
train_cmds() {
  cmd_tile_pretrain
  cmd_tile_finetune
  cmd_entity
}

# ============================================================================
# Data transfer
# ============================================================================
sync_corpus() {
  log "Syncing decoded corpus shards to remote ~/$DATA_SUB ..."
  [ -d "$LOCAL_DATA/corpus_full" ]   || die "missing $LOCAL_DATA/corpus_full (run the local ingest, or use --from-attracts)"
  [ -d "$LOCAL_DATA/corpus_subset" ] || die "missing $LOCAL_DATA/corpus_subset"
  rmt "mkdir -p ~/$DATA_SUB"
  rsync_to "$LOCAL_DATA/corpus_full" "$LOCAL_DATA/corpus_subset" \
           "${SSH_USER}@${INSTANCE_IP}:~/$DATA_SUB/"
  ok "Corpus shards synced."
}
sync_attracts() {
  log "Syncing raw attracts to remote ~/attracts (this is large) ..."
  [ -d "$LOCAL_ATTRACTS" ] || die "missing attracts dir: $LOCAL_ATTRACTS (set LOCAL_ATTRACTS=...)"
  rmt "mkdir -p ~/attracts"
  rsync_to "$LOCAL_ATTRACTS/" "${SSH_USER}@${INSTANCE_IP}:~/attracts/"
  ok "Attracts synced."
}

# ============================================================================
# Subcommands
# ============================================================================
cmd_setup() {
  require_ip
  log "Syncing nclone code to remote ~/$NCLONE_SUB ..."
  local tar="/tmp/nclone-levelgen-${TS}.tar.gz"
  tar -czf "$tar" --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='*.so' --exclude='*.egg-info' --exclude='build' --exclude='dist' \
      --exclude='datasets' --exclude='ckpt' --exclude='logs' --exclude='experiments' \
      -C "$(dirname "$REPO_ROOT")" "$(basename "$REPO_ROOT")" 2>/dev/null || true
  rmt "mkdir -p ~/nclone-levelgen"
  scp_to "$tar" "~/nclone-levelgen/code.tar.gz" >/dev/null
  rmt "cd ~/nclone-levelgen && rm -rf nclone _extract && mkdir _extract && tar -xzf code.tar.gz -C _extract && mv _extract/'$(basename "$REPO_ROOT")' nclone && rm -rf _extract code.tar.gz"
  rm -f "$tar"
  ok "Code synced."

  local arch; arch="$(rmt 'uname -m' | tr -d '\r')"
  log "Remote arch: $arch"
  log "Installing system build deps (best effort; needs sudo) ..."
  rmt "sudo apt-get update -qq && sudo apt-get install -y -qq libcairo2-dev pkg-config build-essential python3-dev python3-venv >/dev/null 2>&1" || warn "apt-get step skipped/failed (continuing)"

  if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
    log "ARM64 detected -> Lambda-Stack venv (--system-site-packages for system PyTorch)."
    rmt "test -d ~/$VENV || python3 -m venv --system-site-packages ~/$VENV"
    rmt "source ~/$VENV/bin/activate && pip install -q --upgrade pip 'setuptools>=64' wheel && \
         pip install -q 'numpy>=1.24,<2.0' 'h5py>=3.7.0' 'tensorboard>=2.12' && \
         cd ~/$NCLONE_SUB && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NCLONE=0.1.0 pip install -q -e . --no-deps && \
         pip install -q 'pygame>=2.1.0' 'gymnasium>=0.26.0' 'pycairo>=1.28.0' 'opencv-python>=4.12.0' 'pillow>=10.0.0' && \
         pip install -q --force-reinstall --no-deps 'numpy>=1.24,<2.0'"
  else
    log "x86_64 detected -> standard pip --user install."
    rmt "python3 -m pip install -q --user --upgrade pip 'setuptools>=64' wheel && \
         python3 -m pip install -q --user 'numpy>=1.24,<2.0' 'h5py>=3.7.0' 'tensorboard>=2.12' && \
         cd ~/$NCLONE_SUB && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NCLONE=0.1.0 python3 -m pip install -q --user -e ."
  fi

  log "Verifying torch + CUDA + h5py + tensorboard + nclone import ..."
  rmt "source ~/$VENV/bin/activate 2>/dev/null || true; cd ~/$NCLONE_SUB && python3 -c \
      'import torch, h5py, numpy; from torch.utils.tensorboard import SummaryWriter; from nclone.levelgen.tokenizer import LevelTokenizer; \
       print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"| numpy\", numpy.__version__, \"| h5py\", h5py.__version__, \"| tensorboard OK\")'" \
    || die "remote import verification failed — fix deps before training"
  rmt "mkdir -p ~/$DATA_SUB ~/$CKPT_SUB ~/$EXP_SUB"
  ok "Setup complete. Next: $0 data   (or $0 pipeline for data+train)."
}

cmd_data() {
  require_ip
  if [ "${FROM_ATTRACTS:-0}" = "1" ]; then
    sync_attracts
    local cmds; cmds="$(cmd_attracts_pipeline)"; cmds="${cmds//WORKERS_TOKEN/$WORKERS}"
    launch_stage "levelgen_data" "data" "$cmds"
  else
    sync_corpus
    launch_stage "levelgen_data" "data" "$(data_cmds)"
  fi
}

cmd_train() {
  require_ip
  warn "Assumes tiles_full.h5 + tiles_subset.h5 already exist on the remote (run '$0 data' first)."
  launch_stage "levelgen_train" "train" "$(train_cmds)"
}

cmd_pipeline() {
  require_ip
  if [ "${FROM_ATTRACTS:-0}" = "1" ]; then
    sync_attracts
    local d; d="$(cmd_attracts_pipeline)"; d="${d//WORKERS_TOKEN/$WORKERS}"
    launch_stage "levelgen_pipeline" "pipeline" "$(printf '%s\n%s\n' "$d" "$(train_cmds)")"
  else
    sync_corpus
    launch_stage "levelgen_pipeline" "pipeline" "$(printf '%s\n%s\n' "$(data_cmds)" "$(train_cmds)")"
  fi
}

cmd_status() {
  require_ip
  log "screen sessions:"; rmt "screen -ls" || true
  echo
  log "GPU:"; rmt "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null" || warn "nvidia-smi unavailable"
  echo
  log "checkpoints on remote (~/$CKPT_SUB):"; rmt "ls -lh ~/$CKPT_SUB 2>/dev/null || echo '(none yet)'"
  log "datasets on remote (~/$DATA_SUB):"; rmt "ls -lh ~/$DATA_SUB/*.h5 2>/dev/null || echo '(no .h5 yet)'"
  echo
  log "TensorBoard runs on remote (~/$TB_SUB):"; rmt "ls -d ~/$TB_SUB/*/ 2>/dev/null || echo '(no tb runs yet)'"
  if rmt "screen -ls 2>/dev/null | grep -q levelgen_tb"; then
    log "TensorBoard server is UP (port $TB_PORT). Tunnel: ssh -i $SSH_KEY -N -L ${TB_PORT}:localhost:${TB_PORT} ${SSH_USER}@${INSTANCE_IP}"
  else
    log "TensorBoard server is not running. Start it with:  $0 tensorboard"
  fi
}

cmd_logs() {
  require_ip
  local stage="${1:-}"; local glob="${stage:+${stage}-}"
  local follow="-n 120"; [ "${FOLLOW:-0}" = "1" ] && follow="-n 120 -f"
  log "Tailing newest ~/$EXP_SUB/${glob}*.log (Ctrl-C to stop) ..."
  rmt "f=\$(ls -t ~/$EXP_SUB/${glob}*.log 2>/dev/null | head -1); \
       [ -n \"\$f\" ] && { echo \"== \$f ==\"; tail $follow \"\$f\"; } || echo 'no log files yet'"
}

cmd_tensorboard() {
  require_ip
  log "Launching TensorBoard on the remote (port $TB_PORT, logdir ~/$TB_SUB) ..."
  rmt "mkdir -p ~/$TB_SUB; screen -S levelgen_tb -X quit >/dev/null 2>&1 || true; \
       screen -dmS levelgen_tb bash -lc '\
         [ -f ~/$VENV/bin/activate ] && source ~/$VENV/bin/activate; \
         exec tensorboard --logdir ~/$TB_SUB --port $TB_PORT --host 0.0.0.0'"
  ok "TensorBoard running in screen session 'levelgen_tb'."
  log "View it locally via an SSH tunnel:"
  echo -e "    ssh -i $SSH_KEY -N -L ${TB_PORT}:localhost:${TB_PORT} ${SSH_USER}@${INSTANCE_IP}"
  log "then open  http://localhost:${TB_PORT}"
  log "Stop it with:  $0 stop levelgen_tb"
}

cmd_stop() {
  require_ip
  local session="${1:-}"
  if [ -n "$session" ]; then
    rmt "screen -S $session -X quit" && ok "stopped '$session'"
  else
    for s in levelgen_pipeline levelgen_train levelgen_data levelgen_tb; do
      rmt "screen -S $s -X quit >/dev/null 2>&1" && log "stopped $s" || true
    done
    ok "stopped all levelgen sessions"
  fi
}

cmd_download() {
  require_ip
  mkdir -p "$LOCAL_CKPT"
  log "Downloading checkpoints + run configs ~/$CKPT_SUB -> $LOCAL_CKPT/ ..."
  scp_from "~/$CKPT_SUB/*.pt" "$LOCAL_CKPT/" || warn "no checkpoints found yet"
  scp_from "~/$CKPT_SUB/*.run_config.json" "$LOCAL_CKPT/" || warn "no run configs found yet"
  log "Downloading TensorBoard events ~/$TB_SUB -> $LOCAL_CKPT/tb/ ..."
  mkdir -p "$LOCAL_CKPT/tb"
  scp_from "~/$TB_SUB/*" "$LOCAL_CKPT/tb/" || warn "no TensorBoard events found yet"
  log "View locally with:  tensorboard --logdir $LOCAL_CKPT/tb"
  if [ "${WITH_DATA:-0}" = "1" ]; then
    log "Downloading datasets ~/$DATA_SUB/*.h5 -> $LOCAL_DATA/ ..."
    mkdir -p "$LOCAL_DATA"; scp_from "~/$DATA_SUB/*.h5" "$LOCAL_DATA/" || warn "no .h5 found"
  fi
  ok "Download complete."
}

cmd_shell() { require_ip; exec ssh -i "$SSH_KEY" "${SSH_OPTS[@]}" -t "${SSH_USER}@${INSTANCE_IP}"; }

usage() {
  cat <<EOF
remote_train.sh — synthetic level-generator training on a remote GH200 (arm64)

USAGE:
  INSTANCE_IP=<ip> $0 <command> [args]   (or pass --ip <ip>)

COMMANDS:
  setup                 sync nclone + install deps on the remote (arm64-aware)
  data                  build tiles_full.h5 + tiles_subset.h5 (detached)
  train                 tile pretrain -> fine-tune -> entity (detached)
  pipeline              data + train in one detached session (full pipeline)
  status                screen sessions, GPU, remote checkpoints/datasets, TB
  tensorboard           launch TensorBoard on the remote (port \$TB_PORT, ~/$TB_SUB)
  logs [stage]          tail newest log (stage = data|train|pipeline); FOLLOW=1 to -f
  stop [session]        kill a screen session (default: all levelgen sessions)
  download              scp checkpoints back (WITH_DATA=1 also pulls *.h5)
  shell                 interactive ssh into the remote
  help                  this message

COMMON FLAGS / ENV:
  --ip <ip>             remote instance IP            (env INSTANCE_IP)
  --from-attracts       re-decode from raw ../attracts instead of syncing
                        the pre-decoded corpus shards (env FROM_ATTRACTS=1)
  SSH_KEY=...           ssh key            (default ~/.ssh/npp-rl.pem)
  SSH_USER=...          ssh user           (default ubuntu)
  TILE_DMODEL/TILE_LAYERS/PRETRAIN_EPOCHS/...   model + schedule overrides
  AMP=0                 disable bf16 autocast
  LABEL_FULL=1          also label the full corpus (difficulty bands on full)
  FOLLOW=1              make 'logs' follow (-f)
  WITH_DATA=1           make 'download' also pull datasets
  TB_PORT=...           remote TensorBoard port            (default 6006)

EXAMPLES:
  INSTANCE_IP=1.2.3.4 $0 setup
  INSTANCE_IP=1.2.3.4 $0 pipeline
  INSTANCE_IP=1.2.3.4 FOLLOW=1 $0 logs pipeline
  INSTANCE_IP=1.2.3.4 $0 tensorboard   # then SSH-tunnel port 6006
  INSTANCE_IP=1.2.3.4 $0 download       # pulls ckpts + run configs + tb/
EOF
}

# ============================================================================
# Arg parse (extract --ip / --from-attracts before the subcommand)
# ============================================================================
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --ip) INSTANCE_IP="$2"; shift 2 ;;
    --from-attracts) FROM_ATTRACTS=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-help}"

case "${1:-help}" in
  setup)    cmd_setup ;;
  data)     cmd_data ;;
  train)    cmd_train ;;
  pipeline) cmd_pipeline ;;
  status)   cmd_status ;;
  tensorboard|tb) cmd_tensorboard ;;
  logs)     shift; cmd_logs "${1:-}" ;;
  stop)     shift; cmd_stop "${1:-}" ;;
  download) cmd_download ;;
  shell)    cmd_shell ;;
  help|-h|--help) usage ;;
  *) die "unknown command '$1' (run: $0 help)" ;;
esac
