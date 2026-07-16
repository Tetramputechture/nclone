# nclone: N++ Simulator for Deep RL

N++ game simulator with a Gymnasium-compatible interface for deep reinforcement learning research.

## Overview

`nclone` is a Python N++ simulator built for DRL training. It provides:

- Physics simulation matching N++ gameplay mechanics
- A Gymnasium-compatible environment
- Headless rendering for fast training
- A structured multi-tensor observation (tiles, entity overlays, hazard field, physics state)

## Installation

```bash
# From source (recommended for development)
git clone https://github.com/Tetramputechture/nclone.git
cd nclone
pip install -e .

# Or directly
pip install git+https://github.com/Tetramputechture/nclone.git
```

Dependencies: Python 3.8+, pygame>=2.5.0, numpy>=1.24.0, opencv-python>=4.8.0, pycairo>=1.25.0, gymnasium>=0.29.0.

## Quick start

### Interactive play

```bash
python -m nclone.test_environment

# Controls:
#   Arrow keys / WASD: move
#   Space / Up: jump
#   R: reset level
```

### RL training

```python
from nclone.gym_environment import NppEnvironment, EnvironmentConfig

# Headless training config
env = NppEnvironment(EnvironmentConfig.for_training())

obs, info = env.reset()
done = False
while not done:
    action = policy(obs)  # your policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
```

`create_training_env(config=None)` is a convenience wrapper around the same constructor.
`EnvironmentConfig` controls rendering, PBRS, graph output, reachability, reward config, sim
backend, and curriculum (see `nclone/gym_environment/config.py`).

## Environment interface

### Action space

Discrete(6): 0 no-op, 1 left, 2 right, 3 jump, 4 left+jump, 5 right+jump.

### Observation space

The default training observation is a `Dict` of five tensors:

| key | shape | dtype | notes |
|---|---|---|---|
| `tile_grid` | `(42, 23)` | int8 | tile-type ids (0 to 37) |
| `entity_overlays` | `(5, 42, 23)` | float32 | per-type entity presence (ninja, switch, exit, deadly/safe mine density) |
| `hazard_map` | `(2, 168, 92)` | float32 | deadly-mine and safe-mine signed distance fields, in [-1, 1] |
| `game_state` | `(261,)` | float32 | flat physics/proprio/goal vector (grows with opt-in exposure flags) |
| `action_mask` | `(6,)` | int8 | valid primitive actions |

The authoritative spec, including the full `game_state` index map and opt-in keys (cost-to-go
map, ego crops, waypoint/lattice nav), is in
[OBSERVATION_SPACE_README.md](OBSERVATION_SPACE_README.md). Validate a live env with:

```bash
python tools/validate_observations.py --episodes 3
```

### Reward structure

Multi-component reward: level completion (+1.0), death (-0.5), switch activation (+0.1), a small
per-step time penalty, distance-based navigation shaping, and exploration bonuses. PBRS shaping and
component weights are configured through `EnvironmentConfig.reward_config`; the training-side
reward design is documented in npp-rl (`docs/REWARD_SYSTEM_DOCUMENTATION.md`).

## Performance

| Mode | FPS | Use case |
|---|---|---|
| Headless | 1000-2000 | RL training |
| Rendered (human) | 60 | Interactive play / debug |
| Multi-process (64 envs) | 30K+ steps/sec | Distributed training |

Reachability analysis runs in under 1ms per step (OpenCV flood fill, cached and updated
incrementally). Memory: ~200MB per environment, ~6GB for 64 parallel environments.

## Datasets

Levels are stored in the simulator `map_data` integer format read from official `.map` binaries
and produced by the map generators. The benchmark dataset and its layout are documented in
[docs/BENCHMARK_DATASET.md](docs/BENCHMARK_DATASET.md); generate one with
`scripts/generate_benchmark_dataset.py`.

## Development

```bash
pytest tests/                                   # test suite
pytest tests/test_native_backend_parity.py -xvs # one module
pytest --cov=nclone --cov-report=html           # with coverage

make dev-setup   # install dev tools
make lint        # lint
make fix         # lint and auto-fix
make imports     # remove unused imports
```

Profiling:
```bash
python -m cProfile -o profile.out -m nclone.test_environment
python -m pstats profile.out
python -m nclone.test_environment --log-frametimes
```

## Integration with npp-rl

This simulator is the training environment for the `npp-rl` framework:

```bash
git clone https://github.com/Tetramputechture/nclone.git
git clone https://github.com/Tetramputechture/npp-rl.git
cd nclone && pip install -e . && cd ..
cd npp-rl && pip install -e . && cd ..

# Train (see npp-rl for full options and the reference v22 launcher)
cd npp-rl
python scripts/train.py \
    --experiment-name quickstart \
    --benchmark-root ~/datasets/mechanic_benchmark \
    --total-timesteps 20_000_000
```

See `npp-rl/README.md` for training instructions.
