"""Horizontal-mirror transform for the policy observation dict and actions.

Companion to :func:`nclone.map_augmentation.transforms.mirror_map_data`. That
primitive reflects a *level*; this module reflects an *observation of* a level,
so that::

    mirror_observation(obs(s)) == obs(mirror_state(s))

holds field by field. The equality is not asserted here -- it is an executable
test (``tests/test_obs_mirror.py``), because the simulator is the authority on
whether it commutes with reflection and several fields are only invariant by
construction (see the scandots note below).

Why the per-field table has to be exhaustive
--------------------------------------------
A missed field does not crash: it produces a plausible-looking observation with
one block describing the wrong direction, i.e. a silently poisoned dataset. The
``game_state`` vector is the dangerous one -- 269 dims in the v26 contract, of
which 180 are ghost-arc bundles whose ``u = -1`` / ``u = +1`` rows must SWAP.

So the spec is built by *marking* every index of the live contract and raising
:class:`UnclassifiedGameStateDims` if any index is left unmarked. An observation
contract that grows a block without a rule here fails loudly at construction
instead of quietly mirroring the new dims as invariant.

Three classes of field
----------------------
* **negate** -- a signed horizontal quantity (``velocity_x``, ``floor_normal_x``,
  a route ``ux``, an arc's landing ``dx``).
* **affine** -- ``position_x`` is normalized to ``[0, 1]``, so its reflection is
  ``1 - x``, not ``-x``.
* **permute** -- the 8 compass raycasts, the ``u = +-1`` arcs within each ghost
  bundle, the per-arc route-progress row, and the discrete action space.

And one trap class that looks like it needs flipping and does not:

* **direction-relative** -- ``GAME_STATE_SCANDOTS`` samples tile columns at
  offsets ``[-1, 0, 1, 2, 3, 4] x travel_direction``. The travel direction flips
  under reflection too, so the block is INVARIANT; "helpfully" reversing it
  would corrupt it. This holds exactly when the travel direction is itself
  reflection-odd, which is what ``mirror_equivariant_scandots`` buys (the legacy
  ``sign(vx) or 1`` tie-break resolves to +1 in BOTH orientations when the ninja
  is at rest, which is an asymmetry -- see ``_compute_scandots`` callers).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..gym_environment.constants import (
    GAME_STATE_ACCEL_IDX,
    GAME_STATE_ARC_PHI_DIM,
    GAME_STATE_ARC_PHI_IDX,
    GAME_STATE_CHANNELS,
    GAME_STATE_CLOSING_SPEED_IDX,
    GAME_STATE_EXIT_VEC_IDX,
    GAME_STATE_FRAMES_AIRBORNE_IDX,
    GAME_STATE_FREE_SPACE_NAV_DIM,
    GAME_STATE_FREE_SPACE_NAV_IDX,
    GAME_STATE_GHOST_ARC_ARCS,
    GAME_STATE_GHOST_ARC_DIM,
    GAME_STATE_GHOST_ARC_IDX,
    GAME_STATE_GHOST_ARC_PER_ARC,
    GAME_STATE_GOLD_TAIL_DIM,
    GAME_STATE_JUMP_ARC_IDX,
    GAME_STATE_JUMP_EXEC_DIM,
    GAME_STATE_JUMP_HELD_IDX,
    GAME_STATE_LANDING_DIST_IDX,
    GAME_STATE_LANDING_NORMAL_IDX,
    GAME_STATE_NEAREST_MINE_IDX,
    GAME_STATE_PASSIVE_ARC_DIM,
    GAME_STATE_PASSIVE_ARC_IDX,
    GAME_STATE_POSITION_START,
    GAME_STATE_RAYCAST_DIM,
    GAME_STATE_RAYCAST_IDX,
    GAME_STATE_SCANDOTS_DIM,
    GAME_STATE_SCANDOTS_IDX,
    GAME_STATE_SWITCH_ACTIVATED_IDX,
    GAME_STATE_SWITCH_VEC_IDX,
    GAME_STATE_TAP_ARC_IDX,
    GAME_STATE_TIME_TO_CONTACT_IDX,
    GAME_STATE_VERTICAL_TERMINAL_IDX,
    GAME_STATE_WALL_ARC_IDX,
    LATTICE_NAV_DIM,
    MINE_PROXIMITY_FEATURES,
    THREAT_PROX_FEATURES,
    THREAT_PROX_IDX_DX,
    THREAT_PROX_IDX_VX,
    NINJA_STATE_DIM,
    NUM_TILE_TYPES,
    PHYSICS_DERIVED_DIM,
    SOLID_MAP_CH_LAUNCHPAD_BX,
    SOLID_MAP_CH_LAUNCHPAD_BY,
    SOLID_MAP_CH_ONEWAY_NX,
    SOLID_MAP_CH_ONEWAY_NY,
    SOLID_MAP_CHANNELS,
    THREAT_TTL_CHANNELS,
    WAYPOINT_NAV_DIM,
)
from .transforms import _TILE_MIRROR

__all__ = [
    "MIRROR_ACTION_MAP",
    "SPATIAL_MIRROR_SPEC",
    "GameStateMirror",
    "ObservationMirror",
    "SpatialMirrorRule",
    "UnclassifiedGameStateDims",
    "mirror_action",
    "mirror_actions",
    "mirror_input_byte",
    "mirror_input_sequence",
    "mirror_tile_ids",
]


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

#: ``base_environment._actions_to_execute`` order:
#: ``0 NOOP, 1 Left, 2 Right, 3 Jump, 4 Jump+Left, 5 Jump+Right``.
#: Left <-> Right and Jump+Left <-> Jump+Right; NOOP and Jump are fixed points.
#: Involutive, and duration bins are untouched by reflection.
MIRROR_ACTION_MAP: Tuple[int, ...] = (0, 2, 1, 3, 5, 4)

_MIRROR_ACTION_LUT = np.array(MIRROR_ACTION_MAP, dtype=np.int64)


def mirror_action(action: int) -> int:
    """Reflect a single discrete action."""
    return MIRROR_ACTION_MAP[int(action)]


def mirror_actions(actions: np.ndarray) -> np.ndarray:
    """Reflect an array of discrete actions (any shape)."""
    return _MIRROR_ACTION_LUT[np.asarray(actions, dtype=np.int64)]


def mirror_input_byte(byte: int) -> int:
    """Reflect a raw replay input byte.

    The encoding is a bit field (``replay_executor.decode_input_to_controls``):
    ``bit0 = jump``, ``bit1 = right``, ``bit2 = left``. Reflection swaps the
    left and right bits and leaves jump alone, which also maps the two invalid
    left+right codes (6, 7) to themselves.
    """
    b = int(byte)
    return (b & 0x01) | ((b & 0x02) << 1) | ((b & 0x04) >> 1)


def mirror_input_sequence(inputs: Iterable[int]) -> list:
    """Reflect a whole replay tape, byte for byte."""
    return [mirror_input_byte(b) for b in inputs]


# --------------------------------------------------------------------------
# Tile IDs
# --------------------------------------------------------------------------

# Flipping a tile array without remapping its IDs produces a level where every
# slope faces the wrong way -- geometrically plausible, physically wrong, and it
# does not crash. The extractor derives per-tile surface normals from the IDs,
# so a correct remap fixes the normals for free.
_TILE_LUT = np.arange(256, dtype=np.int16)
for _src, _dst in _TILE_MIRROR.items():
    _TILE_LUT[_src] = _dst
# Guard the assumption that _TILE_MIRROR only ever renames within the tile range.
assert all(0 <= v < NUM_TILE_TYPES for v in _TILE_MIRROR.values())


def mirror_tile_ids(tiles: np.ndarray) -> np.ndarray:
    """Remap tile IDs to their mirror variants (does NOT flip the array)."""
    arr = np.asarray(tiles)
    return _TILE_LUT[arr.astype(np.int64)].astype(arr.dtype, copy=False)


# --------------------------------------------------------------------------
# game_state
# --------------------------------------------------------------------------


class UnclassifiedGameStateDims(ValueError):
    """Raised when a ``game_state`` dim has no explicit mirror rule.

    The v26 launch died because ``expose_commitment_state`` grew ``game_state``
    and an allow-list in another code path was not updated. The mirror spec is
    the same shape of hazard, so an unclassified dim is a hard error here rather
    than a silent pass-through.
    """


#: Optional ``game_state`` tail blocks, in the exact order
#: ``NppEnvironment._process_observation`` appends them. Each entry is
#: ``(exposure flag, width, negated offsets within the block)``.
_OPTIONAL_TAIL_BLOCKS: Tuple[Tuple[str, int, Tuple[int, ...]], ...] = (
    # gold fraction remaining -- appended UNCONDITIONALLY, hence flag None.
    (None, GAME_STATE_GOLD_TAIL_DIM, ()),
    ("expose_blocked_goal", 1, ()),
    ("expose_wall_contact_tau", 1, ()),
    # precision_state[0] is wall_normal_buffered: a retained wall normal, so it
    # is a signed horizontal direction. [1..9] are states / dwell counters.
    ("expose_precision_state", 10, (0,)),
    ("expose_time_phase", 6, ()),
    # commitment_state aggregates OVER arcs (fatal fractions, no_escape,
    # escape_deficit), so reflection permutes what is counted, not the counts.
    ("expose_commitment_state", 6, ()),
)

#: Reflection of the 8 compass raycasts ``(E, NE, N, NW, W, SW, S, SE)``.
#: ``_RAYCAST_DIRS`` is ``((1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1))``
#: and reflection sends ``(dx, dy) -> (-dx, dy)``.
_RAYCAST_PERM: Tuple[int, ...] = (4, 3, 2, 1, 0, 7, 6, 5)

#: Per-arc offsets to negate inside a 15-dim ghost arc:
#: ``[0] dx``, ``[2] nx``, ``[9] vx_contact``. Everything else is a scalar
#: magnitude, a flag, or a vertical quantity.
_GHOST_ARC_NEGATED: Tuple[int, ...] = (0, 2, 9)


class GameStateMirror:
    """Index/sign/offset arrays that reflect a ``game_state`` vector.

    The transform is ``out[i] = sign[i] * v[perm[i]] + offset[i]``, which covers
    negation, permutation (arc / raycast swaps) and the one affine field
    (``position_x``, normalized to ``[0, 1]``, so it reflects as ``1 - x``) in a
    single gather. It is involutive by construction: ``perm`` is an involution,
    ``sign`` is constant across each swapped pair, and ``offset`` is non-zero
    only where ``perm`` is the identity.
    """

    def __init__(
        self,
        width: int,
        *,
        expose_blocked_goal: bool = False,
        expose_wall_contact_tau: bool = False,
        expose_precision_state: bool = False,
        expose_time_phase: bool = False,
        expose_commitment_state: bool = False,
    ) -> None:
        width = int(width)
        self.width = width
        self.perm = np.arange(width, dtype=np.int64)
        self.sign = np.ones(width, dtype=np.float32)
        self.offset = np.zeros(width, dtype=np.float32)
        self._claimed = np.zeros(width, dtype=bool)

        flags = {
            "expose_blocked_goal": bool(expose_blocked_goal),
            "expose_wall_contact_tau": bool(expose_wall_contact_tau),
            "expose_precision_state": bool(expose_precision_state),
            "expose_time_phase": bool(expose_time_phase),
            "expose_commitment_state": bool(expose_commitment_state),
        }
        self.exposure = flags

        self._build_base()
        cursor = GAME_STATE_CHANNELS
        for flag, block_width, negated in _OPTIONAL_TAIL_BLOCKS:
            if flag is not None and not flags[flag]:
                continue
            if cursor + block_width > width:
                raise UnclassifiedGameStateDims(
                    f"game_state width {width} is too small for the requested "
                    f"exposure blocks (block {flag!r} would end at "
                    f"{cursor + block_width})"
                )
            self._invariant(cursor, block_width)
            for off in negated:
                self._negate(cursor + off)
            cursor += block_width

        unclassified = np.flatnonzero(~self._claimed)
        if unclassified.size:
            raise UnclassifiedGameStateDims(
                f"{unclassified.size} game_state dim(s) have no mirror rule: "
                f"{unclassified.tolist()[:32]}"
                + (" ..." if unclassified.size > 32 else "")
                + ". The observation contract grew without a corresponding rule "
                "in nclone.map_augmentation.obs_mirror -- classify the new dims "
                "as negate / invariant / permute rather than letting them pass "
                "through unmirrored."
            )
        if cursor != width:
            raise UnclassifiedGameStateDims(
                f"game_state width {width} does not match the exposure blocks "
                f"declared ({cursor}). Pass the run's real exposure flags."
            )

    # -- rule primitives -------------------------------------------------

    def _claim(self, start: int, length: int) -> None:
        stop = start + length
        if stop > self.width:
            raise UnclassifiedGameStateDims(
                f"rule [{start}, {stop}) exceeds game_state width {self.width}"
            )
        if self._claimed[start:stop].any():
            overlap = start + int(np.flatnonzero(self._claimed[start:stop])[0])
            raise UnclassifiedGameStateDims(
                f"game_state dim {overlap} is claimed by two mirror rules"
            )
        self._claimed[start:stop] = True

    def _invariant(self, start: int, length: int = 1) -> None:
        self._claim(start, length)

    def _negate(self, index: int, length: int = 1) -> None:
        # Re-negating an already-claimed dim is how block rules compose (claim
        # the whole block invariant, then negate the signed entries inside it).
        if not self._claimed[index : index + length].all():
            self._claim(index, length)
        self.sign[index : index + length] = -1.0

    def _permute(self, start: int, order: Sequence[int]) -> None:
        self._claim(start, len(order))
        for out_i, in_i in enumerate(order):
            self.perm[start + out_i] = start + in_i

    # -- the contract ----------------------------------------------------

    def _build_base(self) -> None:
        # --- ninja state block (NINJA_STATE_DIM) ---
        # 0 velocity_x (negate), 1 velocity_y, 2-5 movement-state one-hots,
        # 6 airborne, 7-9 buffers, 10 jump_duration_remaining,
        # 11-13 floor/wall/ceiling contact, 14-15 floor normal (x negate),
        # 16 wall_normal (negate), 17 walled, 18-20 gravity/drag/friction,
        # 21-22 position (x is normalized to [0, 1] -> affine 1 - x),
        # 23 time_remaining.
        self._invariant(0, NINJA_STATE_DIM)
        self._negate(0)  # velocity_x
        self._negate(14)  # floor_normal_x
        self._negate(16)  # wall_normal
        self._negate(GAME_STATE_POSITION_START)  # position_x, normalized [0, 1]
        self.offset[GAME_STATE_POSITION_START] = 1.0

        # Physics-derived: horizontal RANGE, jump height, time to ground -- all
        # unsigned magnitudes.
        self._invariant(NINJA_STATE_DIM, PHYSICS_DERIVED_DIM)

        self._invariant(GAME_STATE_SWITCH_ACTIVATED_IDX)

        # Goal geometry: ninja-relative (dx, dy) to switch and exit.
        self._invariant(GAME_STATE_SWITCH_VEC_IDX, 2)
        self._negate(GAME_STATE_SWITCH_VEC_IDX)
        self._invariant(GAME_STATE_EXIT_VEC_IDX, 2)
        self._negate(GAME_STATE_EXIT_VEC_IDX)

        # 8 compass raycasts: distances are invariant, the DIRECTIONS permute.
        assert len(_RAYCAST_PERM) == GAME_STATE_RAYCAST_DIM
        self._permute(GAME_STATE_RAYCAST_IDX, _RAYCAST_PERM)

        # Free-space nav hint [dx_hop, dy_hop, cost_to_go_norm, phase].
        self._invariant(GAME_STATE_FREE_SPACE_NAV_IDX, GAME_STATE_FREE_SPACE_NAV_DIM)
        self._negate(GAME_STATE_FREE_SPACE_NAV_IDX)

        # Nearest-mine distance and closing speed toward the goal: both are
        # projections onto reflection-invariant quantities (a distance, and a
        # velocity dotted with a goal direction that flips with it).
        self._invariant(GAME_STATE_NEAREST_MINE_IDX)
        self._invariant(GAME_STATE_CLOSING_SPEED_IDX)

        # Jump-execution block (held / useless / launchpad buffer).
        self._invariant(GAME_STATE_JUMP_HELD_IDX, GAME_STATE_JUMP_EXEC_DIM)

        # Landing surface normal (nx, ny) + distance + vertical terminal speed.
        self._invariant(GAME_STATE_LANDING_NORMAL_IDX, 2)
        self._negate(GAME_STATE_LANDING_NORMAL_IDX)
        self._invariant(GAME_STATE_LANDING_DIST_IDX)
        self._invariant(GAME_STATE_VERTICAL_TERMINAL_IDX)

        # tau, frames airborne, per-frame acceleration (ax, ay).
        self._invariant(GAME_STATE_TIME_TO_CONTACT_IDX)
        self._invariant(GAME_STATE_FRAMES_AIRBORNE_IDX)
        self._invariant(GAME_STATE_ACCEL_IDX, 2)
        self._negate(GAME_STATE_ACCEL_IDX)

        # Scandots: expressed relative to the TRAVEL DIRECTION, which flips
        # under reflection -- so the block is invariant. See module docstring.
        self._invariant(GAME_STATE_SCANDOTS_IDX, GAME_STATE_SCANDOTS_DIM)

        # Four ghost-arc bundles: committed / jump-now / tap / wall-jump.
        for base in (
            GAME_STATE_GHOST_ARC_IDX,
            GAME_STATE_JUMP_ARC_IDX,
            GAME_STATE_TAP_ARC_IDX,
            GAME_STATE_WALL_ARC_IDX,
        ):
            self._build_arc_bundle(base)

        # Per-arc route-progress deltas, one scalar per arc in bundle order
        # (3 committed + 3 jump-now). Steering left becomes steering right, so
        # the +-1 entries swap within each row of three.
        assert GAME_STATE_ARC_PHI_DIM % GAME_STATE_GHOST_ARC_ARCS == 0
        phi_order = []
        for row in range(GAME_STATE_ARC_PHI_DIM // GAME_STATE_GHOST_ARC_ARCS):
            base = row * GAME_STATE_GHOST_ARC_ARCS
            phi_order.extend(
                base + a for a in reversed(range(GAME_STATE_GHOST_ARC_ARCS))
            )
        self._permute(GAME_STATE_ARC_PHI_IDX, phi_order)

        # Passive no-input arc [dx, dy, vx, vy, fatal, none].
        self._invariant(GAME_STATE_PASSIVE_ARC_IDX, GAME_STATE_PASSIVE_ARC_DIM)
        self._negate(GAME_STATE_PASSIVE_ARC_IDX)  # dx
        self._negate(GAME_STATE_PASSIVE_ARC_IDX + 2)  # vx

    def _build_arc_bundle(self, base: int) -> None:
        """Swap the ``u = -1`` / ``u = +1`` arcs and negate the signed dims.

        Forgetting the swap is the highest-cost bug in this file: the four
        bundles are 180 of the 269 v26 dims, and a wrong swap teaches the policy
        that steering left produces the right-steer outcome.
        """
        per = GAME_STATE_GHOST_ARC_PER_ARC
        n_arcs = GAME_STATE_GHOST_ARC_ARCS
        assert n_arcs * per == GAME_STATE_GHOST_ARC_DIM
        order = []
        for out_arc in range(n_arcs):
            in_arc = n_arcs - 1 - out_arc  # u = -1 <-> u = +1, u = 0 fixed
            order.extend(in_arc * per + k for k in range(per))
        self._permute(base, order)
        for out_arc in range(n_arcs):
            for off in _GHOST_ARC_NEGATED:
                self.sign[base + out_arc * per + off] = -1.0

    # -- application -----------------------------------------------------

    def __call__(self, game_state: np.ndarray) -> np.ndarray:
        """Reflect a ``(..., width)`` game_state array."""
        arr = np.asarray(game_state)
        if arr.shape[-1] != self.width:
            raise ValueError(
                f"game_state width {arr.shape[-1]} != spec width {self.width}"
            )
        out = arr[..., self.perm] * self.sign + self.offset
        return out.astype(arr.dtype, copy=False)


# --------------------------------------------------------------------------
# Spatial + vector observation keys
# --------------------------------------------------------------------------


class SpatialMirrorRule:
    """How to reflect one non-``game_state`` observation key.

    ``flip_x`` flips the x axis, which is ALWAYS ``-2`` in this contract: every
    spatial tensor is ``(C, W, H)`` or ``(W, H)`` with ``W`` the x extent. The
    ego crops are odd-sized (21, 31) so the ninja occupies the exact centre cell
    and the flip is clean -- do not change them to even sizes without revisiting.
    """

    __slots__ = ("flip_x", "negate_channels", "tile_remap", "negate_last_axis")

    def __init__(
        self,
        *,
        flip_x: bool = False,
        negate_channels: Tuple[int, ...] = (),
        tile_remap: bool = False,
        negate_last_axis: Tuple[int, ...] = (),
    ) -> None:
        self.flip_x = flip_x
        self.negate_channels = tuple(negate_channels)
        self.tile_remap = tile_remap
        self.negate_last_axis = tuple(negate_last_axis)

    def __call__(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value)
        if self.flip_x:
            arr = np.flip(arr, axis=-2)
        if self.tile_remap:
            arr = mirror_tile_ids(arr)
        if self.negate_channels or self.negate_last_axis:
            arr = np.array(arr, copy=True)
            for ch in self.negate_channels:
                arr[ch] = -arr[ch]
            for idx in self.negate_last_axis:
                arr[..., idx] = -arr[..., idx]
        return np.ascontiguousarray(arr)


#: Every non-``game_state`` key in the v26 policy observation contract.
#: A key absent from the live observation is skipped; a key PRESENT but absent
#: from this table is a hard error (see :class:`ObservationMirror`).
SPATIAL_MIRROR_SPEC: Dict[str, SpatialMirrorRule] = {
    # Tile IDs must be remapped, not just flipped.
    "tile_grid": SpatialMirrorRule(flip_x=True, tile_remap=True),
    "ego_crop": SpatialMirrorRule(flip_x=True, tile_remap=True),
    # Scalar density / SDF / cost fields: a plain x flip.
    "entity_overlays": SpatialMirrorRule(flip_x=True),
    "hazard_map": SpatialMirrorRule(flip_x=True),
    # Drone SDF is a scalar distance field like hazard_map, so the RULE is a
    # plain x flip -- correct given a level mirror that mirrors drone motion.
    #
    # CAVEAT (measured 2026-08-03, pre-existing, NOT fixed here): the level
    # mirror does not currently reflect drone trajectories. Spawn positions
    # mirror exactly, but the paths diverge from frame 1 -- up to 312 px over
    # 200 frames on `official/R/173 variational mechanics`. Root cause found so
    # far is `transforms._ORIENT_FLIP`, which is provably wrong for the four
    # AXIS-ALIGNED orientations: checked against the sim's own
    # `map_orientation_to_vector`, a horizontal mirror must map 0<->4 (right /
    # left) and leave 2 and 6 (down / up) alone, but the table leaves 0 and 4
    # untouched and swaps 2<->6. The four diagonals are right. Its comment shows
    # the misconception -- it calls 0 "up" and 4 "down", but the sim says 0 is
    # (+1, 0) and 4 is (-1, 0). This affects every type in `_ORIENT_TYPES`
    # (5, 6, 8, 10, 11, 14, 15, 20, 23, 26), not just drones; it went unnoticed
    # because mines carry no orientation byte, so a mines-only corpus cannot
    # exercise it. Correcting the orientation shrinks the drone mismatch (312 ->
    # 240 px, and 252 -> 36 px on `R/034 oh, they'll swarm ya`) but does NOT
    # zero it, so at least one further defect remains.
    #
    # Consequence: SYNTHESISING a mirrored observation (the equivariance /
    # aux-loss tiers) will place drones where the mirrored level does not have
    # them. Mirroring at the LEVEL level is unaffected in kind -- it still yields
    # a valid level, just not the true reflection of the original.
    "drone_hazard_map": SpatialMirrorRule(flip_x=True),
    # Phase 1. `threat_ttl` is four scalar fields (SDF + TTL per tier), so a
    # plain x flip -- and it inherits the drone caveat above in full, plus more
    # of it: the field covers thwumps, rockets, death balls and eye bats too, and
    # `_ORIENT_FLIP` is wrong for every type in `_ORIENT_TYPES`. Synthesised
    # mirrors of this key are therefore NOT trustworthy on entity-bearing maps.
    "threat_ttl": SpatialMirrorRule(flip_x=True),
    # `solid_map` -- occupancy masks plus TWO vector fields: the one-way normal
    # (ch 3/4) and the launch-pad boost (ch 8/9). Both x components must be
    # negated, or every mirrored one-way faces the wrong way and every mirrored
    # launch pad fires the wrong way -- while still reading as geometrically
    # valid, which is what makes the error invisible.
    "solid_map": SpatialMirrorRule(flip_x=True, negate_channels=(3, 8)),
    "fall_lethality_map": SpatialMirrorRule(flip_x=True),
    "cost_to_go_map": SpatialMirrorRule(flip_x=True),
    "visitation_map": SpatialMirrorRule(flip_x=True),
    "frontier_map": SpatialMirrorRule(flip_x=True),
    "ego_crop_mines": SpatialMirrorRule(flip_x=True),
    "ego_crop_mines_fine": SpatialMirrorRule(flip_x=True),
    "ego_crop_fall_fine": SpatialMirrorRule(flip_x=True),
    # Route crops are VECTOR fields: flip the window AND negate the ux channel.
    "ego_crop_route": SpatialMirrorRule(flip_x=True, negate_channels=(0,)),
    "ego_crop_route_fine": SpatialMirrorRule(flip_x=True, negate_channels=(0,)),
    # Coarse crop: [0] solid, [1] deadly, [2] safe, [3] objective marker,
    # then optional route (ux, uy) at 4/5. The rule is chosen per-shape in
    # ObservationMirror because the route channels are optional.
    "ego_crop_coarse": SpatialMirrorRule(flip_x=True),
    # Flat vectors.
    "mine_proximity": SpatialMirrorRule(),  # handled by stride, see below
    # waypoint_nav = [ux, uy, ctg_norm, phase, leg_len, tol, airborne_leg, nx, ny]
    "waypoint_nav": SpatialMirrorRule(negate_last_axis=(0, 7)),
    # lattice_nav = [dx_hop, dy_hop, cost_to_go_norm, phase]
    "lattice_nav": SpatialMirrorRule(negate_last_axis=(0,)),
}
assert WAYPOINT_NAV_DIM == 9, "waypoint_nav layout changed; recheck its mirror rule"
# The solid_map rule negates channel 3 by index. Pin the layout so a channel
# reorder fails here instead of silently mirroring the wrong component -- an
# unnegated normal is geometrically valid and therefore invisible.
assert (
    SOLID_MAP_CHANNELS,
    SOLID_MAP_CH_ONEWAY_NX,
    SOLID_MAP_CH_ONEWAY_NY,
    SOLID_MAP_CH_LAUNCHPAD_BX,
    SOLID_MAP_CH_LAUNCHPAD_BY,
) == (
    11, 3, 4, 8, 9,
), "solid_map layout changed; recheck which channels carry a vector's x"
assert THREAT_TTL_CHANNELS == 4, "threat_ttl layout changed; recheck its mirror rule"
assert LATTICE_NAV_DIM == 4, "lattice_nav layout changed; recheck its mirror rule"

#: ``ego_crop_coarse`` with the optional route channels appended.
_EGO_CROP_COARSE_WITH_ROUTE = SpatialMirrorRule(flip_x=True, negate_channels=(4,))

#: Keys deliberately NOT mirrored geometrically -- they permute by the action map.
_ACTION_AXIS_KEYS = ("action_mask", "duration_mask")

#: Keys that carry no spatial or directional content at all.
_PASSTHROUGH_KEYS = frozenset({"switch_activated", "gold_fraction_remaining"})


class ObservationMirror:
    """Reflect a whole policy observation dict.

    Construct once per run from the observation space plus the run's obs-exposure
    flags, then call it per sample/batch. Constructing it validates the
    ``game_state`` contract (see :class:`UnclassifiedGameStateDims`).
    """

    def __init__(
        self,
        game_state_width: int,
        *,
        exposure: Optional[Mapping[str, Any]] = None,
        strict: bool = True,
    ) -> None:
        exposure = dict(exposure or {})
        self.game_state = GameStateMirror(
            game_state_width,
            expose_blocked_goal=bool(exposure.get("expose_blocked_goal", False)),
            expose_wall_contact_tau=bool(
                exposure.get("expose_wall_contact_tau", False)
            ),
            expose_precision_state=bool(
                exposure.get("expose_precision_state", False)
            ),
            expose_time_phase=bool(exposure.get("expose_time_phase", False)),
            expose_commitment_state=bool(
                exposure.get("expose_commitment_state", False)
            ),
        )
        self.strict = bool(strict)

    @classmethod
    def from_observation_space(
        cls,
        observation_space,
        *,
        exposure: Optional[Mapping[str, Any]] = None,
        strict: bool = True,
    ) -> "ObservationMirror":
        """Build from a gym ``spaces.Dict`` (the usual training entry point)."""
        width = int(observation_space["game_state"].shape[-1])
        return cls(width, exposure=exposure, strict=strict)

    def _rule_for(self, key: str, value: np.ndarray) -> Optional[SpatialMirrorRule]:
        if key == "ego_crop_coarse":
            arr = np.asarray(value)
            has_route = arr.shape[-3] > 4 if arr.ndim >= 3 else False
            return (
                _EGO_CROP_COARSE_WITH_ROUTE if has_route
                else SPATIAL_MIRROR_SPEC["ego_crop_coarse"]
            )
        return SPATIAL_MIRROR_SPEC.get(key)

    def __call__(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in obs.items():
            if key == "game_state":
                out[key] = self.game_state(value)
                continue
            if key in _ACTION_AXIS_KEYS:
                arr = np.asarray(value)
                # action_mask is (..., 6); duration_mask is (..., 6, K) -- in
                # both the ACTION axis is the one of length len(MIRROR_ACTION_MAP)
                # counted from the front of the trailing block.
                axis = -1 if arr.shape[-1] == len(MIRROR_ACTION_MAP) else -2
                out[key] = np.take(arr, _MIRROR_ACTION_LUT, axis=axis)
                continue
            if key == "mine_proximity":
                arr = np.asarray(value)
                shaped = arr.reshape(*arr.shape[:-1], -1, MINE_PROXIMITY_FEATURES)
                shaped = np.array(shaped, copy=True)
                # [dx_hat, dy_hat, proximity, closing, ttc]; only the bearing's
                # x component is signed. Slots are ordered nearest-first and
                # distance is reflection-invariant, so the order is preserved.
                shaped[..., 0] = -shaped[..., 0]
                out[key] = shaped.reshape(arr.shape)
                continue
            if key == "threat_proximity":
                arr = np.asarray(value)
                shaped = arr.reshape(*arr.shape[:-1], -1, THREAT_PROX_FEATURES)
                shaped = np.array(shaped, copy=True)
                # [dx_hat, dy_hat, proximity, closing, imminence, vx, vy, spin].
                # TWO signed x components, not one: the bearing AND the threat's
                # own velocity. `closing` is a scalar projection of two vectors
                # that both reflect, so it is invariant; `spin` is a magnitude.
                shaped[..., THREAT_PROX_IDX_DX] = -shaped[..., THREAT_PROX_IDX_DX]
                shaped[..., THREAT_PROX_IDX_VX] = -shaped[..., THREAT_PROX_IDX_VX]
                out[key] = shaped.reshape(arr.shape)
                continue
            if key in _PASSTHROUGH_KEYS or key.startswith("_"):
                out[key] = value
                continue
            rule = self._rule_for(key, value)
            if rule is None:
                if self.strict:
                    raise KeyError(
                        f"observation key {key!r} has no mirror rule. Add one to "
                        "nclone.map_augmentation.obs_mirror.SPATIAL_MIRROR_SPEC "
                        "-- passing an unclassified key through unmirrored "
                        "silently poisons every mirrored sample."
                    )
                out[key] = value
                continue
            out[key] = rule(value)
        return out
