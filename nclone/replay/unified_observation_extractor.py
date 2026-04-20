"""Unified observation extraction for replay parsing.

Produces Layer 1 policy observations (``tile_grid``, ``entity_overlays``,
``hazard_map``, ``game_state``, ``action_mask``) plus legacy JSONL frame data.

The previous implementation rebuilt the full observation pipeline (graph,
reachability, mine SDF, spatial_context, ...) on top of a raw simulator
instance. Since the policy now only consumes Layer 1 tensors, we rely on
``NppEnvironment`` (via ``ReplayExecutor``) to produce observations so replays
and the live env stay byte-for-byte compatible.

The legacy ninja/entity frame-data output is retained for tools that still
consume JSONL exports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from .replay_executor import ReplayExecutor

logger = logging.getLogger(__name__)


class UnifiedObservationExtractor:
    """Replay-side observation extractor aligned with training.

    Observations produced by :meth:`extract_processed_observation` exactly
    match ``NppEnvironment.step`` output (Layer 1 policy contract).
    """

    def __init__(self, enable_visual_observations: bool = False) -> None:
        # ``enable_visual_observations`` is accepted for API compatibility with
        # older callers. Layer 1 observations never include screen pixels, so
        # this flag only toggles render-mode wiring for tools that grab frames
        # via ``env.render()`` themselves.
        self.enable_visual_observations = enable_visual_observations

        self._executor = ReplayExecutor(
            enable_rendering=enable_visual_observations,
            frame_skip=1,
        )

    # ------------------------------------------------------------------
    # Replay simulation
    # ------------------------------------------------------------------
    def execute_replay(
        self,
        map_data: bytes,
        input_sequence: list,
    ) -> list:
        """Return per-frame Layer 1 observations for a full replay."""
        return self._executor.execute_replay(map_data, input_sequence)

    def close(self) -> None:
        """Release the underlying environment."""
        self._executor.close()

    # ------------------------------------------------------------------
    # Per-frame helpers used by SimulationManager (legacy JSONL path)
    # ------------------------------------------------------------------
    def extract_legacy_frame_data(
        self,
        sim,
        frame_number: int,
        level_id: str,
        session_id: str,
        npp_level_id: Optional[int] = None,
        level_name: str = "",
    ) -> Dict[str, Any]:
        """Extract a single simulator frame's state in the legacy JSONL format.

        This walks the simulator directly (no observation pipeline) so that
        tools exporting raw gameplay JSONL continue to work without depending
        on graph / reachability infrastructure.
        """
        ninja = sim.ninja

        # Ninja state used by downstream tools for backward compatibility.
        position = {"x": float(ninja.xpos), "y": float(ninja.ypos)}
        velocity = {"x": float(ninja.xspeed), "y": float(ninja.yspeed)}

        legacy_ninja_state = {
            "position": position,
            "velocity": velocity,
            "state": int(ninja.state),
            "airborne": bool(ninja.airborn),
            "walled": bool(ninja.walled),
            "jump_duration": int(ninja.jump_duration),
            "applied_gravity": float(ninja.applied_gravity),
            "applied_drag": float(ninja.applied_drag),
            "applied_friction": float(ninja.applied_friction),
            "floor_normal": {
                "x": float(ninja.floor_normalized_x),
                "y": float(ninja.floor_normalized_y),
            },
            "ceiling_normal": {
                "x": float(ninja.ceiling_normalized_x),
                "y": float(ninja.ceiling_normalized_y),
            },
            "buffers": {
                "jump": int(ninja.jump_buffer),
                "floor": int(ninja.floor_buffer),
                "wall": int(ninja.wall_buffer),
                "launch_pad": int(ninja.launch_pad_buffer),
            },
            "contact_counts": {
                "floor": int(ninja.floor_count),
                "wall": int(ninja.wall_count),
                "ceiling": int(ninja.ceiling_count),
            },
        }

        entities = []
        for entity_type, entity_list in sim.entity_dic.items():
            for entity in entity_list:
                entity_data = {
                    "type": self._get_entity_type_name(entity_type),
                    "position": {"x": float(entity.xpos), "y": float(entity.ypos)},
                    "active": bool(getattr(entity, "active", True)),
                }
                if hasattr(entity, "state"):
                    entity_data["state"] = entity.state
                if hasattr(entity, "radius"):
                    entity_data["radius"] = float(entity.radius)
                entities.append(entity_data)

        frame_data = {
            "timestamp": frame_number * (1.0 / 60.0),
            "level_id": level_id,
            "frame_number": frame_number,
            "player_state": legacy_ninja_state,
            "player_inputs": {
                "left": ninja.hor_input == -1,
                "right": ninja.hor_input == 1,
                "jump": ninja.jump_input == 1,
                "restart": False,
            },
            "entities": entities,
            "level_bounds": {"width": 1056, "height": 600},
            "meta": {
                "session_id": session_id,
                "player_id": "binary_replay",
                "quality_score": 0.8,
                "completion_status": "in_progress",
                "npp_level_id": npp_level_id,
                "npp_level_name": level_name,
            },
        }

        if ninja.state == 6:  # dead
            frame_data["meta"]["completion_status"] = "failed"
        elif ninja.state == 8:  # celebrating (completed)
            frame_data["meta"]["completion_status"] = "completed"
            frame_data["meta"]["quality_score"] = 0.9

        return frame_data

    @staticmethod
    def _get_entity_type_name(entity_type: int) -> str:
        """Convert entity type ID to name used by legacy JSONL consumers."""
        type_mapping = {
            1: "mine",
            2: "gold",
            3: "exit_door",
            4: "exit_switch",
            5: "door",
            6: "locked_door",
            8: "trap_door",
            10: "launch_pad",
            11: "one_way_platform",
            14: "drone",
            17: "bounce_block",
            20: "thwump",
            21: "toggle_mine",
            25: "death_ball",
            26: "mini_drone",
        }
        return type_mapping.get(entity_type, f"unknown_{entity_type}")

    # Backwards-compatible reset hook — callers can still invoke this between
    # replays to clear any transient state.
    def reset(self) -> None:
        # Nothing to reset explicitly; executor resets per replay.
        # Kept for API compatibility.
        _ = np  # silence 'unused import' for minimal implementations
