"""Single owner of every debug-overlay flag.

Replaces the ~200 lines of per-keypress overlay handling in the old
``test_environment.py`` with one declarative table. The dev panel builds toggle
controls from :data:`OverlayManager.SIMPLE` + the data-fed overlays, and
:meth:`OverlayManager.apply` pushes the current flags (and any per-frame data the
overlay needs) onto the env once per frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


class OverlayManager:
    # Simple toggles: (name, label, env setter method, pygame key, key label).
    # These just forward a bool to the env; no per-frame data feeding.
    SIMPLE: List[Tuple[str, str, str, int, str]] = [
        ("tile_rendering", "Tiles", "set_tile_rendering_enabled", pygame.K_l, "L"),
        ("grid", "Grid", "set_grid_debug_enabled", pygame.K_c, "C"),
        ("tile_types", "Tile types", "set_tile_types_debug_enabled", pygame.K_i, "I"),
        ("action_mask", "Action mask", "set_action_mask_debug_enabled", pygame.K_w, "W"),
        ("mine_sdf", "Mine SDF", "set_mine_sdf_debug_enabled", pygame.K_m, "M"),
        (
            "jump_cost_to_go",
            "Jump cost-to-go",
            "set_jump_cost_to_go_debug_enabled",
            pygame.K_j,
            "J",
        ),
        # Audit II P0-1 + v13/v14: the ballistic ghost arcs the game_state
        # bundles are computed from, with contact class/normal +
        # tau/impact/clear-width annotations. Airborne = the 3 committed
        # steering arcs (u = -1/0/+1); grounded = BOTH counterfactual takeoff
        # rows (bright/solid = v13 jump-now full-hold, dim/thin = v14
        # tap-jump, jf=0).
        ("ghost_arcs", "Ghost/takeoff arcs", "set_ghost_arcs_debug_enabled", pygame.K_t, "T"),
        # v15: the 8 compass surface-distance rays the game_state RAYCAST block
        # carries ([E,NE,N,NW,W,SW,S,SE]); drawn to the exact normalized obs value.
        ("raycasts", "Cardinal raycasts", "set_raycasts_debug_enabled", pygame.K_k, "K"),
        ("hazard_combo", "Hazard combo", "set_hazard_debug_combo_enabled", pygame.K_h, "H"),
        ("reachability", "Reachability", "set_reachability_overlay_enabled", pygame.K_y, "Y"),
    ]

    # Data-fed overlays surfaced as toggles too (name, label, key, key label).
    # All three are drawn from the env's own graph via set_path_aware_data, so
    # they share the per-frame feed in _apply_graph. "reward_path" shows the
    # geometric A* path-distance route the RL reward (PBRS / HWM shaping) actually
    # uses -- via set_show_paths_to_goals / CachedPathDistanceCalculator -- NOT the
    # analytical reachability path, so the overlay matches what the agent trains on.
    GRAPH = [
        ("reward_path", "Reward path (PBRS)", pygame.K_p, "P"),
        ("adjacency", "Adjacency graph", pygame.K_a, "A"),
        ("blocked", "Blocked entities", pygame.K_b, "B"),
    ]
    HEAVY = [
        ("value_function", "Value function", pygame.K_f, "F"),
    ]

    def __init__(self, args=None, demo_system=None):
        self.args = args
        self.demo_system = demo_system  # set when --demo-replays given
        # tile_rendering defaults ON (matches the renderer default); rest OFF.
        self.flags: Dict[str, bool] = {name: False for name, *_ in self.SIMPLE}
        self.flags["tile_rendering"] = True
        for name, *_ in self.GRAPH:
            self.flags[name] = False
        for name, *_ in self.HEAVY:
            self.flags[name] = False
        if demo_system is not None:
            self.flags["demo_checkpoints"] = False
        self._value_pbrs_cache: Dict[str, object] = {}

    # -- toggle API --------------------------------------------------------

    def is_on(self, name: str) -> bool:
        return self.flags.get(name, False)

    def toggle(self, name: str) -> None:
        self.flags[name] = not self.flags.get(name, False)

    def toggle_specs(self) -> List[Tuple[str, str, int, str]]:
        """(name, label, key, key_label) for every toggle, in display order."""
        specs = [(n, lbl, key, kl) for n, lbl, _setter, key, kl in self.SIMPLE]
        specs += [(n, lbl, key, kl) for n, lbl, key, kl in self.GRAPH]
        specs += [(n, lbl, key, kl) for n, lbl, key, kl in self.HEAVY]
        if self.demo_system is not None:
            specs.append(("demo_checkpoints", "Demo checkpoints", pygame.K_q, "Q"))
        return specs

    # -- per-frame application --------------------------------------------

    def apply(self, env) -> None:
        for name, _label, setter, _key, _kl in self.SIMPLE:
            try:
                getattr(env, setter)(self.flags[name])
            except Exception:
                pass

        self._apply_graph(env)
        self._apply_value_function(env)
        self._apply_demo_checkpoints(env)

    def _level_name(self, env) -> Optional[str]:
        name = getattr(getattr(env, "map_loader", None), "current_map_name", None)
        if name is None and getattr(self.args, "map", None):
            name = Path(self.args.map).stem
        return name

    def _apply_graph(self, env) -> None:
        adj = self.flags.get("adjacency", False)
        blk = self.flags.get("blocked", False)
        # reward_path draws the geometric A* route the RL reward uses; the renderer
        # computes it from the same graph + CachedPathDistanceCalculator and honours
        # the switch->exit phase, so it mirrors training (HWM / PBRS) shaping.
        rew = self.flags.get("reward_path", False)
        try:
            env.set_adjacency_graph_debug_enabled(adj)
            env.set_blocked_entities_debug_enabled(blk)
            env.set_show_paths_to_goals(rew)
        except Exception:
            return
        if not (adj or blk or rew):
            return
        # Feed the env's own graph (built during reset) to the overlay renderer.
        try:
            if hasattr(env, "_update_graph_from_env_state"):
                env._update_graph_from_env_state()
            graph_data = getattr(env, "current_graph_data", None)
            if graph_data:
                env.set_path_aware_data(
                    graph_data=graph_data,
                    entity_mask=None,
                    level_data=env.level_data,
                    cached_paths=None,
                )
        except Exception:
            pass

    def _apply_value_function(self, env) -> None:
        on = self.flags.get("value_function", False)
        try:
            env.set_value_function_debug_enabled(on)
        except Exception:
            return
        if not on:
            try:
                env.set_value_function_pbrs(None)
            except Exception:
                pass
            return
        level_name = self._level_name(env)
        if not level_name:
            self.flags["value_function"] = False
            print("Value function: cannot determine level name")
            return
        try:
            if level_name not in self._value_pbrs_cache:
                from ..value_function.two_phase_pbrs import create_two_phase_pbrs

                self._value_pbrs_cache[level_name] = create_two_phase_pbrs(level_name)
            env.set_value_function_pbrs(self._value_pbrs_cache[level_name])
        except FileNotFoundError:
            self.flags["value_function"] = False
            print(
                f"Value function data not found for {level_name}. Precompute with "
                "python -m nclone.value_function.precompute_cli --map-path <level.map>"
            )
        except Exception as exc:
            self.flags["value_function"] = False
            print(f"Value function load failed: {exc}")

    def _apply_demo_checkpoints(self, env) -> None:
        if self.demo_system is None:
            return
        on = self.flags.get("demo_checkpoints", False)
        try:
            env.set_demo_checkpoints_debug_enabled(on)
        except Exception:
            return
        if not on:
            return
        level_name = self._level_name(env)
        if not level_name:
            return
        cache = self.demo_system["checkpoint_cache"]
        if self.demo_system.get("cached_level_name") != level_name:
            try:
                seeder = self.demo_system["seeder"]
                demo_files = seeder.find_matching_demos(level_name)
                checkpoints = []
                for demo_file in demo_files:
                    checkpoints.extend(seeder._extract_positions_from_demo_simple(demo_file))
                cache[level_name] = checkpoints
                self.demo_system["cached_level_name"] = level_name
                print(f"Demo checkpoints: {len(checkpoints)} for {level_name}")
            except Exception as exc:
                print(f"Demo checkpoint load failed: {exc}")
                self.flags["demo_checkpoints"] = False
                return
        try:
            env.set_demo_checkpoints_data(cache.get(level_name, []))
        except Exception:
            pass
