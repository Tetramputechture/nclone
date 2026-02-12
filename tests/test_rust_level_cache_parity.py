"""
Parity tests for Rust LevelCache vs Python LevelBasedPathDistanceCache.

Compares get_distance, get_phi, and get_next_hop between the two implementations
when built from the same graph input. Skips if nclone_native lacks build_from_graph.
"""

import os
import sys
import unittest
import random
import numpy as np

# Prefer loading nclone_native from build dir for dev (has build_from_graph)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NCLONE_ROOT = os.path.join(_SCRIPT_DIR, "..")
_BUILD_SO = os.path.join(_NCLONE_ROOT, "nclone-native", "target", "debug", "libnclone_native.so")


def _load_nclone_native():
    """Load nclone_native, preferring build dir if it has build_from_graph."""
    import importlib.util
    # Try normal import first
    try:
        import nclone_native
        if hasattr(nclone_native.LevelCache, "build_from_graph"):
            return nclone_native
    except ImportError:
        pass
    # Try loading from build dir
    if os.path.isfile(_BUILD_SO):
        spec = importlib.util.spec_from_file_location("nclone_native", _BUILD_SO)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


NCLONE_NATIVE = _load_nclone_native()
RUST_AVAILABLE = (
    NCLONE_NATIVE is not None and hasattr(NCLONE_NATIVE.LevelCache, "build_from_graph")
)


@unittest.skipIf(not RUST_AVAILABLE, "nclone_native.LevelCache.build_from_graph not available")
class TestRustLevelCacheParity(unittest.TestCase):
    """Parity tests between Rust and Python level cache."""

    def setUp(self):
        """Set up simple test graph and build both caches."""
        sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
        from nclone.graph.reachability.graph_serializer import serialize_adjacency_for_rust
        from nclone.graph.reachability.level_data_helpers import extract_goal_positions
        from nclone.graph.level_data import LevelData
        from nclone.graph.reachability.path_distance_cache import LevelBasedPathDistanceCache
        from nclone.graph.reachability.mine_proximity_cache import MineProximityCostCache

        # Simple linear graph: (0,0) -> (12,0) -> (24,0) -> (36,0)
        self.adjacency = {
            (0, 0): [((12, 0), 12.0)],
            (12, 0): [((0, 0), 12.0), ((24, 0), 12.0)],
            (24, 0): [((12, 0), 12.0), ((36, 0), 12.0)],
            (36, 0): [((24, 0), 12.0)],
        }
        self.base_adjacency = self.adjacency
        self.physics_cache = {
            (0, 0): {"grounded": True, "walled": False},
            (12, 0): {"grounded": True, "walled": False},
            (24, 0): {"grounded": True, "walled": False},
            (36, 0): {"grounded": True, "walled": False},
        }

        # Level data: EXIT_SWITCH=4 at (60,24), EXIT_DOOR=3 at (72,24) world space
        from nclone.constants.entity_types import EntityType

        self.level_data = LevelData(
            start_position=(24, 24),
            tiles=np.zeros((23, 42), dtype=np.int32),
            entities=[
                {"type": EntityType.EXIT_SWITCH, "x": 60, "y": 24, "active": True},
                {"type": EntityType.EXIT_DOOR, "x": 72, "y": 24},
            ],
            switch_states={},
        )

        goals = extract_goal_positions(self.level_data)
        mine_prox = {}
        serialized = serialize_adjacency_for_rust(
            self.adjacency, self.physics_cache, goals, mine_prox
        )

        self.rust_cache = NCLONE_NATIVE.LevelCache.build_from_graph(
            node_positions=serialized["node_positions"],
            edges=serialized["edges"],
            physics_grounded=serialized["physics_grounded"],
            goals=serialized["goals"],
        )

        self.py_cache = LevelBasedPathDistanceCache()
        mine_cache = MineProximityCostCache()
        graph_data = {
            "node_physics": self.physics_cache,
            "spatial_hash": None,
            "subcell_lookup": None,
        }
        self.py_cache.build_cache(
            self.level_data,
            self.adjacency,
            self.base_adjacency,
            graph_data=graph_data,
            mine_proximity_cache=mine_cache,
            mine_sdf=None,
        )

        self.node_positions = serialized["node_positions"]

    def test_get_distance_parity(self):
        """Node positions: Rust and Python get_distance match within tolerance."""
        # Query from each graph node position (both caches have data for nodes)
        for (nx, ny) in self.node_positions:
            wx = nx + 24
            wy = ny + 24
            for goal_id in ("switch", "exit", "exit_switch_0", "exit_door_0"):
                goal_pos = self.py_cache._goal_id_to_goal_pos.get(goal_id)
                if goal_pos is None:
                    continue
                py_dist = self.py_cache.get_distance((nx, ny), goal_pos, goal_id)
                rust_dist = self.rust_cache.get_distance(wx, wy, goal_id)
                if py_dist == float("inf") and rust_dist == float("inf"):
                    continue
                self.assertAlmostEqual(
                    py_dist, rust_dist, places=2,
                    msg=f"get_distance mismatch at node ({nx},{ny}) goal_id={goal_id}: "
                    f"py={py_dist} rust={rust_dist}",
                )

    def test_get_phi_parity(self):
        """Phi values match between Rust and Python (rm_state 0=switch, 1=exit)."""
        random.seed(43)
        for _ in range(50):
            x = random.uniform(24, 500)
            y = random.uniform(24, 300)
            for rm_state in (0, 1):
                py_phi = self.py_cache.get_phi(x, y, rm_state)
                rust_phi = self.rust_cache.get_phi(x, y, rm_state)
                self.assertAlmostEqual(
                    py_phi, rust_phi, places=4,
                    msg=f"get_phi mismatch at ({x:.1f},{y:.1f}) rm_state={rm_state}: "
                    f"py={py_phi} rust={rust_phi}",
                )

    def test_get_next_hop_direction_parity(self):
        """Next-hop directions match (Python returns node, Rust returns normalized vec)."""
        # For nodes on the path, direction toward goal should be consistent
        for i, (nx, ny) in enumerate(self.node_positions):
            wx = nx + 24
            wy = ny + 24
            for goal_id in ("switch", "exit"):
                py_nh = self.py_cache.get_next_hop((nx, ny), goal_id)
                rust_nh = self.rust_cache.get_next_hop(wx, wy, goal_id)
                if py_nh is None and rust_nh is None:
                    continue
                if py_nh is not None and rust_nh is not None:
                    # Python: next node pos. Rust: normalized (dx, dy)
                    # Compare direction: from (nx,ny) to py_nh should match rust_nh
                    dx = (py_nh[0] - nx)
                    dy = (py_nh[1] - ny)
                    mag = (dx * dx + dy * dy) ** 0.5
                    if mag > 0.01:
                        py_dir = (dx / mag, dy / mag)
                        self.assertAlmostEqual(
                            py_dir[0], rust_nh[0], places=3,
                            msg=f"next_hop dx at node {i} goal={goal_id}",
                        )
                        self.assertAlmostEqual(
                            py_dir[1], rust_nh[1], places=3,
                            msg=f"next_hop dy at node {i} goal={goal_id}",
                        )
