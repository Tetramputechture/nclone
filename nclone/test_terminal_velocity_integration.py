"""
Integration tests for terminal velocity death prediction in environment.

Tests that terminal velocity predictor is properly initialized and integrated
with the environment and action masking system, including graph optimization.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTerminalVelocityIntegration(unittest.TestCase):
    """Test terminal velocity predictor integration with environment."""

    def test_predictor_accepts_graph_data(self):
        """Test that predictor accepts graph_data parameter and stores graph components."""
        from nclone.terminal_velocity_predictor import TerminalVelocityPredictor
        from nclone.nsim import Simulator
        from nclone.sim_config import SimConfig
        
        # Create minimal simulation
        sim = Simulator(SimConfig())
        
        # Test 1: Predictor works without graph data
        predictor_no_graph = TerminalVelocityPredictor(sim, graph_data=None)
        self.assertIsNone(predictor_no_graph.graph_data)
        self.assertEqual(len(predictor_no_graph.adjacency), 0)
        self.assertEqual(len(predictor_no_graph.reachable), 0)
        # Subcell lookup is always initialized (not None) for potential use
        self.assertIsNotNone(predictor_no_graph.subcell_lookup)
        
        # Test 2: Predictor works with graph data
        graph_data = {
            "adjacency": {(100, 100): [], (112, 100): []},
            "reachable": {(100, 100), (112, 100)}
        }
        predictor_with_graph = TerminalVelocityPredictor(sim, graph_data=graph_data)
        self.assertIsNotNone(predictor_with_graph.graph_data)
        self.assertEqual(len(predictor_with_graph.adjacency), 2)
        self.assertEqual(len(predictor_with_graph.reachable), 2)
        # Subcell lookup may or may not load depending on data file availability
        # Just verify the attribute exists
        self.assertTrue(hasattr(predictor_with_graph, 'subcell_lookup'))

    def test_predictor_initialization_structure(self):
        """Test that predictor initializes with correct structure and attributes."""
        from nclone.terminal_velocity_predictor import TerminalVelocityPredictor
        from nclone.nsim import Simulator
        from nclone.sim_config import SimConfig
        
        sim = Simulator(SimConfig())
        predictor = TerminalVelocityPredictor(sim, graph_data=None)
        
        # Verify core attributes exist
        self.assertTrue(hasattr(predictor, 'sim'))
        self.assertTrue(hasattr(predictor, 'lookup_table'))
        self.assertTrue(hasattr(predictor, 'simulator'))
        self.assertTrue(hasattr(predictor, 'stats'))
        self.assertTrue(hasattr(predictor, 'graph_data'))
        self.assertTrue(hasattr(predictor, 'adjacency'))
        self.assertTrue(hasattr(predictor, 'reachable'))
        self.assertTrue(hasattr(predictor, 'subcell_lookup'))
        
        # Verify stats tracking
        self.assertEqual(predictor.stats.tier1_queries, 0)
        self.assertEqual(predictor.stats.tier2_queries, 0)
        self.assertEqual(predictor.stats.tier3_queries, 0)
        self.assertEqual(predictor.stats.build_time_ms, 0.0)
        self.assertEqual(predictor.stats.lookup_table_size, 0)

    def test_graph_data_storage(self):
        """Test that graph data is properly stored in predictor."""
        from nclone.terminal_velocity_predictor import TerminalVelocityPredictor
        from nclone.nsim import Simulator
        from nclone.sim_config import SimConfig
        
        sim = Simulator(SimConfig())
        
        # Create test graph data
        test_adjacency = {
            (100, 100): [(112, 100), (100, 112)],
            (112, 100): [(100, 100)],
            (100, 112): [(100, 100)]
        }
        test_reachable = {(100, 100), (112, 100), (100, 112), (124, 100)}
        
        graph_data = {
            "adjacency": test_adjacency,
            "reachable": test_reachable
        }
        
        predictor = TerminalVelocityPredictor(sim, graph_data=graph_data)
        
        # Verify graph data is stored correctly
        self.assertEqual(predictor.graph_data, graph_data)
        self.assertEqual(predictor.adjacency, test_adjacency)
        self.assertEqual(predictor.reachable, test_reachable)

    def test_stats_structure(self):
        """Test that stats are properly tracked."""
        from nclone.terminal_velocity_predictor import TerminalVelocityPredictor, TerminalVelocityPredictorStats
        from nclone.nsim import Simulator
        from nclone.sim_config import SimConfig
        
        sim = Simulator(SimConfig())
        predictor = TerminalVelocityPredictor(sim, graph_data=None)
        
        # Verify stats is correct type
        self.assertIsInstance(predictor.stats, TerminalVelocityPredictorStats)
        
        # Verify get_stats returns the stats object
        stats = predictor.get_stats()
        self.assertIsInstance(stats, TerminalVelocityPredictorStats)
        self.assertEqual(stats, predictor.stats)


def run_tests():
    """Run all integration tests."""
    unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()

