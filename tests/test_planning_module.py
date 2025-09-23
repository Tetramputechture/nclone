"""
Test Suite for nclone Planning Module

This test suite validates the planning components that were moved from npp-rl
to nclone, including subgoals, completion planner, and prioritizer.
"""

import unittest
import time
import math
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Tuple

import numpy as np
import torch

# Import the planning components
from nclone.planning import (
    Subgoal,
    NavigationSubgoal,
    SwitchActivationSubgoal,
    CompletionStrategy,
    CompletionStep,
    LevelCompletionPlanner,
    SubgoalPrioritizer,
)
from nclone.constants.entity_types import EntityType


class MockLevelData:
    """Mock level data for testing using actual NppEnvironment data structures."""

    def __init__(self):
        self.level_id = "test_level_001"

        # Use entities structure like actual NppEnvironment
        self.entities = [
            {
                "type": EntityType.EXIT_DOOR,
                "entity_id": "exit_door_1",
                "x": 400,
                "y": 300,
                "active": True,
            },
            {
                "type": EntityType.EXIT_SWITCH,
                "entity_id": "exit_switch_1",
                "x": 200,
                "y": 200,
                "active": True,  # Not activated (inverted logic)
            },
            {
                "type": EntityType.EXIT_SWITCH,
                "entity_id": "switch_1",
                "x": 100,
                "y": 100,
                "active": True,  # Not activated
            },
            {
                "type": EntityType.EXIT_SWITCH,
                "entity_id": "switch_2",
                "x": 300,
                "y": 100,
                "active": True,  # Not activated
            },
            {
                "type": EntityType.EXIT_SWITCH,
                "entity_id": "switch_3",
                "x": 500,
                "y": 200,
                "active": True,  # Not activated
            },
        ]


class MockReachabilitySystem:
    """Mock reachability system for testing."""

    def analyze_reachability(
        self, level_data, ninja_pos, switch_states, performance_target="balanced"
    ):
        return {
            "reachable_positions": [(100, 100), (200, 200), (300, 300)],
            "switch_accessibility": {
                "switch_1": True,
                "switch_2": False,
                "switch_3": True,
            },
            "completion_time_estimate": 45.0,
        }


class MockReachabilityFeatures:
    """Mock reachability features for testing."""

    def encode_reachability(
        self, reachability_result, level_data, objectives, ninja_pos, switch_states
    ):
        # Return 64-dimensional feature vector
        return np.random.rand(64).astype(np.float32)


class TestSubgoalFramework(unittest.TestCase):
    """Test the hierarchical subgoal framework."""

    def setUp(self):
        self.ninja_pos = (150, 150)
        self.level_data = MockLevelData()
        self.switch_states = {"switch_1": False, "switch_2": False, "switch_3": False}

    def test_navigation_subgoal_creation(self):
        """Test NavigationSubgoal creation and basic functionality."""
        subgoal = NavigationSubgoal(
            target_position=(200, 200),
            target_type="exit_door",
            distance=70.7,
            priority=0.8,
            estimated_time=5.0,
            success_probability=0.9,
        )

        self.assertEqual(subgoal.get_target_position(), (200, 200))
        self.assertEqual(subgoal.target_type, "exit_door")
        self.assertAlmostEqual(subgoal.distance, 70.7, places=1)

        # Test completion check
        close_pos = (205, 205)  # Within 24.0 distance
        far_pos = (300, 300)  # Beyond 24.0 distance

        self.assertTrue(
            subgoal.is_completed(close_pos, self.level_data, self.switch_states)
        )
        self.assertFalse(
            subgoal.is_completed(far_pos, self.level_data, self.switch_states)
        )

    def test_switch_activation_subgoal_creation(self):
        """Test SwitchActivationSubgoal creation and completion checking."""
        subgoal = SwitchActivationSubgoal(
            switch_id="switch_1",
            switch_position=(100, 100),
            switch_type="exit_switch",
            reachability_score=0.8,
            priority=0.9,
            estimated_time=3.0,
            success_probability=0.85,
        )

        self.assertEqual(subgoal.get_target_position(), (100, 100))
        self.assertEqual(subgoal.switch_id, "switch_1")
        self.assertAlmostEqual(subgoal.reachability_score, 0.8)

        # Test completion - should be False initially (switch not activated)
        self.assertFalse(
            subgoal.is_completed(self.ninja_pos, self.level_data, self.switch_states)
        )

        # Test completion after activation - modify both switch_states and level_data
        activated_switch_states = {
            "switch_1": True,
            "switch_2": False,
            "switch_3": False,
        }

        # Also update the level_data entities to reflect activation
        for entity in self.level_data.entities:
            if entity.get("entity_id") == "switch_1":
                entity["active"] = False  # Activated (inverted logic)

        self.assertTrue(
            subgoal.is_completed(
                self.ninja_pos, self.level_data, activated_switch_states
            )
        )


class TestLevelCompletionPlanner(unittest.TestCase):
    """Test the level completion planner."""

    def setUp(self):
        self.planner = LevelCompletionPlanner()
        self.ninja_pos = (150, 150)
        self.level_data = MockLevelData()
        self.switch_states = {"switch_1": False, "switch_2": False, "switch_3": False}
        self.reachability_system = MockReachabilitySystem()
        self.reachability_features = MockReachabilityFeatures()

    def test_find_exit_door(self):
        """Test exit door detection using actual NppEnvironment data structures."""
        exit_door = self.planner._find_exit_door(self.level_data)

        self.assertIsNotNone(exit_door)
        self.assertEqual(exit_door["id"], "exit_door_1")
        self.assertEqual(exit_door["position"], (400, 300))
        self.assertEqual(exit_door["type"], "exit_door")

    def test_find_exit_switch(self):
        """Test exit switch detection using actual NppEnvironment data structures."""
        exit_switch = self.planner._find_exit_switch(self.level_data)

        self.assertIsNotNone(exit_switch)
        self.assertEqual(exit_switch["id"], "exit_switch_1")
        self.assertEqual(exit_switch["position"], (200, 200))
        self.assertEqual(exit_switch["type"], "exit_switch")

    def test_objective_reachability_check(self):
        """Test objective reachability using neural features."""
        # Mock reachability features with positive values
        reachability_features = torch.tensor(
            [0.5, 0.3, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0] + [0.0] * 56
        )

        # Test reachability check
        is_reachable = self.planner._is_objective_reachable(
            (200, 200), reachability_features
        )
        self.assertTrue(is_reachable)

        # Test with low reachability features
        low_features = torch.tensor(
            [0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 56
        )
        is_not_reachable = self.planner._is_objective_reachable(
            (200, 200), low_features
        )
        self.assertFalse(is_not_reachable)


class TestSubgoalPrioritizer(unittest.TestCase):
    """Test the subgoal prioritizer."""

    def setUp(self):
        self.prioritizer = SubgoalPrioritizer()
        self.ninja_pos = (150, 150)
        self.level_data = MockLevelData()
        self.reachability_features = torch.rand(64)

    def test_subgoal_prioritization(self):
        """Test subgoal prioritization with mixed subgoal types."""
        subgoals = [
            NavigationSubgoal(
                target_position=(400, 300),
                target_type="exit_door",
                distance=300.0,
                priority=0.5,
                estimated_time=8.0,
                success_probability=0.9,
            ),
            SwitchActivationSubgoal(
                switch_id="switch_1",
                switch_position=(100, 100),
                switch_type="exit_switch",
                reachability_score=0.8,
                priority=0.7,
                estimated_time=3.0,
                success_probability=0.85,
            ),
        ]

        prioritized = self.prioritizer.prioritize(
            subgoals, self.ninja_pos, self.level_data, self.reachability_features
        )

        self.assertEqual(len(prioritized), 2)
        # Check that prioritization occurred (priorities should be updated)
        # The actual order depends on distance and other factors, so just verify structure
        self.assertIsInstance(
            prioritized[0],
            (NavigationSubgoal, SwitchActivationSubgoal),
        )

        # Verify priorities were updated
        for subgoal in prioritized:
            self.assertGreater(subgoal.priority, 0.0)
            self.assertLessEqual(subgoal.priority, 1.0)

    def test_priority_score_calculation(self):
        """Test priority score calculation factors."""
        subgoal = SwitchActivationSubgoal(
            switch_id="switch_1",
            switch_position=(100, 100),
            switch_type="exit_switch",
            reachability_score=0.8,
            priority=0.7,
            estimated_time=3.0,
            success_probability=0.9,
        )

        score = self.prioritizer._calculate_priority_score(
            subgoal, self.ninja_pos, self.level_data, self.reachability_features
        )

        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_reachability_bonus_calculation(self):
        """Test reachability bonus from neural features."""
        switch_subgoal = SwitchActivationSubgoal(
            switch_id="switch_1",
            switch_position=(100, 100),
            switch_type="exit_switch",
            reachability_score=0.8,
            priority=0.7,
            estimated_time=3.0,
            success_probability=0.9,
        )

        bonus = self.prioritizer._get_reachability_bonus(
            switch_subgoal, self.reachability_features
        )
        self.assertGreater(bonus, 0.0)
        self.assertAlmostEqual(bonus, 0.8 * 0.2, places=2)

    def test_empty_subgoal_list(self):
        """Test prioritizer with empty subgoal list."""
        prioritized = self.prioritizer.prioritize(
            [], self.ninja_pos, self.level_data, self.reachability_features
        )
        self.assertEqual(len(prioritized), 0)


if __name__ == "__main__":
    unittest.main()
