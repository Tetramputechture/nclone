#!/usr/bin/env python3
"""
Comprehensive Reachability Test Suite

This test suite validates the reachability analysis system using all maps in test_maps.
It tests both correctness (can the ninja reach the exit) and performance (analysis time).
Also validates the subgoal planning system functionality.

Usage:
    python test_reachability_suite.py [--verbose] [--performance-only] [--map=<name>]
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from nclone.nplay_headless import NPlayHeadless
from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
# Removed legacy trajectory calculator import
from nclone.graph.level_data import LevelData
from nclone.gym_environment.npp_environment import NppEnvironment


class ExpectedOutcome(Enum):
    """Expected reachability outcomes for test maps."""

    COMPLETABLE = "completable"
    UNCOMPLETABLE = "uncompletable"


@dataclass
class TestResult:
    """Result of a single reachability test."""

    map_name: str
    expected_outcome: ExpectedOutcome
    actual_reachable: bool
    analysis_time_ms: float
    reachable_positions: int
    subgoals_found: int
    passed: bool
    error_message: Optional[str] = None
    performance_passed: bool = True


@dataclass
class TestSuiteResults:
    """Results of the complete test suite."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    performance_failures: int
    total_time_ms: float
    results: List[TestResult]


class ReachabilityTestSuite:
    """
    Comprehensive test suite for the reachability analysis system.

    This class loads test maps, runs reachability analysis, and validates
    both correctness and performance of the system.
    """

    # Performance requirements
    MAX_ANALYSIS_TIME_MS = 10.0
    MAX_REPEATED_QUERY_TIME_MS = 1.0

    def __init__(self, verbose: bool = False):
        """
        Initialize the test suite.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.nplay_headless = NPlayHeadless()

        # Initialize tiered reachability system
        self.reachability_analyzer = TieredReachabilitySystem(debug=verbose)

        # Load test map definitions
        self.test_maps = self._load_test_map_definitions()

        # Define expected outcomes based on map descriptions
        self.expected_outcomes = self._define_expected_outcomes()

    def _load_test_map_definitions(self) -> List[Dict[str, str]]:
        """Load test map definitions from maps.json."""
        maps_json_path = "nclone/test_maps/maps.json"
        if not os.path.exists(maps_json_path):
            raise FileNotFoundError(
                f"Test maps definition file not found: {maps_json_path}"
            )

        with open(maps_json_path, "r") as f:
            return json.load(f)

    def _define_expected_outcomes(self) -> Dict[str, ExpectedOutcome]:
        """
        Define expected reachability outcomes based on map descriptions.

        Returns:
            Dictionary mapping map names to expected outcomes
        """
        outcomes = {}

        for map_info in self.test_maps:
            name = map_info["name"]
            completable = map_info["completable"]

            if completable:
                outcomes[name] = ExpectedOutcome.COMPLETABLE
            else:
                outcomes[name] = ExpectedOutcome.UNCOMPLETABLE

        return outcomes

    def _load_map_and_create_level_data(
        self, map_path: str
    ) -> Tuple[LevelData, Tuple[float, float]]:
        """
        Load a map and create LevelData for reachability analysis.

        Args:
            map_path: Path to the map file

        Returns:
            Tuple of (LevelData, ninja_position)
        """
        # Create environment and load the specific map
        env = NppEnvironment(render_mode="rgb_array", custom_map_path=map_path)
        env.reset()

        # Extract level data
        level_data = env.level_data
        ninja_position = env.nplay_headless.ninja_position()

        # Clean up
        env.close()

        return level_data, ninja_position

    def _run_single_test(self, map_info: Dict[str, str]) -> TestResult:
        """
        Run reachability analysis on a single map.

        Args:
            map_info: Map information from maps.json

        Returns:
            TestResult with analysis results
        """
        map_name = map_info["name"]
        map_path = map_info["path"]
        expected_outcome = self.expected_outcomes.get(
            map_name, ExpectedOutcome.COMPLETABLE
        )

        if self.verbose:
            print(f"\nTesting map: {map_name}")
            print(f"  Path: {map_path}")
            print(f"  Expected: {expected_outcome.value}")
            print(f"  Description: {map_info['description']}")

        try:
            # Load map and create level data
            level_data, ninja_position = self._load_map_and_create_level_data(map_path)

            # Run reachability analysis with timing
            start_time = time.perf_counter()
            reachability_result = self.reachability_analyzer.analyze_reachability(
                level_data, ninja_position, switch_states={}, performance_target="balanced"
            )
            end_time = time.perf_counter()

            analysis_time_ms = (end_time - start_time) * 1000

            # Extract results
            reachable_positions = len(reachability_result.reachable_positions)
            subgoals_found = 0  # Subgoals not implemented in tiered system yet

            # Determine if exit is reachable using level completability heuristic
            actual_reachable = reachability_result.is_level_completable()

            # Check if result matches expectation
            passed = self._validate_outcome(expected_outcome, actual_reachable)

            # Check performance
            performance_passed = analysis_time_ms <= self.MAX_ANALYSIS_TIME_MS

            if self.verbose:
                print(f"  Analysis time: {analysis_time_ms:.2f}ms")
                print(f"  Reachable positions: {reachable_positions}")
                print(f"  Subgoals found: {subgoals_found}")
                print(f"  Exit reachable: {actual_reachable}")
                print(f"  Test passed: {passed}")
                print(f"  Performance passed: {performance_passed}")

                # Print subgoal details if any found
                if subgoals_found > 0:
                    print(
                        f"  Subgoals: {[str(sg) for sg in reachability_state.subgoals]}"
                    )

            return TestResult(
                map_name=map_name,
                expected_outcome=expected_outcome,
                actual_reachable=actual_reachable,
                analysis_time_ms=analysis_time_ms,
                reachable_positions=reachable_positions,
                subgoals_found=subgoals_found,
                passed=passed,
                performance_passed=performance_passed,
            )

        except Exception as e:
            error_msg = f"Error testing {map_name}: {str(e)}"
            if self.verbose:
                print(f"  ERROR: {error_msg}")

            return TestResult(
                map_name=map_name,
                expected_outcome=expected_outcome,
                actual_reachable=False,
                analysis_time_ms=0.0,
                reachable_positions=0,
                subgoals_found=0,
                passed=False,
                error_message=error_msg,
                performance_passed=False,
            )

    def _determine_exit_reachability(
        self, level_data: LevelData, reachability_state
    ) -> bool:
        """
        Determine if the exit is reachable based on reachability analysis.

        This checks if the ninja can actually reach exit-related entities.

        Args:
            level_data: Level data
            reachability_state: Reachability analysis results

        Returns:
            True if exit appears to be reachable
        """
        # Find exit entities in the level
        exit_entities = []
        for entity in level_data.entities:
            entity_id = entity.get("entity_id", "")
            if "exit" in entity_id.lower():
                exit_entities.append(entity)

        if self.verbose:
            print(f"  DEBUG: Found {len(exit_entities)} exit entities")
            for i, entity in enumerate(exit_entities):
                print(
                    f"    Exit {i}: {entity.get('entity_id', 'unknown')} at ({entity.get('x', 0)}, {entity.get('y', 0)})"
                )

        if not exit_entities:
            # No exit entities found, fall back to position count heuristic
            if self.verbose:
                print(
                    f"  DEBUG: No exit entities found, using position count heuristic: {len(reachability_state.reachable_positions)} > 1"
                )
            return len(reachability_state.reachable_positions) > 1

        # Check if any exit entity position is reachable using circle-to-circle collision
        from nclone.graph.reachability.position_validator import PositionValidator
        from nclone.constants.physics_constants import NINJA_RADIUS
        import math

        validator = PositionValidator()

        for i, exit_entity in enumerate(exit_entities):
            exit_x = exit_entity.get("x", 0)
            exit_y = exit_entity.get("y", 0)
            exit_radius = exit_entity.get(
                "radius", 12
            )  # Default to 12 if not specified

            if self.verbose:
                print(
                    f"    Exit {i}: {exit_entity.get('entity_id', 'unknown')} at pixel ({exit_x}, {exit_y}), radius={exit_radius}"
                )

            # Calculate the maximum distance at which ninja and exit circles can touch
            max_interaction_distance = NINJA_RADIUS + exit_radius

            # Check all reachable positions to see if any allow ninja to reach the exit
            for sub_row, sub_col in reachability_state.reachable_positions:
                # Convert reachable position to pixel coordinates
                ninja_pixel_x, ninja_pixel_y = validator.convert_sub_grid_to_pixel(
                    sub_row, sub_col
                )

                # Calculate distance between ninja center and exit center
                distance = math.sqrt(
                    (ninja_pixel_x - exit_x) ** 2 + (ninja_pixel_y - exit_y) ** 2
                )

                # If distance is less than sum of radii, circles overlap
                if distance <= max_interaction_distance:
                    if self.verbose:
                        print(
                            f"  DEBUG: Exit {i} is reachable! Ninja at ({sub_row}, {sub_col}) -> pixel ({ninja_pixel_x:.1f}, {ninja_pixel_y:.1f})"
                        )
                        print(
                            f"         Distance {distance:.1f} <= max_interaction {max_interaction_distance:.1f}"
                        )
                    return True

            if self.verbose:
                print(f"    Exit {i} not reachable from any ninja position")

        if self.verbose:
            print(f"  DEBUG: No exit entities are reachable")
        # No exit entities are reachable
        return False

    def _validate_outcome(
        self, expected: ExpectedOutcome, actual_reachable: bool
    ) -> bool:
        """
        Validate if the actual outcome matches the expected outcome.

        Args:
            expected: Expected outcome
            actual_reachable: Whether exit was determined to be reachable

        Returns:
            True if outcome matches expectation
        """
        if expected == ExpectedOutcome.COMPLETABLE:
            return actual_reachable
        elif expected == ExpectedOutcome.UNCOMPLETABLE:
            return not actual_reachable

        return False

    def run_test_suite(
        self, specific_map: Optional[str] = None, performance_only: bool = False
    ) -> TestSuiteResults:
        """
        Run the complete reachability test suite.

        Args:
            specific_map: If provided, only test this specific map
            performance_only: If True, only run performance tests

        Returns:
            TestSuiteResults with complete results
        """
        print("Reachability Analysis Test Suite")
        print("=" * 50)

        if performance_only:
            print("Running performance tests only...")
            return self._run_performance_tests()

        # Filter maps if specific map requested
        maps_to_test = self.test_maps
        if specific_map:
            maps_to_test = [m for m in self.test_maps if m["name"] == specific_map]
            if not maps_to_test:
                raise ValueError(f"Map '{specific_map}' not found in test maps")

        print(f"Testing {len(maps_to_test)} maps...")

        # Run tests on all maps
        results = []
        total_start_time = time.perf_counter()

        for map_info in maps_to_test:
            result = self._run_single_test(map_info)
            results.append(result)

        total_end_time = time.perf_counter()
        total_time_ms = (total_end_time - total_start_time) * 1000

        # Calculate summary statistics
        passed_tests = sum(1 for r in results if r.passed and r.performance_passed)
        failed_tests = len(results) - passed_tests
        performance_failures = sum(1 for r in results if not r.performance_passed)

        return TestSuiteResults(
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            performance_failures=performance_failures,
            total_time_ms=total_time_ms,
            results=results,
        )

    def _run_performance_tests(self) -> TestSuiteResults:
        """Run performance-specific tests."""
        results = []

        # Test repeated query performance
        perf_passed, avg_time = self._test_performance_repeated_queries()

        if not perf_passed:
            results.append(
                TestResult(
                    map_name="repeated_queries_performance",
                    expected_outcome=ExpectedOutcome.COMPLETABLE,
                    actual_reachable=True,
                    analysis_time_ms=avg_time,
                    reachable_positions=0,
                    subgoals_found=0,
                    passed=False,
                    performance_passed=False,
                    error_message=f"Repeated queries too slow: {avg_time:.3f}ms > {self.MAX_REPEATED_QUERY_TIME_MS}ms",
                )
            )

        return TestSuiteResults(
            total_tests=1,
            passed_tests=1 if perf_passed else 0,
            failed_tests=0 if perf_passed else 1,
            performance_failures=0 if perf_passed else 1,
            total_time_ms=avg_time,
            results=results,
        )

    def _test_performance_repeated_queries(self) -> Tuple[bool, float]:
        """
        Test performance of repeated queries on the same map.

        Returns:
            Tuple of (passed, average_time_ms)
        """
        if self.verbose:
            print("\nTesting repeated query performance...")

        # Use a simple map for repeated testing
        simple_map = next(m for m in self.test_maps if m["name"] == "simple-walk")
        level_data, ninja_position = self._load_map_and_create_level_data(
            simple_map["path"]
        )

        # Run multiple queries
        num_queries = 100
        total_time = 0.0

        for i in range(num_queries):
            start_time = time.perf_counter()
            self.reachability_analyzer.analyze_reachability(level_data, ninja_position, switch_states={}, performance_target="balanced")
            end_time = time.perf_counter()
            total_time += end_time - start_time

        average_time_ms = (total_time / num_queries) * 1000
        passed = average_time_ms <= self.MAX_REPEATED_QUERY_TIME_MS

        if self.verbose:
            print(f"  Queries: {num_queries}")
            print(f"  Average time: {average_time_ms:.3f}ms")
            print(f"  Performance passed: {passed}")

        return passed, average_time_ms

    def print_results(self, results: TestSuiteResults):
        """
        Print detailed test results.

        Args:
            results: Test suite results to print
        """
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        print(f"Total tests: {results.total_tests}")
        print(f"Passed: {results.passed_tests}")
        print(f"Failed: {results.failed_tests}")
        print(f"Performance failures: {results.performance_failures}")
        print(f"Total time: {results.total_time_ms:.2f}ms")

        if results.results:
            avg_time = sum(r.analysis_time_ms for r in results.results) / len(
                results.results
            )
            print(f"Average analysis time: {avg_time:.2f}ms")

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("-" * 60)

        for result in results.results:
            status = (
                "✓ PASS" if result.passed and result.performance_passed else "✗ FAIL"
            )
            perf_status = "✓" if result.performance_passed else "✗"

            print(f"{status} {result.map_name}")
            print(f"    Expected: {result.expected_outcome.value}")
            print(
                f"    Actual: {'completable' if result.actual_reachable else 'uncompletable'}"
            )
            print(f"    Time: {result.analysis_time_ms:.2f}ms {perf_status}")
            print(
                f"    Positions: {result.reachable_positions}, Subgoals: {result.subgoals_found}"
            )

            if result.error_message:
                print(f"    Error: {result.error_message}")
            print()

        # Print performance summary
        if results.performance_failures > 0:
            print("PERFORMANCE ISSUES:")
            print("-" * 30)
            for result in results.results:
                if not result.performance_passed:
                    print(
                        f"  {result.map_name}: {result.analysis_time_ms:.2f}ms > {self.MAX_ANALYSIS_TIME_MS}ms"
                    )

        # Overall result
        if results.failed_tests == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"❌ {results.failed_tests} TESTS FAILED")


def main():
    """Main entry point for the test suite."""
    parser = argparse.ArgumentParser(description="Reachability Analysis Test Suite")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--performance-only",
        "-p",
        action="store_true",
        help="Only run performance tests",
    )
    parser.add_argument("--map", "-m", type=str, help="Test only a specific map")

    args = parser.parse_args()

    try:
        # Create and run test suite
        test_suite = ReachabilityTestSuite(verbose=args.verbose)
        results = test_suite.run_test_suite(
            specific_map=args.map, performance_only=args.performance_only
        )

        # Print results
        test_suite.print_results(results)

        # Exit with appropriate code
        sys.exit(0 if results.failed_tests == 0 else 1)

    except Exception as e:
        print(f"Error running test suite: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
