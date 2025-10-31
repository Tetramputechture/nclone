"""
Test environment module for interactive testing and visualization.

This module provides a refactored, modular version of the test_environment
functionality, broken down into manageable components.
"""

from .config import parse_arguments, TestConfig
from .controls import KeyboardController, ActionMapper

# Phase 2-3 modules (TODO)
# from .path_aware_manager import PathAwareManager
# from .recording_manager import RecordingManager
# from .profiling_manager import ProfilingManager
# from .test_suite_manager import TestSuiteManager
# from .generator_manager import GeneratorManager

__all__ = [
    'parse_arguments',
    'TestConfig',
    'KeyboardController',
    'ActionMapper',
    # Phase 2-3 exports
    # 'PathAwareManager',
    # 'RecordingManager',
    # 'ProfilingManager',
    # 'TestSuiteManager',
    # 'GeneratorManager',
]
