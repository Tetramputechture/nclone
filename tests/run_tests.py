#!/usr/bin/env python3
"""
Test runner for nclone graph traversability tests.

This script runs all tests and provides a summary of results.
"""

import sys
import os
import subprocess
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_tests():
    """Run all tests and return success status."""
    print("Running nclone graph traversability tests...")
    print("=" * 60)
    
    test_files = [
        "test_graph_traversability.py",
        "test_debug_overlay_renderer.py",
    ]
    
    all_passed = True
    results = {}
    
    for test_file in test_files:
        print(f"\nRunning {test_file}...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                os.path.join(os.path.dirname(__file__), test_file), 
                "-v", "--tb=short"
            ], capture_output=False, cwd=os.path.dirname(os.path.dirname(__file__)))
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                results[test_file] = ("PASSED", duration)
                print(f"✅ {test_file} PASSED ({duration:.2f}s)")
            else:
                results[test_file] = ("FAILED", duration)
                print(f"❌ {test_file} FAILED ({duration:.2f}s)")
                all_passed = False
                
        except Exception as e:
            results[test_file] = ("ERROR", 0)
            print(f"💥 {test_file} ERROR: {e}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_file, (status, duration) in results.items():
        status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "💥"
        print(f"{status_icon} {test_file:<40} {status:<8} ({duration:.2f}s)")
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
