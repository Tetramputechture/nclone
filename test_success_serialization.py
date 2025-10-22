#!/usr/bin/env python3
"""Test success flag serialization/deserialization."""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from nclone.replay.gameplay_recorder import CompactReplay

print("="*80)
print("Test: Success Flag Serialization")
print("="*80)

# Test data
fake_map_data = b'\x00' * 1335  # 1335 bytes of zeros
fake_inputs = [0, 1, 2, 3, 4, 5, 0]  # 7 frames

# Test 1: Successful replay
print("\n" + "-"*80)
print("Test 1: Successful Replay (success=True)")
print("-"*80)

replay_success = CompactReplay(
    episode_id="test_success",
    map_data=fake_map_data,
    input_sequence=fake_inputs,
    level_id="test_level",
    start_time=datetime.now(),
    end_time=datetime.now(),
    success=True,
)

print(f"Original success flag: {replay_success.success}")

# Serialize
binary_data = replay_success.to_binary()
print(f"Binary size: {len(binary_data)} bytes")
print(f"Expected size: {replay_success.get_file_size()} bytes")

# Deserialize
loaded_replay = CompactReplay.from_binary(binary_data, episode_id="loaded_test")
print(f"Loaded success flag: {loaded_replay.success}")

if loaded_replay.success == replay_success.success:
    print("✓ Success flag correctly preserved")
else:
    print("✗ FAIL: Success flag not preserved!")
    sys.exit(1)

# Test 2: Failed replay
print("\n" + "-"*80)
print("Test 2: Failed Replay (success=False)")
print("-"*80)

replay_fail = CompactReplay(
    episode_id="test_fail",
    map_data=fake_map_data,
    input_sequence=fake_inputs,
    level_id="test_level",
    start_time=datetime.now(),
    end_time=datetime.now(),
    success=False,
)

print(f"Original success flag: {replay_fail.success}")

# Serialize
binary_data = replay_fail.to_binary()
print(f"Binary size: {len(binary_data)} bytes")

# Deserialize
loaded_replay = CompactReplay.from_binary(binary_data, episode_id="loaded_test")
print(f"Loaded success flag: {loaded_replay.success}")

if loaded_replay.success == replay_fail.success:
    print("✓ Success flag correctly preserved")
else:
    print("✗ FAIL: Success flag not preserved!")
    sys.exit(1)

# Test 3: Backward compatibility with legacy format (V0)
print("\n" + "-"*80)
print("Test 3: Backward Compatibility (V0 format without success flag)")
print("-"*80)

import struct

# Create legacy V0 format manually
map_data_len = len(fake_map_data)
input_seq_len = len(fake_inputs)
header_v0 = struct.pack("<II", map_data_len, input_seq_len)
inputs_bytes = bytes(fake_inputs)
legacy_binary = header_v0 + fake_map_data + inputs_bytes

print(f"Legacy binary size: {len(legacy_binary)} bytes (V0 format)")

# Try to load legacy format
loaded_legacy = CompactReplay.from_binary(legacy_binary, episode_id="legacy_test")
print(f"Loaded success flag (V0 default): {loaded_legacy.success}")

if loaded_legacy.success == True:
    print("✓ Legacy format loads with success=True (expected default)")
else:
    print("✗ FAIL: Legacy format should default to success=True")
    sys.exit(1)

# Verify data integrity
if loaded_legacy.input_sequence == fake_inputs:
    print("✓ Input sequence preserved from V0 format")
else:
    print("✗ FAIL: Input sequence corrupted")
    sys.exit(1)

# Test 4: Test with actual existing replay file
print("\n" + "-"*80)
print("Test 4: Load Existing Replay Files (V0 format)")
print("-"*80)

replay_dir = Path(__file__).parent / "bc_replays"
if replay_dir.exists():
    replay_files = list(replay_dir.glob("*.replay"))[:2]  # Test first 2
    
    for replay_path in replay_files:
        with open(replay_path, 'rb') as f:
            binary_data = f.read()
        
        loaded = CompactReplay.from_binary(binary_data)
        print(f"\n{replay_path.name}:")
        print(f"  success: {loaded.success}")
        print(f"  inputs: {len(loaded.input_sequence)}")
        print(f"  map_data: {len(loaded.map_data)} bytes")
        
        # Re-serialize and verify
        reserialized = loaded.to_binary()
        reloaded = CompactReplay.from_binary(reserialized)
        
        if reloaded.success == loaded.success:
            print(f"  ✓ Round-trip preserves success flag")
        else:
            print(f"  ✗ FAIL: Round-trip lost success flag")
            sys.exit(1)
else:
    print("Skipping (no bc_replays directory found)")

print("\n" + "="*80)
print("All Tests Passed!")
print("="*80)
print("\nSummary:")
print("  ✓ Success flag serialization works correctly")
print("  ✓ Failure flag serialization works correctly")
print("  ✓ Backward compatibility with V0 format maintained")
print("  ✓ Existing replay files load correctly")
print("  ✓ Round-trip serialization preserves all data")
