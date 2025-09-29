#!/usr/bin/env python3
"""
Comprehensive analysis of all N++ attract replay files to understand the general format.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
from nclone.replay.binary_replay_parser import BinaryReplayParser
import glob

def analyze_all_replays():
    """Analyze all replay files to understand the general format patterns."""
    
    replay_dir = Path("nclone/example_replays/npp_attract")
    replay_files = sorted(glob.glob(str(replay_dir / "*")), key=lambda x: int(Path(x).name))
    
    print("=" * 80)
    print("COMPREHENSIVE N++ ATTRACT REPLAY FORMAT ANALYSIS")
    print("=" * 80)
    print(f"Analyzing {len(replay_files)} replay files...")
    
    parser = BinaryReplayParser()
    results = []
    
    for replay_file in replay_files:
        replay_path = Path(replay_file)
        replay_id = replay_path.name
        
        try:
            with open(replay_path, "rb") as f:
                data = f.read()
            
            # Parse with current decoder
            inputs, map_data, level_id, level_name = parser.parse_single_replay_file(replay_path)
            
            # Find all valid input sequences
            sequences = []
            current_seq_start = None
            current_seq_length = 0
            
            for i in range(len(data)):
                byte_val = data[i]
                
                if 0 <= byte_val <= 7:
                    if current_seq_start is None:
                        current_seq_start = i
                        current_seq_length = 1
                    else:
                        current_seq_length += 1
                else:
                    if current_seq_start is not None and current_seq_length >= 10:
                        sequences.append((current_seq_start, current_seq_length))
                    current_seq_start = None
                    current_seq_length = 0
            
            if current_seq_start is not None and current_seq_length >= 10:
                sequences.append((current_seq_start, current_seq_length))
            
            # Analyze input distribution
            input_counts = {}
            for inp in inputs:
                input_counts[inp] = input_counts.get(inp, 0) + 1
            
            result = {
                'id': replay_id,
                'file_size': len(data),
                'level_name': level_name,
                'total_inputs': len(inputs),
                'runtime': len(inputs) / 60.0,
                'sequences': sequences,
                'input_distribution': input_counts,
                'success': True
            }
            
            results.append(result)
            
        except Exception as e:
            result = {
                'id': replay_id,
                'file_size': len(data) if 'data' in locals() else 0,
                'level_name': 'ERROR',
                'total_inputs': 0,
                'runtime': 0.0,
                'sequences': [],
                'input_distribution': {},
                'success': False,
                'error': str(e)
            }
            results.append(result)
    
    return results

def analyze_format_patterns(results):
    """Analyze patterns across all replay files."""
    
    print(f"\n" + "=" * 80)
    print("FORMAT PATTERN ANALYSIS")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    print(f"Successfully parsed: {len(successful_results)}/{len(results)} files")
    
    # File size analysis
    file_sizes = [r['file_size'] for r in successful_results]
    print(f"\nFile Size Analysis:")
    print(f"  Range: {min(file_sizes)} - {max(file_sizes)} bytes")
    print(f"  Average: {sum(file_sizes) / len(file_sizes):.1f} bytes")
    
    # Runtime analysis
    runtimes = [r['runtime'] for r in successful_results]
    print(f"\nRuntime Analysis:")
    print(f"  Range: {min(runtimes):.1f} - {max(runtimes):.1f} seconds")
    print(f"  Average: {sum(runtimes) / len(runtimes):.1f} seconds")
    
    # Input count analysis
    input_counts = [r['total_inputs'] for r in successful_results]
    print(f"\nInput Count Analysis:")
    print(f"  Range: {min(input_counts)} - {max(input_counts)} inputs")
    print(f"  Average: {sum(input_counts) / len(input_counts):.1f} inputs")
    
    # Sequence analysis
    print(f"\nSequence Pattern Analysis:")
    for result in successful_results[:5]:  # Show first 5 as examples
        print(f"  File {result['id']}: {len(result['sequences'])} sequences")
        for i, (start, length) in enumerate(result['sequences']):
            if i < 3:  # Show first 3 sequences
                print(f"    Seq {i+1}: Offset {start:4d}-{start+length-1:4d} ({length:3d} bytes)")
    
    # Level name analysis
    level_names = [r['level_name'] for r in successful_results]
    unique_levels = list(set(level_names))
    print(f"\nLevel Analysis:")
    print(f"  Unique levels: {len(unique_levels)}")
    for level in unique_levels[:10]:  # Show first 10
        count = level_names.count(level)
        print(f"    '{level}': {count} file(s)")

def generate_detailed_report(results):
    """Generate detailed report for each file."""
    
    print(f"\n" + "=" * 80)
    print("DETAILED FILE ANALYSIS")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    for result in successful_results:
        print(f"\nFile {result['id']}: '{result['level_name']}'")
        print(f"  Size: {result['file_size']} bytes")
        print(f"  Runtime: {result['runtime']:.1f}s ({result['total_inputs']} inputs)")
        print(f"  Sequences: {len(result['sequences'])}")
        
        # Show sequence details
        for i, (start, length) in enumerate(result['sequences']):
            print(f"    Seq {i+1}: Offset {start:4d}-{start+length-1:4d} ({length:3d} bytes)")
        
        # Show input distribution
        if result['input_distribution']:
            print(f"  Input distribution:")
            total = sum(result['input_distribution'].values())
            for inp_val in sorted(result['input_distribution'].keys()):
                count = result['input_distribution'][inp_val]
                percentage = (count / total) * 100
                print(f"    Input {inp_val}: {count:4d} ({percentage:5.1f}%)")

def identify_common_patterns(results):
    """Identify common patterns across all files."""
    
    print(f"\n" + "=" * 80)
    print("COMMON PATTERN IDENTIFICATION")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    # Analyze sequence count patterns
    sequence_counts = {}
    for result in successful_results:
        seq_count = len(result['sequences'])
        sequence_counts[seq_count] = sequence_counts.get(seq_count, 0) + 1
    
    print(f"Sequence Count Patterns:")
    for seq_count in sorted(sequence_counts.keys()):
        file_count = sequence_counts[seq_count]
        print(f"  {seq_count} sequences: {file_count} files")
    
    # Analyze common offset ranges
    print(f"\nCommon Offset Ranges:")
    offset_ranges = {}
    
    for result in successful_results:
        for start, length in result['sequences']:
            # Group by offset ranges (rounded to nearest 100)
            offset_group = (start // 100) * 100
            if offset_group not in offset_ranges:
                offset_ranges[offset_group] = []
            offset_ranges[offset_group].append((start, length))
    
    for offset_group in sorted(offset_ranges.keys()):
        sequences = offset_ranges[offset_group]
        print(f"  Offset {offset_group:4d}-{offset_group+99:4d}: {len(sequences)} sequences")
        
        # Show range of actual offsets in this group
        starts = [s for s, l in sequences]
        lengths = [l for s, l in sequences]
        if starts:
            print(f"    Actual range: {min(starts):4d}-{max(starts):4d}")
            print(f"    Length range: {min(lengths):3d}-{max(lengths):3d} bytes")

def main():
    results = analyze_all_replays()
    
    if results:
        analyze_format_patterns(results)
        identify_common_patterns(results)
        generate_detailed_report(results)
        
        print(f"\n" + "=" * 80)
        print("SUMMARY FOR DOCUMENTATION")
        print("=" * 80)
        
        successful_results = [r for r in results if r['success']]
        
        print(f"✅ Successfully analyzed {len(successful_results)}/{len(results)} N++ attract replay files")
        
        file_sizes = [r['file_size'] for r in successful_results]
        runtimes = [r['runtime'] for r in successful_results]
        input_counts = [r['total_inputs'] for r in successful_results]
        
        print(f"\n📊 Format Characteristics:")
        print(f"  File sizes: {min(file_sizes)}-{max(file_sizes)} bytes (avg: {sum(file_sizes)/len(file_sizes):.0f})")
        print(f"  Runtimes: {min(runtimes):.1f}-{max(runtimes):.1f}s (avg: {sum(runtimes)/len(runtimes):.1f}s)")
        print(f"  Input counts: {min(input_counts)}-{max(input_counts)} inputs (avg: {sum(input_counts)/len(input_counts):.0f})")
        
        # Count unique levels
        level_names = [r['level_name'] for r in successful_results]
        unique_levels = len(set(level_names))
        print(f"  Unique levels: {unique_levels}")
        
        print(f"\n🎯 This analysis provides comprehensive data for generalizing the format specification!")

if __name__ == "__main__":
    main()