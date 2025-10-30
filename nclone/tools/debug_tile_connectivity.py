"""
Debug tool to visualize and validate tile connectivity.

This tool helps identify issues with tile-to-tile connectivity by:
1. Visualizing which tile pairs are marked as traversable
2. Checking specific problematic connections
3. Generating detailed reports on connectivity decisions
"""

import pickle
import gzip
import numpy as np
import os


def load_connectivity_table():
    """Load the precomputed connectivity table."""
    data_path = os.path.join(
        os.path.dirname(__file__),
        "../data/tile_connectivity.pkl.gz"
    )
    
    with gzip.open(data_path, 'rb') as f:
        table = pickle.load(f)
    
    print(f"Loaded connectivity table: shape {table.shape}")
    return table


def analyze_tile_pair_connectivity(tile_a: int, tile_b: int, table: np.ndarray):
    """Analyze connectivity between two specific tile types."""
    print(f"\n{'='*70}")
    print(f"ANALYZING CONNECTIVITY: Tile {tile_a} → Tile {tile_b}")
    print(f"{'='*70}")
    
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    dir_idx_map = {name: i for i, name in enumerate(directions)}
    
    # For each direction
    for dir_name in directions:
        dir_idx = dir_idx_map[dir_name]
        print(f"\n{dir_name} direction:")
        
        # Check all sub-node combinations
        for src_sub_y in range(2):
            for src_sub_x in range(2):
                src_sub_idx = src_sub_y * 2 + src_sub_x
                src_name = ['TL', 'TR', 'BL', 'BR'][src_sub_idx]
                
                for dst_sub_y in range(2):
                    for dst_sub_x in range(2):
                        dst_sub_idx = dst_sub_y * 2 + dst_sub_x
                        dst_name = ['TL', 'TR', 'BL', 'BR'][dst_sub_idx]
                        
                        is_traversable = table[tile_a, tile_b, dir_idx, src_sub_idx, dst_sub_idx]
                        
                        if is_traversable:
                            print(f"  ✓ {src_name}→{dst_name}: TRAVERSABLE")
                        else:
                            print(f"  ✗ {src_name}→{dst_name}: BLOCKED")


def find_suspicious_connections(table: np.ndarray):
    """Find potentially problematic connections."""
    print(f"\n{'='*70}")
    print("FINDING SUSPICIOUS CONNECTIONS")
    print(f"{'='*70}")
    
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    # Check for patterns that might be wrong
    suspicious = []
    
    # Check Type 0 (empty) to Type 1 (solid) - should always be blocked
    for dir_idx, dir_name in enumerate(directions):
        for src_sub in range(4):
            for dst_sub in range(4):
                if table[0, 1, dir_idx, src_sub, dst_sub]:
                    suspicious.append(
                        f"Empty→Solid {dir_name}: src_sub={src_sub} dst_sub={dst_sub} is TRAVERSABLE (SHOULD BE BLOCKED)"
                    )
                if table[1, 0, dir_idx, src_sub, dst_sub]:
                    suspicious.append(
                        f"Solid→Empty {dir_name}: src_sub={src_sub} dst_sub={dst_sub} is TRAVERSABLE (SHOULD BE BLOCKED)"
                    )
    
    # Check half tiles - certain directions should be blocked
    # Type 2: Top half solid, only bottom accessible
    # Moving North (up) from Type 2 might be problematic from top sub-nodes
    for src_sub_y in [0]:  # Top row
        for src_sub_x in [0, 1]:
            src_sub = src_sub_y * 2 + src_sub_x
            if table[2, 0, 0, src_sub, 0]:  # North direction, to empty
                suspicious.append(
                    f"Type2(TopSolid) North from top sub-node {src_sub}: TRAVERSABLE (might be wrong)"
                )
    
    if suspicious:
        print("\nFound suspicious connections:")
        for s in suspicious[:20]:  # Limit output
            print(f"  ⚠ {s}")
    else:
        print("\n✓ No obviously suspicious connections found")
    
    return suspicious


def generate_connectivity_matrix_text(table: np.ndarray, direction: str = 'E'):
    """Generate a text matrix showing which tile pairs can connect."""
    dir_idx = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'].index(direction)
    
    print(f"\n{'='*70}")
    print(f"CONNECTIVITY MATRIX FOR {direction} DIRECTION")
    print(f"{'='*70}")
    print("Rows = Source Tile, Columns = Dest Tile")
    print("Value = Number of sub-node pairs that can traverse (0-16)")
    print()
    
    # Count traversable sub-node pairs for each tile pair
    matrix = np.zeros((34, 34), dtype=int)
    for tile_a in range(34):
        for tile_b in range(34):
            count = 0
            for src_sub in range(4):
                for dst_sub in range(4):
                    if table[tile_a, tile_b, dir_idx, src_sub, dst_sub]:
                        count += 1
            matrix[tile_a, tile_b] = count
    
    # Print header
    print("   ", end="")
    for tile_b in range(min(34, 20)):  # Limit to first 20 for readability
        print(f"{tile_b:3}", end="")
    print()
    
    # Print matrix
    for tile_a in range(min(34, 20)):
        print(f"{tile_a:3}", end="")
        for tile_b in range(min(34, 20)):
            val = matrix[tile_a, tile_b]
            if val == 0:
                print("  .", end="")
            elif val <= 9:
                print(f"  {val}", end="")
            else:
                print(f" {val}", end="")
        print()
    
    return matrix


def main():
    """Main debug analysis."""
    table = load_connectivity_table()
    
    print(f"\nTable shape: {table.shape}")
    print(f"Total possible connections: {table.size:,}")
    print(f"Traversable connections: {np.sum(table):,}")
    print(f"Percentage traversable: {100 * np.sum(table) / table.size:.2f}%")
    
    # Analyze specific problematic pairs
    print("\n" + "="*70)
    print("ANALYZING COMMON TILE PAIRS")
    print("="*70)
    
    # Empty to empty
    analyze_tile_pair_connectivity(0, 0, table)
    
    # Empty to half-tile (bottom solid)
    analyze_tile_pair_connectivity(0, 4, table)
    
    # Half-tile to empty
    analyze_tile_pair_connectivity(2, 0, table)
    
    # Find suspicious connections
    find_suspicious_connections(table)
    
    # Generate connectivity matrices
    for direction in ['N', 'E', 'S', 'W', 'NE']:
        generate_connectivity_matrix_text(table, direction)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

