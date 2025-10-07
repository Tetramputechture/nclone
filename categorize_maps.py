#!/usr/bin/env python3
"""
Categorize N++ maps based on entity complexity.

This script categorizes maps from maps/official/S and maps/official/SL into two categories:
- Simple: Contains only exit switches, exit doors, toggle mines, locked doors, and gold
- Complex: Contains any other entity types
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

from nclone.constants.entity_types import EntityType

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Simple entity types that qualify a map as "Simple"
SIMPLE_ENTITY_TYPES = {
    EntityType.EXIT_DOOR,
    EntityType.EXIT_SWITCH,
    EntityType.TOGGLE_MINE,
    EntityType.TOGGLE_MINE_TOGGLED,
    EntityType.LOCKED_DOOR,
    EntityType.LOCKED_DOOR_SWITCH,
    EntityType.GOLD,
}

# Entity type name mapping for human-readable output
ENTITY_TYPE_NAMES = {
    EntityType.NINJA: "Ninja",
    EntityType.TOGGLE_MINE: "Toggle Mine (Active)",
    EntityType.GOLD: "Gold",
    EntityType.EXIT_DOOR: "Exit Door",
    EntityType.EXIT_SWITCH: "Exit Switch",
    EntityType.REGULAR_DOOR: "Regular Door",
    EntityType.LOCKED_DOOR: "Locked Door",
    EntityType.LOCKED_DOOR_SWITCH: "Locked Door Switch",  # Locked door switch entity
    EntityType.TRAP_DOOR: "Trap Door",
    9: "Trap Door Switch",  # Trap door switch entity
    EntityType.LAUNCH_PAD: "Launch Pad",
    EntityType.ONE_WAY: "One-Way Platform",
    EntityType.DRONE_ZAP: "Drone (Zap)",
    EntityType.DRONE_CHASER: "Drone (Chaser)",
    EntityType.BOUNCE_BLOCK: "Bounce Block",
    19: "Bounce Block (Extended)",  # Extended bounce block
    EntityType.THWUMP: "Thwump",
    EntityType.TOGGLE_MINE_TOGGLED: "Toggle Mine (Inactive)",
    23: "Laser",  # Laser entity
    EntityType.BOOST_PAD: "Boost Pad",
    EntityType.DEATH_BALL: "Death Ball",
    EntityType.MINI_DRONE: "Mini Drone",
    EntityType.SHWUMP: "Shove Thwump",
}


def read_map_file(map_path: Path) -> List[int]:
    """
    Read a binary map file and return the map data as a list of integers.

    Args:
        map_path: Path to the binary map file

    Returns:
        List of integers representing the map data
    """
    with open(map_path, "rb") as f:
        raw_data = f.read()
    return [int(b) for b in raw_data]


def extract_entity_types(map_data: List[int]) -> Set[int]:
    """
    Extract all unique entity types from map data.

    Args:
        map_data: List of integers representing the map data

    Returns:
        Set of entity type integers found in the map
    """
    entity_types = set()

    # Entity data starts at index 1230
    # Each entity is represented by 5 integers:
    # [entity_type, xcoord, ycoord, orientation, mode]
    index = 1230

    while index < len(map_data):
        entity_type = map_data[index]
        entity_types.add(entity_type)
        index += 5

    return entity_types


def categorize_map(map_path: Path) -> tuple:
    """
    Categorize a map as Simple or Complex based on its entity types.

    Args:
        map_path: Path to the binary map file

    Returns:
        Tuple of (category, all_entity_types) where category is "Simple", "Complex", or "Error"
        and all_entity_types is a sorted list of all entity type integers in the map
    """
    try:
        map_data = read_map_file(map_path)
        entity_types = extract_entity_types(map_data)

        # Remove ninja (type 0) from consideration as it's always present
        entity_types.discard(EntityType.NINJA)

        # Store all entity types for the result
        all_entity_types = sorted(list(entity_types))

        # Check if all entity types are in the simple set
        if entity_types.issubset(SIMPLE_ENTITY_TYPES):
            return "Simple", all_entity_types
        else:
            return "Complex", all_entity_types

    except Exception as e:
        logger.error(f"Error processing {map_path.name}: {e}")
        return "Error", []


def categorize_all_maps(maps_dir: Path, folder_name: str) -> Dict[str, Dict[str, any]]:
    """
    Categorize all maps in the specified directory.

    Args:
        maps_dir: Path to directory containing map files
        folder_name: Name of the folder being processed (for logging)

    Returns:
        Dictionary mapping map names to their categories and entity information
    """
    results = {
        "simple": [],
        "complex": [],
        "errors": [],
        "summary": {
            "total_maps": 0,
            "simple_count": 0,
            "complex_count": 0,
            "error_count": 0,
        },
    }

    # Get all map files in the directory
    map_files = sorted([f for f in maps_dir.iterdir() if f.is_file()])

    logger.info(f"Processing {len(map_files)} maps from {folder_name}")
    logger.info("")

    for map_file in map_files:
        map_name = map_file.name
        results["summary"]["total_maps"] += 1

        category, entity_types = categorize_map(map_file)

        if category == "Simple":
            results["simple"].append(
                {
                    "name": map_name,
                    "folder": folder_name,
                    "entity_types": [str(et) for et in entity_types],
                }
            )
            results["summary"]["simple_count"] += 1
            logger.info(f"✓ Simple:  {map_name}")
        elif category == "Complex":
            results["complex"].append(
                {
                    "name": map_name,
                    "folder": folder_name,
                    "entity_types": [str(et) for et in entity_types],
                }
            )
            results["summary"]["complex_count"] += 1
            logger.info(f"✗ Complex: {map_name}")
        else:
            results["errors"].append({"name": map_name, "folder": folder_name})
            results["summary"]["error_count"] += 1
            logger.warning(f"! Error:   {map_name}")

    return results


def main():
    """Main entry point for the script."""
    # Set up paths
    script_dir = Path(__file__).parent
    official_maps_dir = script_dir / "nclone" / "maps" / "official"
    folders_to_process = ["S", "SI", "SL"]
    output_file = script_dir / "map_categorization.json"

    logger.info("=" * 70)
    logger.info("N++ Map Categorization Tool")
    logger.info("=" * 70)
    logger.info("")
    logger.info(
        f"Source: {official_maps_dir} (folders: {', '.join(folders_to_process)})"
    )
    logger.info(f"Output: {output_file}")
    logger.info("")
    logger.info("Simple maps contain only:")
    logger.info("  - Exit doors and exit switches")
    logger.info("  - Toggle mines (any state)")
    logger.info("  - Locked doors and locked door switches")
    logger.info("  - Gold")
    logger.info("")
    logger.info("Complex maps contain any other entity types")
    logger.info("")
    logger.info("=" * 70)
    logger.info("")

    # Combined results across all folders
    combined_results = {
        "simple": [],
        "complex": [],
        "errors": [],
        "summary": {
            "total_maps": 0,
            "simple_count": 0,
            "complex_count": 0,
            "error_count": 0,
        },
    }

    # Process each folder
    for folder_name in folders_to_process:
        maps_dir = official_maps_dir / folder_name

        if not maps_dir.exists():
            logger.warning(f"Maps directory not found: {maps_dir}")
            logger.info("")
            continue

        # Categorize all maps in this folder
        folder_results = categorize_all_maps(maps_dir, folder_name)

        # Merge results
        combined_results["simple"].extend(folder_results["simple"])
        combined_results["complex"].extend(folder_results["complex"])
        combined_results["errors"].extend(folder_results["errors"])
        combined_results["summary"]["total_maps"] += folder_results["summary"][
            "total_maps"
        ]
        combined_results["summary"]["simple_count"] += folder_results["summary"][
            "simple_count"
        ]
        combined_results["summary"]["complex_count"] += folder_results["summary"][
            "complex_count"
        ]
        combined_results["summary"]["error_count"] += folder_results["summary"][
            "error_count"
        ]

        logger.info("")

    # Save results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"Total maps:   {combined_results['summary']['total_maps']}")
    logger.info(f"Simple maps:  {combined_results['summary']['simple_count']}")
    logger.info(f"Complex maps: {combined_results['summary']['complex_count']}")
    if combined_results["summary"]["error_count"] > 0:
        logger.info(f"Errors:       {combined_results['summary']['error_count']}")
    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info("")


if __name__ == "__main__":
    main()
