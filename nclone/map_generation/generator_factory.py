"""
Generator factory for creating configured map generators.

This module provides a centralized factory for instantiating map generators
with preset configurations. It acts as the bridge between configuration
and generator implementation.
"""

from typing import Dict, Type
from .map import Map
from .map_maze import MazeGenerator
from .map_jump_required import MapJumpRequired
from .map_hills import MapHills
from .map_vertical_corridor import MapVerticalCorridor
from .map_horizontal_corridor import MapHorizontalCorridor
from .map_mine_maze import MapMineMaze
from .map_islands import MapIslands
from .map_jump_platforms import MapJumpPlatforms
from .map_multi_chamber import MultiChamberGenerator
from .map_single_chamber import SingleChamberGenerator
from .map_corridors import MapCorridors
from .map_basic_jump import MapBasicJump
from .map_wall_jump import MapWallJump
from .map_mine_path import MapMinePath
from .map_slopes import MapSlopes
from .map_curves import MapCurves
from .map_multi_jump import MapMultiJump
from .map_composite import MapComposite
from .map_precision import MapPrecision
from .map_open_field import MapOpenField
from .map_round_corner import MapRoundCorner
from .map_loop_corridor import MapLoopCorridor
from .map_simple_corridor import MapSimpleCorridor
from .map_platform_stairs import MapPlatformStairs
from .map_long_jump import MapLongJump
from .map_drop_landing import MapDropLanding
from .map_cavern_chambers import MapCavernChambers
from .map_mine_overlay import MapMineOverlay
from .map_hazard_overlay import MapHazardOverlay
from .generator_configs import GENERATOR_PRESETS


GENERATOR_REGISTRY: Dict[str, Type[Map]] = {
    "maze": MazeGenerator,
    "jump_required": MapJumpRequired,
    "hills": MapHills,
    "vertical_corridor": MapVerticalCorridor,
    "horizontal_corridor": MapHorizontalCorridor,
    "mine_maze": MapMineMaze,
    "islands": MapIslands,
    "jump_platforms": MapJumpPlatforms,
    "multi_chamber": MultiChamberGenerator,
    "single_chamber": SingleChamberGenerator,
    "corridors": MapCorridors,
    "basic_jump": MapBasicJump,
    "wall_jump": MapWallJump,
    "mine_path": MapMinePath,
    "slopes": MapSlopes,
    "curves": MapCurves,
    "multi_jump": MapMultiJump,
    "composite": MapComposite,
    "precision": MapPrecision,
    "open_field": MapOpenField,
    "round_corner": MapRoundCorner,
    "loop_corridor": MapLoopCorridor,
    "simple_corridor": MapSimpleCorridor,
    "platform_stairs": MapPlatformStairs,
    "long_jump": MapLongJump,
    "drop_landing": MapDropLanding,
    "cavern_chambers": MapCavernChambers,
    "mine_overlay": MapMineOverlay,
    "hazard_overlay": MapHazardOverlay,
}


class GeneratorFactory:
    """Factory for creating and configuring map generators."""

    @staticmethod
    def create_generator(generator_type: str, config, seed: int) -> Map:
        """Create and configure a generator with the given config.

        Args:
            generator_type: Key from GENERATOR_REGISTRY
            config: Config dataclass instance with parameters
            seed: Random seed for generation

        Returns:
            Configured Map generator ready to call .generate()

        Raises:
            ValueError: If generator_type is not recognized
        """
        if generator_type not in GENERATOR_REGISTRY:
            raise ValueError(
                f"Unknown generator type: {generator_type}. "
                f"Available: {list(GENERATOR_REGISTRY.keys())}"
            )

        generator_class = GENERATOR_REGISTRY[generator_type]
        generator = generator_class(seed=seed)

        # Apply config to generator attributes (convert to uppercase class constants)
        for key, value in config.__dict__.items():
            attr_name = key.upper()
            setattr(generator, attr_name, value)

        return generator

    @staticmethod
    def create_from_preset(generator_type: str, preset: str, seed: int) -> Map:
        """Convenience method to create generator using preset configs.

        Args:
            generator_type: Key from GENERATOR_REGISTRY
            preset: Preset name from GENERATOR_PRESETS
            seed: Random seed for generation

        Returns:
            Configured Map generator ready to call .generate()

        Raises:
            ValueError: If generator_type or preset is not recognized
        """
        if generator_type not in GENERATOR_PRESETS:
            raise ValueError(
                f"Unknown generator type: {generator_type}. "
                f"Available: {list(GENERATOR_PRESETS.keys())}"
            )

        if preset not in GENERATOR_PRESETS[generator_type]:
            raise ValueError(
                f"Unknown preset '{preset}' for generator '{generator_type}'. "
                f"Available: {list(GENERATOR_PRESETS[generator_type].keys())}"
            )

        config = GENERATOR_PRESETS[generator_type][preset]
        return GeneratorFactory.create_generator(generator_type, config, seed)


from dataclasses import dataclass as _dataclass


@_dataclass
class _NullConfig:
    """Empty config for composable-template generators (no UPPERCASE patching)."""


def _register_compose_templates() -> None:
    """Expose every registered composable template as a ``template_<name>``
    generator with a ``default`` preset, additively. This touches no existing
    generator/preset/catalog entry, so the legacy pipeline is byte-unchanged.
    """
    from .compose.registry import TEMPLATE_REGISTRY
    from .compose.adapters import make_template_generator

    for tname in TEMPLATE_REGISTRY:
        key = f"template_{tname}"
        if key not in GENERATOR_REGISTRY:
            GENERATOR_REGISTRY[key] = make_template_generator(tname)
        GENERATOR_PRESETS.setdefault(key, {"default": _NullConfig()})


_register_compose_templates()
