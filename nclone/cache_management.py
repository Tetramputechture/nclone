"""
Centralized cache management for N++ environment components.

Provides DRY functions for clearing various cache types on level load,
reset, and manual cache busting operations.

This module eliminates code duplication across npp_environment.py,
nplay_headless.py, and test_environment.py by providing a single
source of truth for cache management.
"""

from typing import Any


def clear_level_data_caches(env: Any, verbose: bool = False) -> None:
    """
    Clear level data caches in the environment.

    Clears:
    - _cached_level_data: Level geometry cache
    - _cached_entities: Entity data cache

    Args:
        env: Environment instance (NppEnvironment or similar)
        verbose: If True, print cache clearing operations

    Example:
        >>> clear_level_data_caches(env, verbose=True)
        Cleared level data caches
    """
    if hasattr(env, "_cached_level_data"):
        env._cached_level_data = None
        if verbose:
            print("  - Cleared _cached_level_data")

    if hasattr(env, "_cached_entities"):
        env._cached_entities = None
        if verbose:
            print("  - Cleared _cached_entities")

    if verbose and (
        hasattr(env, "_cached_level_data") or hasattr(env, "_cached_entities")
    ):
        print("Cleared level data caches")


def clear_door_feature_caches(env: Any, verbose: bool = False) -> None:
    """
    Clear door feature caches in the environment.

    Clears:
    - _locked_door_cache: Path distances by switch state/position
    - _last_switch_state_hash: Cache key for switch states
    - _last_ninja_grid_cell: Cache key for ninja position
    - _switch_states_changed: Invalidation flag
    - door_feature_cache: Precomputed door features
    - _has_locked_doors: Level door presence flag
    - _cached_locked_doors: Cached door entities
    - _cached_switch_states: Cached switch collection states

    Args:
        env: Environment instance (NppEnvironment or similar)
        verbose: If True, print cache clearing operations

    Example:
        >>> clear_door_feature_caches(env, verbose=True)
        Cleared door feature caches
    """
    if hasattr(env, "_locked_door_cache"):
        env._locked_door_cache.clear()
        if verbose:
            print("  - Cleared _locked_door_cache")

    if hasattr(env, "_last_switch_state_hash"):
        env._last_switch_state_hash = None
        if verbose:
            print("  - Reset _last_switch_state_hash")

    if hasattr(env, "_last_ninja_grid_cell"):
        env._last_ninja_grid_cell = None
        if verbose:
            print("  - Reset _last_ninja_grid_cell")

    if hasattr(env, "_switch_states_changed"):
        env._switch_states_changed = True
        if verbose:
            print("  - Set _switch_states_changed = True")

    if hasattr(env, "door_feature_cache"):
        env.door_feature_cache.clear()
        if verbose:
            print("  - Cleared door_feature_cache")

    if hasattr(env, "_has_locked_doors"):
        env._has_locked_doors = False
        if verbose:
            print("  - Reset _has_locked_doors")

    if hasattr(env, "_cached_locked_doors"):
        env._cached_locked_doors.clear()
        if verbose:
            print("  - Cleared _cached_locked_doors")

    if hasattr(env, "_cached_switch_states"):
        env._cached_switch_states = None
        if verbose:
            print("  - Reset _cached_switch_states")

    # Exit features cache (position-based caching for performance)
    if hasattr(env, "_cached_exit_features"):
        env._cached_exit_features = None
        if verbose:
            print("  - Reset _cached_exit_features")

    if hasattr(env, "_last_exit_cache_ninja_pos"):
        env._last_exit_cache_ninja_pos = None
        if verbose:
            print("  - Reset _last_exit_cache_ninja_pos")

    if verbose:
        print("Cleared door feature caches")


def clear_render_caches(nplay_headless: Any, verbose: bool = False) -> None:
    """
    Clear rendering caches in NPlayHeadless.

    Clears:
    - cached_render_surface: Cached pygame surface
    - cached_render_buffer: Cached numpy buffer
    - current_tick / last_rendered_tick: Render state tracking

    Args:
        nplay_headless: NPlayHeadless instance
        verbose: If True, print cache clearing operations

    Example:
        >>> clear_render_caches(nplay_headless, verbose=True)
        Cleared render caches
    """
    if hasattr(nplay_headless, "cached_render_surface"):
        nplay_headless.cached_render_surface = None
        if verbose:
            print("  - Cleared cached_render_surface")

    if hasattr(nplay_headless, "cached_render_buffer"):
        nplay_headless.cached_render_buffer = None
        if verbose:
            print("  - Cleared cached_render_buffer")

    if hasattr(nplay_headless, "current_tick"):
        nplay_headless.current_tick = -1
        if verbose:
            print("  - Reset current_tick")

    if hasattr(nplay_headless, "last_rendered_tick"):
        nplay_headless.last_rendered_tick = -1
        if verbose:
            print("  - Reset last_rendered_tick")

    if verbose:
        print("Cleared render caches")


def clear_renderer_surface_caches(sim_renderer: Any, verbose: bool = False) -> None:
    """
    Clear surface caches in NSimRenderer.

    Clears:
    - cached_tile_surface: Cached tile rendering surface
    - cached_entity_surface: Cached entity rendering surface
    - last_entity_state_hash: Entity state hash for cache validation
    - last_init_state: Init state for cache validation

    Args:
        sim_renderer: NSimRenderer instance
        verbose: If True, print cache clearing operations

    Example:
        >>> clear_renderer_surface_caches(sim_renderer, verbose=True)
        Cleared renderer surface caches
    """
    if hasattr(sim_renderer, "cached_tile_surface"):
        sim_renderer.cached_tile_surface = None
        if verbose:
            print("  - Cleared cached_tile_surface")

    if hasattr(sim_renderer, "cached_entity_surface"):
        sim_renderer.cached_entity_surface = None
        if verbose:
            print("  - Cleared cached_entity_surface")

    if hasattr(sim_renderer, "last_entity_state_hash"):
        sim_renderer.last_entity_state_hash = None
        if verbose:
            print("  - Reset last_entity_state_hash")

    if hasattr(sim_renderer, "last_init_state"):
        sim_renderer.last_init_state = None
        if verbose:
            print("  - Reset last_init_state")

    if verbose:
        print("Cleared renderer surface caches")


def clear_pathfinding_caches(
    debug_overlay_renderer: Any, verbose: bool = False
) -> None:
    """
    Clear pathfinding visualization caches in DebugOverlayRenderer.

    Clears:
    - _path_visualization_cache: Pathfinding visualization cache

    Args:
        debug_overlay_renderer: DebugOverlayRenderer instance
        verbose: If True, print cache clearing operations

    Example:
        >>> clear_pathfinding_caches(debug_overlay_renderer, verbose=True)
        Cleared pathfinding caches
    """
    if hasattr(debug_overlay_renderer, "clear_pathfinding_cache"):
        debug_overlay_renderer.clear_pathfinding_cache()
        if verbose:
            print("  - Cleared pathfinding cache")
            print("Cleared pathfinding caches")


def clear_debug_overlay_caches(
    debug_overlay_renderer: Any, verbose: bool = False
) -> None:
    """
    Clear debug overlay surface caches in DebugOverlayRenderer.

    Clears:
    - cached_mine_surface: Cached mine predictor overlay
    - cached_death_surface: Cached death probability overlay
    - cached_exploration_surface: Cached exploration overlay
    - text_cache: Text rendering cache

    Args:
        debug_overlay_renderer: DebugOverlayRenderer instance
        verbose: If True, print cache clearing operations

    Example:
        >>> clear_debug_overlay_caches(debug_overlay_renderer, verbose=True)
        Cleared debug overlay caches
    """
    if hasattr(debug_overlay_renderer, "cached_mine_surface"):
        debug_overlay_renderer.cached_mine_surface = None
        if verbose:
            print("  - Cleared cached_mine_surface")

    if hasattr(debug_overlay_renderer, "cached_death_surface"):
        debug_overlay_renderer.cached_death_surface = None
        if verbose:
            print("  - Cleared cached_death_surface")

    if hasattr(debug_overlay_renderer, "cached_exploration_surface"):
        debug_overlay_renderer.cached_exploration_surface = None
        if verbose:
            print("  - Cleared cached_exploration_surface")

    if hasattr(debug_overlay_renderer, "text_cache"):
        debug_overlay_renderer.text_cache.clear()
        if verbose:
            print("  - Cleared text_cache")

    if verbose:
        print("Cleared debug overlay caches")


def clear_pathfinding_utility_caches(verbose: bool = False) -> None:
    """
    Clear module-level caches in pathfinding_utils.

    Clears:
    - _surface_area_cache: Reachable surface area by level ID

    This should be called on environment reset to prevent stale cache entries
    across level changes.

    Args:
        verbose: If True, print cache clearing operations

    Example:
        >>> clear_pathfinding_utility_caches(verbose=True)
        Cleared pathfinding utility caches
    """
    from nclone.graph.reachability.pathfinding_utils import clear_surface_area_cache

    clear_surface_area_cache()  # Clear all entries

    if verbose:
        print("  - Cleared pathfinding_utils._surface_area_cache")
        print("Cleared pathfinding utility caches")


def reset_graph_state_caches(env: Any, verbose: bool = False) -> None:
    """
    Reset graph and state caches in the environment.

    Calls environment methods to reset:
    - Graph state (_reset_graph_state)
    - Reachability state (_reset_reachability_state)

    Also clears:
    - Pathfinding utility caches (surface area cache)

    Args:
        env: Environment instance (NppEnvironment or similar)
        verbose: If True, print cache clearing operations

    Example:
        >>> reset_graph_state_caches(env, verbose=True)
        Reset graph state caches
    """
    if hasattr(env, "_reset_graph_state"):
        env._reset_graph_state()
        if verbose:
            print("  - Reset graph state")

    if hasattr(env, "_reset_reachability_state"):
        env._reset_reachability_state()
        if verbose:
            print("  - Reset reachability state")

    # Clear pathfinding utility caches
    clear_pathfinding_utility_caches(verbose=verbose)

    if verbose:
        print("Reset graph state caches")


def clear_all_caches_for_new_level(env: Any, verbose: bool = False) -> None:
    """
    Clear all caches when loading a new level.

    This is a comprehensive cache clear that should be called when:
    - Loading a new map
    - Switching to a different level
    - Resetting to a new episode with a different map

    Clears:
    - Level data caches
    - Door feature caches
    - Entity caches
    - Render caches (in nplay_headless)
    - Renderer surface caches (in sim_renderer)
    - Pathfinding caches (in debug_overlay_renderer)
    - Debug overlay caches (in debug_overlay_renderer)
    - Pathfinding utility caches (surface area cache)

    Does NOT reset graph state - call reset_graph_state_caches() separately if needed.

    Args:
        env: Environment instance (NppEnvironment or similar)
        verbose: If True, print all cache clearing operations

    Example:
        >>> clear_all_caches_for_new_level(env, verbose=True)
        Clearing all caches for new level...
        Cleared level data caches
        Cleared door feature caches
        ...
        All caches cleared for new level
    """
    if verbose:
        print("Clearing all caches for new level...")

    # Environment-level caches
    clear_level_data_caches(env, verbose=verbose)
    clear_door_feature_caches(env, verbose=verbose)

    # Clear pathfinding utility caches (module-level caches)
    clear_pathfinding_utility_caches(verbose=verbose)

    # NPlayHeadless caches
    if hasattr(env, "nplay_headless"):
        nplay_headless = env.nplay_headless

        # Render caches
        clear_render_caches(nplay_headless, verbose=verbose)

        # Renderer surface caches
        if hasattr(nplay_headless, "sim_renderer"):
            clear_renderer_surface_caches(nplay_headless.sim_renderer, verbose=verbose)

            # Debug overlay caches
            if hasattr(nplay_headless.sim_renderer, "debug_overlay_renderer"):
                debug_renderer = nplay_headless.sim_renderer.debug_overlay_renderer
                clear_pathfinding_caches(debug_renderer, verbose=verbose)
                clear_debug_overlay_caches(debug_renderer, verbose=verbose)

    if verbose:
        print("All caches cleared for new level")


def clear_all_caches_for_reset(env: Any, verbose: bool = False) -> None:
    """
    Clear all caches when resetting the environment.

    This is similar to clear_all_caches_for_new_level but specifically
    for environment reset operations. Use this when:
    - Calling env.reset()
    - Resetting to the same level (no map change)

    Clears:
    - Level data caches
    - Door feature caches
    - Render caches (in nplay_headless)
    - Pathfinding caches (in debug_overlay_renderer)
    - Pathfinding utility caches (surface area cache)

    Does NOT clear:
    - Entity caches (rebuilt separately)
    - Renderer surface caches (may be reused)
    - Graph state (reset separately via reset_graph_state_caches)

    Args:
        env: Environment instance (NppEnvironment or similar)
        verbose: If True, print all cache clearing operations

    Example:
        >>> clear_all_caches_for_reset(env, verbose=True)
        Clearing all caches for reset...
        Cleared level data caches
        Cleared door feature caches
        ...
        All caches cleared for reset
    """
    if verbose:
        print("Clearing all caches for reset...")

    # Environment-level caches
    clear_level_data_caches(env, verbose=verbose)
    clear_door_feature_caches(env, verbose=verbose)

    # Clear pathfinding utility caches (module-level caches)
    clear_pathfinding_utility_caches(verbose=verbose)

    # NPlayHeadless caches
    if hasattr(env, "nplay_headless"):
        nplay_headless = env.nplay_headless

        # Render caches
        clear_render_caches(nplay_headless, verbose=verbose)

        # Pathfinding caches
        if hasattr(nplay_headless, "sim_renderer"):
            if hasattr(nplay_headless.sim_renderer, "debug_overlay_renderer"):
                debug_renderer = nplay_headless.sim_renderer.debug_overlay_renderer
                clear_pathfinding_caches(debug_renderer, verbose=verbose)

    if verbose:
        print("All caches cleared for reset")
