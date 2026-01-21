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
    - _cached_locked_door_switches: Cached door switch entities
    - _cached_switch_states: Cached switch collection states

    Args:
        env: Environment instance (NppEnvironment or similar)
        verbose: If True, print cache clearing operations

    Example:
        >>> clear_door_feature_caches(env, verbose=True)
        Cleared door feature caches
    """
    if hasattr(env, "_locked_door_cache"):
        if env._locked_door_cache is not None and hasattr(env._locked_door_cache, "clear"):
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
        if env.door_feature_cache is not None and hasattr(env.door_feature_cache, "clear"):
            env.door_feature_cache.clear()
        if verbose:
            print("  - Cleared door_feature_cache")

    if hasattr(env, "_has_locked_doors"):
        env._has_locked_doors = False
        if verbose:
            print("  - Reset _has_locked_doors")

    # CRITICAL: These are Optional[list] types - set to None to invalidate
    # (they get re-populated on first access from nplay_headless)
    if hasattr(env, "_cached_locked_doors"):
        env._cached_locked_doors = None
        if verbose:
            print("  - Cleared _cached_locked_doors")

    if hasattr(env, "_cached_locked_door_switches"):
        env._cached_locked_door_switches = None
        if verbose:
            print("  - Cleared _cached_locked_door_switches")

    if hasattr(env, "_cached_switch_states"):
        env._cached_switch_states = None
        if verbose:
            print("  - Reset _cached_switch_states")

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

    # Clear graph builder cache (_level_graph_cache and _flood_fill_cache)
    # This prevents accumulation of level graphs across resets
    if hasattr(env, "graph_builder") and env.graph_builder is not None:
        env.graph_builder.clear_cache()
        if verbose:
            print("  - Cleared graph builder cache")

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
    
    # Clear static position caches (goal positions from previous level)
    # CRITICAL: These must be cleared when loading a new level to prevent
    # using old goal positions with new level's graph (causes "goal not found" errors)
    if hasattr(env, "_cached_switch_pos"):
        env._cached_switch_pos = None
        if verbose:
            print("  - Cleared _cached_switch_pos")
    if hasattr(env, "_cached_exit_pos"):
        env._cached_exit_pos = None
        if verbose:
            print("  - Cleared _cached_exit_pos")

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


def clear_graph_caches_for_curriculum_load(env: Any, verbose: bool = False) -> None:
    """
    Clear graph and path-related caches when curriculum loads a new level.

    This function should be called AFTER load_map_from_map_data() but BEFORE
    env.reset() to ensure graph caches are cleared before the new level's
    graph is built.

    CRITICAL: This fixes the "graph pollution" bug where goal positions from
    previous levels persist when using curriculum learning with SubprocVecEnv.

    Clears:
    - Level data caches (_cached_level_data, _cached_entities)
    - Static position caches (_cached_switch_pos, _cached_exit_pos)
    - Locked door caches from base_environment.py:
      - _cached_locked_doors, _cached_locked_door_switches, _cached_toggle_mines
    - Locked door caches from npp_environment.py:
      - _has_locked_doors, _cached_switch_states
      - _locked_door_cache, door_feature_cache
      - _last_switch_state_hash, _last_ninja_grid_cell
    - Graph builder cache (_level_graph_cache, _flood_fill_cache)
    - Graph state (current_graph, current_graph_data, _graph_data_cache)
    - Graph debug caches (_graph_debug_cached_door_states)
    - Path calculator cache (level_cache, mine_proximity_cache)
    - Reachability state
    - Pathfinding utility caches (surface area cache)

    Args:
        env: Environment instance (NppEnvironment or similar)
        verbose: If True, print cache clearing operations

    Example:
        >>> # In curriculum wrapper reset:
        >>> base_env.nplay_headless.load_map_from_map_data(map_data)
        >>> clear_graph_caches_for_curriculum_load(base_env, verbose=True)
        >>> obs, info = env.reset(options={"skip_map_load": True})
    """
    if verbose:
        print("Clearing graph caches for curriculum level load...")

    # Clear level data caches first (ensures fresh level_data extraction)
    clear_level_data_caches(env, verbose=verbose)

    # Clear static position caches (goal positions from previous level)
    if hasattr(env, "_cached_switch_pos"):
        env._cached_switch_pos = None
        if verbose:
            print("  - Cleared _cached_switch_pos")

    if hasattr(env, "_cached_exit_pos"):
        env._cached_exit_pos = None
        if verbose:
            print("  - Cleared _cached_exit_pos")

    # Clear locked door caches from base_environment.py (Optional[list] types)
    if hasattr(env, "_cached_locked_doors"):
        # Handle both list and None types - always set to None for fresh reload
        env._cached_locked_doors = None
        if verbose:
            print("  - Cleared _cached_locked_doors (base)")

    if hasattr(env, "_cached_locked_door_switches"):
        env._cached_locked_door_switches = None
        if verbose:
            print("  - Cleared _cached_locked_door_switches")

    if hasattr(env, "_cached_toggle_mines"):
        env._cached_toggle_mines = None
        if verbose:
            print("  - Cleared _cached_toggle_mines")

    # Clear locked door caches from npp_environment.py (level-specific)
    # These are set in _initialize_locked_door_caches() and must be reset
    if hasattr(env, "_has_locked_doors"):
        env._has_locked_doors = False
        if verbose:
            print("  - Reset _has_locked_doors")

    if hasattr(env, "_cached_switch_states"):
        env._cached_switch_states = None
        if verbose:
            print("  - Cleared _cached_switch_states")

    if hasattr(env, "_switch_states_changed"):
        env._switch_states_changed = True  # Mark as changed to force rebuild
        if verbose:
            print("  - Reset _switch_states_changed")

    # Clear door path distance cache (Dict from npp_environment.py)
    if hasattr(env, "_locked_door_cache"):
        if env._locked_door_cache is not None and hasattr(env._locked_door_cache, "clear"):
            env._locked_door_cache.clear()
        if verbose:
            print("  - Cleared _locked_door_cache")

    # Clear precomputed door feature cache (PrecomputedDoorFeatureCache from npp_environment.py)
    if hasattr(env, "door_feature_cache"):
        if env.door_feature_cache is not None and hasattr(env.door_feature_cache, "clear"):
            env.door_feature_cache.clear()
        if verbose:
            print("  - Cleared door_feature_cache")

    # Clear door cache invalidation keys
    if hasattr(env, "_last_switch_state_hash"):
        env._last_switch_state_hash = None
        if verbose:
            print("  - Reset _last_switch_state_hash")

    if hasattr(env, "_last_ninja_grid_cell"):
        env._last_ninja_grid_cell = None
        if verbose:
            print("  - Reset _last_ninja_grid_cell")

    # Clear graph debug door states cache (from graph_mixin.py)
    if hasattr(env, "_graph_debug_cached_door_states"):
        env._graph_debug_cached_door_states = None
        if verbose:
            print("  - Cleared _graph_debug_cached_door_states")

    # Clear graph builder cache (prevents stale graph nodes/edges)
    if hasattr(env, "graph_builder") and env.graph_builder is not None:
        env.graph_builder.clear_cache()
        if verbose:
            print("  - Cleared graph builder cache")

    # Clear graph state
    if hasattr(env, "current_graph"):
        env.current_graph = None
        if verbose:
            print("  - Cleared current_graph")

    if hasattr(env, "current_graph_data"):
        env.current_graph_data = None
        if verbose:
            print("  - Cleared current_graph_data")

    # Clear graph data cache (per-level GraphData cache)
    if hasattr(env, "_graph_data_cache"):
        env._graph_data_cache.clear()
        if verbose:
            print("  - Cleared _graph_data_cache")

    if hasattr(env, "_current_level_id"):
        env._current_level_id = None
        if verbose:
            print("  - Cleared _current_level_id")

    # Clear reachable area scale cache
    if hasattr(env, "_reachable_area_scale_cache"):
        env._reachable_area_scale_cache.clear()
        if verbose:
            print("  - Cleared _reachable_area_scale_cache")

    # Clear path calculator cache (goal positions stored here!)
    if hasattr(env, "_path_calculator") and env._path_calculator is not None:
        env._path_calculator.clear_cache()
        if verbose:
            print("  - Cleared path calculator cache (goal positions)")

    # Clear door feature cache
    if hasattr(env, "door_feature_cache"):
        env.door_feature_cache.clear()
        if verbose:
            print("  - Cleared door_feature_cache")

    # Clear switch/mine state tracking
    if hasattr(env, "last_switch_states"):
        env.last_switch_states.clear()
        if verbose:
            print("  - Cleared last_switch_states")

    if hasattr(env, "last_mine_states"):
        env.last_mine_states = {}
        if verbose:
            print("  - Cleared last_mine_states")

    # Clear pathfinding utility caches (module-level surface area cache)
    clear_pathfinding_utility_caches(verbose=verbose)

    if verbose:
        print("Graph caches cleared for curriculum level load")


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
                # CRITICAL FIX: Clear debug overlay caches including text_cache
                # This prevents unbounded growth of text_cache across episodes
                clear_debug_overlay_caches(debug_renderer, verbose=verbose)

    if verbose:
        print("All caches cleared for reset")


def clear_episode_caches_only(env: Any, verbose: bool = False) -> None:
    """Clear only episode-specific caches for fast same-level resets.
    
    This minimal cache clearing function is used with fast_reset() to clear only
    the caches that MUST be cleared between episodes on the same level.
    
    CRITICAL: This should be used ONLY when the level has not changed. For new
    levels, use clear_all_caches_for_new_level() instead.
    
    Clears (episode-specific only):
    - _prev_obs_cache: Previous observation cache
    - current_ep_reward: Episode reward accumulator
    - current_route: Episode trajectory tracking
    - _cached_observation: Current observation cache
    - Render caches: Must be cleared as entity states change
    
    Does NOT clear (level-persistent):
    - _cached_level_data: Level geometry (unchanged)
    - _cached_entities: Entity data (unchanged)
    - Graph caches: Level graph (unchanged)
    - Pathfinding utility caches: Surface area cache (unchanged)
    - Door feature caches: Can be invalidated via switch state tracking
    - Renderer surface caches: Tile rendering (unchanged)
    
    Args:
        env: Environment instance (NppEnvironment or similar)
        verbose: If True, print cache clearing operations
    
    Example:
        >>> # In fast reset path:
        >>> env.nplay_headless.fast_reset()
        >>> clear_episode_caches_only(env, verbose=True)
        Cleared episode-specific caches (fast reset)
    """
    if verbose:
        print("Clearing episode-specific caches (fast reset)...")
    
    # Clear observation cache
    if hasattr(env, "_prev_obs_cache"):
        env._prev_obs_cache = None
        if verbose:
            print("  - Cleared _prev_obs_cache")
    
    # Reset episode reward accumulator
    if hasattr(env, "current_ep_reward"):
        env.current_ep_reward = 0
        if verbose:
            print("  - Reset current_ep_reward")
    
    # Clear episode route tracking
    if hasattr(env, "current_route"):
        if hasattr(env.current_route, "clear"):
            env.current_route.clear()
        else:
            env.current_route = []
        if verbose:
            print("  - Cleared current_route")
    
    # Clear current observation cache
    if hasattr(env, "_cached_observation"):
        env._cached_observation = None
        if verbose:
            print("  - Cleared _cached_observation")
    
    # Clear render caches (entity positions/states change each episode)
    # Even though we're on the same level, entities move during the episode
    # and render caches must be cleared to show the reset state
    if hasattr(env, "nplay_headless"):
        nplay_headless = env.nplay_headless
        
        # Render caches (surface and buffer)
        if hasattr(nplay_headless, "cached_render_surface"):
            nplay_headless.cached_render_surface = None
            if verbose:
                print("  - Cleared cached_render_surface")
        
        if hasattr(nplay_headless, "cached_render_buffer"):
            nplay_headless.cached_render_buffer = None
            if verbose:
                print("  - Cleared cached_render_buffer")
    
    if verbose:
        print("Cleared episode-specific caches (fast reset)")
