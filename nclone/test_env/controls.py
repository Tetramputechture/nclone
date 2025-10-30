"""Keyboard controls and action mapping for test environment."""

import pygame
from typing import Callable, Dict, Optional


class ActionMapper:
    """Maps keyboard inputs to environment actions."""
    
    # Action constants
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    JUMP_LEFT = 4
    JUMP_RIGHT = 5
    
    @staticmethod
    def get_action_from_keys(headless: bool = False) -> int:
        """Get environment action from current keyboard state.
        
        Args:
            headless: If True, always returns NOOP
            
        Returns:
            Action integer (0-5)
        """
        if headless:
            return ActionMapper.NOOP
        
        keys = pygame.key.get_pressed()
        
        # Check jump first (up arrow)
        if keys[pygame.K_UP]:
            if keys[pygame.K_LEFT]:
                return ActionMapper.JUMP_LEFT
            elif keys[pygame.K_RIGHT]:
                return ActionMapper.JUMP_RIGHT
            else:
                return ActionMapper.JUMP
        else:
            if keys[pygame.K_LEFT]:
                return ActionMapper.LEFT
            elif keys[pygame.K_RIGHT]:
                return ActionMapper.RIGHT
        
        return ActionMapper.NOOP


class KeyboardController:
    """Handles keyboard event processing for test environment.
    
    This class manages all keyboard shortcuts and event handling for the
    test environment, routing events to appropriate handlers.
    """
    
    def __init__(self):
        """Initialize keyboard controller."""
        self.event_handlers: Dict[int, Callable] = {}
        self.modifier_handlers: Dict[tuple, Callable] = {}
        
    def register_key(self, key: int, handler: Callable, 
                    modifiers: Optional[tuple] = None):
        """Register a key handler.
        
        Args:
            key: pygame key constant (e.g., pygame.K_r)
            handler: Callback function to execute when key is pressed
            modifiers: Optional tuple of modifier keys (e.g., (pygame.KMOD_CTRL,))
        """
        if modifiers:
            self.modifier_handlers[(key, modifiers)] = handler
        else:
            self.event_handlers[key] = handler
    
    def handle_keydown(self, event: pygame.event.Event) -> bool:
        """Handle KEYDOWN event.
        
        Args:
            event: pygame KEYDOWN event
            
        Returns:
            True if event was handled, False otherwise
        """
        # Check modifier key combinations first
        for (key, modifiers), handler in self.modifier_handlers.items():
            if event.key == key:
                # Check if all required modifiers are pressed
                if all(event.mod & mod for mod in modifiers):
                    handler()
                    return True
        
        # Check simple key handlers
        if event.key in self.event_handlers:
            self.event_handlers[event.key]()
            return True
        
        return False
    
    def clear_handlers(self):
        """Clear all registered handlers."""
        self.event_handlers.clear()
        self.modifier_handlers.clear()


def setup_default_controls(controller: KeyboardController, 
                          env, 
                          config,
                          state: Dict) -> KeyboardController:
    """Setup default keyboard controls for test environment.
    
    Args:
        controller: KeyboardController instance to configure
        env: Test environment instance
        config: TestConfig with configuration settings
        state: Dictionary containing runtime state (flags, managers, etc.)
        
    Returns:
        Configured KeyboardController
    """
    # Reset (R key)
    def handle_reset():
        if state.get('recorder') and state['recorder'].is_recording:
            player_won = env.nplay_headless.ninja_has_won()
            state['recorder'].stop_recording(success=player_won, save=player_won)
        
        if state.get('generator_tester'):
            try:
                new_map = state['generator_tester'].generate_map()
                env.nplay_headless.load_map_from_map_data(new_map.map_data())
                env.observation_processor.reset()
                env.reward_calculator.reset()
                env.truncation_checker.reset()
                env.current_ep_reward = 0
                initial_obs = env._get_observation()
                observation = env._process_observation(initial_obs)
                print(f"Generated new map: {state['generator_tester'].get_info_string()}")
                if state.get('show_ascii_on_reset'):
                    print("\nASCII Visualization:")
                    print(new_map.to_ascii(show_coords=False))
                    print()
            except Exception as e:
                print(f"Error generating map: {e}")
                import traceback
                traceback.print_exc()
        else:
            observation, info = env.reset()
    
    controller.register_key(pygame.K_r, handle_reset)
    
    # Recording controls (B key)
    if config.record:
        def toggle_recording():
            recorder = state.get('recorder')
            if not recorder:
                return
            
            if not recorder.is_recording:
                current_map_data = list(env.nplay_headless.current_map_data)
                env.nplay_headless.load_map_from_map_data(current_map_data)
                env.observation_processor.reset()
                env.reward_calculator.reset()
                env.truncation_checker.reset()
                env.current_ep_reward = 0
                initial_obs = env._get_observation()
                observation = env._process_observation(initial_obs)
                recorder.start_recording(env.current_map_name, current_map_data)
                print("🔴 Recording started")
            else:
                player_won = env.nplay_headless.ninja_has_won()
                recorder.stop_recording(success=player_won, save=player_won)
                print("⏹️  Recording stopped")
        
        controller.register_key(pygame.K_b, toggle_recording)
    
    # Test suite navigation
    if config.test_suite:
        def next_level():
            if not state.get('test_suite_loader'):
                return
            test_suite_level_ids = state['test_suite_level_ids']
            current_level_idx = state['current_level_idx']
            current_level_idx = (current_level_idx + 1) % len(test_suite_level_ids)
            level_id = test_suite_level_ids[current_level_idx]
            level = state['test_suite_loader'].get_level(level_id)
            env.nplay_headless.load_map_from_map_data(level["map_data"])
            print(f"Loaded level {current_level_idx + 1}/{len(test_suite_level_ids)}: {level_id}")
            state['current_level_idx'] = current_level_idx
        
        def prev_level():
            if not state.get('test_suite_loader'):
                return
            test_suite_level_ids = state['test_suite_level_ids']
            current_level_idx = state['current_level_idx']
            current_level_idx = (current_level_idx - 1) % len(test_suite_level_ids)
            level_id = test_suite_level_ids[current_level_idx]
            level = state['test_suite_loader'].get_level(level_id)
            env.nplay_headless.load_map_from_map_data(level["map_data"])
            print(f"Loaded level {current_level_idx + 1}/{len(test_suite_level_ids)}: {level_id}")
            state['current_level_idx'] = current_level_idx
        
        controller.register_key(pygame.K_n, next_level)
        controller.register_key(pygame.K_p, prev_level)
    
    # Debug overlay toggles
    def toggle_entities():
        state['entities_debug_enabled'] = not state.get('entities_debug_enabled', False)
        env.set_entity_debug_enabled(state['entities_debug_enabled'])
        print(f"Entity debug: {'ON' if state['entities_debug_enabled'] else 'OFF'}")
    
    def toggle_collision():
        state['collision_debug_enabled'] = not state.get('collision_debug_enabled', False)
        env.set_collision_debug_enabled(state['collision_debug_enabled'])
        print(f"Collision debug: {'ON' if state['collision_debug_enabled'] else 'OFF'}")
    
    def list_generators():
        if state.get('generator_tester'):
            state['generator_tester'].list_current_generators()
    
    def toggle_interactive_graph():
        state['interactive_graph_enabled'] = not state.get('interactive_graph_enabled', False)
        print(f"Interactive graph: {'ON' if state['interactive_graph_enabled'] else 'OFF'}")
    
    controller.register_key(pygame.K_e, toggle_entities)
    controller.register_key(pygame.K_c, toggle_collision)
    controller.register_key(pygame.K_l, list_generators)
    controller.register_key(pygame.K_i, toggle_interactive_graph)
    
    # Generator testing controls
    if config.test_generators:
        def next_generator():
            if state.get('generator_tester'):
                state['generator_tester'].next_generator()
        
        def prev_generator():
            if state.get('generator_tester'):
                state['generator_tester'].previous_generator()
        
        def next_category():
            if state.get('generator_tester'):
                state['generator_tester'].next_category()
        
        def prev_category():
            if state.get('generator_tester'):
                state['generator_tester'].previous_category()
        
        def toggle_ascii():
            state['show_ascii_on_reset'] = not state.get('show_ascii_on_reset', False)
            print(f"ASCII visualization: {'ON' if state['show_ascii_on_reset'] else 'OFF'}")
        
        controller.register_key(pygame.K_g, next_generator)
        controller.register_key(pygame.K_k, next_category)
        controller.register_key(pygame.K_v, toggle_ascii)
        
        # Number keys for generator selection
        for num_key, gen_idx in {
            pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2,
            pygame.K_4: 3, pygame.K_5: 4, pygame.K_6: 5,
            pygame.K_7: 6, pygame.K_8: 7, pygame.K_9: 8,
        }.items():
            controller.register_key(
                num_key,
                lambda idx=gen_idx: state.get('generator_tester') and 
                                   state['generator_tester'].jump_to_generator(idx)
            )
    
    # Reachability controls
    if config.visualize_reachability or config.reachability_from_ninja:
        def toggle_reachability():
            state['reachability_debug_enabled'] = not state.get('reachability_debug_enabled', False)
            print(f"Reachability: {'ON' if state['reachability_debug_enabled'] else 'OFF'}")
        
        def update_reachability():
            # Update reachability from current ninja position
            print("Updating reachability analysis...")
        
        def toggle_subgoals():
            state['show_subgoals'] = not state.get('show_subgoals', False)
            print(f"Subgoals: {'ON' if state['show_subgoals'] else 'OFF'}")
        
        def toggle_frontiers():
            state['show_frontiers'] = not state.get('show_frontiers', False)
            print(f"Frontiers: {'ON' if state['show_frontiers'] else 'OFF'}")
        
        controller.register_key(pygame.K_t, toggle_reachability)
        controller.register_key(pygame.K_n, update_reachability)
        controller.register_key(pygame.K_u, toggle_subgoals)
        controller.register_key(pygame.K_f, toggle_frontiers)
    
    # Subgoal visualization controls
    if config.visualize_subgoals:
        def toggle_subgoal_viz():
            state['subgoal_debug_enabled'] = not state.get('subgoal_debug_enabled', False)
            env.set_subgoal_debug_enabled(state['subgoal_debug_enabled'])
            print(f"Subgoal visualization: {'ON' if state['subgoal_debug_enabled'] else 'OFF'}")
        
        def cycle_viz_mode():
            modes = ["basic", "detailed", "reachability"]
            current_mode = state.get('subgoal_mode', 'detailed')
            current_idx = modes.index(current_mode)
            new_idx = (current_idx + 1) % len(modes)
            state['subgoal_mode'] = modes[new_idx]
            print(f"Visualization mode: {state['subgoal_mode']}")
        
        controller.register_key(pygame.K_s, toggle_subgoal_viz)
        controller.register_key(pygame.K_m, cycle_viz_mode)
    
    # Path-aware controls
    if config.test_path_aware:
        def toggle_path_distances():
            if not state.get('subgoal_debug_enabled'):
                state['path_distances_debug_enabled'] = not state.get('path_distances_debug_enabled', False)
                print(f"Path distances: {'ON' if state['path_distances_debug_enabled'] else 'OFF'}")
        
        def toggle_adjacency_graph():
            state['adjacency_graph_debug_enabled'] = not state.get('adjacency_graph_debug_enabled', False)
            print(f"Adjacency graph: {'ON' if state['adjacency_graph_debug_enabled'] else 'OFF'}")
        
        def toggle_blocked_entities():
            if not state.get('recorder'):
                state['blocked_entities_debug_enabled'] = not state.get('blocked_entities_debug_enabled', False)
                print(f"Blocked entities: {'ON' if state['blocked_entities_debug_enabled'] else 'OFF'}")
        
        controller.register_key(pygame.K_p, toggle_path_distances)
        controller.register_key(pygame.K_a, toggle_adjacency_graph)
        controller.register_key(pygame.K_b, toggle_blocked_entities)
    
    return controller
