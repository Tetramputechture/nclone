"""Test suite for physics discovery rewards system."""

import pytest
import math
from collections import deque
from nclone.gym_environment.reward_calculation.physics_discovery_rewards import PhysicsDiscoveryRewards


class TestPhysicsDiscoveryRewards:
    """Test physics discovery reward system functionality."""

    def test_initialization(self):
        """Test that PhysicsDiscoveryRewards initializes correctly."""
        rewards = PhysicsDiscoveryRewards()
        
        assert rewards.movement_efficiency_baseline == 0.5
        assert isinstance(rewards.energy_utilization_history, deque)
        assert rewards.energy_utilization_history.maxlen == 100
        assert isinstance(rewards.visited_physics_states, set)
        assert rewards.wall_jump_count == 0
        assert rewards.buffered_jump_count == 0

    def test_efficiency_bonus_calculation(self):
        """Test energy efficiency bonus calculation."""
        rewards = PhysicsDiscoveryRewards()
        
        # Create test observations with enhanced physics state (40D)
        prev_obs = {
            'player_x': 100.0,
            'player_y': 100.0,
            'game_state': [0.0] * 40  # 40D state with kinetic energy at index 32
        }
        prev_obs['game_state'][32] = 0.5  # Previous kinetic energy
        
        # Current observation with movement and energy change
        current_obs = {
            'player_x': 110.0,  # Moved 10 units horizontally
            'player_y': 100.0,
            'game_state': [0.0] * 40
        }
        current_obs['game_state'][32] = 0.6  # Slightly higher kinetic energy
        
        # Calculate efficiency bonus
        efficiency_reward = rewards._calculate_efficiency_bonus(current_obs, prev_obs)
        
        # Should be positive since we moved distance with little energy change
        assert efficiency_reward >= 0.0
        assert efficiency_reward <= 1.0  # Bonus is capped at 1.0

    def test_efficiency_bonus_with_high_efficiency(self):
        """Test efficiency bonus with high movement efficiency."""
        rewards = PhysicsDiscoveryRewards()
        
        # Scenario: Large distance with small energy change (very efficient)
        prev_obs = {
            'player_x': 100.0, 
            'player_y': 100.0,
            'game_state': [0.0] * 40
        }
        prev_obs['game_state'][32] = 0.5
        
        current_obs = {
            'player_x': 150.0,  # Moved 50 units (large distance)
            'player_y': 100.0,
            'game_state': [0.0] * 40  
        }
        current_obs['game_state'][32] = 0.51  # Tiny energy change
        
        efficiency_reward = rewards._calculate_efficiency_bonus(current_obs, prev_obs)
        
        # Should give high reward for efficient movement
        assert efficiency_reward > 0.5
        assert rewards.energy_efficient_moves == 1

    def test_efficiency_bonus_fallback_without_enhanced_state(self):
        """Test efficiency bonus calculation with fallback velocity estimation."""
        rewards = PhysicsDiscoveryRewards()
        
        # Test with basic observation without enhanced physics state
        prev_obs = {
            'player_x': 100.0,
            'player_y': 100.0,
            'player_xspeed': 2.0,
            'player_yspeed': 0.0,
            'game_state': [0.0] * 32  # Basic 32D state
        }
        
        current_obs = {
            'player_x': 110.0,
            'player_y': 100.0,
            'player_xspeed': 2.5,
            'player_yspeed': 0.0,
            'game_state': [0.0] * 32
        }
        
        # Should work with fallback calculation
        efficiency_reward = rewards._calculate_efficiency_bonus(current_obs, prev_obs)
        assert efficiency_reward >= 0.0

    def test_diversity_bonus_calculation(self):
        """Test physics state diversity bonus."""
        rewards = PhysicsDiscoveryRewards()
        
        # Test with enhanced 40D physics state
        game_state_1 = [0.0] * 40
        game_state_1[0] = 0.5   # velocity_mag
        game_state_1[3] = 1.0   # ground_movement
        game_state_1[7] = -1.0  # airborne = False
        game_state_1[32] = 0.3  # kinetic_energy
        
        # First visit to this state should give diversity bonus
        diversity_reward_1 = rewards._calculate_diversity_bonus(game_state_1)
        assert diversity_reward_1 > 0.0
        assert len(rewards.visited_physics_states) == 1
        
        # Second visit to same state should give no bonus
        diversity_reward_2 = rewards._calculate_diversity_bonus(game_state_1)
        assert diversity_reward_2 == 0.0
        assert len(rewards.visited_physics_states) == 1
        
        # Different state should give bonus again
        game_state_2 = game_state_1.copy() 
        game_state_2[7] = 1.0  # airborne = True (different state)
        diversity_reward_3 = rewards._calculate_diversity_bonus(game_state_2)
        assert diversity_reward_3 > 0.0
        assert len(rewards.visited_physics_states) == 2

    def test_diversity_bonus_diminishing_returns(self):
        """Test that diversity bonus has diminishing returns."""
        rewards = PhysicsDiscoveryRewards()
        
        diversity_rewards = []
        
        # Visit many different states and track rewards
        for i in range(10):
            game_state = [0.0] * 40
            game_state[0] = i * 0.1  # Different velocity each time
            game_state[32] = i * 0.05  # Different energy
            
            reward = rewards._calculate_diversity_bonus(game_state)
            diversity_rewards.append(reward)
        
        # Later rewards should be smaller (diminishing returns)
        assert diversity_rewards[0] > diversity_rewards[-1]

    def test_utilization_bonus_wall_jump_detection(self):
        """Test wall jump detection and bonus."""
        rewards = PhysicsDiscoveryRewards()
        
        # Previous state: against wall, not airborne  
        prev_obs = {
            'ninja_airborne': False,
            'ninja_walled': True,
            'player_xspeed': 0.0,
            'game_state': [0.0] * 40
        }
        
        # Current state: airborne with velocity change (successful wall jump)
        current_obs = {
            'ninja_airborne': True,
            'ninja_walled': False,
            'player_xspeed': 3.0,  # Significant velocity change
            'game_state': [0.0] * 40
        }
        
        # Jump action (JUMP+LEFT)
        action = 4
        
        utilization_reward = rewards._calculate_utilization_bonus(current_obs, prev_obs, action)
        
        assert utilization_reward >= 0.5  # Should get wall jump bonus
        assert rewards.wall_jump_count == 1

    def test_utilization_bonus_buffered_jump_detection(self):
        """Test buffered jump detection and bonus."""
        rewards = PhysicsDiscoveryRewards()
        
        prev_obs = {'game_state': [0.0] * 40}
        
        # Current state with active buffer and jump action
        current_obs = {
            'game_state': [0.0] * 40
        }
        current_obs['game_state'][10] = 0.5  # Active jump buffer (> -0.8)
        
        action = 3  # JUMP action
        
        utilization_reward = rewards._calculate_utilization_bonus(current_obs, prev_obs, action)
        
        assert utilization_reward >= 0.3  # Should get buffered jump bonus
        assert rewards.buffered_jump_count == 1

    def test_utilization_bonus_movement_transitions(self):
        """Test bonus for complex movement transitions."""
        rewards = PhysicsDiscoveryRewards()
        
        # Previous state: ground movement
        prev_obs = {
            'game_state': [0.0] * 40
        }
        prev_obs['game_state'][3] = 1.0  # ground_movement = True
        
        # Current state: wall interaction  
        current_obs = {
            'game_state': [0.0] * 40
        }
        current_obs['game_state'][5] = 1.0  # wall_interaction = True
        
        action = 1  # LEFT action
        
        utilization_reward = rewards._calculate_utilization_bonus(current_obs, prev_obs, action)
        
        assert utilization_reward >= 0.2  # Should get transition bonus

    def test_calculate_physics_rewards_integration(self):
        """Test complete physics rewards calculation."""
        rewards = PhysicsDiscoveryRewards()
        
        # Setup test observations
        prev_obs = {
            'player_x': 100.0,
            'player_y': 100.0,
            'ninja_airborne': False,
            'ninja_walled': True,
            'player_xspeed': 0.0,
            'game_state': [0.0] * 40
        }
        prev_obs['game_state'][32] = 0.5  # kinetic energy
        
        current_obs = {
            'player_x': 120.0,  # Good distance
            'player_y': 100.0, 
            'ninja_airborne': True,
            'ninja_walled': False,
            'player_xspeed': 3.0,  # Velocity change for wall jump
            'game_state': [0.0] * 40
        }
        current_obs['game_state'][32] = 0.51  # Small energy change (efficient)
        current_obs['game_state'][0] = 0.8    # Different velocity for diversity
        
        action = 4  # JUMP+LEFT for wall jump
        
        reward_components = rewards.calculate_physics_rewards(current_obs, prev_obs, action)
        
        # Should have all three components
        assert 'efficiency' in reward_components
        assert 'diversity' in reward_components  
        assert 'utilization' in reward_components
        
        # All should be non-negative
        for component, value in reward_components.items():
            assert value >= 0.0, f"Component {component} should be non-negative, got {value}"

    def test_reset_functionality(self):
        """Test episode reset functionality."""
        rewards = PhysicsDiscoveryRewards()
        
        # Accumulate some state
        rewards.wall_jump_count = 5
        rewards.buffered_jump_count = 3
        rewards.energy_efficient_moves = 10
        rewards.energy_utilization_history.extend([0.6, 0.7, 0.8])
        
        # Add many entries to test history pruning
        for i in range(60):
            rewards.energy_utilization_history.append(0.5 + i * 0.01)
        
        rewards.reset()
        
        # Episode-specific counters should be reset
        assert rewards.wall_jump_count == 0
        assert rewards.buffered_jump_count == 0
        assert rewards.energy_efficient_moves == 0
        
        # Long-term learning should persist but be pruned
        assert len(rewards.energy_utilization_history) <= 50
        
        # visited_physics_states should persist (long-term learning)
        # This is not reset, so we can't test it specifically without first populating it

    def test_get_physics_stats(self):
        """Test physics statistics retrieval."""
        rewards = PhysicsDiscoveryRewards()
        
        # Set some values
        rewards.wall_jump_count = 3
        rewards.buffered_jump_count = 2
        rewards.energy_efficient_moves = 5
        rewards.energy_utilization_history.extend([0.4, 0.6, 0.8])
        rewards.visited_physics_states.add((1, 2, 0, 0, 1))
        rewards.visited_physics_states.add((2, 1, 1, 1, 2))
        
        stats = rewards.get_physics_stats()
        
        assert stats['wall_jumps_used'] == 3
        assert stats['buffered_jumps_used'] == 2
        assert stats['energy_efficient_moves'] == 5
        assert stats['unique_physics_states_visited'] == 2
        assert stats['average_efficiency'] == 0.6  # (0.4 + 0.6 + 0.8) / 3
        assert stats['efficiency_baseline'] == 0.5

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        rewards = PhysicsDiscoveryRewards()
        
        # Test with minimal distance movement
        prev_obs = {
            'player_x': 100.0,
            'player_y': 100.0,
            'game_state': [0.0] * 40
        }
        prev_obs['game_state'][32] = 0.5
        
        current_obs = {
            'player_x': 100.001,  # Tiny movement
            'player_y': 100.0,
            'game_state': [0.0] * 40
        }
        current_obs['game_state'][32] = 0.5
        
        efficiency_reward = rewards._calculate_efficiency_bonus(current_obs, prev_obs)
        assert efficiency_reward >= 0.0  # Should not crash
        
        # Test with zero energy change (division by small number)
        current_obs['game_state'][32] = 0.5  # Same energy
        efficiency_reward = rewards._calculate_efficiency_bonus(current_obs, prev_obs)
        assert efficiency_reward >= 0.0  # Should handle division by small number

    def test_reward_component_weights(self):
        """Test that reward components are properly weighted."""
        rewards = PhysicsDiscoveryRewards()
        
        # Create scenario that triggers all bonuses
        prev_obs = {
            'player_x': 100.0,
            'player_y': 100.0,
            'ninja_airborne': False,
            'ninja_walled': True,
            'player_xspeed': 0.0,
            'game_state': [0.0] * 40
        }
        prev_obs['game_state'][32] = 0.5
        
        current_obs = {
            'player_x': 140.0,  # Large efficient movement
            'player_y': 100.0,
            'ninja_airborne': True,
            'ninja_walled': False,
            'player_xspeed': 4.0,
            'game_state': [0.0] * 40
        }
        current_obs['game_state'][32] = 0.51  # Efficient
        current_obs['game_state'][0] = 0.9    # Novel state
        current_obs['game_state'][10] = 0.0   # Active buffer
        
        action = 4  # JUMP+LEFT
        
        reward_components = rewards.calculate_physics_rewards(current_obs, prev_obs, action)
        
        # Check weights are applied correctly
        # Efficiency: weight 0.2, Diversity: weight 0.1, Utilization: weight 0.3
        assert reward_components['utilization'] > reward_components['efficiency']  # 0.3 > 0.2
        assert reward_components['efficiency'] > reward_components['diversity']    # 0.2 > 0.1
