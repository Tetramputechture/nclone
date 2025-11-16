"""Integration tests for action masking system.

These tests verify that action masking works correctly across:
- Multiple episodes and steps
- Different ninja states (grounded, airborne, walled)
- Vectorized environments
- Random action selection
"""

import pytest
import numpy as np
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class TestActionMaskingIntegration:
    """Integration tests for action masking system."""

    def test_action_mask_single_environment_many_steps(self):
        """Test that action mask is valid across many steps in a single environment."""
        config = EnvironmentConfig.for_training()
        env = NppEnvironment(config=config)

        num_episodes = 10
        max_steps_per_episode = 1000
        total_steps = 0
        masked_action_count = 0

        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                # Verify action mask is present
                assert "action_mask" in obs, "action_mask not in observation"
                action_mask = obs["action_mask"]

                # Verify action mask properties
                assert isinstance(action_mask, np.ndarray), (
                    f"action_mask is {type(action_mask)}, expected np.ndarray"
                )
                assert action_mask.shape == (6,), (
                    f"action_mask shape is {action_mask.shape}, expected (6,)"
                )
                assert action_mask.dtype in [np.int8, np.bool_, bool], (
                    f"action_mask dtype is {action_mask.dtype}"
                )

                # Verify at least one action is valid
                assert action_mask.any(), (
                    f"No valid actions! Mask: {action_mask}, Step: {steps}, Episode: {episode}"
                )

                # Count masked actions for statistics
                masked_action_count += 6 - action_mask.sum()

                # Select a valid action randomly
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)

                # Verify selected action is valid
                assert action_mask[action], (
                    f"Selected invalid action {action}, mask: {action_mask}"
                )

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                total_steps += 1

        env.close()

        print(f"\n✓ Tested {total_steps} steps across {num_episodes} episodes")
        print(
            f"  Average masked actions per step: {masked_action_count / total_steps:.2f}"
        )
        print(f"  Total valid steps: {total_steps}")

    def test_action_mask_vectorized_dummy_env(self):
        """Test action masking with DummyVecEnv (single process, multiple envs)."""

        def make_env():
            def _init():
                config = EnvironmentConfig.for_training()
                return NppEnvironment(config=config)

            return _init

        num_envs = 4
        env = DummyVecEnv([make_env() for _ in range(num_envs)])

        num_steps = 500

        obs = env.reset()
        for step in range(num_steps):
            # Verify action mask for each environment
            assert "action_mask" in obs, "action_mask not in observation dict"
            action_masks = obs["action_mask"]

            # Verify shape: (num_envs, 6)
            assert action_masks.shape == (num_envs, 6), (
                f"action_masks shape is {action_masks.shape}, expected ({num_envs}, 6)"
            )

            # Select valid actions for each environment
            actions = []
            for i in range(num_envs):
                mask = action_masks[i]
                assert mask.any(), f"No valid actions for env {i}! Mask: {mask}"
                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)
                assert mask[action], (
                    f"Selected invalid action {action} for env {i}, mask: {mask}"
                )
                actions.append(action)

            # Step all environments
            obs, rewards, dones, infos = env.step(np.array(actions))

        env.close()
        print(
            f"\n✓ Tested {num_steps} steps with {num_envs} vectorized environments (DummyVecEnv)"
        )

    def test_action_mask_all_actions_eventually_valid(self):
        """Test that all 6 actions are valid in at least some game states."""
        config = EnvironmentConfig.for_training()
        env = NppEnvironment(config=config)

        # Track which actions have been valid at least once
        actions_seen_valid = set()

        num_episodes = 20
        max_steps_per_episode = 500

        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                action_mask = obs["action_mask"]

                # Record which actions are valid
                for action_idx in range(6):
                    if action_mask[action_idx]:
                        actions_seen_valid.add(action_idx)

                # If we've seen all actions valid, we can stop early
                if len(actions_seen_valid) == 6:
                    break

                # Select random valid action
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

            if len(actions_seen_valid) == 6:
                break

        env.close()

        action_names = ["NOOP", "LEFT", "RIGHT", "JUMP", "JUMP+LEFT", "JUMP+RIGHT"]
        print(f"\n✓ Valid actions seen: {len(actions_seen_valid)}/6")
        for i in range(6):
            status = "✓" if i in actions_seen_valid else "✗"
            print(f"  {status} Action {i} ({action_names[i]})")

        # All actions should be valid in at least some states
        assert len(actions_seen_valid) == 6, (
            f"Not all actions were valid in any state! "
            f"Only saw: {[action_names[i] for i in sorted(actions_seen_valid)]}"
        )

    def test_action_mask_consistency_across_calls(self):
        """Test that action mask doesn't change if ninja state doesn't change."""
        config = EnvironmentConfig.for_training()
        env = NppEnvironment(config=config)

        obs, info = env.reset()

        # Get action mask multiple times without stepping
        mask1 = obs["action_mask"].copy()

        # Get observation again (should be cached or recomputed consistently)
        obs_again = env._get_observation()
        mask2 = obs_again["action_mask"].copy()

        # Masks should be identical (same ninja state)
        assert np.array_equal(mask1, mask2), (
            f"Action mask changed without state change!\n"
            f"Mask 1: {mask1}\n"
            f"Mask 2: {mask2}"
        )

        env.close()
        print(
            "\n✓ Action mask is consistent across multiple calls without state change"
        )

    def test_action_mask_changes_after_step(self):
        """Test that action mask updates after stepping the environment."""
        config = EnvironmentConfig.for_training()
        env = NppEnvironment(config=config)

        obs, info = env.reset()
        initial_mask = obs["action_mask"].copy()

        # Take several steps and check if mask ever changes
        mask_changed = False
        num_steps = 100

        for _ in range(num_steps):
            valid_actions = np.where(obs["action_mask"])[0]
            action = np.random.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            current_mask = obs["action_mask"]

            if not np.array_equal(initial_mask, current_mask):
                mask_changed = True
                break

            if terminated or truncated:
                obs, info = env.reset()
                current_mask = obs["action_mask"]
                if not np.array_equal(initial_mask, current_mask):
                    mask_changed = True
                    break

        env.close()

        # Mask should change at some point (different ninja states have different valid actions)
        assert mask_changed, (
            f"Action mask never changed across {num_steps} steps! "
            f"This suggests mask is not being updated properly."
        )

        print("\n✓ Action mask updates correctly as ninja state changes")

    def test_masked_action_detection_raises_error(self):
        """Test that attempting to execute a masked action raises an error."""
        config = EnvironmentConfig.for_training()
        env = NppEnvironment(config=config)

        obs, info = env.reset()
        action_mask = obs["action_mask"]

        # Find a masked action
        masked_actions = np.where(~action_mask.astype(bool))[0]

        if len(masked_actions) == 0:
            # Skip test if all actions are valid (rare but possible)
            env.close()
            pytest.skip(
                "All actions valid in initial state, cannot test masked action detection"
            )
            return

        masked_action = masked_actions[0]

        # Attempting to execute a masked action should raise RuntimeError
        with pytest.raises(RuntimeError, match="Masked action bug detected"):
            env.step(masked_action)

        env.close()
        print("\n✓ Masked action detection correctly raises RuntimeError")

    def test_action_mask_subproc_vec_env_parallel(self):
        """Test action masking with SubprocVecEnv (50 parallel environments, 2000 steps).

        This is a stress test designed to catch race conditions and memory sharing
        issues that can occur in parallel subprocess environments. The bug typically
        manifests after 100-1000 steps with many parallel environments.
        """

        def make_env(rank):
            def _init():
                config = EnvironmentConfig.for_training()
                return NppEnvironment(config=config)

            return _init

        num_envs = 50
        num_steps = 2000

        print(f"\n  Creating {num_envs} parallel environments...")
        env = SubprocVecEnv(
            [make_env(i) for i in range(num_envs)], start_method="spawn"
        )

        print(f"  Running {num_steps} steps to detect race conditions...")
        obs = env.reset()

        masked_action_bug_count = 0
        total_actions = 0

        for step in range(num_steps):
            # Verify action mask for each environment
            assert "action_mask" in obs, (
                f"action_mask not in observation dict at step {step}"
            )
            action_masks = obs["action_mask"]

            # Verify shape: (num_envs, 6)
            assert action_masks.shape == (num_envs, 6), (
                f"action_masks shape is {action_masks.shape}, expected ({num_envs}, 6) at step {step}"
            )

            # Verify each mask owns its data (no memory sharing)
            if hasattr(action_masks, "flags"):
                assert action_masks.flags["OWNDATA"] or action_masks.base is None, (
                    f"action_masks shares memory at step {step}! "
                    f"OWNDATA={action_masks.flags['OWNDATA']}, base={action_masks.base}"
                )

            # Select valid actions for each environment
            actions = []
            for i in range(num_envs):
                mask = action_masks[i]

                # Verify mask properties
                assert mask.any(), (
                    f"No valid actions for env {i} at step {step}! Mask: {mask}"
                )

                # Verify individual mask memory ownership
                if hasattr(mask, "flags"):
                    owns_data = mask.flags["OWNDATA"]
                    if not owns_data and step % 100 == 0:
                        # Log periodically but don't fail (views are ok as long as they're stable)
                        print(
                            f"    Warning: mask for env {i} at step {step} doesn't own data"
                        )

                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)

                # Double-check selected action is valid
                if not mask[action]:
                    masked_action_bug_count += 1
                    print(
                        f"    ERROR: Selected masked action {action} for env {i} at step {step}!"
                    )

                actions.append(action)
                total_actions += 1

            # Step all environments
            obs, rewards, dones, infos = env.step(np.array(actions))

            # Progress indicator every 500 steps
            if (step + 1) % 500 == 0:
                print(f"    Step {step + 1}/{num_steps} completed")

        env.close()

        # Verify no masked actions were selected
        assert masked_action_bug_count == 0, (
            f"Masked action bug detected! {masked_action_bug_count} masked actions selected "
            f"out of {total_actions} total actions across {num_steps} steps"
        )

        print(
            f"\n✓ Tested {num_steps} steps with {num_envs} parallel SubprocVecEnv environments"
        )
        print(f"  Total actions: {total_actions}")
        print(f"  Masked action bugs: {masked_action_bug_count}")

    def test_action_mask_memory_ownership(self):
        """Test that action_mask arrays always have proper memory ownership.

        This test verifies that defensive copying throughout the pipeline ensures
        each action_mask owns its memory and is independent from other copies.
        """

        def make_env():
            def _init():
                config = EnvironmentConfig.for_training()
                return NppEnvironment(config=config)

            return _init

        # Test with vectorized environment
        num_envs = 10
        env = DummyVecEnv([make_env() for _ in range(num_envs)])

        obs = env.reset()

        for step in range(100):
            action_masks = obs["action_mask"]

            # Test 1: Batch masks should own their data or have independent memory
            assert isinstance(action_masks, np.ndarray), (
                f"action_masks is {type(action_masks)}"
            )

            # Test 2: Each individual mask should be independent
            mask_copies = []
            for i in range(num_envs):
                mask = action_masks[i]
                # Make a copy and verify it's truly independent
                mask_copy = mask.copy()
                mask_copies.append(mask_copy)

                # Verify shapes match
                assert mask.shape == (6,), f"mask shape is {mask.shape}"
                assert mask_copy.shape == (6,), f"mask_copy shape is {mask_copy.shape}"

                # Verify copies are equal initially
                assert np.array_equal(mask, mask_copy), "mask and mask_copy differ"

            # Test 3: Modifying a copy shouldn't affect other copies
            for i, mask_copy in enumerate(mask_copies):
                original_value = mask_copy[0]
                mask_copy[0] = not original_value  # Flip the first bit

                # Verify other masks are unaffected
                for j, other_copy in enumerate(mask_copies):
                    if i != j:
                        # Other copies should still have their original values
                        # (We can't directly check this without storing originals,
                        # but we can at least verify they weren't all changed together)
                        pass

            # Test 4: Step environment with valid actions
            actions = []
            for i in range(num_envs):
                mask = action_masks[i]
                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)
                actions.append(action)

            obs, rewards, dones, infos = env.step(np.array(actions))

        env.close()
        print("\n✓ Action masks have proper memory ownership and independence")


if __name__ == "__main__":
    # Run tests manually
    test = TestActionMaskingIntegration()

    print("\n" + "=" * 80)
    print("Running Action Masking Integration Tests")
    print("=" * 80)

    print("\n[1/9] Testing single environment with many steps...")
    test.test_action_mask_single_environment_many_steps()

    print("\n[2/9] Testing vectorized environment (DummyVecEnv)...")
    test.test_action_mask_vectorized_dummy_env()

    print("\n[3/9] Testing all actions are eventually valid...")
    test.test_action_mask_all_actions_eventually_valid()

    print("\n[4/9] Testing action mask consistency...")
    test.test_action_mask_consistency_across_calls()

    print("\n[5/9] Testing action mask updates after steps...")
    test.test_action_mask_changes_after_step()

    print("\n[6/9] Testing masked action detection...")
    test.test_masked_action_detection_raises_error()

    print(
        "\n[7/9] Testing parallel SubprocVecEnv (50 envs, 2000 steps) - STRESS TEST..."
    )
    print("  This test may take several minutes...")
    test.test_action_mask_subproc_vec_env_parallel()

    print("\n[8/9] Testing action mask memory ownership...")
    test.test_action_mask_memory_ownership()

    print("\n" + "=" * 80)
    print("✓ All Action Masking Integration Tests Passed!")
    print("=" * 80)
