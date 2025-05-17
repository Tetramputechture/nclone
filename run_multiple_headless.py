import argparse
import multiprocessing
import time
import imageio # For saving videos
import os
import numpy as np
# Updated import for BasicLevelNoGold, assuming nclone is an installed package
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold


def random_action():
    return np.random.randint(0, 6) # Assuming 6 discrete actions

def run_single_simulation(simulation_id, num_steps, video_path_prefix=None):
    """
    Runs a single headless simulation instance.
    Optionally records the simulation if video_path_prefix is provided.
    """
    print(f"Starting simulation {simulation_id}...")
    # pygame.init() removed

    env = BasicLevelNoGold(render_mode='rgb_array',
                           enable_frame_stack=False,
                           enable_debug_overlay=False, # Usually False for headless performance
                           eval_mode=True, # Or False, depending on desired behavior
                           seed=None) # Or pass a specific seed if needed

    observation, info = env.reset()
    frames = []

    if video_path_prefix:
        frame_array = env.render() # Returns (H, W, C) NumPy array
        if frame_array is not None:
            # imageio expects (H, W, C) for standard video frames.
            # No transpose needed if env.render() provides this directly.
            frames.append(frame_array.copy()) 
    
    start_time = time.perf_counter()
    for step in range(num_steps):
        action = random_action()
        observation, reward, terminated, truncated, info = env.step(action)

        if video_path_prefix:
            frame_array = env.render()
            if frame_array is not None:
                frames.append(frame_array.copy())

        if (step + 1) % 1000 == 0:
            print(f"Simulation {simulation_id}: Step {step + 1}/{num_steps}")

        if terminated or truncated:
            observation, info = env.reset()
            if video_path_prefix: # Capture frame after reset too
                frame_array = env.render()
                if frame_array is not None:
                    frames.append(frame_array.copy())
    
    end_time = time.perf_counter()
    print(f"Simulation {simulation_id} finished {num_steps} steps in {end_time - start_time:.2f} seconds.")

    if video_path_prefix and frames:
        video_filename = f"{video_path_prefix}{simulation_id}.mp4"
        print(f"Simulation {simulation_id}: Saving video to {video_filename} ({len(frames)} frames)...")
        try:
            imageio.mimsave(video_filename, frames, fps=60) # fps can be adjusted
            print(f"Simulation {simulation_id}: Video saved successfully.")
        except Exception as e:
            print(f"Simulation {simulation_id}: Error saving video: {e}")
            print(f"Details: Frames list length: {len(frames)}. First frame shape (if any): {frames[0].shape if frames else 'N/A'}")

    env.close() # Handles cleanup of the environment and its renderer
    # pygame.quit() removed

def main():
    parser = argparse.ArgumentParser(description="Run multiple N++ headless simulations concurrently.")
    parser.add_argument(
        '--num-simulations',
        type=int,
        default=multiprocessing.cpu_count(), # Default to number of CPU cores
        help='Number of concurrent simulations to run.'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=10000, 
        help='Number of steps each simulation should run.'
    )
    parser.add_argument(
        '--record-dir',
        type=str,
        default=None,
        help='Directory to save simulation recordings. If not specified, no recording occurs.'
    )
    args = parser.parse_args()

    if args.num_simulations <= 0:
        print("Number of simulations must be positive.")
        return

    video_path_prefix = None
    if args.record_dir:
        if not os.path.exists(args.record_dir):
            os.makedirs(args.record_dir)
            print(f"Created recording directory: {args.record_dir}")
        video_path_prefix = os.path.join(args.record_dir, "sim_")

    print(f"Starting {args.num_simulations} headless simulations, each for {args.num_steps} steps...")
    if video_path_prefix:
        print(f"Recordings will be saved with prefix: {video_path_prefix}")

    processes = []
    for i in range(args.num_simulations):
        # Pass arguments to run_single_simulation correctly
        p = multiprocessing.Process(target=run_single_simulation, args=(i, args.num_steps, video_path_prefix))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All simulations completed.")

if __name__ == '__main__':
    # Set start method for multiprocessing for consistency across platforms.
    # 'spawn' is generally safer than 'fork' with complex libraries.
    # This must be in the if __name__ == '__main__': block.
    try:
        # Check if the start method has already been set to avoid error if run multiple times in same session
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn')
        elif multiprocessing.get_start_method() != 'spawn':
             # If set to something else and we want to force it (be careful with this)
             # For now, let's just note it or try to set if it's fork and we are not on windows
             if os.name != 'nt' and multiprocessing.get_start_method() == 'fork':
                 print("Warning: Multiprocessing start method is 'fork'. Attempting to set to 'spawn'.")
                 multiprocessing.set_start_method('spawn', force=True)
             else:
                 print(f"Multiprocessing start method already set to {multiprocessing.get_start_method()}") 
    except RuntimeError as e:
        print(f"Note: Could not set multiprocessing start method to 'spawn' (possibly already set or context issue): {e}")
        pass 
    main()
