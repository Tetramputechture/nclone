import pygame
import argparse
import multiprocessing
import time
import imageio
import os
import numpy as np
from .nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold


def random_action():
    return np.random.randint(0, 6)

def run_single_simulation(simulation_id, num_steps, video_path_prefix=None):
    """
    Runs a single headless simulation instance.
    Optionally records the simulation if video_path_prefix is provided.
    """
    print(f"Starting simulation {simulation_id}...")
    pygame.init()

    env = BasicLevelNoGold(render_mode='rgb_array',
                           enable_frame_stack=False,
                           enable_debug_overlay=False,
                           eval_mode=True,
                           seed=None)

    observation, info = env.reset()
    frames = []

    if video_path_prefix:
        frame_array = env.render()
        if frame_array is not None:
            copied_frame = frame_array.copy() # Make a copy
            transposed_frame = copied_frame.transpose(1, 0, 2) # Transpose H, W
            frames.append(transposed_frame)

    start_time = time.perf_counter()
    for step in range(num_steps):
        action = random_action()
        observation, reward, terminated, truncated, info = env.step(action)

        if video_path_prefix:
            frame_array = env.render()
            if frame_array is not None:
                copied_frame = frame_array.copy() # Make a copy
                transposed_frame = copied_frame.transpose(1, 0, 2) # Transpose H, W
                frames.append(transposed_frame)

        if (step + 1) % 1000 == 0:
            print(f"Simulation {simulation_id}: Step {step + 1}/{num_steps}")

        if terminated or truncated:
            observation, info = env.reset()
            if video_path_prefix:
                frame_array = env.render()
                if frame_array is not None:
                    copied_frame = frame_array.copy() # Make a copy
                    transposed_frame = copied_frame.transpose(1, 0, 2) # Transpose H, W
                    frames.append(transposed_frame)

    end_time = time.perf_counter()
    print(f"Simulation {simulation_id} finished {num_steps} steps in {end_time - start_time:.2f} seconds.")

    if video_path_prefix and frames:
        video_filename = f"{video_path_prefix}{simulation_id}.mp4"
        print(f"Simulation {simulation_id}: Saving video to {video_filename} ({len(frames)} frames)...")
        try:
            imageio.mimsave(video_filename, frames, fps=60)
            print(f"Simulation {simulation_id}: Video saved successfully.")
        except Exception as e:
            print(f"Simulation {simulation_id}: Error saving video: {e}")

    env.close()
    pygame.quit()

def main():
    parser = argparse.ArgumentParser(description="Run multiple N++ headless simulations concurrently.")
    parser.add_argument(
        '--num-simulations',
        type=int,
        default=2,
        help='Number of concurrent simulations to run.'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=10000, # Default to a smaller number for testing recording
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
        p = multiprocessing.Process(target=run_single_simulation, args=(i, args.num_steps, video_path_prefix))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All simulations completed.")

if __name__ == '__main__':
    # It's good practice to set the start method for multiprocessing,
    # especially on platforms where 'fork' is the default but might cause issues
    # with libraries like Pygame. 'spawn' or 'forkserver' are generally safer.
    # 'spawn' is available on all platforms.
    # This should be called only once, in the main module.
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Might have already been set by another module or in a different context
        pass
    main() 