import pyglet
from pyglet.window import key
import argparse
import time
import cProfile
import pstats
import io

# Attempt to import BasicLevelNoGold from its new location
try:
    from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
except ModuleNotFoundError:
    from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

parser = argparse.ArgumentParser(description="Test N++ environment with frametime logging.")
parser.add_argument('--log-frametimes', action='store_true', help='Enable frametime logging to stdout.')
parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI).')
parser.add_argument('--profile-frames', type=int, default=None, help='Run for a specific number of frames and then exit (for profiling).')
args = parser.parse_args()

render_mode = 'rgb_array' if args.headless else 'human'
debug_overlay_enabled = not args.headless
env = BasicLevelNoGold(render_mode=render_mode,
                       enable_frame_stack=False, enable_debug_overlay=debug_overlay_enabled, eval_mode=False, seed=42)

running = True
last_time = time.perf_counter()
profiler = cProfile.Profile()
frame_counter = 0

window = None
keys = key.KeyStateHandler()
observation, info = None, None

def get_action_from_keys(current_keys):
    action = 0
    if current_keys[key.SPACE] or current_keys[key.UP]:
        if current_keys[key.A] or current_keys[key.LEFT]:
            action = 4
        elif current_keys[key.D] or current_keys[key.RIGHT]:
            action = 5
        else:
            action = 3
    else:
        if current_keys[key.A] or current_keys[key.LEFT]:
            action = 1
        elif current_keys[key.D] or current_keys[key.RIGHT]:
            action = 2
    return action

def update_human_mode(dt):
    global running, observation, info, frame_counter, last_time, window
    if not running:
        if window and hasattr(window, 'has_exit') and not window.has_exit:
            window.close()
        pyglet.app.exit()
        return

    action = get_action_from_keys(keys)
    new_observation, reward, terminated, truncated, new_info = env.step(action)
    observation, info = new_observation, new_info

    if terminated or truncated:
        observation, info = env.reset()

    current_time = time.perf_counter()
    if args.log_frametimes:
        frame_time_ms = (current_time - last_time) * 1000
        print(f"Frametime: {frame_time_ms:.2f} ms")
    last_time = current_time
    
    frame_counter += 1
    if args.profile_frames is not None and frame_counter >= args.profile_frames:
        running = False

profiler.enable()

if args.headless:
    observation, info = env.reset()
    while running:
        action = 0
        new_observation, reward, terminated, truncated, new_info = env.step(action)
        observation, info = new_observation, new_info

        if terminated or truncated:
            observation, info = env.reset()

        current_time = time.perf_counter()
        if args.log_frametimes:
            frame_time_ms = (current_time - last_time) * 1000
            print(f"Frametime: {frame_time_ms:.2f} ms")
        last_time = current_time

        frame_counter += 1
        if args.profile_frames is not None and frame_counter >= args.profile_frames:
            running = False
else:
    observation, info = env.reset()
    if hasattr(env, 'sim_renderer') and hasattr(env.sim_renderer, 'window') and env.sim_renderer.window is not None:
        window = env.sim_renderer.window
        window.push_handlers(keys)

        @window.event
        def on_key_press(symbol, modifiers):
            global observation, info, running
            if symbol == key.R:
                observation, info = env.reset()
            if symbol == key.ESCAPE:
                running = False
        
        @window.event
        def on_close():
            global running
            running = False

        pyglet.clock.schedule_interval(update_human_mode, 1/60.0)
        pyglet.app.run()
    else:
        print("Error: Could not get Pyglet window from environment. Human mode needs NSimRenderer to expose its window.")
        running = False

profiler.disable()
env.close()

with open('profiling_stats.txt', 'w') as f:
    ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
    ps.print_stats()

print("Profiling stats saved to profiling_stats.txt")
