import pyglet
import argparse
from pyglet.window import key, mouse
import os.path
import zlib
from .nsim import *
from .nsim_renderer import NSimRenderer
from .map_generation.map_generator import generate_map, random_official_map
from .sim_config import SimConfig
import random

# Input handling constants (previously in ntrace.py/ntrace_manual.py)
COMPRESSED_INPUTS = False # Defaulting to False, adjust if input files are compressed
HOR_INPUTS_DIC = {0:0, 1:0, 2:1, 3:1, 4:-1, 5:-1, 6:-1, 7:-1}
JUMP_INPUTS_DIC = {0:0, 1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:1}


# Create a new map
# map_data = generate_map() # This seems unused later, commenting out for now

SRCWIDTH = 1056
SRCHEIGHT = 600

# Create argument parser so that we can pass parameters when executing the tool
# Run the tool with the -h option to see the complete help
parser = argparse.ArgumentParser(description='N++ physics clone')
parser.add_argument('--basic-sim', action='store_true',
                    help='Only simulate entities with physical collision')
parser.add_argument('--full-export', action='store_true',
                    help="Export coordinates of moving entities")
parser.add_argument('-t', '--tolerance', type=float, default=1.0,
                    help='Minimum units to consider an entity moved')
ARGUMENTS = parser.parse_args()

sim = Simulator(SimConfig())

window = pyglet.window.Window(SRCWIDTH, SRCHEIGHT, "N++", resizable=True)
# screen variable is no longer needed, window will be used directly or passed to renderer
# clock is handled by pyglet.clock
# running is handled by pyglet.app.run() and window events

running_mode = "playing"

# with open("maps/basics", "rb") as f:
#     mapdata = [int(b) for b in f.read()]
# sim.load(mapdata)
level_type = random.choice(["MAZE", "SIMPLE_HORIZONTAL_NO_BACKTRACK"])
width = random.randint(4, 42)
height = random.randint(4, 23)
map_gen = generate_map(level_type=level_type,
                       width=width, height=height)
sim.load_from_created(map_gen)
inputs = None
hor_inputs = [] # Initialize to avoid NameError if "inputs" file doesn't exist
jump_inputs = []
inp_len = 0
if os.path.isfile("inputs"):
    with open("inputs", "rb") as f:
        if COMPRESSED_INPUTS:
            inputs_raw = [int(b) for b in zlib.decompress(f.read())]
        else:
            inputs_raw = [int(b) for b in f.read()][215:] # Original had [215:] slice
    # Assuming HOR_INPUTS_DIC and JUMP_INPUTS_DIC are defined in nsim or globally accessible
    hor_inputs = [HOR_INPUTS_DIC[inp] for inp in inputs_raw]
    jump_inputs = [JUMP_INPUTS_DIC[inp] for inp in inputs_raw]
    inp_len = len(inputs_raw)

sim_renderer = NSimRenderer(sim, render_mode='human', window=window) # Pass window to renderer

keys_pressed = set()
first_draw_done = False

@window.event
def on_draw():
    global first_draw_done
    window.clear()
    # The init flag for the renderer is True for the first draw.
    sim_renderer.draw(init=(not first_draw_done), debug_info=None) # Pass init flag and placeholder debug_info
    if not first_draw_done:
        first_draw_done = True

@window.event
def on_resize(width, height):
    # Pass resize event to renderer if it needs to adjust viewport or projections
    # For now, nsim_renderer's draw method takes a resize hint.
    # We might need a more direct way to inform the renderer about the new size.
    # The old code called sim_renderer.draw(True) on resize.
    # We'll rely on the on_draw call to handle this for now,
    # potentially making the next call to sim_renderer.draw pass True.
    # This might need refinement based on how nsim_renderer is structured.
    # For simplicity, let's assume sim_renderer.draw() can handle being called after a resize.
    # The `needs_resize_handling` in on_draw will be True for the very first frame.
    # Subsequent resizes will trigger this on_resize, but on_draw will then be called.
    # We might need to set a flag here if nsim_renderer.draw() needs an explicit resize signal
    # beyond the first frame.
    pass # pyglet handles window resizing; viewport/projection is internal to renderer or on_draw

@window.event
def on_key_press(symbol, modifiers):
    global running_mode, sim, hor_inputs, jump_inputs, inp_len # Ensure access (removed inputs)
    keys_pressed.add(symbol)
    if symbol == key.SPACE:
        sim.reset()
        running_mode = "playing"
    elif symbol == key.R:
        if inp_len > 0: # Check if replay data is available
            sim.reset()
            running_mode = "replaying"
    elif symbol == key.G:
        sim.reset()
        new_level_type = random.choice(
            ["MAZE", "SIMPLE_HORIZONTAL_NO_BACKTRACK"])
        new_width = random.randint(4, 42)
        new_height = random.randint(4, 23)
        if random.random() < 0.5:
            print("Generating a random map")
            new_map_gen = generate_map(level_type=new_level_type,
                                   width=new_width, height=new_height)
            sim.load_from_created(new_map_gen)
        else:
            print("Loading a random official map")
            new_map_data = random_official_map()
            sim.load(new_map_data)
    elif symbol == key.ESCAPE: # Allow closing with ESC
        window.close()


@window.event
def on_key_release(symbol, modifiers):
    if symbol in keys_pressed:
        keys_pressed.remove(symbol)

def update(dt):
    global running_mode, sim, hor_inputs, jump_inputs, inp_len # Ensure access (removed inputs)
    hor_input = 0
    jump_input = 0

    if running_mode == "playing":
        if key.RIGHT in keys_pressed:
            hor_input = 1
        if key.LEFT in keys_pressed:
            hor_input = -1
        if key.Z in keys_pressed: # Assuming Z key for jump
            jump_input = 1
    elif running_mode == "replaying":
        if sim.frame < inp_len:
            hor_input = hor_inputs[sim.frame]
            jump_input = jump_inputs[sim.frame]
        else: # Stop replaying if end of inputs is reached
            running_mode = "playing"


    sim.tick(hor_input, jump_input)
    # Rendering is handled by on_draw

# Schedule the update function
pyglet.clock.schedule_interval(update, 1/60.0)

# Run the application
try:
    pyglet.app.run()
finally:
    if sim_renderer: # sim_renderer is defined in the global scope of nplay.py
        sim_renderer.close()
# pygame.quit() is no longer needed
