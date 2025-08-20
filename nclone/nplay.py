import pygame
import os.path
import zlib
from nsim import *
from nsim_renderer import NSimRenderer
from map_generation.map_generator import generate_map, random_official_map
from sim_config import SimConfig
import random
import argparse
from . import render_utils

# Create a new map
map_data = generate_map()

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

pygame.init()
pygame.display.set_caption("N++")
screen = pygame.display.set_mode((render_utils.SRCWIDTH, render_utils.SRCHEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
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
if os.path.isfile("inputs"):
    with open("inputs", "rb") as f:
        if COMPRESSED_INPUTS:
            inputs = [int(b) for b in zlib.decompress(f.read())]
        else:
            inputs = [int(b) for b in f.read()][215:]
    hor_inputs = [HOR_INPUTS_DIC[inp] for inp in inputs]
    jump_inputs = [JUMP_INPUTS_DIC[inp] for inp in inputs]
    inp_len = len(inputs)

sim_renderer = NSimRenderer(sim, render_mode='human')

# # Print space of NplayHeadless state, only ninja and exit and switch
# np_headless = NPlayHeadless()
# np_headless.load_random_map(seed=42)
# ninja_state = np_headless.get_ninja_state()
# entity_states = np_headless.get_entity_states(only_one_exit_and_switch=True)
# total_state = np.concatenate([ninja_state, entity_states])
# # Print the total state
# print(total_state.shape)


while running:
    resize = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.VIDEORESIZE or sim.frame == 0:
            resize = True

    keys = pygame.key.get_pressed()
    hor_input = 0
    jump_input = 0
    if running_mode == "playing":
        if keys[pygame.K_RIGHT]:
            hor_input = 1
        if keys[pygame.K_LEFT]:
            hor_input = -1
        if keys[pygame.K_z]:
            jump_input = 1
    elif running_mode == "replaying":
        if sim.frame < inp_len:
            hor_input = hor_inputs[sim.frame]
            jump_input = jump_inputs[sim.frame]
    if keys[pygame.K_SPACE]:
        sim.reset()
        running_mode = "playing"
    if keys[pygame.K_r]:
        if inputs:
            sim.reset()
            running_mode = "replaying"
    if keys[pygame.K_g]:
        sim.reset()
        # Randomly choose between maze and simple horizontal level
        level_type = random.choice(
            ["MAZE", "SIMPLE_HORIZONTAL_NO_BACKTRACK"])
        # level_type = "MULTI_CHAMBER"
        width = random.randint(4, 42)
        height = random.randint(4, 23)
        # 50% chance to generate a random map, 50% chance to load a random official map
        if random.random() < 0.5:
            print("Generating a random map")
            map_gen = generate_map(level_type=level_type,
                                   width=width, height=height)
            sim.load_from_created(map_gen)
        else:
            print("Loading a random official map")
            map_data = random_official_map()
            sim.load(map_data)
    sim_renderer.draw(resize)

    sim.tick(hor_input, jump_input)

    clock.tick(60)

pygame.quit()
