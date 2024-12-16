from PIL import Image
import cairo
import pygame
import math
import os.path
import zlib

from nsim import *
from nsim_renderer import NSimRenderer
from map import Map
from sim_config import init_sim_config

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

sim_config = init_sim_config(ARGUMENTS)


sim = Simulator(sim_config)

# Create a new map
map = Map()

# Add entities if desired
map.add_entity(2, 10, 10)  # Add gold at (10,10)

map.set_tile(4, 7, 1)
# Set the ninja spawn point
map.set_ninja_spawn(24, 28)  # Ninja will spawn at coordinates (3,3)

pygame.init()
pygame.display.set_caption("N++")
screen = pygame.display.set_mode((SRCWIDTH, SRCHEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
running_mode = "playing"

# with open("maps/map_data_simple", "rb") as f:
#     mapdata = [int(b) for b in f.read()]
sim.load_from_created(map)
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

sim_renderer = NSimRenderer(sim)


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

    sim_renderer.draw(resize)

    sim.tick(hor_input, jump_input)

    clock.tick(60)

pygame.quit()
