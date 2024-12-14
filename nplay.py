import cairo
import pygame
import math
import os.path
import zlib

from nsim import *
from nsim_renderer import NSimRenderer


SRCWIDTH = 1056
SRCHEIGHT = 600


pygame.init()
pygame.display.set_caption("N++")
screen = pygame.display.set_mode((SRCWIDTH, SRCHEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
running_mode = "playing"

sim = Simulator()
with open("map_data", "rb") as f:
    mapdata = [int(b) for b in f.read()]
sim.load(mapdata)
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
        sim.load(mapdata)
        running_mode = "playing"
    if keys[pygame.K_r]:
        if inputs:
            sim.load(mapdata)
            running_mode = "replaying"

    sim_renderer.draw(resize)

    sim.tick(hor_input, jump_input)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
