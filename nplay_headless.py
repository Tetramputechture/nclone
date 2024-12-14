import pygame
import os

from nsim import Simulator
from nsim_renderer import NSimRenderer

# set pygame env to headless

os.environ["SDL_VIDEODRIVER"] = "dummy"


class NPlayHeadless:
    """
    This class is used to run the simulation in headless mode,
    while supporting rendering a frame to a NumPy array. Has manual
    control over the simulation and rendering, and is intended to be
    used for training machine learning agents.

    Has a simple API for interacting with the simulation (moving horizontally
    or jumping, loading a map, resetting and ticking the simulation).
    """

    def __init__(self):
        """
        Initialize the simulation and renderer, as well as the headless pygame
        interface and display.
        """
        self.sim = Simulator()
        self.sim_renderer = NSimRenderer(self.sim)
        self.current_map_data = None

        # init pygame
        pygame.init()

        # init headless display
        pygame.display.init()

    def load_map(self, map_path: str):
        """
        Load a map from a file.
        """
        with open(map_path, "rb") as map_file:
            mapdata = [int(b) for b in map_file.read()]
        self.sim.load(mapdata)
        self.current_map_data = mapdata

    def reset(self):
        """
        Reset the simulation to the initial state.
        """
        self.sim.reset()

    def tick(self, horizontal_input: int, jump_input: int):
        """
        Tick the simulation with the given horizontal and jump inputs.
        """
        self.sim.tick(horizontal_input, jump_input)

    def render(self):
        """
        Render the current frame to a NumPy array.
        """
        init = self.sim.frame >= 1
        surface = self.sim_renderer.draw(init)
        return pygame.surfarray.array2d(surface)

    def ninja_has_won_or_died(self):
        """
        Check if the ninja has won or died.
        """
        return self.sim.ninja.has_won_or_died()
