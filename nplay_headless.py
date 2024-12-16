import pygame
import os

from nsim import Simulator
from nsim_renderer import NSimRenderer

# set pygame env to headless
SRCWIDTH = 1056
SRCHEIGHT = 600


class NPlayHeadless:
    """
    This class is used to run the simulation in headless mode,
    while supporting rendering a frame to a NumPy array. Has manual
    control over the simulation and rendering, and is intended to be
    used for training machine learning agents.

    Has a simple API for interacting with the simulation (moving horizontally
    or jumping, loading a map, resetting and ticking the simulation).
    """

    def __init__(self, render_mode: str = 'rgb_array'):
        """
        Initialize the simulation and renderer, as well as the headless pygame
        interface and display.
        """
        self.render_mode = render_mode
        if render_mode == 'rgb_array':
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            pygame.display.set_mode((SRCWIDTH, SRCHEIGHT))

        self.sim = Simulator()
        self.sim_renderer = NSimRenderer(self.sim)
        self.current_map_data = None

        # init pygame
        pygame.init()

        # init display
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

    def render(self, render_mode: str = 'rgb_array'):
        """
        Render the current frame to a NumPy array.
        """
        init = self.sim.frame <= 1
        surface = self.sim_renderer.draw(init)
        if render_mode == 'rgb_array':
            imgdata = pygame.surfarray.array3d(surface)
            imgdata = imgdata.swapaxes(0, 1)
            return imgdata
        else:
            return surface

    def ninja_has_won(self):
        return self.sim.ninja.has_won()

    def ninja_has_died(self):
        return self.sim.ninja.has_died()

    def ninja_position(self):
        return self.sim.ninja.xpos, self.sim.ninja.ypos

    def ninja_velocity(self):
        return self.sim.ninja.xspeed, self.sim.ninja.yspeed

    def ninja_is_in_air(self):
        return self.sim.ninja.airborn

    def ninja_is_walled(self):
        return self.sim.ninja.walled

    def _sim_exit_switch(self):
        # We want to get the last entry of self.sim.entity_dic[3];
        # this is the exit switch.
        # First, check if the exit switch exists.
        if len(self.sim.entity_dic[3]) == 0:
            return None

        return self.sim.entity_dic[3][-1]

    def exit_switch_activated(self):
        return not self._sim_exit_switch().active

    def exit_switch_position(self):
        return self._sim_exit_switch().xpos, self._sim_exit_switch().ypos

    def _sim_exit_door(self):
        # We want to get the .parent attribute of the exit switch.
        exit_switch = self._sim_exit_switch()
        if exit_switch is None:
            return None
        return exit_switch.parent

    def exit_door_position(self):
        return self._sim_exit_door().xpos, self._sim_exit_door().ypos

    def mines(self):
        # We want to return a list of all mines in the simulation with state == 0 (toggled)
        # of type 1.
        return [entity for entity in self.sim.entity_dic[1] if entity.active and entity.state == 0]

    def exit(self):
        pygame.quit()
