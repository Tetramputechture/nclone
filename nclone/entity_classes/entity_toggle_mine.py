
from ..entities import Entity
from ..physics import *
from ..ninja import NINJA_RADIUS

class EntityToggleMine(Entity):
    """This class handles both toggle mines (untoggled state) and regular mines (toggled state)."""
    ENTITY_TYPE = 1  # Also handles type 21 for toggled state
    RADII = {0: 4, 1: 3.5, 2: 4.5}  # 0:toggled, 1:untoggled, 2:toggling
    MAX_COUNT_PER_LEVEL = 8192

    def __init__(self, entity_type, sim, xcoord, ycoord, state):
        super().__init__(entity_type, sim, xcoord, ycoord)
        self.is_thinkable = True
        self.is_logical_collidable = True
        self.set_state(state)

    def think(self):
        """Handle interactions between the ninja and the untoggled mine"""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if self.state == 1:  # Set state to toggling if ninja touches untoggled mine
                if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                            ninja.xpos, ninja.ypos, NINJA_RADIUS):
                    self.set_state(2)
            elif self.state == 2:  # Set state to toggled if ninja stops touching toggling mine
                if not overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                                ninja.xpos, ninja.ypos, NINJA_RADIUS):
                    self.set_state(0)
        else:  # Set state to untoggled if ninja dies while toggling a mine
            if self.state == 2 and ninja.state == 6:
                self.set_state(1)

    def logical_collision(self):
        """Kill the ninja if it touches a toggled mine"""
        ninja = self.sim.ninja
        if ninja.is_valid_target() and self.state == 0:
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, NINJA_RADIUS):
                self.set_state(1)
                ninja.kill(0, 0, 0, 0, 0)

    def set_state(self, state):
        """Set the state of the toggle. 0:toggled, 1:untoggled, 2:toggling."""
        if state in (0, 1, 2):
            self.state = state
            self.RADIUS = self.RADII[state]
            self.log_collision(state)

    def get_state(self):
        state = super().get_state()
        # Normalize state (0, 1, or 2)
        # state.append(max(0.0, min(1.0, float(self.state) / 2)))
        return state
