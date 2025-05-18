# distutils: language = c++
# distutils: sources = ../src/sim_wrapper.cpp
# distutils: include_dirs = ../src/

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
import numpy as np
cimport numpy as np

# Need to declare unsigned char for vector
ctypedef unsigned char uchar

# Declare the C++ class
cdef extern from "sim_wrapper.hpp":
    cdef cppclass SimWrapper:
        SimWrapper(bool, bool, bool, float, bool, bool, string) except +
        void loadMap(vector[uchar]&)
        void reset()
        void tick(float, int)
        bool hasWon()
        bool hasDied()
        pair[float, float] getNinjaPosition()
        pair[float, float] getNinjaVelocity()
        bool isNinjaInAir()
        bool isNinjaWalled()
        int getGoldCollected()
        int getDoorsOpened()
        int getSimFrame()
        int getTotalGoldAvailable()
        bool exitSwitchActivated()
        pair[float, float] getExitSwitchPosition()
        pair[float, float] getExitDoorPosition()
        vector[float] getNinjaState()
        vector[float] getEntityStates(bool)
        vector[float] getStateVector(bool)
        void render(vector[float]&, vector[float]&, int, int, int, int)
        bool isWindowOpen()

# Python wrapper class
cdef class NPlayHeadlessCpp:
    cdef unique_ptr[SimWrapper] _sim

    def __cinit__(self, bool enable_debug_overlay=False, bool basic_sim=False, bool full_export=False, float tolerance=1.0, bool enable_anim=True, bool log_data=False, str render_mode="rgb_array"):
        self._sim.reset(new SimWrapper(enable_debug_overlay, basic_sim, full_export, tolerance, enable_anim, log_data, render_mode.encode('utf-8')))

    def load_map(self, bytes map_data):
        cdef vector[uchar] cpp_map_data
        cpp_map_data.resize(len(map_data))
        for i in range(len(map_data)):
            cpp_map_data[i] = map_data[i]
        self._sim.get().loadMap(cpp_map_data)

    def reset(self):
        self._sim.get().reset()

    def tick(self, float hor_input, int jump_input):
        self._sim.get().tick(hor_input, jump_input)

    def has_won(self):
        return self._sim.get().hasWon()

    def has_died(self):
        return self._sim.get().hasDied()

    def get_ninja_position(self):
        cdef pair[float, float] pos = self._sim.get().getNinjaPosition()
        return (pos.first, pos.second)

    def get_ninja_velocity(self):
        cdef pair[float, float] vel = self._sim.get().getNinjaVelocity()
        return (vel.first, vel.second)

    def is_ninja_in_air(self):
        return self._sim.get().isNinjaInAir()

    def is_ninja_walled(self):
        return self._sim.get().isNinjaWalled()

    def get_gold_collected(self):
        """Returns the total gold collected by the ninja."""
        return self._sim.get().getGoldCollected()

    def get_doors_opened(self):
        """Returns the total doors opened by the ninja."""
        return self._sim.get().getDoorsOpened()

    def get_total_gold_available(self):
        """Returns count of all gold (entity type 2) in the map."""
        return self._sim.get().getTotalGoldAvailable()

    def get_sim_frame(self):
        return self._sim.get().getSimFrame()

    def get_ninja_state(self):
        """Get ninja state information as a 10-element list of floats, all normalized between 0 and 1."""
        cdef vector[float] state = self._sim.get().getNinjaState()
        return np.array(state, dtype=np.float32)

    def get_entity_states(self, bool only_exit_and_switch=False):
        """Get all entity states as a list of floats with fixed length, all normalized between 0 and 1."""
        cdef vector[float] state = self._sim.get().getEntityStates(only_exit_and_switch)
        return np.array(state, dtype=np.float32)

    def get_state_vector(self, bool only_exit_and_switch=False):
        """Get a complete state representation of the game environment as a vector of float values."""
        cdef vector[float] state = self._sim.get().getStateVector(only_exit_and_switch)
        return np.array(state, dtype=np.float32)

    def render(self):
        """Render both the global view and player-centered view of the game.
        
        Returns:
            tuple: (global_view, player_view) where:
                - global_view: numpy array of shape (RENDERED_VIEW_HEIGHT, RENDERED_VIEW_WIDTH, 3)
                - player_view: numpy array of shape (84, 84, 3)
        """
        cdef vector[float] global_buffer
        cdef vector[float] player_buffer
        cdef int full_width = 176  # RENDERED_VIEW_WIDTH
        cdef int full_height = 100  # RENDERED_VIEW_HEIGHT
        cdef int player_width = 84
        cdef int player_height = 84

        self._sim.get().render(global_buffer, player_buffer, 
                              full_width, full_height,
                              player_width, player_height)
        
        # Convert to numpy arrays with appropriate shapes
        global_view = np.array(global_buffer, dtype=np.float32).reshape(full_height, full_width, 3)
        player_view = np.array(player_buffer, dtype=np.float32).reshape(player_height, player_width, 3)
        
        return global_view, player_view

    def exit_switch_activated(self):
        """Return whether the exit switch is activated."""
        return self._sim.get().exitSwitchActivated()

    def exit_switch_position(self):
        """Return the position of the exit switch."""
        cdef pair[float, float] pos = self._sim.get().getExitSwitchPosition()
        return (pos.first, pos.second)

    def exit_door_position(self):
        """Return the position of the exit door."""
        cdef pair[float, float] pos = self._sim.get().getExitDoorPosition()
        return (pos.first, pos.second)

    def is_window_open(self):
        """Check if the SFML window is still open (if in human render mode)."""
        return self._sim.get().isWindowOpen() 