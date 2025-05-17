import os
import random
from typing import Optional, List
import numpy as np

from nclone.nsim import Simulator
from nclone.nsim_renderer import NSimRenderer
from nclone.map_generation.map_generator import generate_map
from nclone.sim_config import SimConfig
from nclone.entities import (
    EntityToggleMine, EntityGold, EntityExit, EntityExitSwitch,
    EntityDoorRegular, EntityDoorLocked, EntityDoorTrap, EntityLaunchPad,
    EntityOneWayPlatform, EntityDroneZap, EntityBounceBlock,
    EntityThwump, EntityBoostPad, EntityDeathBall, EntityMiniDrone,
    EntityShoveThwump
)

SRCWIDTH = 1056
SRCHEIGHT = 600

class NPlayHeadless:
    def __init__(self,
                 render_mode: str = 'rgb_array',
                 enable_animation: bool = False,
                 enable_logging: bool = False,
                 enable_debug_overlay: bool = False,
                 seed: Optional[int] = None):
        self.render_mode = render_mode
        if self.render_mode != 'rgb_array':
            print("Warning: NPlayHeadless is primarily intended for 'rgb_array' mode. Forcing to 'rgb_array'.")
            self.render_mode = 'rgb_array'

        self.sim = Simulator(
            SimConfig(enable_anim=enable_animation, log_data=enable_logging))
        self.sim_renderer = NSimRenderer(
            self.sim, self.render_mode, enable_debug_overlay)
        self.current_map_data = None
        self.enable_debug_overlay = enable_debug_overlay
        self.seed = seed
        self.rng = random.Random(seed)

    def load_map_from_map_data(self, map_data: List[int]):
        self.sim.load(map_data)
        self.current_map_data = map_data

    def load_map(self, map_path: str):
        with open(map_path, "rb") as map_file:
            map_data = [int(b) for b in map_file.read()]
        self.load_map_from_map_data(map_data)

    def load_random_map(self, map_type: Optional[str] = "SIMPLE_HORIZONTAL_NO_BACKTRACK"):
        map_data = generate_map(level_type=map_type, seed=self.seed).map_data()
        self.load_map_from_map_data(map_data)

    def load_random_official_map(self):
        script_abspath = os.path.abspath(__file__)
        nclone_package_dir = os.path.dirname(script_abspath)
        src_dir = os.path.dirname(nclone_package_dir)
        project_root = os.path.dirname(src_dir)
        
        base_map_path = os.path.join(project_root, 'maps', 'official')

        if not os.path.isdir(base_map_path):
            print(f"Warning: Maps directory not found at primary expected location: {base_map_path}. Trying CWD based.")
            base_map_path = os.path.join(os.getcwd(), 'maps', 'official')
            if not os.path.isdir(base_map_path):
                 raise FileNotFoundError(f"Could not locate maps/official directory. Project root: {project_root}, CWD: {os.getcwd()}")

        subfolders = [f for f in os.listdir(base_map_path) if os.path.isdir(os.path.join(base_map_path, f))]
        if not subfolders:
            raise FileNotFoundError(f"No subfolders found in maps directory: {base_map_path}")
        subfolder = self.rng.choice(subfolders)
        subfolder_path = os.path.join(base_map_path, subfolder)
        
        map_files_in_subfolder = [mf for mf in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, mf)) and not mf.startswith('.')]
        if not map_files_in_subfolder:
            raise FileNotFoundError(f"No map files found in subfolder: {subfolder_path}")
        map_file = self.rng.choice(map_files_in_subfolder)
        map_path = os.path.join(subfolder_path, map_file)

        self.load_map(map_path)
        return os.path.join(subfolder, map_file)

    def reset(self):
        self.sim.reset()

    def tick(self, horizontal_input: int, jump_input: int):
        self.sim.tick(horizontal_input, jump_input)

    def render(self, debug_info: Optional[dict] = None):
        init = self.sim.frame <= 1
        rendered_array = self.sim_renderer.draw(init, debug_info)
        return rendered_array

    def render_collision_map(self):
        init = self.sim.frame <= 1
        collision_map_array = self.sim_renderer.draw_collision_map(init)
        if collision_map_array is None:
            print("Warning: NSimRenderer.draw_collision_map() returned None. Placeholder used.")
            return np.zeros((SRCHEIGHT, SRCWIDTH, 3), dtype=np.uint8)
        return collision_map_array

    def ninja_has_won(self): return self.sim.ninja.has_won()
    def ninja_has_died(self): return self.sim.ninja.has_died()
    def ninja_position(self): return self.sim.ninja.xpos, self.sim.ninja.ypos
    def ninja_velocity(self): return self.sim.ninja.xspeed, self.sim.ninja.yspeed
    def ninja_is_in_air(self): return self.sim.ninja.airborn
    def ninja_is_walled(self): return self.sim.ninja.walled

    def _sim_exit_switch(self):
        entities = self.sim.entity_dic.get(3, []) # Entity type 3 for ExitSwitch
        return entities[-1] if entities else None

    def exit_switch_activated(self):
        switch = self._sim_exit_switch()
        return switch is not None and hasattr(switch, 'active') and not switch.active

    def exit_switch_position(self):
        switch = self._sim_exit_switch()
        return (switch.xpos, switch.ypos) if switch and hasattr(switch, 'xpos') and hasattr(switch, 'ypos') else (None, None)

    def _sim_exit_door(self):
        exit_switch = self._sim_exit_switch()
        return exit_switch.parent if exit_switch and hasattr(exit_switch, 'parent') else None

    def exit_door_position(self):
        door = self._sim_exit_door()
        return (door.xpos, door.ypos) if door and hasattr(door, 'xpos') and hasattr(door, 'ypos') else (None, None)

    def mines(self):
        # Entity type 1 for ToggleMine
        return [e for e in self.sim.entity_dic.get(1, []) if hasattr(e, 'active') and e.active and hasattr(e, 'state') and e.state == 0]

    def get_tile_data(self): return self.sim.tile_dic
    def get_segment_data(self): return self.sim.segment_dic
    def get_grid_edges(self): return {'horizontal': self.sim.hor_grid_edge_dic, 'vertical': self.sim.ver_grid_edge_dic}
    def get_segment_edges(self): return {'horizontal': self.sim.hor_segment_dic, 'vertical': self.sim.ver_segment_dic}

    def exit(self):
        if self.sim_renderer:
            self.sim_renderer.close()

    def get_state_vector(self, only_exit_and_switch: bool = False):
        state = list(self.get_ninja_state())
        state.extend(self.get_entity_states(only_exit_and_switch))
        return np.array(state, dtype=np.float32)

    def get_gold_collected(self): return self.sim.ninja.gold_collected
    def get_doors_opened(self): return self.sim.ninja.doors_opened
    def get_total_gold_available(self): 
        # Entity type 2 for Gold
        return sum(1 for _ in self.sim.entity_dic.get(2, []))

    def get_ninja_state(self):
        ninja = self.sim.ninja
        def safe_div(num, den, default=0.0): return num / den if den and den != 0 else default
        
        max_hor_speed = ninja.MAX_HOR_SPEED if hasattr(ninja, 'MAX_HOR_SPEED') else 1.0 # Default if attr missing
        max_jump_duration = ninja.MAX_JUMP_DURATION if hasattr(ninja, 'MAX_JUMP_DURATION') else 1.0
        gravity_jump = ninja.GRAVITY_JUMP if hasattr(ninja, 'GRAVITY_JUMP') else 0.0
        gravity_fall = ninja.GRAVITY_FALL if hasattr(ninja, 'GRAVITY_FALL') else 1.0
        drag_slow = ninja.DRAG_SLOW if hasattr(ninja, 'DRAG_SLOW') else 0.0
        drag_regular = ninja.DRAG_REGULAR if hasattr(ninja, 'DRAG_REGULAR') else 1.0
        friction_wall = ninja.FRICTION_WALL if hasattr(ninja, 'FRICTION_WALL') else 0.0
        friction_ground = ninja.FRICTION_GROUND if hasattr(ninja, 'FRICTION_GROUND') else 1.0

        return [
            safe_div(ninja.xpos, SRCWIDTH),
            safe_div(ninja.ypos, SRCHEIGHT),
            safe_div(ninja.xspeed / max_hor_speed + 1, 2),
            safe_div(ninja.yspeed / max_hor_speed + 1, 2), 
            float(ninja.airborn),
            float(ninja.walled),
            safe_div(ninja.jump_duration, max_jump_duration),
            safe_div(ninja.applied_gravity - gravity_jump, gravity_fall - gravity_jump),
            safe_div(ninja.applied_drag - drag_slow, drag_regular - drag_slow),
            safe_div(ninja.applied_friction - friction_wall, friction_ground - friction_wall)
        ]

    def get_entity_states(self, only_one_exit_and_switch: bool = False):
        state = []
        exit_switch_entity = self._sim_exit_switch()
        exit_door_entity = self._sim_exit_door()

        if only_one_exit_and_switch:
            return [
                float(exit_switch_entity.active) if exit_switch_entity and hasattr(exit_switch_entity, 'active') else 0.0,
                float(exit_door_entity.active) if exit_door_entity and hasattr(exit_door_entity, 'active') else 0.0
            ]

        MAX_ATTRIBUTES = 4
        # Ensure keys in MAX_COUNTS are the entity type values (integers)
        MAX_COUNTS = {
            EntityToggleMine.ENTITY_TYPE: 128, EntityGold.ENTITY_TYPE: 128,
            EntityExit.ENTITY_TYPE: 1, EntityExitSwitch.ENTITY_TYPE: 1,
            EntityDoorRegular.ENTITY_TYPE: 32, EntityDoorLocked.ENTITY_TYPE: 32, EntityDoorTrap.ENTITY_TYPE: 32,
            EntityLaunchPad.ENTITY_TYPE: 32, EntityOneWayPlatform.ENTITY_TYPE: 32, EntityDroneZap.ENTITY_TYPE: 32,
            EntityBounceBlock.ENTITY_TYPE: 32, EntityThwump.ENTITY_TYPE: 32, EntityBoostPad.ENTITY_TYPE: 32,
            EntityDeathBall.ENTITY_TYPE: 32, EntityMiniDrone.ENTITY_TYPE: 32, EntityShoveThwump.ENTITY_TYPE: 32
        }

        # Iterate through a fixed order of entity types for consistent state vector
        # This order should match the definition of MAX_COUNTS if that's intended as the canonical order.
        # Or define a canonical list: e.g., sorted(MAX_COUNTS.keys())
        sorted_entity_types = sorted(list(MAX_COUNTS.keys()))

        for entity_type_val in sorted_entity_types: 
            entities = self.sim.entity_dic.get(entity_type_val, [])
            max_count = MAX_COUNTS[entity_type_val]
            state.append(float(len(entities)) / max_count if max_count > 0 else 0.0)

            for i in range(max_count):
                if i < len(entities):
                    entity = entities[i]
                    try:
                        raw_state_values = entity.get_state()
                        if not isinstance(raw_state_values, (list, tuple)): raw_state_values = [raw_state_values]
                    except AttributeError: raw_state_values = [] 

                    current_entity_state = []
                    for val in raw_state_values:
                        if isinstance(val, bool): current_entity_state.append(float(val))
                        elif isinstance(val, (int, float)):
                            current_entity_state.append(float(val))
                        else: current_entity_state.append(0.0) 
                    
                    current_entity_state.extend([0.0] * (MAX_ATTRIBUTES - len(current_entity_state)))
                    state.extend(current_entity_state[:MAX_ATTRIBUTES])
                else:
                    state.extend([0.0] * MAX_ATTRIBUTES)
        return state

    def _get_geometry_state(self):
        state = []
        def safe_div(num, den, default=0.0): return num / den if den and den != 0 else default

        for x in range(44):
            for y in range(25):
                state.append(safe_div(float(self.sim.tile_dic.get((x,y),0)), 37.0))
        for x in range(88):
            for y in range(51):
                state.append(safe_div(float(self.sim.hor_grid_edge_dic.get((x,y),0)) + 1.0, 2.0))
        for x in range(89):
            for y in range(50):
                state.append(safe_div(float(self.sim.ver_grid_edge_dic.get((x,y),0)) + 1.0, 2.0))
        for x in range(88):
            for y in range(51):
                state.append(safe_div(float(self.sim.hor_segment_dic.get((x,y),0)) + 1.0, 2.0))
        for x in range(89):
            for y in range(50):
                state.append(safe_div(float(self.sim.ver_segment_dic.get((x,y),0)) + 1.0, 2.0))
        return state
