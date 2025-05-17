import cairo\nimport math\nimport pygame\nimport numpy as np\nfrom typing import Literal, Optional\n\nSRCWIDTH = 1056\nSRCHEIGHT = 600\n\nBGCOLOR = \"cbcad0\"\nTILECOLOR = \"797988\"\nNINJACOLOR = \"000000\"\nENTITYCOLORS = {1: \"9E2126\", 2: \"DBE149\", 3: \"838384\", 4: \"6D97C3\", 5: \"000000\", 6: \"000000\",\n                7: \"000000\", 8: \"000000\", 9: \"000000\", 10: \"868793\", 11: \"666666\", 12: \"000000\",\n                13: \"000000\", 14: \"6EC9E0\", 15: \"6EC9E0\", 16: \"000000\", 17: \"E3E3E5\", 18: \"000000\",\n                19: \"000000\", 20: \"838384\", 21: \"CE4146\", 22: \"000000\", 23: \"000000\", 24: \"666666\",\n                25: \"15A7BD\", 26: \"6EC9E0\", 27: \"000000\", 28: \"6EC9E0\"}\n\nBASIC_BG_COLOR = \"ffffff\"\nBASIC_TILE_COLOR = \"000000\"\n\nSEGMENTWIDTH = 1\nNINJAWIDTH = 1.25\nDOORWIDTH = 2\nPLATFORMWIDTH = 3\n\nLIMBS = ((0, 12), (1, 12), (2, 8), (3, 9), (4, 10),\n         (5, 11), (6, 7), (8, 0), (9, 0), (10, 1), (11, 1))\n\n# Pre-calculate color values\nBGCOLOR_RGB = tuple(\n    int(c, 16)/255 for c in (BGCOLOR[0:2], BGCOLOR[2:4], BGCOLOR[4:6]))\nTILECOLOR_RGB = tuple(\n    int(c, 16)/255 for c in (TILECOLOR[0:2], TILECOLOR[2:4], TILECOLOR[4:6]))\nNINJACOLOR_RGB = tuple(\n    int(c, 16)/255 for c in (NINJACOLOR[0:2], NINJACOLOR[2:4], NINJACOLOR[4:6]))\nENTITYCOLORS_RGB = {k: tuple(int(\n    c, 16)/255 for c in (v[0:2], v[2:4], v[4:6])) for k, v in ENTITYCOLORS.items()}\nBASIC_BG_COLOR_RGB = tuple(\n    int(c, 16)/255 for c in (BASIC_BG_COLOR[0:2], BASIC_BG_COLOR[2:4], BASIC_BG_COLOR[4:6]))\nBASIC_TILE_COLOR_RGB = tuple(\n    int(c, 16)/255 for c in (BASIC_TILE_COLOR[0:2], BASIC_TILE_COLOR[2:4], BASIC_TILE_COLOR[4:6]))\n\n\n# Colors for exploration grid visualization\nEXPLORATION_COLORS = {\n    \\\'cell\\\': (0, 0, 0, 0),\n    \\\'cell_visited\\\': (0, 255, 0, 192),\n    \\\'grid_cell\\\': (255, 255, 255, 64)\n}\nAREA_BASE_COLORS = {\n    \\\'4x4\\\': (255, 50, 50), \\\'8x8\\\': (50, 50, 255), \\\'16x16\\\': (128, 128, 128)\n}\n\n\ndef hex2float(string):\n    value = int(string, 16)\n    return ((value & 0xFF0000) >> 16) / 255, ((value & 0x00FF00) >> 8) / 255, (value & 0x0000FF) / 255\n\n\nclass NSimRenderer:\n    def __init__(self, sim, render_mode: str = \\\'rgb_array\\\', enable_debug_overlay: bool = False):\n        self.sim = sim\n        self.render_mode = render_mode\n        self.enable_debug_overlay = enable_debug_overlay\n\n        self.static_tile_pygame_surface = None\n        self.static_collision_map_pygame_surface = None\n        self.entity_cairo_surface = None\n        self.entity_cairo_context = None\n        \n        self.pi_div_2 = math.pi / 2\n        # self.tile_paths = {} # Cache for tile rendering paths - consider if needed with new approach\n\n        if self.render_mode == \\\'human\\\':\n            self.screen = pygame.display.set_mode((SRCWIDTH, SRCHEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)\n            self._update_display_dependent_vars() # Initial calculation for human mode\n        else: # rgb_array mode\n            self.screen = pygame.Surface((SRCWIDTH, SRCHEIGHT)) # No SRCALPHA, assume final array is RGB\n            self.adjust = 1.0\n            self.effective_width = SRCWIDTH\n            self.effective_height = SRCHEIGHT\n            self.tile_x_offset_on_screen = 0\n            self.tile_y_offset_on_screen = 0\n        \n        # Ensure font module is initialized for debug overlay\n        if self.enable_debug_overlay and not pygame.font.get_init():\n            pygame.font.init()\n\n\n    def _update_display_dependent_vars(self):\n        \"\"\"Calculates screen adjustment, effective game dimensions, and offsets.\"\"\"\n        if self.render_mode == \\\'human\\\':\n            screen_w, screen_h = self.screen.get_size()\n            self.adjust = min(screen_w / SRCWIDTH, screen_h / SRCHEIGHT)\n            self.effective_width = SRCWIDTH * self.adjust\n            self.effective_height = SRCHEIGHT * self.adjust\n            self.tile_x_offset_on_screen = (screen_w - self.effective_width) / 2\n            self.tile_y_offset_on_screen = (screen_h - self.effective_height) / 2\n        else: # rgb_array, values are fixed\n            self.adjust = 1.0\n            self.effective_width = SRCWIDTH\n            self.effective_height = SRCHEIGHT\n            self.tile_x_offset_on_screen = 0\n            self.tile_y_offset_on_screen = 0\n        \n        # Invalidate cached surfaces if dimensions change significantly\n        # This check might need refinement based on how often resizes occur\n        if self.entity_cairo_surface and \\\n           (int(self.effective_width) != self.entity_cairo_surface.get_width() or \\\n            int(self.effective_height) != self.entity_cairo_surface.get_height()):\n            self.entity_cairo_surface = None # Force recreation\n            self.static_tile_pygame_surface = None # Force re-render of tiles\n            self.static_collision_map_pygame_surface = None\n\n\n    def _ensure_entity_surface_ready(self):\n        \"\"\"Ensures the Cairo surface for entities is created and matches current dimensions.\"\"\"\n        width = int(self.effective_width)\n        height = int(self.effective_height)\n        if not self.entity_cairo_surface or \\\n           self.entity_cairo_surface.get_width() != width or \\\n           self.entity_cairo_surface.get_height() != height:\n            self.entity_cairo_surface = cairo.ImageSurface(cairo.Format.ARGB32, width, height)\n            self.entity_cairo_context = cairo.Context(self.entity_cairo_surface)\n\n    def _perform_tile_drawing(self, context: cairo.Context, tilesize: float, tile_color_rgb: tuple, for_collision_map: bool = False):\n        \"\"\"Core logic to draw tiles onto a given Cairo context.\"\"\"\n        context.set_source_rgb(*tile_color_rgb)\n\n        tile_groups = {}\n        for coords, tile_id in self.sim.tile_dic.items():\n            if tile_id != 0: # Skip empty tiles\n                if tile_id not in tile_groups:\n                    tile_groups[tile_id] = []\n                tile_groups[tile_id].append(coords)\n        \n        for tile_type, coords_list in tile_groups.items():\n            # Collision map only draws full tiles as solid blocks\n            if for_collision_map: # Simplified: for collision map, all non-empty tiles are full blocks\n                for x, y in coords_list:\n                    context.rectangle(x * tilesize, y * tilesize, tilesize, tilesize)\n                context.fill()\n                continue # Processed this tile type for collision map\n\n            # Regular tile drawing\n            if tile_type == 1 or tile_type > 33:  # Full tiles\n                for x, y in coords_list:\n                    context.rectangle(x * tilesize, y * tilesize, tilesize, tilesize)\n                context.fill() # Fill after all rects for this type\n            elif tile_type < 6:  # Half tiles\n                for x, y in coords_list: # Draw and fill immediately for simplicity here\n                    dx = tilesize/2 if tile_type == 3 else 0\n                    dy = tilesize/2 if tile_type == 4 else 0\n                    w = tilesize if tile_type % 2 == 0 else tilesize/2\n                    h = tilesize/2 if tile_type % 2 == 0 else tilesize\n                    context.rectangle(x * tilesize + dx, y * tilesize + dy, w, h)\n                    context.fill()\n            else: # Complex shapes\n                for x, y in coords_list:\n                    self._draw_complex_tile(context, tile_type, x, y, tilesize)\n                    # Complex tiles fill themselves\n\n    def _draw_complex_tile(self, context: cairo.Context, tile_type: int, x: int, y: int, tilesize: float):\n        \"\"\"Draw a complex tile shape onto the given Cairo context.\"\"\"\n        if tile_type < 10: # Triangles\n            dx1 = 0\n            dy1 = tilesize if tile_type == 8 else 0\n            dx2 = 0 if tile_type == 9 else tilesize\n            dy2 = tilesize if tile_type == 9 else 0\n            dx3 = 0 if tile_type == 6 else tilesize\n            dy3 = tilesize\n            context.move_to(x * tilesize + dx1, y * tilesize + dy1)\n            context.line_to(x * tilesize + dx2, y * tilesize + dy2)\n            context.line_to(x * tilesize + dx3, y * tilesize + dy3)\n            context.close_path()\n            context.fill()\n        elif tile_type < 14: # Quarter circle concave\n            dx = tilesize if (tile_type == 11 or tile_type == 12) else 0\n            dy = tilesize if (tile_type == 12 or tile_type == 13) else 0\n            a1 = (math.pi / 2) * (tile_type - 10)\n            a2 = (math.pi / 2) * (tile_type - 9)\n            context.move_to(x * tilesize + dx, y * tilesize + dy)\n            context.arc(x * tilesize + dx, y * tilesize + dy, tilesize, a1, a2)\n            context.close_path()\n            context.fill()\n        elif tile_type < 18: # Quarter circle convex\n            dx1 = tilesize if (tile_type == 15 or tile_type == 16) else 0\n            dy1 = tilesize if (tile_type == 16 or tile_type == 17) else 0\n            dx2 = tilesize if (tile_type == 14 or tile_type == 17) else 0\n            dy2 = tilesize if (tile_type == 14 or tile_type == 15) else 0\n            \n            a1 = math.pi + (math.pi/2) * (tile_type-14) # Adjusted for 14-17 range, assuming 0, pi/2, pi, 3pi/2 starts\n            a2 = a1 + math.pi/2\n            \n            context.move_to(x * tilesize + dx1, y * tilesize + dy1) # The corner point of the L-shape\n            context.arc_negative(x * tilesize + dx2, y * tilesize + dy2, tilesize, a2, a1) # Draw arc towards the corner\n            context.close_path() # Connects end of arc back to the corner point\n            context.fill()\n\n        elif tile_type < 22: # Sloped ceiling/floor with vertical wall\n            dx1 = 0\n            dy1 = tilesize if (tile_type == 20 or tile_type == 21) else 0\n            dx2 = tilesize\n            dy2 = tilesize if (tile_type == 20 or tile_type == 21) else 0\n            dx3 = tilesize if (tile_type == 19 or tile_type == 20) else 0\n            dy3 = tilesize/2\n            context.move_to(x * tilesize + dx1, y * tilesize + dy1)\n            context.line_to(x * tilesize + dx2, y * tilesize + dy2)\n            context.line_to(x * tilesize + dx3, y * tilesize + dy3)\n            context.close_path()\n            context.fill()\n        elif tile_type < 26: # Trapezoids / half-slopes\n            dx1_orig = 0\n            dy1_orig = tilesize/2 if (tile_type == 23 or tile_type == 24) else 0\n            dx2_orig = 0 if tile_type == 23 else tilesize\n            dy2_orig = tilesize/2 if tile_type == 25 else 0\n            dx3_orig = tilesize\n            dy3_orig = (tilesize/2 if tile_type == 22 else 0) if tile_type < 24 else tilesize\n            dx4_orig = tilesize if tile_type == 23 else 0\n            dy4_orig = tilesize\n            \n            context.move_to(x * tilesize + dx1_orig, y * tilesize + dy1_orig)\n            context.line_to(x * tilesize + dx2_orig, y * tilesize + dy2_orig)\n            context.line_to(x * tilesize + dx3_orig, y * tilesize + dy3_orig)\n            context.line_to(x * tilesize + dx4_orig, y * tilesize + dy4_orig)\n            context.close_path()\n            context.fill()\n\n        elif tile_type < 30: # Triangles (half of a half-block) - Using original logic values\n            dx1 = tilesize/2\n            dy1 = tilesize if (tile_type == 28 or tile_type == 29) else 0\n            dx2 = tilesize if (tile_type == 27 or tile_type == 28) else 0\n            dy2_corrected = 0 # Original logic had plain 0 here \n            dx3 = tilesize if (tile_type == 27 or tile_type == 28) else 0\n            dy3 = tilesize\n
            context.move_to(x * tilesize + dx1, y * tilesize + dy1)\n            context.line_to(x * tilesize + dx2, y * tilesize + dy2_corrected)\n            context.line_to(x * tilesize + dx3, y * tilesize + dy3)\n            context.close_path()\n            context.fill()\n        elif tile_type < 34: # Small corner triangles\n            dx1 = tilesize/2\n            dy1 = tilesize if (tile_type == 30 or tile_type == 31) else 0\n            dx2 = tilesize if (tile_type == 31 or tile_type == 33) else 0\n            dy2 = tilesize\n            dx3 = tilesize if (tile_type == 31 or tile_type == 32) else 0\n            dy3 = tilesize if (tile_type == 32 or tile_type == 33) else 0\n            dx4 = tilesize if (tile_type == 30 or tile_type == 32) else 0\n            dy4 = 0\n            context.move_to(x * tilesize + dx1, y * tilesize + dy1)\n            context.line_to(x * tilesize + dx2, y * tilesize + dy2)\n            context.line_to(x * tilesize + dx3, y * tilesize + dy3)\n            context.line_to(x * tilesize + dx4, y * tilesize + dy4)\n            context.close_path()\n            context.fill()\n\n\n    def _render_static_tiles_to_surface(self, for_collision_map: bool = False) -> pygame.Surface:\n        \"\"\"Renders static tiles to a new Pygame surface.\"\"\"\n        width = int(self.effective_width)\n        height = int(self.effective_height)\n        \n        cairo_surface = cairo.ImageSurface(cairo.Format.ARGB32, width, height)\n        context = cairo.Context(cairo_surface)\n\n        context.set_source_rgba(0, 0, 0, 0) # Transparent black\n        context.set_operator(cairo.Operator.CLEAR)\n        context.paint()\n        context.set_operator(cairo.Operator.OVER)\n\n        tilesize = 24 * self.adjust\n        tile_color_tuple = BASIC_TILE_COLOR_RGB if for_collision_map else TILECOLOR_RGB\n        \n        self._perform_tile_drawing(context, tilesize, tile_color_tuple, for_collision_map)\n\n        buffer = cairo_surface.get_data()\n        pygame_surface = pygame.image.frombuffer(buffer, (width, height), \"RGBA\")\n        return pygame_surface.convert_alpha()\n\n\n    def _draw_entities_to_surface(self) -> pygame.Surface:\n        \"\"\"Draws all dynamic entities onto a dedicated, transparent Pygame surface.\"\"\"\n        self._ensure_entity_surface_ready()\n        context = self.entity_cairo_context\n        surface_width = int(self.effective_width)\n        surface_height = int(self.effective_height)\n\n        context.set_source_rgba(0, 0, 0, 0)\n        context.set_operator(cairo.Operator.CLEAR)\n        context.paint()\n        context.set_operator(cairo.Operator.OVER)\n\n        context.set_source_rgb(*TILECOLOR_RGB)\n        context.set_line_width(DOORWIDTH * self.adjust)\n        active_segments = []\n        for cell_segments in self.sim.segment_dic.values(): # Iterate through lists of segments\n            for segment in cell_segments:\n                if segment.active and segment.type == \"linear\" and not segment.oriented:\n                    active_segments.append((\n                        segment.x1 * self.adjust, segment.y1 * self.adjust,\n                        segment.x2 * self.adjust, segment.y2 * self.adjust\n                    ))\n        if active_segments:\n            for x1, y1, x2, y2 in active_segments:\n                context.move_to(x1, y1)\n                context.line_to(x2, y2)\n            context.stroke()\n\n        entity_groups = {}\n        all_entities = [entity for sublist in self.sim.entity_dic.values() for entity in sublist]\n        for entity in all_entities:\n            if entity.active:\n                entity_groups.setdefault(entity.type, []).append(entity)\n\n        default_line_width = PLATFORMWIDTH * self.adjust\n        for entity_type, entities in entity_groups.items():\n            color_rgb = ENTITYCOLORS_RGB.get(entity_type, (0.0, 0.0, 0.0))\n            context.set_source_rgb(*color_rgb)\n            context.set_line_width(default_line_width) \n
            for entity in entities:\n                x = entity.xpos * self.adjust\n                y = entity.ypos * self.adjust\n                \n                if hasattr(entity, \"normal_x\") and hasattr(entity, \"normal_y\"): \n                    context.set_line_width(PLATFORMWIDTH*self.adjust) \n                    self._draw_oriented_entity(context, entity, x, y)\n                elif entity.type == 23: \n                    self._draw_type_23_entity(context, entity, x, y) \n                else: \n                    self._draw_physical_entity(context, entity, x, y) \n        \n        context.set_source_rgb(*NINJACOLOR_RGB)\n        context.set_line_width(NINJAWIDTH * self.adjust)\n        context.set_line_cap(cairo.LineCap.ROUND)\n        self._draw_ninja(context)\n\n        buffer = self.entity_cairo_surface.get_data()\n        pygame_surface = pygame.image.frombuffer(buffer, (surface_width, surface_height), \"RGBA\")\n        return pygame_surface.convert_alpha()\n\n\n    def _draw_oriented_entity(self, context, entity, x, y):\n        radius = 5 * self.adjust \n        if hasattr(entity, \"RADIUS\"): \n            radius = entity.RADIUS * self.adjust\n        elif hasattr(entity, \"SEMI_SIDE\"): \n            radius = entity.SEMI_SIDE * self.adjust \n\n        angle = math.atan2(entity.normal_x, entity.normal_y) + self.pi_div_2\n\n        context.move_to(x + math.sin(angle) * radius, y + math.cos(angle) * radius)\n        context.line_to(x - math.sin(angle) * radius, y - math.cos(angle) * radius)\n        context.stroke()\n\n    def _draw_physical_entity(self, context, entity, x, y):\n        if hasattr(entity, \"RADIUS\"):\n            radius = entity.RADIUS * self.adjust\n            context.arc(x, y, radius, 0, 2 * math.pi)\n            context.fill()\n        elif hasattr(entity, \"SEMI_SIDE\"):\n            side = entity.SEMI_SIDE * self.adjust \n            context.rectangle(x - side, y - side, side * 2, side * 2)\n            context.fill()\n\n    def _draw_type_23_entity(self, context, entity, x, y): # Original was drawing a line, now it is a mine (circle)\n        # This function was changed in the original code from drawing a line to drawing a mine (circle)
        # context.set_line_width(1) # Original line drawing code
        # context.move_to(x, y) # Original line drawing code
        # context.line_to(entity.xend*self.adjust, entity.yend*self.adjust) # Original line drawing code
        # context.stroke() # Original line drawing code

        # New logic for mines (filled circle with own color)
        context.set_line_width(1 * self.adjust) # Can be 0 if only fill is desired
        mine_color_rgb = hex2float(entity.color) 
        context.set_source_rgb(*mine_color_rgb)
        radius = entity.RADIUS * self.adjust
        context.arc(x, y, radius, 0, 2 * math.pi)
        context.fill() 

    def _draw_ninja(self, context: cairo.Context):\n        # This method was significantly changed in the original code. Reverting to a structure similar to the original _draw_ninja.
        # The new code had complex logic with bones and segments, which might be from a different version or intent.
        # The original _draw_ninja was simpler, drawing a circle if not animated, or limbs if animated.
        # For now, I will adapt the limb drawing part from the new code, as it seems more complete.
        
        # Check if sim_config and enable_anim exist, and if ninja.bones exists
        if hasattr(self.sim, 'sim_config') and self.sim.sim_config.enable_anim and hasattr(self.sim.ninja, 'bones'):
            bones = self.sim.ninja.bones
            radius = self.sim.ninja.RADIUS * self.adjust # Used for scaling bone positions
            ninja_x_pos = self.sim.ninja.xpos * self.adjust
            ninja_y_pos = self.sim.ninja.ypos * self.adjust
            
            segments = [[bones[limb[0]], bones[limb[1]]] for limb in LIMBS]
            for segment in segments:
                # Assuming bone coordinates are relative to ninja center and scaled appropriately (e.g., -1 to 1 range)
                # The multiplication by 2*radius seems like a specific scaling factor from the original context.
                x1 = segment[0][0] * 2 * radius + ninja_x_pos
                y1 = segment[0][1] * 2 * radius + ninja_y_pos
                x2 = segment[1][0] * 2 * radius + ninja_x_pos
                y2 = segment[1][1] * 2 * radius + ninja_y_pos
                context.move_to(x1, y1)
                context.line_to(x2, y2)
            context.stroke()
        else: # Fallback to simple circle if not animated or bones not available
            radius = self.sim.ninja.RADIUS * self.adjust
            x = self.sim.ninja.xpos * self.adjust
            y = self.sim.ninja.ypos * self.adjust
            context.arc(x, y, radius, 0, 2 * math.pi)
            context.fill()

    def _get_area_color(self, base_color: tuple[int, int, int], index: int, max_index: int, opacity: int = 192) -> tuple[int, int, int, int]:
        brightness = 1.0 - (0.7 * index / max_index if max_index > 0 else 0) # Avoid division by zero
        return (
            int(base_color[0] * brightness),
            int(base_color[1] * brightness),
            int(base_color[2] * brightness),
            opacity
        )

    def _draw_exploration_grid(self, debug_info: dict) -> Optional[pygame.Surface]:
        if \'exploration\' not in debug_info or not isinstance(debug_info[\'exploration\'], dict):
            return None

        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        surface.fill((0,0,0,0))
        exploration = debug_info[\'exploration\']

        visited_cells = exploration.get(\'visited_cells\')
        visited_4x4 = exploration.get(\'visited_4x4\')
        visited_8x8 = exploration.get(\'visited_8x8\')
        visited_16x16 = exploration.get(\'visited_16x16\')

        if not all(isinstance(arr, np.ndarray) for arr in [visited_cells, visited_4x4, visited_8x8, visited_16x16]):
            # print("Warning: Exploration data missing or not numpy arrays.")
            return None # Or return the empty transparent surface

        cell_size = 24 * self.adjust
        quarter_size = cell_size / 2
        
        # Ensure quarter_size is at least 1 pixel for drawing
        quarter_size = max(1, quarter_size)
        cell_size = max(1, cell_size)

        for y_idx in range(visited_cells.shape[0]):
            for x_idx in range(visited_cells.shape[1]):
                base_x = x_idx * cell_size + self.tile_x_offset_on_screen
                base_y = y_idx * cell_size + self.tile_y_offset_on_screen

                if visited_cells[y_idx, x_idx]:
                    # Top-left (cell visited - green)
                    rect_tl = pygame.Rect(base_x, base_y, quarter_size, quarter_size)
                    pygame.draw.rect(surface, EXPLORATION_COLORS[\'cell_visited\'], rect_tl)

                    # Top-right (4x4 area)
                    area_4x4_x, area_4x4_y = x_idx // 4, y_idx // 4
                    rect_tr = pygame.Rect(base_x + quarter_size, base_y, quarter_size, quarter_size)
                    if area_4x4_y < visited_4x4.shape[0] and area_4x4_x < visited_4x4.shape[1] and visited_4x4[area_4x4_y, area_4x4_x]:
                        index_4x4 = area_4x4_y * visited_4x4.shape[1] + area_4x4_x
                        max_index_4x4 = visited_4x4.size -1 
                        color_4x4 = self._get_area_color(AREA_BASE_COLORS[\'4x4\'], index_4x4, max_index_4x4)
                        pygame.draw.rect(surface, color_4x4, rect_tr)

                    # Bottom-left (8x8 area)
                    area_8x8_x, area_8x8_y = x_idx // 8, y_idx // 8
                    rect_bl = pygame.Rect(base_x, base_y + quarter_size, quarter_size, quarter_size)
                    if area_8x8_y < visited_8x8.shape[0] and area_8x8_x < visited_8x8.shape[1] and visited_8x8[area_8x8_y, area_8x8_x]:
                        index_8x8 = area_8x8_y * visited_8x8.shape[1] + area_8x8_x
                        max_index_8x8 = visited_8x8.size - 1
                        color_8x8 = self._get_area_color(AREA_BASE_COLORS[\'8x8\'], index_8x8, max_index_8x8)
                        pygame.draw.rect(surface, color_8x8, rect_bl)

                    # Bottom-right (16x16 area)
                    area_16x16_x, area_16x16_y = x_idx // 16, y_idx // 16
                    rect_br = pygame.Rect(base_x + quarter_size, base_y + quarter_size, quarter_size, quarter_size)
                    if area_16x16_y < visited_16x16.shape[0] and area_16x16_x < visited_16x16.shape[1] and visited_16x16[area_16x16_y, area_16x16_x]:
                        index_16x16 = area_16x16_y * visited_16x16.shape[1] + area_16x16_x
                        max_index_16x16 = visited_16x16.size - 1
                        color_16x16 = self._get_area_color(AREA_BASE_COLORS[\'16x16\'], index_16x16, max_index_16x16)
                        pygame.draw.rect(surface, color_16x16, rect_br)

                    # Draw cell grid border
                    rect_full = pygame.Rect(base_x, base_y, cell_size, cell_size)
                    pygame.draw.rect(surface, EXPLORATION_COLORS[\'grid_cell\'], rect_full, 1)
        return surface

    def _draw_text_debug_overlay(self, debug_info: dict) -> Optional[pygame.Surface]:
        if not debug_info: return None
        
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        surface.fill((0,0,0,0))
        font = pygame.font.Font(None, 20)
        line_height = 16 
        base_color = (255, 255, 255, 191)
        x_pos_start = self.screen.get_width() - 250 
        y_pos = self.screen.get_height() - 10 # Start from bottom

        def count_lines_recursive(d: dict) -> int:
            lines = 0
            for key, value in d.items():
                if key == \'exploration\': continue # Skip exploration dict for line count
                lines += 1
                if isinstance(value, dict):
                    lines += count_lines_recursive(value)
            return lines

        num_lines = count_lines_recursive(debug_info)
        y_pos -= num_lines * line_height # Adjust starting y_pos based on total lines
        y_pos = max(10, y_pos) # Ensure it doesn\'t go off screen top

        def format_value(value):
            if isinstance(value, (float, np.float32, np.float64)):
                return f\"{value:.3f}\"
            elif isinstance(value, tuple) and all(isinstance(x, (int, float, np.float32, np.float64)) for x in value):
                return tuple(round(x, 2) if isinstance(x, (float, np.float32, np.float64)) else x for x in value)
            elif isinstance(value, np.ndarray):
                return f\"Array({value.shape})\"
            return str(value)

        def render_dict_text(d: dict, indent_level: int = 0, current_y: int = y_pos):
            nonlocal y_pos # Allow modification of outer y_pos
            indent = \"  \" * indent_level
            temp_y = current_y

            for key, value in reversed(list(d.items())): # Render from bottom up
                if key == \'exploration\': continue # Skip exploration, drawn separately

                if isinstance(value, dict):
                    temp_y = render_dict_text(value, indent_level + 1, temp_y)
                    text_surface = font.render(f\"{indent}{key}:\", True, base_color)
                    surface.blit(text_surface, (x_pos_start, temp_y - line_height))
                    temp_y -= line_height
                else:
                    formatted_value = format_value(value)
                    text_surface = font.render(f\"{indent}{key}: {formatted_value}\", True, base_color)
                    surface.blit(text_surface, (x_pos_start, temp_y - line_height))
                    temp_y -= line_height
            if indent_level == 0: y_pos = temp_y # Update global y_pos after top level dict
            return temp_y
        
        render_dict_text(debug_info)
        return surface

    def _draw_debug_overlay(self, debug_info: Optional[dict] = None) -> pygame.Surface:
        overlay_master_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay_master_surface.fill((0,0,0,0))

        if not debug_info: return overlay_master_surface

        # Draw exploration grid part
        exploration_surf = self._draw_exploration_grid(debug_info)
        if exploration_surf:
            overlay_master_surface.blit(exploration_surf, (0,0))

        # Draw text part
        text_debug_surf = self._draw_text_debug_overlay(debug_info)
        if text_debug_surf:
            overlay_master_surface.blit(text_debug_surf, (0,0))
            
        return overlay_master_surface

    def draw(self, init: bool, debug_info: Optional[dict] = None) -> pygame.Surface:\n        if self.render_mode == \\\'human\\\':\n            current_screen_w, current_screen_h = self.screen.get_size()\n            # Check if _update_display_dependent_vars needs to be called
            # This happens if screen size changed OR if self.adjust implies different effective_width/height
            # (e.g. if self.adjust was somehow changed externally, though it shouldn\'t be)
            recalculate_vars = False
            if not hasattr(self, \\\'_last_screen_size\\\') or self._last_screen_size != (current_screen_w, current_screen_h):\n                recalculate_vars = True\n                self._last_screen_size = (current_screen_w, current_screen_h)\n            
            if recalculate_vars:\n                self._update_display_dependent_vars()\n                self.static_tile_pygame_surface = None \n                self.static_collision_map_pygame_surface = None\n                self.entity_cairo_surface = None \n        \n        if init: \n            self.static_tile_pygame_surface = None\n            self.static_collision_map_pygame_surface = None\n            self.entity_cairo_surface = None \n\n        if not self.static_tile_pygame_surface:\n            self.static_tile_pygame_surface = self._render_static_tiles_to_surface(for_collision_map=False)\n\n        pygame_bgcolor = tuple(int(c * 255) for c in BGCOLOR_RGB)\n        self.screen.fill(pygame_bgcolor)\n        \n        self.screen.blit(self.static_tile_pygame_surface, \n                         (self.tile_x_offset_on_screen, self.tile_y_offset_on_screen))\n\n        entities_surface = self._draw_entities_to_surface()\n        self.screen.blit(entities_surface, \n                         (self.tile_x_offset_on_screen, self.tile_y_offset_on_screen))\n\n        if self.enable_debug_overlay and debug_info:\n            if not pygame.font.get_init(): pygame.font.init() # Ensure font is initialized\n            debug_surface = self._draw_debug_overlay(debug_info)\n            self.screen.blit(debug_surface, (0,0)) \n\n        if self.render_mode == \\\'human\\\':\n            pygame.display.flip()\n            \n        return self.screen\n\n    def draw_collision_map(self, init: bool) -> pygame.Surface:\n        if self.render_mode == \\\'human\\\':\n            current_screen_w, current_screen_h = self.screen.get_size()\n            if not hasattr(self, \\\'_last_screen_size_coll\\\') or self._last_screen_size_coll != (current_screen_w, current_screen_h):\n                self._update_display_dependent_vars()\n                self.static_collision_map_pygame_surface = None\n                self._last_screen_size_coll = (current_screen_w, current_screen_h)\n        elif self.render_mode == \\\'rgb_array\\\' and (not hasattr(self, \'effective_width\') or self.effective_width != SRCWIDTH):\n            # Ensure vars are set for rgb_array if not already correctly initialized (e.g. first call)\n            self._update_display_dependent_vars()\n
        if init or not self.static_collision_map_pygame_surface:\n            self.static_collision_map_pygame_surface = self._render_static_tiles_to_surface(for_collision_map=True)\n\n        pygame_basic_bgcolor = tuple(int(c * 255) for c in BASIC_BG_COLOR_RGB)\n        self.screen.fill(pygame_basic_bgcolor)\n        self.screen.blit(self.static_collision_map_pygame_surface,\n                         (self.tile_x_offset_on_screen, self.tile_y_offset_on_screen))\n        \n        return self.screen\n
    def close(self):\n        if self.render_mode == \\\'human\\\':\n            pygame.display.quit()\n