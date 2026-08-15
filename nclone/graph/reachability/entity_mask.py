"""
Entity-based dynamic masking for path-aware reward shaping.

Handles dynamic entity states (doors, mines) that change during gameplay,
applying them as lightweight masks over the static tile connectivity graph.
"""

from typing import Dict, Set, Tuple, List, Any, Optional
from ...constants.entity_types import EntityType

# Hardcoded cell size as per N++ constants
CELL_SIZE = 24

# One-way platform geometry (mirrors physics: semi-side 12, ninja radius 10). The
# blocking barrier is modelled as a line segment of half-length SEMI+RADIUS so the
# graph severs roughly the same lateral span the physics depenetration captures.
_ONE_WAY_SEMI_SIDE = 12.0
_ONE_WAY_LATERAL_PAD = 10.0  # NINJA_RADIUS


def _orient(a, b, c) -> float:
    """Signed area * 2 of triangle (a, b, c); sign gives orientation."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if segment p1p2 intersects segment p3p4 (proper or touching)."""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    # Collinear/touching cases: check on-segment for any zero orientation.
    def _on(a, b, c) -> bool:
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    if d1 == 0 and _on(p3, p4, p1):
        return True
    if d2 == 0 and _on(p3, p4, p2):
        return True
    if d3 == 0 and _on(p1, p2, p3):
        return True
    if d4 == 0 and _on(p1, p2, p4):
        return True
    return False


class EntityMask:
    """
    Manages dynamic entity states and generates blocking masks.

    Handles:
    - Locked doors (open/closed based on switch collection)
    - Toggle mines (deadly/safe/toggling states)

    Key insight: Entity POSITIONS never change during a run, only STATES change.
    This allows efficient caching of position-based structures.
    """

    def __init__(self, level_data: Dict[str, Any]):
        """
        Initialize entity mask from level data.

        Args:
            level_data: Level data containing tiles, entities, switch_states
        """
        self.entities = level_data.get("entities", [])
        self.switch_states = level_data.get("switch_states", {})

        # Parse entities by type (positions are cached - never change!)
        self.doors = self._extract_doors()
        self.mines = self._extract_mines()
        self.one_ways = self._extract_one_ways()
        self.solid_bodies = self._extract_solid_bodies()

        # Build switch->door mapping (static per level)
        self.switch_to_door = self._build_switch_door_mapping()

    def _extract_doors(self) -> List[Dict]:
        """Extract locked door entities only."""
        doors = []
        for entity in self.entities:
            entity_type = entity.get("type")
            if entity_type == EntityType.LOCKED_DOOR:
                doors.append(entity)
        return doors

    def _extract_mines(self) -> List[Dict]:
        """Extract toggle mine entities."""
        mines = []
        for entity in self.entities:
            entity_type = entity.get("type")
            # Toggle mines: TOGGLE_MINE (untoggled) and TOGGLE_MINE_TOGGLED (toggled)
            if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
                mines.append(entity)
        return mines

    def _extract_one_ways(self) -> List[Dict]:
        """Extract one-way platform entities (type 11)."""
        return [e for e in self.entities if e.get("type") == EntityType.ONE_WAY]

    #: Solid entity bodies the ninja cannot pass through, with their half-extent
    #: in px. These occupy space the tile grid does not know about, so without
    #: them the reachability graph routes STRAIGHT THROUGH a thwump or a bounce
    #: block and claims a route that does not exist -- the dangerous direction
    #: for a potential, because it shapes the agent toward a phantom path.
    #:
    #: Blocking is the conservative half of modelling these. The permissive half
    #: -- that their tops are standable surfaces, which opens routes -- is a
    #: separate change to the support model, deliberately not bundled here: this
    #: one can only REMOVE phantom edges, so it cannot invent reachability.
    _SOLID_BODY_SEMI = {
        EntityType.THWUMP: 9.0,
        EntityType.SHWUMP: 12.0,
        EntityType.BOUNCE_BLOCK: 9.0,
    }

    def _extract_solid_bodies(self) -> List[Dict]:
        """Extract entities with a solid, ninja-blocking body.

        Positions are taken as given, which for level_data built at load time are
        the entities' RESTING positions. That is deliberate and matches the
        potential's design: a thwump's phase and a bounce block's displacement
        are agent- or time-dependent, and a graph that tracked them would make
        Phi oscillate for reasons unrelated to progress -- the v39 hazard-PBRS
        pathology arriving through a different door. A bounce block additionally
        only ever moves because the agent pushed it, so anchoring at rest also
        removes any incentive to farm shaping by shoving one toward the goal.
        """
        return [
            e for e in self.entities if e.get("type") in self._SOLID_BODY_SEMI
        ]

    def _build_switch_door_mapping(self) -> Dict[str, Dict]:
        """
        Build switch ID -> door mapping.
        One switch always corresponds to one locked door (1:1 relationship).
        """
        mapping = {}

        for door in self.doors:
            if door.get("type") == EntityType.LOCKED_DOOR:
                switch_id = door.get("controlled_by") or door.get("switch_id")
                if switch_id:
                    mapping[switch_id] = door

        return mapping

    def get_blocked_positions(self) -> Set[Tuple[int, int]]:
        """
        Get set of tile positions currently blocked by entities.

        Returns:
            Set of (tile_x, tile_y) positions that are blocked
        """
        blocked = set()

        # Check doors
        for door in self.doors:
            if self._is_door_closed(door):
                door_tiles = self._get_door_tile_positions(door)
                blocked.update(door_tiles)

        # Check mines (only deadly state blocks)
        for mine in self.mines:
            if self._is_mine_deadly(mine):
                mine_tile = self._get_entity_tile_position(mine)
                if mine_tile:
                    blocked.add(mine_tile)

        return blocked

    def get_blocked_pixel_positions(
        self, all_nodes: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """
        Get set of pixel positions (sub-nodes) blocked by entities.

        Uses radius-based blocking for mines:
        - Finds all graph nodes within (mine_radius + ninja_radius) of deadly mines
        - Blocks entire tiles for closed doors

        Args:
            all_nodes: Set of all graph node positions in pixel coordinates

        Returns:
            Set of (pixel_x, pixel_y) positions that are blocked
        """
        from ...constants.physics_constants import NINJA_RADIUS

        blocked = set()

        # Block solid entity bodies (thwump / shove thwump / bounce block).
        # A node is blocked when the ninja's disc would overlap the body, i.e.
        # within `semi + NINJA_RADIUS` on BOTH axes -- the bodies are
        # axis-aligned squares, not circles, so this is a box test rather than
        # the radial one the mines use.
        for body in self.solid_bodies:
            pos = self._get_entity_pixel_position(body)
            if pos is None:
                continue
            semi = self._SOLID_BODY_SEMI[body.get("type")] + NINJA_RADIUS
            bx, by = float(pos[0]), float(pos[1])
            for node in all_nodes:
                if abs(node[0] - bx) < semi and abs(node[1] - by) < semi:
                    blocked.add(node)

        # Block doors (tile-based, all 4 sub-nodes per tile)
        blocked_tiles = set()
        for door in self.doors:
            if self._is_door_closed(door):
                door_tiles = self._get_door_tile_positions(door)
                blocked_tiles.update(door_tiles)

        # Convert blocked tiles to sub-node pixel positions
        sub_offsets = [(6, 6), (18, 6), (6, 18), (18, 18)]
        for tile_x, tile_y in blocked_tiles:
            for offset_x, offset_y in sub_offsets:
                pixel_x = tile_x * CELL_SIZE + offset_x
                pixel_y = tile_y * CELL_SIZE + offset_y
                blocked.add((pixel_x, pixel_y))

        # Block mines (radius-based). A mine only blocks nodes within
        # block_radius (= NINJA_RADIUS + mine_radius, ~14px < one CELL_SIZE), so
        # the naive "every mine vs every node" scan is O(mines x nodes) — on
        # mine-dense levels (e.g. 13k mines x 5k nodes ~= 70M Python distance
        # checks, ~12s per graph build) it dominates env reset. Spatially hash
        # the nodes into CELL_SIZE buckets once, then probe only the buckets a
        # mine could possibly reach. Output is identical to the naive scan; only
        # the candidate set per mine shrinks from all-nodes to a 3x3 cell window.
        from collections import defaultdict

        node_buckets: Dict[Tuple[int, int], list] = defaultdict(list)
        for node_pos in all_nodes:
            nx, ny = node_pos
            node_buckets[(nx // CELL_SIZE, ny // CELL_SIZE)].append(node_pos)

        for mine in self.mines:
            if self._is_mine_deadly(mine):
                mine_pos = self._get_entity_pixel_position(mine)
                if mine_pos is None:
                    continue

                mine_x, mine_y = mine_pos
                mine_radius = mine.get("radius", 4.0)

                # Collision occurs when: distance(ninja_center, mine_center) < ninja_radius + mine_radius
                block_radius = NINJA_RADIUS + mine_radius
                block_radius_sq = block_radius * block_radius

                # Probe only buckets the block_radius circle can overlap. The
                # cell span is ceil(block_radius / CELL_SIZE) in each direction.
                cell_span = int(block_radius // CELL_SIZE) + 1
                base_cx = int(mine_x // CELL_SIZE)
                base_cy = int(mine_y // CELL_SIZE)
                for cx in range(base_cx - cell_span, base_cx + cell_span + 1):
                    for cy in range(base_cy - cell_span, base_cy + cell_span + 1):
                        bucket = node_buckets.get((cx, cy))
                        if not bucket:
                            continue
                        for node_pos in bucket:
                            nx, ny = node_pos
                            dist_sq = (nx - mine_x) ** 2 + (ny - mine_y) ** 2
                            if dist_sq < block_radius_sq:
                                blocked.add(node_pos)

        return blocked

    def state_signature(self) -> Tuple:
        """Hashable signature of the dynamic entity state that determines node
        blocking: the set of deadly mines and closed doors (both by pixel
        position). Independent of ninja position, so it is identical across
        resets of a static level — used to memoize the O(mines) blocked-pixel
        computation in :class:`GraphBuilder`, which otherwise dominates env
        reset on mine-dense levels (13k+ mines ~= 9s/reset). A change in any
        mine/door state yields a different signature and forces a recompute, so
        the cache stays correct mid-episode (e.g. toggling mines)."""
        mines: list = []
        for mine in self.mines:
            if self._is_mine_deadly(mine):
                pos = self._get_entity_pixel_position(mine)
                if pos is not None:
                    mines.append((int(pos[0]), int(pos[1])))
        doors: list = []
        for door in self.doors:
            if self._is_door_closed(door):
                pos = self._get_entity_pixel_position(door)
                doors.append((int(pos[0]), int(pos[1])) if pos is not None else id(door))
        # Solid bodies are anchored at rest, so they never change within a
        # level -- but they MUST still appear here: the signature keys a shared
        # cache, and two levels that differ only in their thwump layout would
        # otherwise collide on it.
        bodies: list = []
        for body in self.solid_bodies:
            pos = self._get_entity_pixel_position(body)
            if pos is not None:
                bodies.append((int(pos[0]), int(pos[1]), int(body.get("type", 0))))
        return (tuple(sorted(mines)), tuple(sorted(doors)), tuple(sorted(bodies)))

    def get_blocked_edges(
        self, adjacency: Optional[Dict] = None
    ) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get the set of directed graph edges blocked by directional entities.

        One-way platforms (type 11) block a traversal that crosses the platform
        from its blocking (+normal) side: a directed edge ``(a, b)`` is blocked iff
        the move opposes the platform normal (``(b-a)·n < 0``) and the segment
        ``a→b`` crosses the platform's barrier span. The reverse edge ``(b, a)``
        moves *with* the normal and stays open — modelling pass-through from the
        permissive side. Mines/doors block via node masking (see
        ``get_blocked_pixel_positions``), not here.

        Args:
            adjacency: Base adjacency ``{node: [(neighbor, cost), ...]}`` in graph
                (inner) pixel space. Required to test one-ways; if ``None`` (or no
                one-ways present) an empty set is returned.

        Returns:
            Set of ``((x1, y1), (x2, y2))`` directed pixel-space edges to remove.
        """
        blocked: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        if not self.one_ways or adjacency is None:
            return blocked

        from ...physics import map_orientation_to_vector

        # Precompute each one-way's inner-space barrier: center + normal + the two
        # tangential endpoints of the blocking span.
        barriers = []
        half = _ONE_WAY_SEMI_SIDE + _ONE_WAY_LATERAL_PAD
        for ow in self.one_ways:
            center = self._get_entity_pixel_position(ow)
            if center is None:
                continue
            cx, cy = float(center[0]), float(center[1])
            nx, ny = map_orientation_to_vector(int(ow.get("orientation", 0)))
            tx, ty = -ny, nx  # tangent (perpendicular to normal)
            e1 = (cx - tx * half, cy - ty * half)
            e2 = (cx + tx * half, cy + ty * half)
            barriers.append((nx, ny, e1, e2))

        for a, neighbors in adjacency.items():
            ax, ay = float(a[0]), float(a[1])
            for neighbor_cost in neighbors:
                b = neighbor_cost[0]
                bx, by = float(b[0]), float(b[1])
                dx, dy = bx - ax, by - ay
                for nx, ny, e1, e2 in barriers:
                    # Only block moves that oppose the normal (into the solid side).
                    if dx * nx + dy * ny >= 0.0:
                        continue
                    if _segments_intersect((ax, ay), (bx, by), e1, e2):
                        blocked.add((a, b))
                        break

        return blocked

    def _is_door_closed(self, door: Dict) -> bool:
        """Check if locked door is currently closed (blocking)."""
        # Only handling EntityType.LOCKED_DOOR
        switch_id = door.get("controlled_by") or door.get("switch_id")
        if switch_id:
            # Door open if switch collected (one-way: never closes again)
            return not self.switch_states.get(switch_id, False)
        return door.get("closed", True)  # Default closed

    def _is_mine_deadly(self, mine: Dict) -> bool:
        """
        Check if toggle mine is in deadly (toggled) state.

        State mapping:
        - 0: Toggled (deadly) - blocks movement
        - 1: Untoggled (safe) - passable
        - 2: Toggling (transitioning) - passable but preparing

        TOGGLE_MINE_TOGGLED entities start toggled and NEVER become untoggled.
        """
        entity_type = mine.get("type")

        # TOGGLE_MINE_TOGGLED are permanently toggled (deadly)
        if entity_type == EntityType.TOGGLE_MINE_TOGGLED:
            return True

        # TOGGLE_MINE entities can change state
        state = mine.get("state", 1)  # Default untoggled (safe)

        # Handle both numeric and float states
        if isinstance(state, (int, float)):
            return int(state) == 0  # Only state 0 is deadly

        return False

    def _get_door_tile_positions(self, door: Dict) -> List[Tuple[int, int]]:
        """Get all tile positions occupied by door segments in inner tile space.

        Door positions from entities are in world space (with 1-tile padding).
        Convert to inner space by subtracting NODE_WORLD_COORD_OFFSET.
        """
        from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

        pos = door.get("position", (0, 0))
        orientation = door.get("orientation", 0)

        # Convert from world space to inner space
        inner_x = pos[0] - NODE_WORLD_COORD_OFFSET
        inner_y = pos[1] - NODE_WORLD_COORD_OFFSET

        # Door length is CELL_SIZE pixels (1 tile)
        # Doors can be horizontal (orientation 0) or vertical (orientation 1)
        tile_x = int(inner_x // CELL_SIZE)
        tile_y = int(inner_y // CELL_SIZE)

        positions = [(tile_x, tile_y)]

        # Doors may span adjacent tile depending on precise position
        if orientation == 0:  # Horizontal
            # Check if extends to adjacent tile
            if (
                inner_x % CELL_SIZE
            ) + CELL_SIZE > CELL_SIZE:  # Extends past tile boundary
                positions.append((tile_x + 1, tile_y))
        else:  # Vertical
            if (inner_y % CELL_SIZE) + CELL_SIZE > CELL_SIZE:
                positions.append((tile_x, tile_y + 1))

        return positions

    def _get_entity_tile_position(self, entity: Dict) -> Optional[Tuple[int, int]]:
        """Get tile position for point entity (mine, switch, etc.) in inner tile space.

        Entities from simulator are in world space (with 1-tile padding).
        Tiles array is in inner space (without padding).
        Convert by subtracting NODE_WORLD_COORD_OFFSET (24px) before dividing by CELL_SIZE.
        """
        from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

        # Handle both formats: dict with 'x'/'y' keys or 'position' tuple
        if "position" in entity:
            pos = entity["position"]
            tile_x = int((pos[0] - NODE_WORLD_COORD_OFFSET) // CELL_SIZE)
            tile_y = int((pos[1] - NODE_WORLD_COORD_OFFSET) // CELL_SIZE)
        elif "x" in entity and "y" in entity:
            tile_x = int((entity["x"] - NODE_WORLD_COORD_OFFSET) // CELL_SIZE)
            tile_y = int((entity["y"] - NODE_WORLD_COORD_OFFSET) // CELL_SIZE)
        else:
            return None

        return (tile_x, tile_y)

    def _get_entity_pixel_position(self, entity: Dict) -> Optional[Tuple[int, int]]:
        """Get pixel position for point entity in graph coordinate space.

        Entities from simulator are in world space (with 1-tile padding).
        Graph nodes are in inner space (without padding).
        Convert by subtracting NODE_WORLD_COORD_OFFSET (24px).
        """
        from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

        if "position" in entity:
            pos = entity["position"]
            return (
                int(pos[0]) - NODE_WORLD_COORD_OFFSET,
                int(pos[1]) - NODE_WORLD_COORD_OFFSET,
            )
        elif "x" in entity and "y" in entity:
            return (
                int(entity["x"]) - NODE_WORLD_COORD_OFFSET,
                int(entity["y"]) - NODE_WORLD_COORD_OFFSET,
            )
        return None

    def update_switch_state(self, switch_id: str, activated: bool):
        """
        Update switch state (affects corresponding door states).

        Args:
            switch_id: Switch identifier
            activated: True if switch collected, False otherwise
        """
        self.switch_states[switch_id] = activated

    def update_mine_state(self, mine_id: str, state: int):
        """
        Update toggle mine state.

        Args:
            mine_id: Mine identifier
            state: New state (0=toggled/deadly, 1=untoggled/safe, 2=toggling)
        """
        # Find mine and update its state
        for mine in self.mines:
            if mine.get("id") == mine_id or mine.get("entity_id") == mine_id:
                mine["state"] = state
                break

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about entities and current states."""
        total_doors = len(self.doors)
        closed_doors = sum(1 for door in self.doors if self._is_door_closed(door))

        total_mines = len(self.mines)
        deadly_mines = sum(1 for mine in self.mines if self._is_mine_deadly(mine))

        blocked_positions = self.get_blocked_positions()
        blocked_edges = self.get_blocked_edges()

        return {
            "total_doors": total_doors,
            "closed_doors": closed_doors,
            "total_mines": total_mines,
            "deadly_mines": deadly_mines,
            "blocked_tile_count": len(blocked_positions),
            "blocked_edge_count": len(blocked_edges),
            "switch_count": len(self.switch_states),
            "active_switches": sum(1 for v in self.switch_states.values() if v),
        }
