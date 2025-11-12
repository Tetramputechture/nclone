from ..entities import Entity
from ..physics import overlap_circle_vs_circle
from ..ninja import NINJA_RADIUS
from ..constants.physics_constants import TOGGLE_MINE_RADII


class EntityToggleMine(Entity):
    """Toggle Mine Entity (Types 1, 21)

    A dynamic hazard that changes state based on ninja interaction. Toggle mines can be
    used to create temporary safe passages, block routes, or force specific movement
    patterns. Their three-state system and state-dependent collision radii create
    complex tactical decisions.

    Physical Properties:
        - Variable Radius:
            * Untoggled (Safe): 3.5 pixels
            * Toggling: 4.5 pixels
            * Toggled (Deadly): 4.0 pixels
        - Max Per Level: 8192 instances
        - Entity Types:
            * Type 1: Initial untoggled state
            * Type 21: Initial toggled state

    Behavior:
        - State System:
            * 0: Toggled (deadly)
            * 1: Untoggled (safe)
            * 2: Toggling (transitioning)
        - State Transitions:
            * Untoggled → Toggling: Ninja contact
            * Toggling → Toggled: Ninja breaks contact
            * Toggled → Untoggled: Ninja contact (dies)
            * Toggling → Untoggled: Ninja dies
        - Collision Response:
            * Untoggled: Safe to touch
            * Toggling: Safe but preparing
            * Toggled: Lethal on contact

    AI Strategy Notes:
        - Tactical Elements:
            * Can create temporary paths
            * Force specific routes
            * Block retreat options
            * Create timing challenges
        - Consider:
            * State of surrounding mines
            * Toggle sequence planning
            * Safe approach angles
            * Alternative routes
        - Used for:
            * Path manipulation
            * Area denial
            * Forced commitments
            * Timing puzzles

    Technical Implementation:
        - State Management:
            * Tracks current state
            * Updates collision radius
            * Handles state transitions
            * Supports logging
        - Collision Detection:
            * Uses circular collision
            * State-dependent radius
            * Ninja state awareness
            * Death handling
        - Interaction Logic:
            * Contact monitoring
            * State validation
            * Death state checks
            * Transition timing
    """

    RADII = TOGGLE_MINE_RADII  # 0:toggled, 1:untoggled, 2:toggling
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
                if overlap_circle_vs_circle(
                    self.xpos,
                    self.ypos,
                    self.RADIUS,
                    ninja.xpos,
                    ninja.ypos,
                    NINJA_RADIUS,
                ):
                    self.set_state(2)
            elif (
                self.state == 2
            ):  # Set state to toggled if ninja stops touching toggling mine
                if not overlap_circle_vs_circle(
                    self.xpos,
                    self.ypos,
                    self.RADIUS,
                    ninja.xpos,
                    ninja.ypos,
                    NINJA_RADIUS,
                ):
                    self.set_state(0)
        else:  # Set state to untoggled if ninja dies while toggling a mine
            if self.state == 2 and ninja.state == 6:
                self.set_state(1)

    def logical_collision(self):
        """Kill the ninja if it touches a toggled mine"""
        ninja = self.sim.ninja
        if ninja.is_valid_target() and self.state == 0:
            if overlap_circle_vs_circle(
                self.xpos, self.ypos, self.RADIUS, ninja.xpos, ninja.ypos, NINJA_RADIUS
            ):
                self.set_state(1)
                ninja.kill(0, 0, 0, 0, 0, cause="mine")

    def set_state(self, state):
        """Set the state of the toggle. 0:toggled, 1:untoggled, 2:toggling."""
        if state in (0, 1, 2):
            old_state = getattr(self, "state", None)
            self.state = state
            self.RADIUS = self.RADII[state]
            self.log_collision(state)

            # Invalidate mine-dependent caches if state actually changed
            # PERFORMANCE: Updates mine death predictor and invalidates debug overlay caches
            if old_state is not None and old_state != state:
                # Update mine death predictor and clear debug overlay caches
                from nclone.cache_management import invalidate_mine_dependent_caches

                # Access renderer through gym_env -> nplay_headless -> sim_renderer
                if hasattr(self.sim, "gym_env") and self.sim.gym_env:
                    env = self.sim.gym_env
                    if hasattr(env, "nplay_headless") and hasattr(
                        env.nplay_headless, "sim_renderer"
                    ):
                        invalidate_mine_dependent_caches(
                            env.nplay_headless.sim_renderer, self.sim
                        )

    def get_state(self):
        state = super().get_state()
        # Normalize state (0, 1, or 2)
        # state.append(max(0.0, min(1.0, float(self.state) / 2)))
        return state
