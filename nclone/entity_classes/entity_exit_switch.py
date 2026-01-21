from ..entities import Entity
from ..physics import overlap_circle_vs_circle
from ..ninja import NINJA_RADIUS


class EntityExitSwitch(Entity):
    """Exit Switch Entity (Type 4)

    A critical progression element that activates its associated exit door when collected.
    Exit switches are the first objective in most levels, creating a two-stage completion
    requirement that influences route planning and risk assessment.

    Physical Properties:
        - Radius: 6 pixels
        - Collision Type: Circle
        - Parent Link: Associated with specific exit door
        - Collection: Single use, disappears after activation

    Behavior:
        - Initial State:
            * Active and triggerable
            * Visible in entity grid
            * Parent exit door inactive
        - On Collection:
            * Becomes inactive and disappears
            * Activates parent exit door
            * Adds exit door to entity grid
            * Logs collection event
        - Parent Activation:
            * Marks switch as hit on parent
            * Makes exit door interactable
            * Creates permanent state change
            * No way to deactivate

    AI Strategy Notes:
        - Primary initial objective
        - Consider:
            * Safest path to switch
            * Post-collection route to exit
            * Hazard positioning relative to switch
            * Optional collectibles en route
        - Critical for:
            * Level progression
            * Route planning
            * Risk evaluation
            * Speedrun optimization

    Technical Implementation:
        - Maintains parent door reference
        - Handles entity grid management
        - Updates parent door state
        - Manages collection logging
    """

    RADIUS = 6

    def __init__(self, type, sim, xcoord, ycoord, parent):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.parent = parent

        # Store initial values for fast reset
        self._initial_xpos = self.xpos
        self._initial_ypos = self.ypos
        self._initial_cell = self.cell

    def logical_collision(self):
        """If the ninja is colliding with the switch, open its associated door. This is done in practice
        by adding the parent door entity to the entity grid.
        """
        ninja = self.sim.ninja

        # Calculate distance for diagnostics
        # dx = ninja.xpos - self.xpos
        # dy = ninja.ypos - self.ypos
        # distance = (dx * dx + dy * dy) ** 0.5
        # collision_threshold = self.RADIUS + NINJA_RADIUS

        if overlap_circle_vs_circle(
            self.xpos, self.ypos, self.RADIUS, ninja.xpos, ninja.ypos, NINJA_RADIUS
        ):
            # DIAGNOSTIC: Log switch activation with full details
            frame = getattr(self.sim, "frame", -1)
            # print(
            #     f"\n{'*' * 60}\n"
            #     f"[SWITCH_COLLISION] Frame {frame}\n"
            #     f"  Switch pos: ({self.xpos:.1f}, {self.ypos:.1f})\n"
            #     f"  Ninja pos: ({ninja.xpos:.1f}, {ninja.ypos:.1f})\n"
            #     f"  Distance: {distance:.1f}px (threshold: {collision_threshold:.1f}px)\n"
            #     f"  Switch was active: {self.active}\n"
            #     f"{'*' * 60}\n"
            # )

            # NOTE: At stage 0, switch IS supposed to be at spawn (distance=0)
            # So frame 1 activation at stage 0 is CORRECT behavior
            # Only warn if we expect the switch to be far from spawn
            if frame <= 1:
                # Get curriculum stage to check if this is expected
                curriculum_stage = None
                if hasattr(self.sim, "gym_env") and self.sim.gym_env:
                    env = self.sim.gym_env
                    if (
                        hasattr(env, "goal_curriculum_manager")
                        and env.goal_curriculum_manager
                    ):
                        curriculum_stage = (
                            env.goal_curriculum_manager.state.unified_stage
                        )

                if curriculum_stage is not None and curriculum_stage > 0:
                    print(
                        f"[CURRICULUM_BUG] Switch collected on FRAME {frame} at STAGE {curriculum_stage}!\n"
                        f"  At stage > 0, switch should be away from spawn.\n"
                        f"  Check that _pending_curriculum_stage was applied before reset.\n"
                        f"  Or check _episode_start_curriculum_stage tracking."
                    )

            self.active = False
            # Add door to the entity grid so the ninja can touch it
            self.sim.grid_entity[self.parent.cell].append(self.parent)
            self.parent.switch_hit = (
                True  # Mark the switch as hit on the parent Exit door
            )
            self.log_collision()

            # Invalidate gym environment caches (switch state changed)
            # PERFORMANCE: Enables efficient cache invalidation
            if hasattr(self.sim, "gym_env") and self.sim.gym_env:
                self.sim.gym_env.invalidate_switch_cache()

    def reset_state(self):
        """Reset exit switch to initial state for fast environment reset.

        IMPORTANT: This method resets the switch to ORIGINAL position.
        The environment's curriculum manager will call apply_to_simulator()
        after fast_reset() to move this switch to the curriculum position.
        """
        self.xpos = self._initial_xpos
        self.ypos = self._initial_ypos
        self.cell = self._initial_cell
        self.active = True  # Switch must be active (not collected) at episode start
