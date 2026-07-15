"""Build the ``debug_info['tas']`` payload consumed by the renderer.

Keeps all the engine -> render-data shaping in one place. The payload carries
trajectory points in *sim* coordinates (the renderer applies the sim->screen
transform it already maintains) plus the HUD/piano-roll state.
"""

from typing import Optional

from .engine import TASEngine
from .layout import ACTION_NAME


def build_tas_payload(engine: TASEngine, view: "TASViewState", prediction=None) -> dict:
    """Assemble the render payload from the engine + transient editor view."""
    frames = engine.frames
    # Trajectory points: (sim_x, sim_y, airborn) so the renderer can colour
    # grounded vs airborne segments.
    traj = [(f.x, f.y, f.airborn) for f in frames]

    cur = engine.current_frame_state()
    status = "WON" if cur.won else "DEAD" if cur.dead else "EDIT"

    return {
        "prediction": list(prediction) if prediction else [],
        "predict_on": view.show_prediction,
        "trajectory": traj,
        "cursor": engine.cursor,
        "length": engine.length,
        "cursor_point": (cur.x, cur.y),
        "cursor_state": cur.state,
        "cursor_vel": (cur.vx, cur.vy),
        "status": status,
        "playing": view.playing,
        "pending_action": view.pending_action(),
        "pending_name": ACTION_NAME.get(view.pending_action(), "?"),
        "batch": view.batch_count or 1,
        "batch_typed": view.batch_count,
        "actions": engine.actions,
        "message": view.message,
    }


class TASViewState:
    """Transient editor UI state (not part of the deterministic movie)."""

    def __init__(self):
        self.pending_hor = 0       # -1 / 0 / +1
        self.pending_jump = False
        self.playing = False
        self.batch_count = 0       # 0 == unset (treated as 1 on commit)
        self.message = ""
        self.show_prediction = True   # NOOP fall-arc ghost ahead of the cursor
        self.predict_frames = 45      # how far to project the ghost

    def pending_action(self) -> int:
        from .layout import inputs_to_action

        return inputs_to_action(self.pending_hor, self.pending_jump)

    def toggle_dir(self, direction: int):
        """Toggle pending horizontal; pressing the active direction clears it."""
        self.pending_hor = 0 if self.pending_hor == direction else direction

    def toggle_jump(self):
        self.pending_jump = not self.pending_jump

    def clear_pending(self):
        self.pending_hor = 0
        self.pending_jump = False

    def push_digit(self, d: int):
        self.batch_count = min(99999, self.batch_count * 10 + d)

    def reset_batch(self):
        self.batch_count = 0

    def commit_batch(self) -> int:
        n = self.batch_count or 1
        self.batch_count = 0
        return n

    def notify(self, msg: str):
        self.message = msg
