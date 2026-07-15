"""Deterministic TAS engine: seek, edit and replay an input movie.

The simulator is bit-deterministic and exposes a full state snapshot
(``nplay_headless.get_state_snapshot`` / ``set_state_snapshot``). The engine
leans on that to make a frame-by-frame editor cheap:

* ``actions``   - the movie: one discrete action (0-5) per simulator tick.
* ``cursor``    - how many ticks the *live* sim currently reflects, in [0, len].
* ``keyframes`` - sparse snapshots (every ``KEY_INTERVAL`` ticks, plus 0) so
  seeking anywhere is "restore nearest keyframe <= target, then replay the few
  remaining ticks" instead of replaying from spawn.
* ``frames``    - per-tick :class:`FrameState` (position/velocity/flags) used to
  draw the trajectory and the HUD without re-stepping the sim.

Edits to ``actions`` mark the engine dirty; :meth:`ensure_fresh` lazily rebuilds
the trajectory (and keyframes) from spawn while preserving the live cursor
state, so callers can edit freely and only pay for a rebuild once per frame.
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

from .layout import ACTION_TO_INPUTS
from .movie import TASMovie

KEY_INTERVAL = 30  # snapshot cadence (ticks) for fast seeking


@dataclass
class FrameState:
    """Ninja state captured after a given number of ticks."""

    frame: int
    x: float
    y: float
    vx: float
    vy: float
    state: int
    airborn: bool
    won: bool
    dead: bool


class TASEngine:
    """Frame-accurate editor over a deterministic episode."""

    def __init__(self, env, actions: Optional[List[int]] = None):
        self.env = env
        self.headless = env.nplay_headless
        self.actions: List[int] = list(actions or [])
        self.cursor: int = 0
        self.keyframes: Dict[int, object] = {}
        self.frames: List[FrameState] = []
        self._dirty = True
        # Snapshot spawn (frame 0) as the anchor keyframe.
        self._reset_to_spawn()
        self.rebuild()

    # ---- low-level sim helpers ------------------------------------------

    def _reset_to_spawn(self):
        """Return the sim to a clean spawn state (frame 0) without reloading map."""
        self.env.reset(options={"skip_map_load": True})
        self.keyframes = {0: self.headless.get_state_snapshot()}

    def _apply_tick(self, action: int):
        hor, jump = ACTION_TO_INPUTS.get(action, (0, 0))
        self.headless.tick(hor, jump)

    def _restore(self, blob):
        """Restore a keyframe snapshot, keeping the stored blob reusable.

        ``set_state_snapshot`` installs the snapshot's objects directly into the
        sim (no copy), so subsequent ticks would mutate the cached keyframe in
        place. Deep-copy on restore so each keyframe stays pristine for repeated
        seeks.
        """
        self.headless.set_state_snapshot(copy.deepcopy(blob))

    def _capture(self) -> FrameState:
        ninja = self.headless.sim.ninja
        return FrameState(
            frame=self.headless.sim.frame,
            x=ninja.xpos,
            y=ninja.ypos,
            vx=ninja.xspeed,
            vy=ninja.yspeed,
            state=ninja.state,
            airborn=bool(ninja.airborn),
            won=self.headless.ninja_has_won(),
            dead=self.headless.ninja_has_died(),
        )

    # ---- rebuild / seek --------------------------------------------------

    def rebuild(self):
        """Replay the whole movie from spawn, recomputing frames + keyframes.

        Leaves the live sim at the cursor afterwards (via :meth:`seek`).
        """
        spawn = self.keyframes[0]
        self._restore(spawn)
        self.keyframes = {0: spawn}
        self.frames = [self._capture()]
        terminated_at: Optional[int] = None
        for i, action in enumerate(self.actions):
            if terminated_at is None:
                self._apply_tick(action)
                fs = self._capture()
                self.frames.append(fs)
                if (i + 1) % KEY_INTERVAL == 0:
                    self.keyframes[i + 1] = self.headless.get_state_snapshot()
                if fs.won or fs.dead:
                    terminated_at = i + 1
            else:
                # Episode already ended; freeze the terminal state for the tail
                # so the timeline still shows the full edited movie length.
                self.frames.append(self.frames[-1])
        self._dirty = False
        self.cursor = min(self.cursor, len(self.actions))
        self.seek(self.cursor)

    def seek(self, target: int):
        """Move the live sim to ``target`` ticks (clamped to [0, len])."""
        target = max(0, min(int(target), len(self.actions)))
        key = max(k for k in self.keyframes if k <= target)
        self._restore(self.keyframes[key])
        for i in range(key, target):
            self._apply_tick(self.actions[i])
            if (i + 1) % KEY_INTERVAL == 0 and (i + 1) not in self.keyframes:
                self.keyframes[i + 1] = self.headless.get_state_snapshot()
        self.cursor = target

    def ensure_fresh(self):
        """Rebuild the trajectory lazily if the movie was edited."""
        if self._dirty:
            self.rebuild()

    # ---- editing ---------------------------------------------------------

    def perform(self, action: int, repeat: int = 1):
        """Commit ``action`` at the cursor for ``repeat`` frames (overwrite mode).

        At the tip this appends; in the middle it overwrites existing frames.
        Steps the live sim incrementally so single frame-advances stay cheap;
        marks dirty only when a downstream tail exists that must be recomputed.
        """
        repeat = max(1, int(repeat))
        tail_touched = False
        for _ in range(repeat):
            # A terminal (dead/won) ninja can't take inputs: don't extend the
            # movie with no-op frames past the end of the run. The user undoes
            # (scrubs back) to revive instead of mashing inputs into a corpse.
            if self.headless.ninja_has_died() or self.headless.ninja_has_won():
                break
            if self.cursor < len(self.actions):
                self.actions[self.cursor] = action
                tail_touched = True
            else:
                self.actions.append(action)
            self._apply_tick(action)
            self.cursor += 1
            fs = self._capture()
            if self.cursor < len(self.frames):
                self.frames[self.cursor] = fs
            else:
                self.frames.append(fs)
            if (self.cursor % KEY_INTERVAL) == 0:
                self.keyframes[self.cursor] = self.headless.get_state_snapshot()
            if fs.won or fs.dead:
                break
        if tail_touched and self.cursor < len(self.actions):
            # Frames after the cursor were computed from the old inputs upstream.
            self._dirty = True

    def predict(self, action: int = 0, n: int = 40):
        """Simulate ``n`` ticks of ``action`` from the cursor WITHOUT committing.

        Returns a list of ``(x, y, airborn)`` points for the resulting path, then
        restores the live sim to exactly the cursor state. Defaults to NOOP so
        the editor can show the natural falling/coasting arc ahead of the cursor
        (handy for lining up jumps and landings). Stops early on death/win.
        """
        if n <= 0:
            return []
        if self.headless.ninja_has_died() or self.headless.ninja_has_won():
            return []
        snap = self.headless.get_state_snapshot()
        points = []
        for _ in range(n):
            self._apply_tick(action)
            ninja = self.headless.sim.ninja
            points.append((ninja.xpos, ninja.ypos, bool(ninja.airborn)))
            if self.headless.ninja_has_died() or self.headless.ninja_has_won():
                break
        # Restore the live state (snap is a one-shot fresh copy, alias is fine).
        self.headless.set_state_snapshot(snap)
        return points

    def back(self, n: int = 1):
        """Scrub the cursor back ``n`` frames (history preserved)."""
        self.seek(self.cursor - n)

    def forward(self, n: int = 1):
        """Scrub the cursor forward ``n`` frames, replaying stored inputs."""
        self.seek(self.cursor + n)

    def set_action(self, frame: int, action: int):
        if 0 <= frame < len(self.actions):
            self.actions[frame] = action
            self._dirty = True

    def insert(self, frame: int, action: int):
        frame = max(0, min(frame, len(self.actions)))
        self.actions.insert(frame, action)
        self._dirty = True

    def delete(self, frame: int):
        if 0 <= frame < len(self.actions):
            del self.actions[frame]
            if self.cursor > len(self.actions):
                self.cursor = len(self.actions)
            self._dirty = True

    def truncate(self, frame: int):
        """Drop every frame at or after ``frame`` (cuts the tail)."""
        frame = max(0, min(frame, len(self.actions)))
        if frame < len(self.actions):
            del self.actions[frame:]
            self.cursor = min(self.cursor, len(self.actions))
            self._dirty = True

    def clear(self):
        """Reset to an empty movie at spawn."""
        self.actions = []
        self.cursor = 0
        self._dirty = True
        self.rebuild()

    # ---- views -----------------------------------------------------------

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def at_tip(self) -> bool:
        return self.cursor >= len(self.actions)

    def action_at(self, frame: int) -> Optional[int]:
        if 0 <= frame < len(self.actions):
            return self.actions[frame]
        return None

    def current_frame_state(self) -> FrameState:
        idx = min(self.cursor, len(self.frames) - 1)
        return self.frames[idx]

    @property
    def won(self) -> bool:
        return self.current_frame_state().won

    @property
    def dead(self) -> bool:
        return self.current_frame_state().dead

    # ---- movie I/O -------------------------------------------------------

    def to_movie(self, level_id: Optional[str] = None) -> TASMovie:
        map_data = bytes(
            max(0, min(255, int(b))) for b in self.headless.current_map_data
        )
        return TASMovie(map_data, self.actions, level_id=level_id)

    def load_actions(self, actions: List[int]):
        """Replace the movie with ``actions`` and rebuild from spawn."""
        self.actions = list(actions)
        self.cursor = 0
        self._dirty = True
        self.rebuild()
