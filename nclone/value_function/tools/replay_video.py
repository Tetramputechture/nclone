#!/usr/bin/env python3
"""Render a verified planner replay to mp4, with the door-aware potential on screen.

Why replay a tape rather than re-plan
-------------------------------------
``route_video`` plans a fresh route and films it.  That is the right tool for
"can the planner solve this", and the wrong one here: the ``.replay`` corpus is
already **validated winning play**, so replaying a tape films the route we
actually reason about elsewhere, and cannot fail or time out mid-render.

What the overlay is for
-----------------------
On a locked-door level the interesting claim is not that the ninja wins — it is
that the phase-aware potential treats each door as progress.  So the HUD carries
the quantities that claim is made of:

* the current **leg** (which necessary door the agent is being sent to),
* how many locked doors are actually open in the live simulation, and
* **Phi**, the door-aware cost-to-completion at the ninja's position.

Phi should fall monotonically across every door, which is exactly the invariant
``test_door_aware_field`` pins numerically — this renders it so it can be seen.

Door identity across backends
-----------------------------
The plan indexes doors in the NATIVE simulator's order; the render runs on the
Python simulator.  The two are matched by **door segment midpoint**, which is
unique per door and identical in both.  Matching on the switch position would be
wrong: several doors routinely share one switch (``014 cheap thrills`` has four
doors on a single switch), so it is not an identity.

Usage::

    python -m nclone.value_function.tools.replay_video \\
        --replay datasets/phase2_planner_demos/456_one_too_many_dramatic_flourishes.replay \\
        --output videos/456_locked_door_sequence.mp4
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless (WSL/CI)

import numpy as np  # noqa: E402

_HUD_H = 52
_PAD = 10


def load_replay(path: str) -> Optional[Tuple[List[int], List[Tuple[int, int]]]]:
    """``(map_data, [(horizontal, jump), ...])`` from a compact V1 ``.replay``."""
    data = open(path, "rb").read()
    _ver, mlen, ilen = struct.unpack("<III", data[0:12])
    map_data = list(data[13:13 + mlen])
    raw = data[13 + mlen:13 + mlen + ilen]
    if not map_data or not raw:
        return None
    return map_data, [(1 if b & 2 else (-1 if b & 4 else 0), b & 1) for b in raw]


def _to_hwc(frame) -> np.ndarray:
    """(H, W, 3) uint8 for imageio, whatever the renderer handed back."""
    a = np.asarray(frame)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _door_index_map(nh, native_doors) -> Dict[int, int]:
    """``python_entity_position -> plan door index``, matched by segment midpoint."""
    out: Dict[int, int] = {}
    py = nh.sim.entity_dic.get(6, [])
    for pi, d in enumerate(py):
        seg = getattr(d, "segment", None)
        if seg is None:
            continue
        mx = (seg.x1 + seg.x2) / 2.0
        my = (seg.y1 + seg.y2) / 2.0
        for ni, (dx, dy, _o, _sx, _sy) in enumerate(native_doors):
            if abs(dx - mx) < 1e-6 and abs(dy - my) < 1e-6:
                out[pi] = ni
                break
    return out


def _open_indices(nh, idx_map: Dict[int, int]) -> List[int]:
    """Plan indices of the locked doors that are open right now."""
    py = nh.sim.entity_dic.get(6, [])
    return [idx_map[i] for i, d in enumerate(py)
            if i in idx_map and not getattr(d, "closed", True)]


#: Candidate HUD fonts, largest-effort first.  PIL's built-in bitmap font is
#: ~11px on a 1056px-wide frame, which is legible only when zoomed; a real
#: TrueType face makes the overlay readable at native size.  Every entry is
#: optional and the built-in is always a working fallback, so this never turns a
#: missing font into a failed render.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_FONT_SIZE = 17
_font_cache: list = []


def _hud_font():
    if _font_cache:
        return _font_cache[0]
    from PIL import ImageFont
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, _FONT_SIZE)
                break
            except Exception:
                font = None
    if font is None:
        font = ImageFont.load_default()
    _font_cache.append(font)
    return font


def _hud(frame: np.ndarray, lines: List[str], bar: Optional[float]) -> np.ndarray:
    """Stack a text strip under the frame; ``bar`` in [0,1] draws a Phi meter."""
    from PIL import Image, ImageDraw

    h, w = frame.shape[:2]
    canvas = Image.new("RGB", (w, h + _HUD_H), (14, 14, 18))
    canvas.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((_PAD, h + 6), "    ".join(lines), fill=(235, 235, 240),
           font=_hud_font())
    if bar is not None:
        x0, x1 = _PAD, w - _PAD
        y = h + 34
        d.rectangle([x0, y, x1, y + 8], outline=(70, 70, 80))
        # Phi is a cost, so a FULL bar is far from the goal and it drains as the
        # route progresses — the direction a viewer expects "progress" to move.
        fill = x0 + int((x1 - x0) * max(0.0, min(1.0, bar)))
        if fill > x0:
            d.rectangle([x0, y, fill, y + 8], fill=(90, 160, 230))
    return np.asarray(canvas)


def render(replay_path: str, output: str, fps: int = 60,
           overlay: bool = True, hold_s: float = 1.0) -> dict:
    import imageio
    from ...nplay_headless import NPlayHeadless
    from ..door_aware_field import DoorAwareFreeSpaceField

    loaded = load_replay(replay_path)
    if loaded is None:
        raise SystemExit(f"unreadable replay: {replay_path}")
    map_data, inputs = loaded

    field = DoorAwareFreeSpaceField(map_data) if overlay else None
    native_doors = []
    if field is not None and field._base is not None:
        native_doors = list(field._base.locked_doors)
    if field is not None:
        print(f"[video] door_aware={field.door_aware} "
              f"necessary={field.necessary_order} "
              f"locked_doors={len(native_doors)} reason={field.fallback_reason}")

    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    nh.load_map_from_map_data(list(map_data))
    nh.reset()
    idx_map = _door_index_map(nh, native_doors) if native_doors else {}

    n_need = len(field.necessary_order) if field else 0
    phi0 = None

    def cap(i: int, total: int) -> np.ndarray:
        raw = _to_hwc(nh.render())
        if not overlay or field is None:
            return raw
        nonlocal phi0
        x, y = nh.ninja_position()
        opened = _open_indices(nh, idx_map)
        sw = bool(nh.exit_switch_activated())
        phi = field.cost_to_completion(float(x), float(y), sw, opened)
        if phi0 is None and np.isfinite(phi):
            phi0 = phi
        ph = field.door_phase(opened)
        if ph is not None:
            goal = f"leg {ph + 1}/{n_need} -> door switch"
        else:
            goal = "-> EXIT" if sw else "-> exit switch"
        lines = [f"frame {i}/{total}", goal,
                 f"doors open {len(opened)}/{len(native_doors)}",
                 f"Phi {phi:8.1f}" if np.isfinite(phi) else "Phi      inf"]
        bar = (phi / phi0) if (phi0 and np.isfinite(phi)) else None
        return _hud(raw, lines, bar)

    total = len(inputs)
    frames = [cap(0, total)]
    won = died = False
    for i, (h, j) in enumerate(inputs, 1):
        nh.tick(h, j)
        frames.append(cap(i, total))
        if nh.ninja_has_won():
            won = True
            break
        if nh.ninja_has_died():
            died = True
            break

    if not won:
        # Filming a loss as though it were a win is the one failure that would
        # make this output actively misleading, so it is loud.
        print(f"[video] WARNING: tape did not win (won={won} died={died}) — "
              f"the replay is stale or the simulator has moved under it")

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    frames.extend([frames[-1]] * int(fps * hold_s))
    imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
    size = os.path.getsize(output)
    print(f"[video] won={won} died={died} frames={len(frames)} @ {fps}fps "
          f"-> {output} ({size/1e6:.2f} MB)")
    return {"won": won, "died": died, "n_frames": len(frames), "output": output}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", required=True, help="a verified .replay tape")
    p.add_argument("--output", required=True, help="output .mp4 path")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--hold-s", type=float, default=1.0,
                   help="seconds to hold the final frame")
    a = p.parse_args(argv)
    r = render(a.replay, a.output, fps=a.fps, overlay=not a.no_overlay,
               hold_s=a.hold_s)
    return 0 if r["won"] else 1


if __name__ == "__main__":
    sys.exit(main())
