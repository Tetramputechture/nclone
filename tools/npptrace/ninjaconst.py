"""Read the ninja's constructor-computed constants straight out of the captures.

`Ninja::Ninja` (`docs/decomp/ninja_105137B0.c`) makes FOUR `_libm_sse2_pow_precise`
calls and stores the results at `+0xA8`, `+0xB0`, `+0xB8`, `+0xC0`. Ghidra drops
the arguments and the return of every one, so no decompile can show these values
-- exactly the situation the rocket's `+0x30 = pow(1.1, 2/3)` was in.

They are inside the 0xD40 ninja slab, so the captures have them. Dump every
double in the drag/friction neighbourhood and match against our constants.
"""
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.constants.physics_constants import (  # noqa: E402
    DRAG_REGULAR, DRAG_SLOW, FRICTION_GROUND, FRICTION_GROUND_SLOW,
    FRICTION_WALL, GRAVITY_FALL, GRAVITY_JUMP, MAX_HOR_SPEED,
)

OURS = {
    "DRAG_REGULAR": DRAG_REGULAR,
    "DRAG_SLOW": DRAG_SLOW,
    "FRICTION_GROUND": FRICTION_GROUND,
    "FRICTION_GROUND_SLOW": FRICTION_GROUND_SLOW,
    "FRICTION_WALL": FRICTION_WALL,
    "GRAVITY_FALL": GRAVITY_FALL,
    "GRAVITY_JUMP": GRAVITY_JUMP,
    "MAX_HOR_SPEED": MAX_HOR_SPEED,
}
bits = lambda x: struct.unpack("<q", struct.pack("<d", float(x)))[0]

stems = sorted(set(
    s.replace(".layout.json", "")
    for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
    + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")))

seen = {}
for stem in stems:
    try:
        cap = Capture(stem)
    except Exception:
        continue
    slab = cap.layout.get("ninjaSlab", 0x200)
    if slab < 0xD0:
        continue
    rec = cap.frame(1)
    for off in range(0x90, 0xD0, 8):
        try:
            v = struct.unpack_from("<d", rec, 4 + off)[0]
        except struct.error:
            continue
        seen.setdefault(off, set()).add(v)

print("ninja slab doubles at +0x90..+0xC8, frame 1, over %d captures\n" % len(stems))
for off in sorted(seen):
    vals = seen[off]
    if len(vals) != 1:
        print("  +0x%02X  VARIES (%d distinct) -- not a constant" % (off, len(vals)))
        continue
    v = next(iter(vals))
    match = ""
    for name, ours in OURS.items():
        if v == ours:
            match = "  == %s (exact)" % name
            break
        if ours and abs(v - ours) < 1e-12 * max(1.0, abs(ours)):
            match = "  ~= %s  (ours %.20g, %+d ulp)" % (name, ours, bits(ours) - bits(v))
            break
    print("  +0x%02X  %.20g%s" % (off, v, match))
