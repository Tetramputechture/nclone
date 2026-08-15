"""Find plausible xpos/ypos double pairs inside captured entity slabs."""
import glob
import os
import re
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "discover")

# Level interior in pixels: 42x23 cells of 24 -> x in (0, 1056), y in (0, 600).
XMAX, YMAX = 1100.0, 620.0


def plausible(v):
    return 0.0 < v < XMAX and abs(v) > 1e-9


for fn in sorted(glob.glob(os.path.join(OUT, "pool*_stride*_n*.bin"))):
    m = re.search(r"pool(\d+)_stride(\d+)_n(\d+)", os.path.basename(fn))
    pool, stride, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
    data = open(fn, "rb").read()
    print("\n=== pool %d  stride %d  n %d  (%d bytes) ===" % (pool, stride, n, len(data)))

    # candidate (x, y) offsets: consecutive doubles both in range for EVERY entity
    cands = []
    for off in range(0, stride - 15, 4):
        ok = True
        vals = []
        for e in range(n):
            b = data[e * stride:(e + 1) * stride]
            try:
                x = struct.unpack_from("<d", b, off)[0]
                y = struct.unpack_from("<d", b, off + 8)[0]
            except struct.error:
                ok = False
                break
            if not (plausible(x) and 0.0 < y < YMAX):
                ok = False
                break
            vals.append((x, y))
        if ok:
            cands.append((off, vals))

    for off, vals in cands:
        print("  offset 0x%02X / 0x%02X looks like (x, y):" % (off, off + 8))
        for e, (x, y) in enumerate(vals):
            print("      [%d] (%9.3f, %9.3f)   cell (%5.2f, %5.2f)"
                  % (e, x, y, x / 24.0, y / 24.0))
    if not cands:
        print("  no consistent (x, y) pair; first entity dwords:")
        b = data[:stride]
        for off in range(0, min(stride, 64), 4):
            print("      +0x%02X %08x" % (off, struct.unpack_from("<I", b, off)[0]))
