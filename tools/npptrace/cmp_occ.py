"""Compare TILE_EDGE_OCCUPANCY against the decompiled table at 0x10585F40.

§A of docs/decomp/TILE_TABLES_2.txt lists, per tile, eight half-edge occupancy
bits in the order (left | bottom | right | top). If the port's table disagrees,
that is a §10-shaped bug: a table that was reconstructed rather than read.
"""
import re
import sys

sys.path.insert(0, "/home/tetra/projects/nclone")
from nclone.tile_definitions import TILE_EDGE_OCCUPANCY  # noqa: E402

TXT = "/home/tetra/projects/nclone/docs/decomp/TILE_TABLES_2.txt"
rows = {}
for line in open(TXT):
    m = re.match(r"\s*(\d+)\s*\|((?:\s*\d){2})\s*\|((?:\s*\d){2})"
                 r"\s*\|((?:\s*\d){2})\s*\|((?:\s*\d){2})\s*$", line)
    if not m:
        continue
    tile = int(m.group(1))
    bits = []
    for g in m.groups()[1:]:
        bits.extend(int(c) for c in g.split())
    rows[tile] = tuple(bits)
    if tile >= 43:
        break

print("parsed %d tiles from the decompiled table" % len(rows))
same = diff = missing = 0
for tile in sorted(rows):
    game = rows[tile]
    ours = TILE_EDGE_OCCUPANCY.get(tile)
    if ours is None:
        missing += 1
        continue
    ours = tuple(ours)
    if ours == game:
        same += 1
    else:
        diff += 1
        print("  tile %-3d ours %s" % (tile, ours))
        print("           game %s" % (game,))
print("\nmatch %d, differ %d, missing %d" % (same, diff, missing))
