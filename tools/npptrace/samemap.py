"""Find captured demos that run on the SAME MAP as a given demo.

A capture on the same map gives the ninja ground truth for a map family that
has none of its own.

Usage: samemap.py <demo> [<demo> ...]
"""
import glob
import hashlib
import os
import sys

sys.path.insert(0, "/home/tetra/projects/nclone")
from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402

dec = NppAttractDecoder()


def sig(demo):
    md = list(dec.create_nclone_map("/home/tetra/projects/attracts/%s" % demo))
    tiles = bytes(md[:1230])
    ents = bytes(md[1230:])
    return (hashlib.sha1(tiles).hexdigest()[:12],
            hashlib.sha1(ents).hexdigest()[:12])


caps = sorted({os.path.basename(s).replace(".layout.json", "")
               for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
               + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")})
caps = [c for c in caps if c.isdigit()
        and os.path.exists("/home/tetra/projects/attracts/%s" % c)]
capsig = {}
for c in caps:
    try:
        capsig[c] = sig(c)
    except Exception:
        pass
print("%d captured demos with a decodable map\n" % len(capsig))

for demo in sys.argv[1:]:
    t, e = sig(demo)
    same_tiles = [c for c, (ct, ce) in capsig.items() if ct == t]
    same_all = [c for c in same_tiles if capsig[c][1] == e]
    print("== %s tiles=%s ents=%s" % (demo, t, e))
    print("   captures on the same TILES:    %s" % (same_tiles or "none"))
    print("   captures on the same FULL map: %s" % (same_all or "none"))
