"""Repackage a split (map file, zlib'd input tape) pair as an N++ attract file.

    python tools/npptrace/pack_attract.py MET-S-E-11-04 S-E-11-04_0th_replay \
        --out /home/tetra/projects/attracts/9000001

The whole trace toolchain -- `sim_trace.py`, `ninjadiv.py`, `entscore.py`,
`install_attracts.py` -- takes an attract file, which is map and tape in one
container. A split pair cannot be fed to any of it. Container layout, from
`nclone/replay/npp_attract_decoder.py`:

    +0        u32  map_data_length      (= len(map blob) - 8)
    +4        u32  demo_data_length
    +8             map blob from its 9th byte on
    +8+mdl         demo section:
                     +0   u8   0
                     +1   u32  section length   (= 30 + frame count)
                     +5   u32  1
                     +9   u32  frame count
                     +13  u32  level id
                     +17  13B  fixed tail, copied from a real attract
                     +30        one input byte per frame

The map blob's own first 8 bytes are overwritten by the two length fields. That
is what the real files do -- nclone reads tiles at absolute 184 and entities at
absolute 1230 of the *whole file*, so the container header lives inside the map
blob's unused 184-byte header.
"""
import argparse
import os
import struct
import sys
import zlib

sys.path.insert(0, "/home/tetra/projects/nclone")

#: `+17..30` of the demo section, verbatim from `attracts/1000622`. Thirteen
#: bytes this decoder never reads; copied rather than zeroed so the file is as
#: close to a real one as possible for the game's own parser.
TAIL = bytes.fromhex("000000000000000001ffffffff")


def pack(map_path, tape_path, level_id=0):
    blob = open(map_path, "rb").read()
    raw = open(tape_path, "rb").read()
    try:
        inputs = zlib.decompress(raw)
    except zlib.error:
        inputs = raw                      # already plain
    bad = sorted({b for b in inputs if b > 7})
    if bad:
        raise SystemExit("tape has bytes outside 0..7: %s" % bad)

    demo = (bytes([0])
            + struct.pack("<I", 30 + len(inputs))
            + struct.pack("<I", 1)
            + struct.pack("<I", len(inputs))
            + struct.pack("<I", level_id)
            + TAIL
            + inputs)
    assert len(demo) == 30 + len(inputs)
    out = struct.pack("<II", len(blob) - 8, len(demo)) + blob[8:] + demo
    return out, blob, inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("tape")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level-id", type=int, default=0)
    a = ap.parse_args()

    out, blob, inputs = pack(a.map, a.tape, a.level_id)
    with open(a.out, "wb") as fh:
        fh.write(out)
    print("wrote %s: %d bytes (map blob %d, %d input frames)"
          % (a.out, len(out), len(blob), len(inputs)))

    # Round-trip: the decoder must recover the same tiles, entities and tape.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    dec = NppAttractDecoder()
    got_tape = dec.decode_npp_attract_file(a.out).get("demo_data") or []
    nclone_map = bytes(dec.create_nclone_map(a.out))

    print("round-trip:")
    print("   tape   %d frames, identical: %s"
          % (len(got_tape), bytes(got_tape) == inputs))
    print("   tiles  identical: %s" % (nclone_map[184:1150] == blob[184:1150]))
    print("   ents   identical: %s" % (nclone_map[1230:] == blob[1230:]))
    print("   spawn  map (%d,%d)  packed (%d,%d)"
          % (blob[0x4C7] * 6, blob[0x4C8] * 6,
             nclone_map[0x4C7] * 6, nclone_map[0x4C8] * 6))


if __name__ == "__main__":
    main()
