"""Capture per-frame ninja ground truth from a running N++.

Segments by Simulator::reset, identifies each demo by fingerprinting the loaded
tile array against the attract source files, and writes one JSONL trace per demo.
"""
import json
import os
import struct
import sys
import time

import frida

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(HERE, "agent_trace.js")
OUT = os.path.join(HERE, "out")
ATTRACT_SRC = r"D:\attracts"
INSTALLED = r"C:\Users\Nick\Documents\Metanet\N++\attract"

SLAB = 0x200
REC = 4 + SLAB
TILE_N = 1100
GRID_W, GRID_H = 44, 25

# ninja field offsets within the slab
F_XPOS, F_YPOS = 0x20, 0x28
F_XSPEED, F_YSPEED = 0x30, 0x38
F_STATE = 0xD0
F_FLOORX, F_FLOORY = 0x120, 0x128

RUN_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def expected_tiles(path):
    """Rebuild the 44x25 tile array the loader would produce from an attract file."""
    b = open(path, "rb").read()
    maplen = struct.unpack_from("<I", b, 0)[0]
    m = b[8:8 + maplen]
    g = bytearray(GRID_W * GRID_H)
    for y in range(GRID_H):
        for x in range(GRID_W):
            if y == 0:
                v = 36
            elif y == GRID_H - 1:
                v = 34
            elif x == 0:
                v = 35
            elif x == GRID_W - 1:
                v = 37
            else:
                v = 0
            g[x + GRID_W * y] = v
    for i in range(42 * 23):
        t = m[0xB0 + i]
        x, y = i % 42, i // 42
        g[(x + 1) + GRID_W * (y + 1)] = t
    return bytes(g)


SPAWN_OFF = 0x4C7  # (x, y) bytes in the map section; pixels = raw * 6


def spawn_of(path):
    b = open(path, "rb").read()
    maplen = struct.unpack_from("<I", b, 0)[0]
    m = b[8:8 + maplen]
    return (m[SPAWN_OFF] * 6, m[SPAWN_OFF + 1] * 6)


def identify(first_rec, catalogue):
    """Match a capture to its demo by the frame-1 ninja position.

    Frame 1 lands on the spawn exactly in x, and spawn+gravity in y.
    """
    xp, yp = first_rec
    hits = [d for d, (sx, sy) in catalogue.items()
            if sx == xp and abs(yp - sy) < 1.0]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "AMBIGUOUS(%s)" % "|".join(sorted(hits))
    return None


class Capture(object):
    def __init__(self, catalogue):
        self.catalogue = catalogue
        self.cur = None
        self.records = []
        self.log = []
        self.index = []

    def note(self, s):
        self.log.append(s)
        print(s, flush=True)

    def on_message(self, message, data):
        if message["type"] == "error":
            self.note("ERROR %s" % message.get("description"))
            return
        p = message["payload"]
        t = p.get("t")
        if t == "module":
            self.note("npp.dll @ %s size=%d" % (p["base"], p["size"]))
        elif t == "installed":
            self.note("hooks installed")
        elif t == "demo":
            self.cur = {"slot": p["demo"], "id": None, "sim": p["sim"]}
            self.records = []
        elif t == "frames":
            if self.cur is None:
                return
            n = p["n"]
            for i in range(n):
                o = i * REC
                fr = struct.unpack_from("<I", data, o)[0]
                s = data[o + 4:o + 4 + SLAB]
                self.records.append((fr, s))
        elif t == "end":
            self.finish(p.get("frames"))

    def finish(self, frames):
        if self.cur is None or not self.records:
            self.cur = None
            self.records = []
            return
        f0 = self.records[0][1]
        first = (struct.unpack_from("<d", f0, F_XPOS)[0],
                 struct.unpack_from("<d", f0, F_YPOS)[0])
        hit = identify(first, self.catalogue)
        # Several demos in a batch can share a map, so the spawn is not always a
        # unique key. Always write under the capture index and record the
        # candidates; `match_traces.py` resolves them offline by replaying.
        demo_id = hit if (hit and not hit.startswith("AMBIGUOUS")) \
            else "cap%02d" % self.cur["slot"]
        self.note("--- capture %d -> %s  spawn=(%.1f, %.1f) frames=%d  %s"
                  % (self.cur["slot"], demo_id, first[0], first[1],
                     len(self.records), hit or "no spawn match"))
        self.index.append({"capture": self.cur["slot"], "id": demo_id,
                           "spawn": first, "frames": len(self.records),
                           "candidates": hit})
        path = os.path.join(OUT, "%s.jsonl" % demo_id)
        # attract demos loop; keep the longest capture we have seen
        if os.path.exists(path):
            try:
                prev = sum(1 for _ in open(path))
            except Exception:
                prev = 0
            if prev >= len(self.records):
                self.cur = None
                self.records = []
                return
        with open(path, "w") as fh:
            for fr, s in self.records:
                xp = struct.unpack_from("<d", s, F_XPOS)[0]
                yp = struct.unpack_from("<d", s, F_YPOS)[0]
                xs = struct.unpack_from("<d", s, F_XSPEED)[0]
                ys = struct.unpack_from("<d", s, F_YSPEED)[0]
                st = struct.unpack_from("<i", s, F_STATE)[0]
                fx = struct.unpack_from("<d", s, F_FLOORX)[0]
                fy = struct.unpack_from("<d", s, F_FLOORY)[0]
                rec = {
                    "frame": fr,
                    "xpos": xp, "ypos": yp,
                    "xspeed": xs, "yspeed": ys,
                    "state": st,
                    "floor_x": fx, "floor_y": fy,
                    "hex": {
                        "xpos": "%016x" % struct.unpack_from("<Q", s, F_XPOS)[0],
                        "ypos": "%016x" % struct.unpack_from("<Q", s, F_YPOS)[0],
                        "xspeed": "%016x" % struct.unpack_from("<Q", s, F_XSPEED)[0],
                        "yspeed": "%016x" % struct.unpack_from("<Q", s, F_YSPEED)[0],
                        "floor_x": "%016x" % struct.unpack_from("<Q", s, F_FLOORX)[0],
                        "floor_y": "%016x" % struct.unpack_from("<Q", s, F_FLOORY)[0],
                    },
                }
                fh.write(json.dumps(rec) + "\n")
        raw = os.path.join(OUT, "%s.raw" % demo_id)
        with open(raw, "wb") as fh:
            for fr, s in self.records:
                fh.write(struct.pack("<I", fr))
                fh.write(s)
        self.note("    wrote %s  %d frames" % (path, len(self.records)))
        self.cur = None
        self.records = []


def main():
    os.makedirs(OUT, exist_ok=True)
    # slot -> demo id, written by manifest.py at install time
    man = {}
    mp = os.path.join(HERE, "manifest.json")
    if os.path.exists(mp):
        man = json.load(open(mp))
    catalogue = {}
    for name in sorted(os.listdir(INSTALLED)):
        demo_id = man.get(name, "slotfile_%s" % name)
        catalogue[demo_id] = spawn_of(os.path.join(INSTALLED, name))
    print("catalogue: %d demos" % len(catalogue), flush=True)

    dev = frida.get_local_device()
    target = None
    for proc in dev.enumerate_processes():
        if proc.name.lower().startswith("n++"):
            target = proc
            break
    if target is None:
        print("N++ is not running", flush=True)
        return
    print("attaching to pid=%d" % target.pid, flush=True)

    cap = Capture(catalogue)
    session = dev.attach(target.pid)
    script = session.create_script(open(AGENT, "r", encoding="utf-8").read())
    script.on("message", cap.on_message)
    script.load()

    deadline = time.time() + RUN_SECONDS
    while time.time() < deadline:
        time.sleep(0.25)
    try:
        script.exports_sync.flush()
    except Exception:
        pass
    time.sleep(0.5)
    session.detach()
    json.dump(cap.index, open(os.path.join(OUT, "_index.json"), "w"), indent=1)
    print("--- detached, %d captures ---" % len(cap.index), flush=True)


main()
