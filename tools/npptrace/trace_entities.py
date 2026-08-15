"""Round 9 capture: ninja + every entity, per frame, segmented by Simulator::reset.

Writes per demo:
    <cap>.ninja.jsonl   the round-8 schema, unchanged
    <cap>.ent.bin.gz    <u32 frame><512 ninja><entity slabs> per frame
    <cap>.layout.json   pool table: index, count, stride, vtable RVA, byte offset
"""
import gzip
import json
import os
import struct
import sys
import time

import frida

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(HERE, "agent_entities.js")
# Overridable so a new capture session does not land on top of an old one.
OUT = os.environ.get("NCLONE_CAPTURE_OUT") or os.path.join(HERE, "out9")
INSTALLED = os.path.join(os.path.expanduser("~"), "Documents", "Metanet", "N++", "attract")

NINJA_SLAB = 0xD40
F_XPOS, F_YPOS = 0x20, 0x28
F_XSPEED, F_YSPEED = 0x30, 0x38
F_STATE = 0xD0
F_FLOORX, F_FLOORY = 0x120, 0x128
SPAWN_OFF = 0x4C7

RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 600


def spawn_of(path):
    b = open(path, "rb").read()
    n = struct.unpack_from("<I", b, 0)[0]
    m = b[8:8 + n]
    return (m[SPAWN_OFF] * 6, m[SPAWN_OFF + 1] * 6)


def hx(b, off):
    return "%016x" % struct.unpack_from("<Q", b, off)[0]


class Cap(object):
    def __init__(self, catalogue):
        self.cat = catalogue
        self.layout = None
        self.rec = 0
        self.frames = []
        self.slot = 0
        self.index = []

    def on_message(self, message, data):
        if message["type"] == "error":
            print("ERROR", message.get("description"), flush=True)
            return
        p = message["payload"]
        t = p.get("t")
        if t == "layout":
            self.layout = p
            self.rec = p["recBytes"]
            self.slot = p["demo"]
            self.frames = []
            print("  layout: %d pools, %d entity bytes/frame, rec=%d"
                  % (len(p["pools"]), p["entityBytes"], p["recBytes"]), flush=True)
        elif t == "frames":
            if self.rec == 0:
                return
            n = p["n"]
            for i in range(n):
                self.frames.append(data[i * self.rec:(i + 1) * self.rec])
        elif t == "end":
            self.finish()
        elif t in ("module", "installed", "err"):
            print("%s %s" % (t, {k: v for k, v in p.items() if k != "t"}), flush=True)

    def finish(self):
        if not self.frames or self.layout is None:
            self.frames = []
            self.layout = None
            return
        f0 = self.frames[0][4:4 + NINJA_SLAB]
        sx = struct.unpack_from("<d", f0, F_XPOS)[0]
        sy = struct.unpack_from("<d", f0, F_YPOS)[0]
        hits = [d for d, (cx, cy) in self.cat.items()
                if cx == sx and abs(sy - cy) < 1.0]
        name = hits[0] if len(hits) == 1 else "cap%02d" % self.slot
        print("--- capture %d -> %-12s spawn=(%.1f, %.1f) frames=%d"
              % (self.slot, name, sx, sy, len(self.frames)), flush=True)

        with open(os.path.join(OUT, "%s.ninja.jsonl" % name), "w") as fh:
            for rec in self.frames:
                fr = struct.unpack_from("<I", rec, 0)[0]
                s = rec[4:4 + NINJA_SLAB]
                o = {
                    "frame": fr,
                    "xpos": struct.unpack_from("<d", s, F_XPOS)[0],
                    "ypos": struct.unpack_from("<d", s, F_YPOS)[0],
                    "xspeed": struct.unpack_from("<d", s, F_XSPEED)[0],
                    "yspeed": struct.unpack_from("<d", s, F_YSPEED)[0],
                    "state": struct.unpack_from("<i", s, F_STATE)[0],
                    "floor_x": struct.unpack_from("<d", s, F_FLOORX)[0],
                    "floor_y": struct.unpack_from("<d", s, F_FLOORY)[0],
                    "hex": {"xpos": hx(s, F_XPOS), "ypos": hx(s, F_YPOS),
                            "xspeed": hx(s, F_XSPEED), "yspeed": hx(s, F_YSPEED),
                            "floor_x": hx(s, F_FLOORX), "floor_y": hx(s, F_FLOORY)},
                }
                fh.write(json.dumps(o) + "\n")

        with gzip.open(os.path.join(OUT, "%s.ent.bin.gz" % name), "wb") as fh:
            for rec in self.frames:
                fh.write(rec)
        meta = dict(self.layout)
        meta["ninjaSlab"] = NINJA_SLAB
        meta["frames"] = len(self.frames)
        meta["demo"] = name
        json.dump(meta, open(os.path.join(OUT, "%s.layout.json" % name), "w"), indent=1)
        self.index.append({"id": name, "frames": len(self.frames),
                           "rec": self.rec, "pools": len(self.layout["pools"])})
        self.frames = []
        self.layout = None


def main():
    os.makedirs(OUT, exist_ok=True)
    man = {}
    mp = os.path.join(HERE, "manifest.json")
    if os.path.exists(mp):
        man = json.load(open(mp))
    cat = {}
    for nm in sorted(os.listdir(INSTALLED)):
        cat[man.get(nm, "slot%s" % nm)] = spawn_of(os.path.join(INSTALLED, nm))
    print("catalogue: %d demos" % len(cat), flush=True)

    dev = frida.get_local_device()
    target = None
    for pr in dev.enumerate_processes():
        if pr.name.lower().startswith("n++"):
            target = pr
            break
    if target is None:
        raise SystemExit("N++ is not running")
    print("attaching to pid=%d" % target.pid, flush=True)
    cap = Cap(cat)
    session = dev.attach(target.pid)
    script = session.create_script(open(AGENT, "r", encoding="utf-8").read())
    script.on("message", cap.on_message)
    script.load()
    end = time.time() + RUN
    while time.time() < end:
        time.sleep(0.25)
    try:
        script.exports_sync.flush()
    except Exception:
        pass
    time.sleep(0.8)
    session.detach()
    json.dump(cap.index, open(os.path.join(OUT, "_index.json"), "w"), indent=1)
    print("--- detached, %d captures ---" % len(cap.index), flush=True)


main()
