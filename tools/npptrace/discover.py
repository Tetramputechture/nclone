"""Run the entity-pool discovery agent against a running N++."""
import json
import os
import struct
import sys
import time

import frida

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(HERE, "agent_discover.js")
OUT = os.path.join(HERE, "discover")
RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 25

os.makedirs(OUT, exist_ok=True)
state = {"pools": None}


def on_message(message, data):
    if message["type"] == "error":
        print("ERROR", message.get("description"), flush=True)
        return
    p = message["payload"]
    t = p.get("t")
    if t == "pools":
        state["pools"] = p
        json.dump(p, open(os.path.join(OUT, "pools.json"), "w"), indent=1)
        print("sim=%s npp.dll=%s" % (p["sim"], p["base"]), flush=True)
        print("%4s %6s %12s %12s %8s" % ("pool", "count", "ptr", "vtable_rva", "stride"))
        for e in p["pools"]:
            if e["count"] == 0 and not e.get("vtable"):
                continue
            print("%4d %6d %12s %12s %8s"
                  % (e["i"], e["count"], e["ptr"],
                     ("0x%X" % e["vtableRva"]) if e.get("vtableRva") is not None else "-",
                     e["stride"]), flush=True)
    elif t == "simhead":
        open(os.path.join(OUT, "simhead.bin"), "wb").write(data)
        c = struct.unpack_from("<40H", data, 0x10)
        print("\ncounts @0x10: %s" % list(c[:32]), flush=True)
    elif t == "poolslab":
        fn = os.path.join(OUT, "pool%02d_stride%d_n%d.bin" % (p["i"], p["stride"], p["n"]))
        open(fn, "wb").write(data)
        print("  slab -> %s" % os.path.basename(fn), flush=True)
    else:
        print("%s %s" % (t, p), flush=True)


dev = frida.get_local_device()
target = None
for proc in dev.enumerate_processes():
    if proc.name.lower().startswith("n++"):
        target = proc
        break
if target is None:
    raise SystemExit("N++ is not running")
print("attaching to pid=%d" % target.pid, flush=True)
session = dev.attach(target.pid)
script = session.create_script(open(AGENT, "r", encoding="utf-8").read())
script.on("message", on_message)
script.load()
deadline = time.time() + RUN
while time.time() < deadline and state["pools"] is None:
    time.sleep(0.25)
time.sleep(1.0)
session.detach()
print("--- detached ---", flush=True)
