"""Write a slot -> demo-id manifest for whatever is installed in the attract cache.

Matches installed slot files against the corpus by (size, sha256), but only
hashes corpus files whose SIZE matches an installed slot -- hashing all 250k
files takes minutes and buys nothing.
"""
import hashlib
import json
import os

CACHE = os.path.join(os.path.expanduser("~"), "Documents", "Metanet", "N++", "attract")
SRC = r"D:\attracts"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


slots = {}
want = {}
for slot in os.listdir(CACHE):
    p = os.path.join(CACHE, slot)
    data = open(p, "rb").read()
    slots[slot] = (len(data), hashlib.sha256(data).hexdigest())
    want.setdefault(len(data), set()).add(slots[slot][1])

by_hash = {}
with os.scandir(SRC) as it:
    for e in it:
        if not e.is_file():
            continue
        sz = e.stat().st_size
        if sz not in want:
            continue
        h = sha(e.path)
        if h in want[sz]:
            by_hash.setdefault(h, e.name)

man = {slot: by_hash.get(h, "unknown_slot%s" % slot) for slot, (_, h) in slots.items()}
json.dump(man, open(OUT, "w"), indent=1, sort_keys=True)
print("wrote %s (%d slots)" % (OUT, len(man)))
for k in sorted(man, key=lambda s: int(s) if s.isdigit() else 1 << 30):
    print("  slot %-4s -> %s" % (k, man[k]))
