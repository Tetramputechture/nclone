"""Aim the chaser's acquisition at the PREVIOUS frame's xtarget/ytarget.

§62: every fact checks out (decomp reads +0x30/+0x38; slab is a raw memcpy; those
offsets hold xtarget/ytarget; `this` is not shifted) yet using the CURRENT
xtarget measures 35.91% against 51.58%. The remaining candidate is *when* the
value is read -- our think and the game's may sit on opposite sides of the target
update, which is exactly the one-frame staleness that turned out to be the whole
answer for the rocket (§49).

Cache the pair at the top of `think` and aim with the cached one.
NCLONE_AB_CHASER_TARGET_AIM=1 selects the target origin;
NCLONE_AB_CHASER_STALE_TARGET=1 makes it the previous frame's.
"""
P = "/home/tetra/projects/nclone/nclone/entity_classes/entity_drone_base.py"
s = open(P).read()
OLD = """        if _TARGET_AIM and getattr(self, "USES_TARGET_ORIGIN", False):
            ox, oy = self.xtarget, self.ytarget"""
NEW = """        if _TARGET_AIM and getattr(self, "USES_TARGET_ORIGIN", False):
            prev = getattr(self, "_prev_target", None)
            if _STALE_TARGET and prev is not None:
                ox, oy = prev
            else:
                ox, oy = self.xtarget, self.ytarget"""
if OLD not in s:
    raise SystemExit("base anchor missing")
s = s.replace(OLD, NEW, 1)
s = s.replace('_TARGET_AIM = os.environ.get("NCLONE_AB_CHASER_TARGET_AIM") == "1"',
              '_TARGET_AIM = os.environ.get("NCLONE_AB_CHASER_TARGET_AIM") == "1"\n'
              '_STALE_TARGET = os.environ.get("NCLONE_AB_CHASER_STALE_TARGET") == "1"', 1)
open(P, "w").write(s)
print("ok base")

P = "/home/tetra/projects/nclone/nclone/entity_classes/entity_drone_chaser.py"
s = open(P).read()
OLD2 = """        if self.chasing and not _INTERRUPT:
            return"""
NEW2 = """        _cached = getattr(self, "_prev_target", None)
        self._prev_target = (self.xtarget, self.ytarget)
        if self.chasing and not _INTERRUPT:
            return
        # `ninja_in_corridor` reads `_prev_target`; restore the pre-update value
        # for the duration of this think so the acquisition sees what the game's
        # `+0x30`/`+0x38` held at its own read point.
        self._prev_target = _cached if _cached is not None else self._prev_target"""
if OLD2 not in s:
    raise SystemExit("chaser anchor missing")
s = s.replace(OLD2, NEW2, 1)
open(P, "w").write(s)
print("ok chaser")
