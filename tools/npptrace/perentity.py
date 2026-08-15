"""Per-entity hit test: the chaingun uses the bare circle test, others keep limbs.

§73: `FUN_1050c690` (the chaingun's lethality test, per §72) is a pure
circle-vs-segment test with no skeleton. `NCLONE_AB_NO_LASER_LIMBS` switches the
test for turret, both shooter drones AND the dual laser at once, so flipping it
gains `drone_shot` and loses `laser_surface`. Split it.

`beam_hits_ninja` gains an explicit `use_limbs` override (None = today's global
behaviour, so nothing changes unless a caller opts in).

Behind NCLONE_AB_PER_ENTITY_LIMBS.
"""
P = "/home/tetra/projects/nclone/nclone/physics.py"
s = open(P).read()

OLD = "def beam_hits_ninja(x1, y1, x2, y2, ninja, seg_len=None):"
NEW = "def beam_hits_ninja(x1, y1, x2, y2, ninja, seg_len=None, use_limbs=None):"
if OLD not in s:
    raise SystemExit("signature anchor missing")
s = s.replace(OLD, NEW, 1)

OLD2 = """    if os.environ.get("NCLONE_AB_NO_LASER_LIMBS") == "1" or not getattr(
        ninja, "bones", None
    ):"""
NEW2 = """    # `use_limbs=False` is an explicit per-entity opt-out: §73 shows the
    # chaingun's `FUN_1050c690` is a bare circle test with no skeleton, while the
    # global env var cannot distinguish entity types. None = global behaviour.
    if (
        use_limbs is False
        or os.environ.get("NCLONE_AB_NO_LASER_LIMBS") == "1"
        or not getattr(ninja, "bones", None)
    ):"""
if OLD2 not in s:
    raise SystemExit("branch anchor missing")
s = s.replace(OLD2, NEW2, 1)
open(P, "w").write(s)
print("ok physics")

P = "/home/tetra/projects/nclone/nclone/entity_classes/entity_drone_shooter_base.py"
s = open(P).read()
OLD3 = """        if beam_hits_ninja(
            self.xpos, self.ypos, self.xend, self.yend, ninja,
            getattr(self, "_beam_len", None),
        ):"""
NEW3 = """        if beam_hits_ninja(
            self.xpos, self.ypos, self.xend, self.yend, ninja,
            getattr(self, "_beam_len", None),
            use_limbs=(False if (_PER_ENTITY_LIMBS
                                 and not self.BEAM_USES_LIMBS) else None),
        ):"""
if OLD3 not in s:
    raise SystemExit("beam_kill anchor missing")
s = s.replace(OLD3, NEW3, 1)

OLD4 = "    CHARGE_DURATION = 53"
NEW4 = ("    #: §73: the chaingun's kill (`FUN_1050c690`) is a bare circle test.\n"
        "    #: Subclasses that genuinely thread between limbs override this.\n"
        "    BEAM_USES_LIMBS = False\n"
        "    CHARGE_DURATION = 53")
if OLD4 not in s:
    raise SystemExit("class anchor missing")
s = s.replace(OLD4, NEW4, 1)

if "_PER_ENTITY_LIMBS" not in s.split("class ")[0]:
    s = s.replace("\nclass ",
                  '\n_PER_ENTITY_LIMBS = (\n'
                  '    os.environ.get("NCLONE_AB_PER_ENTITY_LIMBS") == "1"\n'
                  ')\n\n\nclass ', 1)
open(P, "w").write(s)
print("ok shooter base")
