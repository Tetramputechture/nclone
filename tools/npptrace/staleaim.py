"""The rocket aims at a ninja position one frame staler than we use.

`launchdir.py` inverts the game's own launch direction over 105 launches: 88.6%
best-fit the ninja **2 frames before** the launch frame with **no lead**. Our
`think` runs before the ninja's physics, so it already sees the end of the
previous frame -- one frame stale. The game is one frame staler again.

Cache the ninja position each think and aim with the cached (previous) one.
Behind NCLONE_AB_ROCKET_STALE_AIM.
"""
P = "/home/tetra/projects/nclone/nclone/entity_classes/entity_rocket.py"
s = open(P).read()

OLD_INIT = "        self._initial_cell = self.cell\n\n    def think(self):"
NEW_INIT = ("        self._initial_cell = self.cell\n"
            "        #: Ninja position as of the PREVIOUS think, for the launch aim.\n"
            "        self._ninja_prev = None\n\n"
            "    def think(self):")
if OLD_INIT not in s:
    raise SystemExit("init anchor missing")
s = s.replace(OLD_INIT, NEW_INIT, 1)

OLD_T = """    def think(self):
        prev = self.state
        self.prev_state = prev
"""
NEW_T = """    def think(self):
        prev = self.state
        self.prev_state = prev
        _n = self.sim.ninja
        _aim_at_pos = self._ninja_prev
        if _n is not None:
            self._ninja_prev = (_n.xpos, _n.ypos)
"""
if OLD_T not in s:
    raise SystemExit("think anchor missing")
s = s.replace(OLD_T, NEW_T, 1)

OLD_L = """            target_x = ninja.xpos if ninja is not None else self.xpos
            target_y = ninja.ypos if ninja is not None else self.ypos"""
NEW_L = """            if _STALE_AIM and _aim_at_pos is not None:
                target_x, target_y = _aim_at_pos
            else:
                target_x = ninja.xpos if ninja is not None else self.xpos
                target_y = ninja.ypos if ninja is not None else self.ypos"""
if OLD_L not in s:
    raise SystemExit("launch anchor missing")
s = s.replace(OLD_L, NEW_L, 1)

if "_STALE_AIM" not in s.split("class EntityRocket")[0]:
    s = s.replace("class EntityRocket",
                  '_STALE_AIM = os.environ.get("NCLONE_AB_ROCKET_STALE_AIM") == "1"'
                  "\n\n\nclass EntityRocket", 1)

# reset_state must clear the cache too
s = s.replace("        self.charge = 0.0\n        self.active = True",
              "        self.charge = 0.0\n        self._ninja_prev = None\n"
              "        self.active = True", 1)
open(P, "w").write(s)
print("ok")
