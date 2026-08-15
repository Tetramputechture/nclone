"""Port FUN_103ff1b0 whole: all four acquisition gates, in the game's order.

§63 measured eight PARTIAL configurations, each bolting one gate onto a model
that disagrees elsewhere; none fixed a loss. This does the whole thing at once,
with each gate reading the field the decompilation says it reads:

  gate 1  forward        dot(ninja - TARGET, DIR[s]) > 0        +0x30/+0x38
  gate 2  perpendicular  |cross(ninja - TARGET, DIR[s])| <= W   +0x30/+0x38
  gate 3  extent         ninja cell within [target cell, extent] +0x30/+0x38
  gate 4  legal step     test_next_direction_and_goal(s)         +0x10/+0x18

Gate 4 is `FUN_103fe0d0`, which §59 established is the one-step move probe and
which uses POSITION, not target -- that mixture is the part no previous
configuration had.

NCLONE_AB_CHASER_PORT=1.
"""
P = "/home/tetra/projects/nclone/nclone/entity_classes/entity_drone_chaser.py"
s = open(P).read()

OLD = """        for k in range(-1, 2):
            # (k + 4 + dir) & 3 -- forward and both sides, never backward.
            s = (k + 4 + self.dir) % 4
            if not self.ninja_in_corridor(s, ninja):
                continue"""
NEW = """        for k in range(-1, 2):
            # (k + 4 + dir) & 3 -- forward and both sides, never backward.
            s = (k + 4 + self.dir) % 4
            if _PORT_THINK:
                if not self._acquire_gates(s, ninja):
                    continue
            elif not self.ninja_in_corridor(s, ninja):
                continue"""
if OLD not in s:
    raise SystemExit("loop anchor missing")
s = s.replace(OLD, NEW, 1)

HELPER = '''
    def _acquire_gates(self, direction, ninja):
        """All four gates of `FUN_103ff1b0`, each on the field it actually reads.

        Gates 1-3 measure from `+0x30`/`+0x38` -- `xtarget`/`ytarget`, confirmed
        against the capture in §60 -- while gate 4 (`FUN_103fe0d0`, the one-step
        move probe) measures from `+0x10`/`+0x18`, the position. Mixing the two
        origins is the part §63's partial configurations never had.
        """
        xdir, ydir = self.DIR_TO_VEC[direction]
        ox, oy = self.xtarget, self.ytarget
        dx = ninja.xpos - ox
        dy = ninja.ypos - oy
        if dy * ydir + dx * xdir <= 0.0:
            return False
        if abs(dy * xdir - dx * ydir) > self.CORRIDOR_HALF_WIDTH:
            return False
        extent = self._extent_from(direction, ox, oy)
        if ydir:
            here, there = math.floor(oy / 12), math.floor(ninja.ypos / 12)
        else:
            here, there = math.floor(ox / 12), math.floor(ninja.xpos / 12)
        if not (min(extent, here) <= there <= max(extent, here)):
            return False
        return bool(self.test_next_direction_and_goal(direction))

    def _extent_from(self, direction, ox, oy):
        """`FUN_103ff070` / `FUN_103ff100`, walked from (ox, oy)."""
        xdir, ydir = self.DIR_TO_VEC[direction]
        sign = 1 if direction < 2 else -1
        if not ydir:
            span = math.floor(oy / 12)
            start = cur = math.floor(ox / 12)
            for _ in range(self.MAX_CELL_WALK):
                if not is_empty_column(self.sim, cur, span, span, sign):
                    return cur
                cur += sign
            return start
        span = math.floor(ox / 12)
        start = cur = math.floor(oy / 12)
        for _ in range(self.MAX_CELL_WALK):
            if not is_empty_row(self.sim, span, span, cur, sign):
                return cur
            cur += sign
        return start
'''

anchor = "    def _end_chase(self):"
if anchor not in s:
    raise SystemExit("helper anchor missing")
s = s.replace(anchor, HELPER + "\n" + anchor, 1)

s = s.replace('_INTERRUPT = os.environ.get("NCLONE_AB_CHASER_INTERRUPT") == "1"',
              '_INTERRUPT = os.environ.get("NCLONE_AB_CHASER_INTERRUPT") == "1"\n'
              '_PORT_THINK = os.environ.get("NCLONE_AB_CHASER_PORT") == "1"', 1)
if "from ..physics import" not in s:
    s = s.replace("import math",
                  "import math\nfrom ..physics import is_empty_column, is_empty_row", 1)
open(P, "w").write(s)
print("ok")
