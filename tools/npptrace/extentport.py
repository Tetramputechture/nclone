"""Port FUN_103ff070 / FUN_103ff100: the chaser's corridor EXTENT test.

The game does not walk toward the ninja. It walks outward from the drone's own
cell until blocked, and then requires the ninja's CELL to lie between the drone's
cell and that extent (`t15_Chaser_think.c`). Two differences from
`corridor_clear` beyond that: no radius offset on the coordinate, and a ONE-CELL
span rather than a radius footprint. Both make our version more permissive, which
is §53's false acquisition.

Array mapping (§55/§57): extentA checks `+0x18` / `+0x8998` = ver + door_ver
(vertical edges block horizontal movement); extentB checks `+0x44d8` / `+0xce58`
= hor + door_hor. `is_empty_column` / `is_empty_row` already test exactly those
pairs.

Behind NCLONE_AB_CHASER_EXTENT.
"""
P = "/home/tetra/projects/nclone/nclone/entity_classes/entity_drone_base.py"
s = open(P).read()

OLD = """        if abs(dy * xdir - dx * ydir) > self.CORRIDOR_HALF_WIDTH:
            return False  # outside the corridor
        return self.corridor_clear(direction, ninja.xpos, ninja.ypos)"""

NEW = """        if abs(dy * xdir - dx * ydir) > self.CORRIDOR_HALF_WIDTH:
            return False  # outside the corridor
        if _EXTENT_GATE:
            extent = self.corridor_extent(direction)
            if ydir:
                here = math.floor(self.ypos / 12)
                there = math.floor(ninja.ypos / 12)
            else:
                here = math.floor(self.xpos / 12)
                there = math.floor(ninja.xpos / 12)
            return min(extent, here) <= there <= max(extent, here)
        return self.corridor_clear(direction, ninja.xpos, ninja.ypos)

    def corridor_extent(self, direction):
        \"\"\"How far the open corridor runs from this drone along `direction`.

        Literal port of `FUN_103ff070` (horizontal) and `FUN_103ff100`
        (vertical), from `docs/decomp/A10_chaser_extentA.c` / `...extentB.c`:
        step one cell at a time from the drone's OWN cell, testing a single-cell
        span, and return the last cell reached. On overrun the game returns the
        unmoved start, so a runaway walk cannot widen the corridor.
        \"\"\"
        xdir, ydir = self.DIR_TO_VEC[direction]
        sign = 1 if direction < 2 else -1
        if not ydir:
            span = math.floor(self.ypos / 12)
            start = cur = math.floor(self.xpos / 12)
            for _ in range(self.MAX_CELL_WALK):
                if not is_empty_column(self.sim, cur, span, span, sign):
                    return cur
                cur += sign
            return start
        span = math.floor(self.xpos / 12)
        start = cur = math.floor(self.ypos / 12)
        for _ in range(self.MAX_CELL_WALK):
            if not is_empty_row(self.sim, span, span, cur, sign):
                return cur
            cur += sign
        return start"""

if OLD not in s:
    raise SystemExit("anchor missing")
s = s.replace(OLD, NEW, 1)

if "_EXTENT_GATE" not in s.split("class ")[0]:
    s = s.replace(
        "CORRIDOR_HALF_WIDTH = float(",
        "_EXTENT_GATE_PLACEHOLDER\n    CORRIDOR_HALF_WIDTH = float(", 1)
    s = s.replace("_EXTENT_GATE_PLACEHOLDER\n    ", "")
    # module-level flag, after the imports
    s = s.replace("\nclass ",
                  '\n_EXTENT_GATE = os.environ.get("NCLONE_AB_CHASER_EXTENT") == "1"\n\n\nclass ',
                  1)
open(P, "w").write(s)
print("ok")
