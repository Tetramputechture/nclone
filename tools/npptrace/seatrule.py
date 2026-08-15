"""Port the game's segment-selection rule for the dual laser seat (§70).

Game (`t23_DualLaser_mode_B.c`, LAB_104018f7):

    d2 = dy*dy + dx*dx
    if   d2 <  best                                     -> take
    elif d2 == best and best_back_facing and not back   -> take

Strict `<`, with the front-facing preference applied ONLY on an exact tie. This
port instead biases every front-facing segment by 0.1 px, which changes the
winner in a band the game treats as decided.

Behind NCLONE_AB_LASER_SEAT_RULE.
"""
P = "/home/tetra/projects/nclone/nclone/entity_classes/entity_dual_laser.py"
s = open(P).read()

OLD = """            is_back_facing, a, b = segment.get_closest_point(self.xpos, self.ypos)
            raw_sq = (self.xpos - a) ** 2 + (self.ypos - b) ** 2
            distance_sq = raw_sq - (0.0 if is_back_facing else 0.1)
            if raw_sq == 0.0:
                flush.append((segment, a, b, is_back_facing))
            if best is None or distance_sq < best[0]:
                best = (distance_sq, segment, a, b, is_back_facing)"""

NEW = """            is_back_facing, a, b = segment.get_closest_point(self.xpos, self.ypos)
            raw_sq = (self.xpos - a) ** 2 + (self.ypos - b) ** 2
            if raw_sq == 0.0:
                flush.append((segment, a, b, is_back_facing))
            if _SEAT_RULE:
                # The game's rule: strict `<` on the RAW distance, with the
                # front-facing preference applied only on an EXACT tie.
                if best is None or raw_sq < best[0]:
                    best = (raw_sq, segment, a, b, is_back_facing)
                elif (raw_sq == best[0] and best[4] and not is_back_facing):
                    best = (raw_sq, segment, a, b, is_back_facing)
            else:
                distance_sq = raw_sq - (0.0 if is_back_facing else 0.1)
                if best is None or distance_sq < best[0]:
                    best = (distance_sq, segment, a, b, is_back_facing)"""

if OLD not in s:
    raise SystemExit("anchor missing")
s = s.replace(OLD, NEW, 1)
s = s.replace("class EntityDualLaser",
              '_SEAT_RULE = os.environ.get("NCLONE_AB_LASER_SEAT_RULE") == "1"\n\n\n'
              "class EntityDualLaser", 1)
open(P, "w").write(s)
print("ok")
