"""Close the loop on the denormal: exact dx, exact distance, exact quotient.

Hypotheses under test:
  * the contact is evaluated AFTER gravity has moved the ninja into the floor,
    so the distance is `264 - fl(254 + GRAVITY_FALL)`, not 10;
  * the floor is one 24 px segment `(72,264)->(48,264)`, not two 12 px halves,
    and that winding is the one whose `x1 + u*px` lands one ulp low.
"""
GRAV = 0.06666666666666665
TARGET = 7.1531147895312075e-16
X = 49.11705634311928

ypos_after_gravity = 254.0 + GRAV
dist = 264.0 - ypos_after_gravity
print("ypos after gravity      = %.17g" % ypos_after_gravity)
print("distance to the floor   = %.17g" % dist)

print("\nclosest point by winding, at x = %.17g" % X)
CANDS = [("two 12px halves, (60,264)->(48,264)", 60.0, -12.0),
         ("      same, other winding", 48.0, 12.0),
         ("one 24px, (72,264)->(48,264)", 72.0, -24.0),
         ("      same, other winding", 48.0, 24.0),
         ("one 24px, (48,264)->(72,264)", 48.0, 24.0),
         ("one 48px, (96,264)->(48,264)", 96.0, -48.0)]
for name, x1, px in CANDS:
    t = (X - x1) * px
    lensq = px * px
    if not (0 < t < lensq):
        print("  %-38s outside" % name)
        continue
    u = t / lensq
    a = x1 + u * px
    dx = X - a
    print("  %-38s a=%.17g  dx=%.17g  dx/dist=%.17g%s"
          % (name, a, dx, dx / dist,
             "   <== MATCHES THE GAME" if dx / dist == TARGET else ""))

print("\ntarget = %.17g" % TARGET)
