"""Don't fire on the frame the discharge ends.

`FIRE_DURATION` is 80 and shots go every 4 frames, so `fire_timer == 80` is both
a shot frame AND the frame `is_fire_done()` becomes true: we fire a 20th shot and
transition to cooldown in the same tick. On `10136116` frame 298 that is exactly
the false kill -- every drone is in state 0 or 3 in BOTH engines, nobody is
discharging, and our beam end has already been reset to the drone by `on_abort`.

Behind NCLONE_AB_DRONE_NO_LAST_SHOT.
"""
P = "/home/tetra/projects/nclone/nclone/entity_classes/entity_drone_shooter_base.py"
s = open(P).read()

OLD = """            if (self._shot_delay_ab
                    and self.fire_timer == self.SHOT_INTERVAL):"""
NEW = """            # The last frame of the discharge is also a multiple of
            # SHOT_INTERVAL, so without this the burst gets a 20th shot on the
            # very frame it ends -- and `on_abort` then wipes the beam, leaving a
            # kill with no drone in state 2 to explain it (§42).
            _ending = self._no_last_shot_ab and self.is_fire_done()
            if _ending:
                pass
            elif (self._shot_delay_ab
                    and self.fire_timer == self.SHOT_INTERVAL):"""
if OLD not in s:
    raise SystemExit("firing anchor missing")
s = s.replace(OLD, NEW, 1)

OLD3 = '        self._rearm_ab = _full'
NEW3 = ('        self._rearm_ab = _full\n'
        '        #: Suppress the shot on the frame the discharge ends (§42).\n'
        '        self._no_last_shot_ab = (\n'
        '            os.environ.get("NCLONE_AB_DRONE_NO_LAST_SHOT") == "1"\n'
        '        )')
if OLD3 not in s:
    raise SystemExit("flag anchor missing")
s = s.replace(OLD3, NEW3, 1)
open(P, "w").write(s)
print("ok")
