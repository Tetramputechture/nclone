"""Delay only the first SHOT, not the sweep setup.

§40's rejected attempt moved `_begin_sweep()` to +4, which changed the ninja
state it samples and walked the whole burst off (97.2% -> 13.3%). Keep the sweep
set up on the transition frame -- same sampling as today -- and delay only
`fire()`:

  transition : _begin_sweep()          (no shot)
  +4         : fire()                  along the swept angle, NO extra step
  +8, +12 .. : step the sweep, fire()

That puts our first impact at +4 along exactly the angle `_begin_sweep` produced,
which is what `10131718` shows: `+0x50` jumps to -1.1504 and `+0xC0` first reads
(599.328, 120.0) -- bearing -1.150 -- on the same frame, f94.

Behind NCLONE_AB_DRONE_SHOT_DELAY.
"""
P = "/home/tetra/projects/nclone/nclone/entity_classes/entity_drone_shooter_base.py"
s = open(P).read()

OLD = """                self.fire()
                if self._sweep_ab:
                    self._begin_sweep()
                return"""
NEW = """                if self._shot_delay_ab:
                    # Sweep is still set up HERE -- moving it to the first shot
                    # changes the ninja state it samples and walks the whole
                    # burst off (§40). Only the shot itself is delayed.
                    if self._sweep_ab:
                        self._begin_sweep()
                else:
                    self.fire()
                    if self._sweep_ab:
                        self._begin_sweep()
                return"""
if OLD not in s:
    raise SystemExit("transition anchor missing")
s = s.replace(OLD, NEW, 1)

OLD2 = """            self.fire_timer += 1
            if self._burst_ab and self.fire_timer % self.SHOT_INTERVAL == 0:"""
NEW2 = """            self.fire_timer += 1
            if (self._shot_delay_ab
                    and self.fire_timer == self.SHOT_INTERVAL):
                # First shot: along the angle `_begin_sweep` already produced,
                # with NO further step -- otherwise it lands one increment past
                # where the game's first impact is recorded.
                self.fire()
            elif self._burst_ab and self.fire_timer % self.SHOT_INTERVAL == 0:"""
if OLD2 not in s:
    raise SystemExit("firing anchor missing")
s = s.replace(OLD2, NEW2, 1)

OLD3 = '        self._rearm_ab = _full'
NEW3 = ('        self._rearm_ab = _full\n'
        '        #: Delay only the first shot to +SHOT_INTERVAL (§40).\n'
        '        self._shot_delay_ab = (\n'
        '            os.environ.get("NCLONE_AB_DRONE_SHOT_DELAY") == "1"\n'
        '        )')
if OLD3 not in s:
    raise SystemExit("flag anchor missing")
s = s.replace(OLD3, NEW3, 1)
open(P, "w").write(s)
print("ok")
