"""Reproducible 2-corner CDJ "staircase" chain map.

The Corner Double Jump (CDJ) was first promoted to a lattice *trick edge* on
``di-required`` (a single CDJ from spawn).  To validate that the global ``d*``
**composes tricks** — routing through more than one CDJ in sequence — we need a
map that requires two CDJs back-to-back:

    spawn floor --CDJ#1--> mid platform (switch) --CDJ#2--> top shelf (exit)

A CDJ fires only where the geometry supports the buffer intersection: a wall
with a ledge ~one floor-jump-apex high (a **1-tile step** — the floor jump rises
~32px, clears the 1-tile ledge, and re-grounds while wall contact is still live,
arming both buffers).  Each platform is therefore a low step the ninja runs into;
the CDJ itself then climbs ~212px to the next platform.

This builder is the committed source of the map the lattice-audit / chain tests
use (``map_data`` list form, consumed via ``LatticeAudit(map_data=...)`` /
``PhysicsStepWrapper(map_data=...)`` — NOT the ``.map`` byte format; see
``docs/CDJ_REPLAY_HANDOFF.md`` for the coordinate-frame notes).
"""

from __future__ import annotations

from typing import List

from ...map_generation.map import Map


def _fill(m: Map, c0: int, c1: int, r0: int, r1: int) -> None:
    """Fill solid tiles over the half-open column/row rectangle [c0,c1)×[r0,r1)."""
    for c in range(c0, c1):
        for r in range(r0, r1):
            m.set_tile(c, r, 1)


def build_cdj_chain_map() -> List[int]:
    """Build the open 2-corner CDJ chain staircase and return its ``map_data``.

    Layout (grid tiles, 42×23 canvas):
      * spawn floor   : cols 0..19,  rows 18..22  (lowest)
      * step #1 / riser: cols 20..25, rows 17..22 (1 tile higher — corner #1)
      * mid platform  : cols 22..34, rows 9..10   (+CDJ#1 climb; holds the switch)
      * step #2 / riser: cols 28..34, row 8       (1 tile higher — corner #2)
      * top shelf     : cols 29..41, rows 1..2    (+CDJ#2 climb; holds the exit)

    The exit *door* sits on the top shelf and its paired *switch* on the mid
    platform, so a solve requires: CDJ#1 (spawn→mid) → touch switch → CDJ#2
    (mid→shelf) → reach exit.

    Shelf left edge (col 29): the CDJ is a near-vertical trick whose apex lands
    at ~launch_x+50 with little drift, so the landing platform must START at (or
    left of) the landing column or the ninja lands just OFF it, balanced on the
    left face one tile below the surface, and cannot get up.  CDJ#2 lands at
    x≈732 (col 29); col 30 (the obvious "exit is over there" choice) put the
    landing one tile short and the chain could not finish.  Col 28 also works;
    col 27 starts intruding on the ascent column and kills the climb.
    """
    m = Map()
    _fill(m, 0, 20, 18, 23)    # spawn floor
    _fill(m, 20, 26, 17, 23)   # step up to corner #1
    _fill(m, 22, 35, 9, 11)    # mid platform
    _fill(m, 28, 35, 8, 9)     # step up to corner #2
    _fill(m, 29, 42, 1, 3)     # top shelf (left edge col 29 — see note above)
    m.set_ninja_spawn(19, 17)
    # Exit door on the top shelf (grid 34,0); paired switch on the mid platform.
    m.add_entity(3, 34, 0, switch_x=25, switch_y=8)
    return [int(round(x)) for x in m.map_data()]


if __name__ == "__main__":
    md = build_cdj_chain_map()
    print(f"cdj-chain map_data: {len(md)} ints")
