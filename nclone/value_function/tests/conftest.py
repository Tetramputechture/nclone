"""Shared fixtures/guards for the value-function tests.

``require_intact_map_fixture`` exists because two committed "map" fixtures are
not maps.  See its docstring for the evidence.
"""

from __future__ import annotations

import os

import pytest

# Repo root, from this file: nclone/value_function/tests/ -> up 3.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _exit_door_count(map_path: str) -> int:
    """Number of type-3 (exit door) entities the native loader finds.

    Any playable level has at least one.  Zero means the bytes did not parse as
    a level at all.
    """
    import nclone_native as Nnative

    with open(map_path, "rb") as f:
        map_data = [int(b) for b in f.read()]
    st = Nnative.Simulator.load(map_data).get_state()
    return sum(1 for e in st.entities if int(e.entity_type) == 3)


def require_intact_map_fixture(rel_path: str) -> str:
    """Skip the calling test unless ``rel_path`` is a genuine, parseable level.

    ``nclone/test_maps/mines-extremely-difficult`` and
    ``nclone/test_maps/mine-difficult-two`` are **byte-identical** 68,309-byte
    blobs (md5 ``8d4df26b90fb1babd8db10d0e3c5b34d``) that both carry the single
    embedded name ``[#t] pretending to be human``.  They are not the mine levels
    the tests that load them describe, and they are not levels at all:

    * The native loader finds **0 exit doors** in them, alongside ~86 each of
      almost every other entity type (4, 5, 6, 8, 10-28) and 820 "mines" spaced
      exactly 6px apart along one row.  That near-uniform spread is what you get
      from reading non-entity bytes as entity records, not from a level design.
    * The byte distribution after the tile section agrees.  Measuring the tail
      past offset 1150: a real level (``simple-walk``) has 0.82 bits/byte of
      entropy and is 88% zeros, because its entity section is mostly empty
      padding.  These two blobs read 4.42 bits/byte over 172 distinct byte
      values -- essentially identical to the known replay file
      ``test_maps/12345.attract`` (3.66 bits/byte, 175 values).  They are replay
      or attract payloads that were committed under map filenames.

    The code those tests cover is fine, which is the important part and the
    reason this is a skip rather than an xfail on the source.  Checked directly
    against the two REAL mine levels in the repo, ``MineThreadingDistanceSource``
    builds a connected, mine-faithful field on both:

        minefield            29 deadly mines   available=True  spawn_finite=True
        mine-very-difficult  174 deadly mines  available=True  spawn_finite=True

    To restore these tests, commit an actual mine-saturated level whose surface
    lattice is severed -- that severing is the property under test, and neither
    ``minefield`` nor ``mine-very-difficult`` has it (``FreeSpaceDistanceSource``
    reports ``available=True`` on both), so they cannot stand in.  The guard is
    deliberately a content check rather than a filename blocklist: drop a real
    level in at the same path and the test starts running again on its own.
    """
    path = os.path.join(_REPO_ROOT, rel_path)
    if not os.path.exists(path):
        pytest.skip(f"map fixture missing: {rel_path}")
    try:
        n_exits = _exit_door_count(path)
    except Exception as exc:  # noqa: BLE001 — a fixture we cannot parse is a skip
        pytest.skip(f"map fixture {rel_path} did not parse ({type(exc).__name__}: {exc})")
    if n_exits == 0:
        pytest.skip(
            f"map fixture {rel_path} is CORRUPT: the native loader finds no exit "
            f"door in it. It is a replay/attract payload committed under a map "
            f"filename, byte-identical to the other 68,309-byte 'mine' fixture. "
            f"See require_intact_map_fixture.__doc__ for the evidence and for "
            f"what a replacement level has to demonstrate."
        )
    return path
