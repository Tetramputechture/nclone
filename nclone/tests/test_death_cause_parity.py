"""Both engines must write the SAME death-cause strings, and only known ones.

This is a source-level test, not a behavioural one: it scans every `kill(...)`
site in the pure-Python entity classes, in `ninja.py`, and in the Rust sources,
and checks the literals against `constants/death_causes.ALL_DEATH_CAUSES`.

Why static rather than staging 18 real deaths: constructing a level that gets
the ninja killed by, say, a surface-mode dual laser is a fixture in its own
right, and the failure this guards against is not behavioural. It is somebody
adding an entity and typing a new literal, or typing `"drone"` where Python says
`"drone_zap"` -- which is exactly what had happened, silently, and which no
behavioural test would have caught either unless it happened to cover that
entity on that backend.

The failure mode this prevents is a diagnostic that lies: a native-backed run
reporting a single `drone` bucket while the taxonomy, the reward table and the
threat-class rollup all expect `drone_zap` / `mini_drone` / `drone_chaser`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nclone.constants.death_causes import (
    ALL_DEATH_CAUSES,
    DEATH_CAUSE_CLASS,
    DEATH_CAUSE_TO_ENTITY_TYPE,
    THREAT_CLASSES,
    CLASS_ENVIRONMENT,
    classify_death_cause,
    normalize_death_cause,
    UNKNOWN,
)

_REPO = Path(__file__).resolve().parents[2]
_PY_SOURCES = sorted((_REPO / "nclone" / "entity_classes").glob("entity_*.py")) + [
    _REPO / "nclone" / "ninja.py"
]
_RS_SOURCES = sorted((_REPO / "nclone-native" / "src" / "entities").glob("*.rs")) + [
    _REPO / "nclone-native" / "src" / "ninja.rs"
]

# Neither engine passes every cause as an inline literal at the kill site, so a
# single pattern per language under-reports. Both indirections are matched
# explicitly rather than by loosening the literal scan, which would sweep in
# every unrelated string in the file:
#
#   Python  `ninja.kill(..., cause="mine")`         -- inline
#           `KILL_CAUSE = "drone_zap"`              -- class attribute, passed
#                                                      as `cause=self.KILL_CAUSE`
#   Rust    `.kill(..., Some("mine"))`              -- inline
#           `fn kill_cause(self) -> &'static str`   -- per-variant match arms
_PY_PATTERNS = (
    re.compile(r'cause\s*=\s*"([a-z_]+)"'),
    re.compile(r'KILL_CAUSE\s*=\s*"([a-z_]+)"'),
)
_RS_PATTERNS = (
    re.compile(r'\.kill\([^)]*?Some\("([a-z_]+)"\)'),
    # The body of `fn kill_cause`, up to the closing brace at method indent.
    re.compile(
        r"fn kill_cause\(self\)[^{]*\{(.*?)\n    \}", re.DOTALL
    ),
)
_ARM_LITERAL = re.compile(r'"([a-z_]+)"')


def _literals(paths, patterns):
    found = {}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text()
        for pattern in patterns:
            for match in pattern.finditer(text):
                group = match.group(1)
                # A block-body match yields the whole body; a literal match
                # yields the cause itself.
                names = (
                    _ARM_LITERAL.findall(group) if "\n" in group else [group]
                )
                for name in names:
                    found.setdefault(name, []).append(path.name)
    return found


def test_python_kill_sites_use_known_causes():
    found = _literals(_PY_SOURCES, _PY_PATTERNS)
    assert found, "found no cause= literals at all -- the scan pattern is stale"
    unknown = {c: v for c, v in found.items() if c not in ALL_DEATH_CAUSES}
    assert not unknown, (
        f"pure-Python engine writes death causes absent from ALL_DEATH_CAUSES: "
        f"{unknown}. Add them to constants/death_causes.py (and emit the same "
        f"literal from the Rust engine)."
    )


def test_rust_kill_sites_use_known_causes():
    found = _literals(_RS_SOURCES, _RS_PATTERNS)
    assert found, "found no Some(\"...\") kill literals at all -- the scan pattern is stale"
    unknown = {c: v for c, v in found.items() if c not in ALL_DEATH_CAUSES}
    assert not unknown, (
        f"nclone-native writes death causes absent from ALL_DEATH_CAUSES: "
        f"{unknown}. Add them to constants/death_causes.py (and emit the same "
        f"literal from the Python engine)."
    )


def test_the_two_engines_agree():
    """The set of causes each engine can emit must be identical.

    A cause only one engine can produce means a metric that changes meaning when
    the backend changes -- the defect this module was written for.
    """
    py = set(_literals(_PY_SOURCES, _PY_PATTERNS))
    rs = set(_literals(_RS_SOURCES, _RS_PATTERNS))
    assert py == rs, (
        "death-cause sets diverge between engines.\n"
        f"  only in pure-Python: {sorted(py - rs)}\n"
        f"  only in nclone-native: {sorted(rs - py)}"
    )


def test_every_declared_cause_is_actually_emitted():
    """No dead entries: a cause nothing writes is a metric row that stays 0.0
    forever and reads as 'the agent never dies this way'."""
    emitted = set(_literals(_PY_SOURCES, _PY_PATTERNS)) | set(
        _literals(_RS_SOURCES, _RS_PATTERNS)
    )
    orphans = set(ALL_DEATH_CAUSES) - emitted
    assert not orphans, (
        f"declared in ALL_DEATH_CAUSES but no kill site writes them: {sorted(orphans)}"
    )


def test_taxonomy_is_total_and_stable():
    assert len(set(ALL_DEATH_CAUSES)) == len(ALL_DEATH_CAUSES), "duplicate cause"
    for cause in ALL_DEATH_CAUSES:
        assert cause in DEATH_CAUSE_CLASS, f"{cause} has no threat class"
        assert DEATH_CAUSE_CLASS[cause] in THREAT_CLASSES + (CLASS_ENVIRONMENT,)
    for cause in DEATH_CAUSE_TO_ENTITY_TYPE:
        assert cause in ALL_DEATH_CAUSES, f"{cause} maps an entity but is not a cause"


@pytest.mark.parametrize("bad", [None, "", "not_a_real_cause", "DRONE_ZAP  "])
def test_lookups_never_raise_on_garbage(bad):
    """A corrupt cause must not create an unbounded set of TB rows, and must not
    inflate a threat-class rate either."""
    assert classify_death_cause(bad) in THREAT_CLASSES + (CLASS_ENVIRONMENT,)
    assert normalize_death_cause(bad) in set(ALL_DEATH_CAUSES) | {UNKNOWN}


def test_case_is_normalized_but_whitespace_is_not_silently_accepted():
    assert normalize_death_cause("DRONE_ZAP") == "drone_zap"
    assert normalize_death_cause("drone_zap") == "drone_zap"
    # A padded string is a corrupt value, not a variant spelling.
    assert normalize_death_cause(" drone_zap ") == UNKNOWN
