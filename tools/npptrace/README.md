# `tools/npptrace` — ground truth from the running game

Captures N++'s own per-frame state and diffs nclone against it. This is what took
the native gate from 62.28% to 94.93%.

**Start with [`docs/GAME_TRACE_HANDOFF.md`](../../docs/GAME_TRACE_HANDOFF.md)** —
environment, method, everything already recovered, and the open problem.

## Two halves

**Capture** runs on the **Windows host** (the game is there), needs
`pip install "frida==16.7.19"`, and talks to a live `N++.exe`:

```bash
python install_attracts.py 10138417 10139490 1000622   # pick the demos
python manifest.py                                     # slot -> attract id
# launch through Steam (steam://rungameid/230270), wait for the window
python trace_entities.py 420                           # capture
python install_attracts.py --restore                   # put the machine back
```

**Analysis** runs in **WSL** against the repo's `.venv`.

| file | role |
| --- | --- |
| `agent_entities.js` / `trace_entities.py` | the capture: ninja + every entity, every frame |
| `agent_trace.js` / `trace.py` | ninja only (round-8 harness, superseded) |
| `agent_discover.js` / `discover.py` | re-derive the entity pool layout on a new build |
| `install_attracts.py` | swap the game's attract cache so it loops chosen demos |
| `manifest.py` | slot → attract id, by content hash |
| `analyze_pools.py` | find position fields in a fresh pool slab |
| `read_entities.py` | where was everything on frame N |
| `compare_entities.py` | nclone's entities of a type vs the game's |
| `compare_laser.py` | every dual-laser field at once |
| `compare_drones.py` | which drone diverges first |
| `compare_bones.py` | the 13 ninja bones (needs the wide slab) |
| `field_hunt.py` | **locate** a field by matching nclone's value into the slab |
| `identify_pools.py` | which pool index is which entity type |
| `drone_dirs.py` | orientation → first move |
| `shooterscore.py <pool>` | **field-by-field score for the shooter drones** (10 = type 12, 11 = type 13) |
| `chaserfields.py` | field-by-field score for the chaser (type 15) |
| `death_cause.py` | how each demo ends; fastest signal, no capture needed |
| `sim_trace.py` / `diff_trace.py` | replay nclone and diff vs a capture |
| `match_traces.py` | resolve ambiguous captures by replay scoring |
| `verify_all.sh` | one engine over every trace, prints each divergence |

The regression guard is `scripts/verify_game_traces.py` (both engines plus a
cross-engine check, baselined in `docs/game_traces/expected.json`). Use that in a
commit gate; `verify_all.sh` is the diagnostic view.

## The measurement that actually finds bugs

`shooterscore.py` / `chaserfields.py` are the shape worth copying for any entity:
replay each capture, compare **every struct field the game records** against
ours, and count only the frames on which our ninja is still exactly on the
game's. One run localises a bug to a field, and the "frames dropped after
divergence" line keeps the filter from hiding a collapse.

Handoff §75-§78 were all found this way. Each fix was taken to ~100% on the
captured fields *before* the gate was run, and each then cost zero demos —
whereas a decade of tuning against demo counts had produced nine chaser
configurations that fixed nothing.

## Three things that will bite you

* **Launch through Steam.** `frida.spawn` on `N++.exe` loses the process — it
  relaunches via Steam and the spawned copy exits. Attach instead.
* **Never install two demos from the same map in one batch.** Neither spawn
  matching nor replay matching can separate them; that is how `10138724` once
  shipped mislabelled as `10138730`.
* **The default ninja slab (`0x200`) does not reach the bones** at `ninja+0xC60`.
  Set `NINJA_SLAB = 0xD40` in both `agent_entities.js` and `trace_entities.py`
  before capturing anything skeleton-related.

Point analysis tools at a non-default capture directory with `NCLONE_TRACES=/path`
and a different corpus with `NCLONE_ATTRACTS=/path`.
