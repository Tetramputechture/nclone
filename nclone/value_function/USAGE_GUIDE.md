# Retired — see `README.md`

This guide documented an older CLI (`--goal {switch,exit} --precompute-transitions`)
that has since been replaced by the single-shot surface-graph pipeline in
`precompute_cli.py`.

The current one-command workflow precomputes both goals, the arc cache, and
the shortest-path tree in a single pass:

```bash
python -m nclone.value_function.precompute_cli \
    --map-path nclone/test_maps/simple-walk \
    --output-dir value_functions \
    --num-workers 8 \
    -v
```

See **`README.md`** in this directory for:

- Full CLI flag reference (including the new memory knobs
  `--arc-waypoint-interval`, `--max-arc-frames`, `--no-waypoints`)
- Per-level performance expectations and the 8 GB RAM budget
- Runtime API usage (`SurfaceGraphPBRS`, `TwoPhaseValuePBRS`, `RoutePBRS`)
- Supported mechanics and currently unmodeled game features

For batch precomputation across all benchmark levels in isolated
subprocesses (so heap fragmentation is reclaimed between levels), use
`npp-rl/scripts/precompute_all.py`.
