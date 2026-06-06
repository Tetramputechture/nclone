"""Render a map to a verification image (tiles + player + mines + exit/switch).

Two backends:

* :func:`render_schematic` (default) -- a fast, dependency-light matplotlib
  schematic. Tile solidity is sampled from the AUTHORITATIVE collision predicate
  ``graph_builder._check_subnode_validity_simple`` (the same geometry the physics
  sim / surface planner use), so slopes, curves and half-tiles draw with their
  true shape and there is no renderer y-flip ambiguity. Entities are overlaid
  with a labelled legend, so it is unambiguous which marker is the player, a
  mine, the exit door, and the exit switch.

* :func:`render_ground_truth` -- the exact game frame via the headless GL
  renderer (``NPlayHeadless``). Use this when you want pixel-faithful output.

Both accept a serialized ``map_data`` list (``Map.map_data()``), so they work for
composed templates, the NL builder, legacy generators, and dataset pickles alike.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from .visualize import _tiles, parse_entities

# Per-entity marker style: (matplotlib marker, color, size, label)
_ENTITY_STYLE = {
    "ninja": ("o", "#1f77b4", 90, "player"),
    "toggle_mine": ("X", "#d62728", 70, "mine"),
    "gold": ("o", "#ffcc00", 40, "gold"),
    "exit_door": ("s", "#2ca02c", 90, "exit door"),
    "exit_switch": ("*", "#9467bd", 150, "exit switch"),
    "locked_door": ("s", "#8c564b", 80, "locked door"),
    "trap_door": ("s", "#7f7f7f", 80, "trap door"),
    "locked_door_switch": ("P", "#e377c2", 80, "door switch"),
}


def _solid_mask(map_data: List[int], sub: int = 6) -> np.ndarray:
    """Boolean solid mask at ``sub`` samples/tile, from the collision predicate."""
    from ...graph.reachability.graph_builder import _check_subnode_validity_simple as valid

    tiles = _tiles(map_data)
    h, w = MAP_TILE_HEIGHT * sub, MAP_TILE_WIDTH * sub
    mask = np.zeros((h, w), dtype=bool)
    step = 24 // sub
    centers = [int((k + 0.5) * step) for k in range(sub)]
    # cache per tile-type sub-grid so we only evaluate 34 patterns
    cache = {}
    for ty in range(MAP_TILE_HEIGHT):
        for tx in range(MAP_TILE_WIDTH):
            tid = tiles[tx + ty * MAP_TILE_WIDTH]
            pat = cache.get(tid)
            if pat is None:
                pat = np.array(
                    [[not valid(tid, px, py) for px in centers] for py in centers],
                    dtype=bool,
                )
                cache[tid] = pat
            mask[ty * sub:(ty + 1) * sub, tx * sub:(tx + 1) * sub] = pat
    return mask


def render_schematic(map_data: List[int], out_path: str,
                     title: Optional[str] = None, sub: int = 6,
                     dpi: int = 130) -> str:
    """Draw a labelled schematic of the level and save it to ``out_path``.

    Returns ``out_path``. Solid terrain is shaded; the player, mines, exit door
    and exit switch are drawn as distinct, legended markers at their true grid
    positions (offset-correct via ``compose.coords``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = _solid_mask(map_data, sub=sub)
    fig, ax = plt.subplots(figsize=(MAP_TILE_WIDTH / 4.0, MAP_TILE_HEIGHT / 4.0))
    # extent in tile units; origin top-left (row 0 at top) to match ASCII/physics
    ax.imshow(mask, cmap="Greys", extent=[0, MAP_TILE_WIDTH, MAP_TILE_HEIGHT, 0],
              interpolation="nearest", vmin=0, vmax=1, alpha=0.9)

    # faint tile grid
    ax.set_xticks(range(0, MAP_TILE_WIDTH + 1, 5))
    ax.set_yticks(range(0, MAP_TILE_HEIGHT + 1, 5))
    ax.grid(True, which="major", color="#cccccc", linewidth=0.4, alpha=0.6)

    seen_labels = set()
    for e in parse_entities(map_data):
        style = _ENTITY_STYLE.get(e["name"])
        if style is None:
            continue
        marker, color, size, label = style
        gx, gy = e["tile"]
        lbl = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.scatter([gx + 0.5], [gy + 0.5], marker=marker, c=color, s=size,
                   edgecolors="black", linewidths=0.6, zorder=5, label=lbl)
        if e["name"] == "ninja":  # show facing
            dx = 0.9 if e.get("orientation") == "right" else -0.9
            ax.annotate("", xy=(gx + 0.5 + dx, gy + 0.5), xytext=(gx + 0.5, gy + 0.5),
                        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.3),
                        zorder=6)

    ax.set_xlim(0, MAP_TILE_WIDTH)
    ax.set_ylim(MAP_TILE_HEIGHT, 0)
    ax.set_aspect("equal")
    ax.set_title(title or "map schematic", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=5,
              fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_ground_truth(map_data: List[int], out_path: str) -> str:
    """Save the exact game frame (headless GL renderer). Returns ``out_path``."""
    from .visualize import render_png

    render_png(list(map_data), out_path)
    return out_path


def _load_map_data(args) -> List[int]:
    """Resolve a ``map_data`` list from the CLI args (template / phrase / pickle)."""
    if args.template:
        from . import compose

        m, _ = compose(args.template, seed=args.seed)
        return m.map_data()
    if args.phrase:
        from . import nl

        m, _, warnings = nl.build_from_phrase(args.phrase, seed=args.seed)
        for w in warnings:
            print(f"[nl warning] {w}")
        return m.map_data()
    if args.pickle:
        from .mapio import load_map_data

        return load_map_data(args.pickle, fmt="pickle")
    if args.json:
        from .mapio import load_map_data

        return load_map_data(args.json, fmt="json")
    if args.map:
        from .mapio import load_map_data

        return load_map_data(args.map, fmt="binary")
    raise SystemExit("provide one of --template / --phrase / --pickle / --json / --map")


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Render an nclone map to a verification image."
    )
    src = p.add_argument_group("source (choose one)")
    src.add_argument("--template", help="registered composable template name")
    src.add_argument("--phrase", help="natural-language description")
    src.add_argument("--pickle", help="dataset .pkl path (dict with map_data, or raw list)")
    src.add_argument("--json", help="path to a JSON map_data file")
    src.add_argument("--map", help="binary .map file (raw N++ bytes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["schematic", "ground_truth"], default="schematic")
    p.add_argument("--out", default="map.png")
    p.add_argument("--save", help="also save the resolved map here (.map/.json/.pkl)")
    p.add_argument("--no-facts", action="store_true", help="skip the describe_level summary")
    args = p.parse_args(argv)

    md = _load_map_data(args)
    if args.save:
        from .mapio import save_map

        save_map(md, args.save)
        print(f"saved {args.save}")
    if args.backend == "ground_truth":
        render_ground_truth(md, args.out)
    else:
        title = args.template or (args.phrase[:48] if args.phrase else "map")
        render_schematic(md, args.out, title=f"{title} (seed {args.seed})")
    print(f"wrote {args.out}")

    if not args.no_facts:
        import json
        from .visualize import describe_level

        facts = describe_level(md)
        print(json.dumps({k: facts[k] for k in ("dims", "tile_histogram", "entities")},
                         indent=1))
        try:
            from ..solvability import is_map_solvable

            print("solvable:", is_map_solvable(md))
        except Exception as exc:  # pragma: no cover
            print("solvable: (check skipped)", exc)


if __name__ == "__main__":
    main()
