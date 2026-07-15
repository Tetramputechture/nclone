"""Interactive developer tool for ``test_environment.py``.

Provides a clickable "dev panel" to the right of the game window (so debug
overlays and map-source navigation no longer require memorising single-letter
keyboard shortcuts) and an interactive browser over every curriculum map
generator that reports whether each generated level is solvable.

The package mirrors the structure of :mod:`nclone.tas_tool`:

* :mod:`~nclone.dev_tool.layout` -- dependency-free panel geometry.
* :mod:`~nclone.dev_tool.controls` -- declarative control registry shared by
  the mouse (panel clicks) and the keyboard (hotkeys).
* :mod:`~nclone.dev_tool.panel` -- pygame drawing + hit-testing of the panel.
* :mod:`~nclone.dev_tool.overlays` -- one place that owns every debug-overlay
  flag and applies it to the env each frame.
* :mod:`~nclone.dev_tool.generators` -- the all-generators browser + solvability.
* :mod:`~nclone.dev_tool.loaders` -- official-map / benchmark map sources and the
  reset / hot-reload helpers.
* :mod:`~nclone.dev_tool.app` -- the main loop / orchestrator.
"""

from .app import DevToolApp, run_dev_tool

__all__ = ["DevToolApp", "run_dev_tool"]
