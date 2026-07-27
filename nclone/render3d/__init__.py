"""3D frontend for the nclone simulation.

A pure frontend wrapper: it reads simulation state each frame (``sim.ninja``,
``sim.entity_dic``, ``sim.tile_dic``) and renders a 2.5D extruded diorama with
moderngl. It never touches physics.

Requires the optional ``render3d`` extra (``pip install nclone[render3d]``).
See ``render3d.app`` for the entry point and ``README.md`` for run commands.
"""
