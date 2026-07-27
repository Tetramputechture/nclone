# nclone 3D frontend

A frontend over the existing simulation. It reads sim state each frame
(`sim.ninja`, `sim.entity_dic`, `sim.tile_dic`) and renders a 2.5D extruded
diorama with moderngl. Physics is never touched: the seam is
`NPlayHeadless(enable_rendering=False)` -> `tick()` -> read state -> draw.

## Install

The 3D frontend needs moderngl, pyrr and imageio, which are not part of the base
install:

```bash
pip install -e ".[render3d]"
```

## Run

```bash
# interactive window
#   move: <-/-> or A/D    jump: Z/Space/Up/W
#   cam: C cycle mode (orbit/follow/first-person), mouse-drag or Q/E rotate,
#        +/- zoom, F follow, O spin; R reset, P pause, ESC quit
python -m nclone.render3d.app --map nclone/test_maps/squiggly --follow

# first-person: ride the agent. A/D + jump drive the ninja, the camera follows
python -m nclone.render3d.app --map nclone/test_maps/basics --fpv --anim

# record a scripted win on a curved level (run-left + bhop, follow cam)
python -m nclone.render3d.app --map nclone/test_maps/open-round-corner \
    --hold left --jump-period 45 --jump-len 15 --anim --follow --zoom 2.2 \
    --record /tmp/solve.mp4

# record an agent-perspective flythrough with edge motion blur
python -m nclone.render3d.app --map nclone/test_maps/basics --hold right \
    --fpv --anim --record /tmp/fpv.mp4 --width 960 --height 540

# record an attract-mode replay (follow cam implied)
python -m nclone.render3d.app --replay nclone/example_replays/npp_attract/0 \
    --record /tmp/replay.mp4 --width 960 --height 540

# cinematic orbit showcase of the tile geometry (fixed length)
python -m nclone.render3d.app --map nclone/test_maps/squiggly --record /tmp/orbit.mp4 \
    --azimuth -25 --elevation 34 --spin --spin-rate 0.0028 --max-frames 320

# single offscreen still
python -m nclone.render3d.app --map nclone/test_maps/basics \
    --headless --out /tmp/level.png --azimuth 15

# the 2D dev tool with a live 3D view of the same sim beside it
python -m nclone.test_environment --map nclone/test_maps/basics --render3d

# stacked multi-level: play a FOLDER of maps, warping to the next on each win
python -m nclone.render3d.app --stack nclone/test_maps --bg light

# record a stacked auto-cycle video (scripted right-hold + bhop drives each level)
python -m nclone.render3d.app --stack /path/to/maps --record /tmp/stack.mp4 \
    --stack-hold right --jump-period 22 --jump-len 4 --bg light --max-frames 400

# stacked mode in first-person: fly through each exit into the next level
python -m nclone.render3d.app --stack /path/to/maps --fpv --anim \
    --stack-hold right --jump-period 22 --jump-len 4 --bg light --record /tmp/stack_fpv.mp4
```

## What it renders

- Every tile type extruded to its exact footprint. Slopes become wedges, quarter
  circles and quarter pipes become tessellated arcs with smooth per-vertex
  normals. See `geometry.tile_polygon`.
- Circular entities (gold, mines, drones, launch pads, death balls) as spheres at
  their physical radius; square entities (thwumps, bounce blocks, one-way
  platforms) as boxes. Mines get spiky geometry. The exit door vanishes on a win.
- The ninja as a solid black body, or a bone skeleton with `--anim`.
- Cel shading: quantized light bands plus a thin fresnel ink outline on tiles,
  with entities rendered flat and vibrant so gold reads yellow and mines red.
- An animated monochrome background of 3D cuboids, visible through the level's
  translucent open cells and around it. It reacts to horizontal run speed
  (EMA-smoothed) and vertical jump impulses (peak-hold decay) on separate
  channels. `--plain-bg` disables it.
- Scattered 3D blocks radiating out from the play area, thinning with distance
  (`--no-scatter`), and static white road-dash lane markings on tile tops that
  tint with the speed the ninja ran over them.
- A speed ribbon streaming from the ninja, blue to magenta by speed, fluttering
  faster the faster you go (`--no-trail`), and motion-blur echoes trailing along
  its velocity (`--no-ninja-blur`).

## Cameras

Three modes, cycled with `C` in a window: orbit, dead-zone follow with
look-ahead, and agent-perspective first person.

First person (`--fpv`) rides just behind and above the ninja looking down the
corridor, and the keyboard (or `--hold` / `--replay`) drives the *ninja*: you
play the agent and the camera comes along. FOV widens with speed (`--fov-speed`)
while the camera dollies in to keep the agent the same on-screen size.

Camera flags: `--azimuth`, `--elevation`, `--zoom`, `--follow`, `--spin`,
`--spin-rate`. First person adds `--fov`, `--fp-back`, `--fp-up`, `--fp-out`,
`--fp-ahead`, `--fov-speed`.

Edge motion blur is sharp at the screen centre and smears toward the edges,
scaled by how fast the camera is moving. `--motion-blur` defaults to `-1`, which
means auto: on in first person only. `0` disables it, and a positive value both
scales the strength and enables it in every camera mode.
`--motion-blur-strength` is an additional multiplier.

## Stacked multi-level mode

`--stack <folder>` plays a folder of maps stacked along the depth axis, up to
three at a time. The live level sits at the play plane; its neighbours render as
faint depth-offset translucent panes, the next level pushed into the screen and
tinted cool, the previous pulled toward the camera and faint.

On a win the camera dives forward through the stack while a cyan warp streak
carries the agent from the exit into the next level: the current level dissolves,
the next fades up to solid, then the window advances (next becomes live, live
becomes prev, a fresh next loads) and the camera resets. The hand-off is seamless
because the geometry is laid out symmetrically around the live level. The folder
cycles forever, so N maps stream past three at a time.

Only the live level owns a simulation; neighbours are static snapshots, so the
cost over single-level mode is two extra blended draws. Death retries the level,
`N` skips, `C` toggles orbit against first person (where the teleport becomes a
fly-through). See `stack.py` (`StackDirector`, `_StackCam`).

Flags: `--stack-spacing` (depth between slabs, default 320), `--stack-teleport`
(warp frames, default 34), `--stack-hold` (auto-drive the live level; keyboard by
default in a window, `right` when recording), `--stack-next-alpha` /
`--stack-prev-alpha` (neighbour pane opacity), `--stack-desat` (neighbour
desaturation), `--stack-azimuth` / `--stack-elevation` (angle that reveals the
depth stack), `--stack-bg-anim` (opt back into the animated background, off by
default so the panes read), `--stack-scatter` (opt back into the scatter blocks,
off by default because they triple up across slabs).

## Dev-tool 3D panel

`--render3d` on `test_environment` adds a live 3D view of the very simulation the
2D debug interface is driving, in a strip between the game and the dev panel.
`--render3d-width` sizes it and `--render3d-fpv` uses the first-person camera. It
renders offscreen through a moderngl EGL context and blits into the pygame
window. See `render3d/devpanel.py` (`Render3DPanel`); the hooks are in
`dev_tool/app.py`.

## Appearance and performance

`--bg` takes a preset (`white`, `light`, `slate`, `dark`) or `r,g,b` floats in
[0,1], defaulting to white, which is mostly hidden behind the animated
background. `--msaa {0,2,4,8}` sets the sample count, clamped to the driver
maximum.

Measured on software GL (llvmpipe) at 1280x720: MSAA 4x costs about 4.5 ms/frame
over MSAA 0 (2.8 ms to 7.3 ms). MSAA and the edge-blur post pass are the two
tunable costs. The per-frame entity mesh build is about 1.5 ms and the one-off
level mesh build about 80 ms.

## Replay fidelity

The bundled npp_attract demos replay through nclone's own decoder, but none of
the 22 currently reach the exit. The demo-input decoder in
`npp_attract_decoder._extract_demo_data` uses a heuristic longest-valid-run
search that drifts from the true frames; the map and tile decode is exact. For a
clean solved video use a scripted `--hold [--jump-period ...]` win, for example
the `open-round-corner` command above, which genuinely reaches the exit.

## Coordinate mapping

The sim is a 2D plane in pixels (tile = 24px), origin top-left, +y down. This
maps to `world_x = sim_x`, `world_y = -sim_y` (flipped for Y-up), with `world_z`
the extruded depth.

## GL under WSLg

Under WSLg, on-screen GLX context creation fails
(`X_GLXCreateContext BadValue`), which breaks GLX-based windows. The working
paths are:

- Window: SDL Wayland backend plus moderngl. This needs `SDL_VIDEODRIVER=wayland`
  and an unversioned `libGL.so` on `LD_LIBRARY_PATH`. When `app.py` detects a
  Linux Wayland session it creates the symlink shim under
  `$XDG_CACHE_HOME/nclone/gl_shims` and re-execs itself once with the right
  environment, so the plain module invocation works. Set
  `NCLONE_RENDER3D_NO_BOOTSTRAP=1` to skip this.
- Headless: `moderngl.create_standalone_context(backend="egl")` needs none of the
  above.

## Files

| File | Responsibility |
|------|----------------|
| `geometry.py` | `MeshBuilder`, tile footprints and extrusion, smooth arc normals, level mesh assembly |
| `scene.py` | per-frame ninja and entity geometry (the sim-reading seam), vectorized instancing, speed ribbon, motion-blur echoes |
| `camera.py` | `OrbitCamera` |
| `postfx.py` | `SceneTarget`: MSAA resolve and edge motion-blur composite |
| `renderer.py` | moderngl programs, mesh slots, cel and background shaders |
| `input_source.py` | `KeyboardSource` / `HoldSource` / `PatternSource` / `ReplaySource` |
| `replay_loader.py` | npp_attract demo file to (map_data, inputs, name) |
| `stack.py` | stacked multi-level mode: `StackDirector` plus window and record drivers |
| `devpanel.py` | `Render3DPanel`, the offscreen view embedded in the 2D dev tool |
| `app.py` | entry point, CLI, window/record/headless loops, GL bootstrap |
