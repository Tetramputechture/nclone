# Surface Graph Edge Enumeration Bug

## Finding

**Verdict: H2 confirmed.** Exit goal nodes exist (105 nodes at pts 530–542, all within 18px of the exit door at (900, 84)) but have **zero incoming AND outgoing edges**. The entire right half of the level's floor is an isolated island in the edge graph.

## Root Cause

`nclone/value_function/edge_enumeration.py` lines 34–39 define incorrect state code constants:

```python
# Current (WRONG) — state 1 is Running, state 2 is GroundSlide (both GROUND states)
_STATE_GROUNDED, _STATE_JUMPING, _STATE_FALLING = 0, 1, 2
_AIRBORNE_STATES = {_STATE_JUMPING, _STATE_FALLING}  # = {1, 2}  ← actually ground states!
_SURFACE_STATES = {_STATE_GROUNDED, _STATE_WALL_SLIDING}  # = {0, 5}  ← missing running/slide
```

Actual Rust simulator state codes (`nclone-native/src/ninja.rs`):

| Code | Meaning      | Category         |
|------|--------------|------------------|
| 0    | Immobile     | Ground (surface) |
| 1    | Running      | Ground (surface) |
| 2    | GroundSlide  | Ground (surface) |
| 3    | Jumping      | Airborne         |
| 4    | Falling      | Airborne         |
| 5    | WallSliding  | Surface          |
| 6    | Dead         | Terminal         |
| 7    | AwaitingDeath| Terminal         |

The names `_STATE_JUMPING = 1` and `_STATE_FALLING = 2` are completely wrong. The actual jumping state is 3 and falling is 4.

## Cascading Failures

### Run arcs generate no edges

`_simulate_arc` (used for surface runs):

1. Start at state 0 (immobile)  
2. First tick with `h_input` → state 1 (running) → **treated as "airborne"** → `was_airborne = True`  
3. Ninja continues running (state 1) for all `max_frames` ticks  
4. State 1 is not in `_SURFACE_STATES = {0, 5}` → the "landed" condition never fires  
5. Loop times out → returns `None` → **no edge created**

**Result:** Every run arc (11 durations × 3 h_inputs = 33 attempts per node) generates zero edges on flat floor, which is the primary traversal surface for most N++ levels.

### Jump arcs miss running/sliding landings

`_simulate_jump_arc` detects landings via `if state_code in _SURFACE_STATES`. With `_SURFACE_STATES = {0, 5}`, landings in state 1 (running — the typical landing state when the ninja has nonzero horizontal velocity) and state 2 (groundslide) are **not detected**. Only landings in state 0 (immobile, very low velocity) or 5 (wall slide) produce edges.

This is why 91,678 edges exist but are clustered around jump-to-wall/stop scenarios, and the flat floor areas reachable by running are completely disconnected.

## Evidence from Diagnostic Run

```
Check 2: Incoming/Outgoing Edge Count for Exit Goal Nodes
  Total edges in graph: 91,678
  Incoming to exit goal nodes: 0
  Outgoing from exit goal nodes: 0

Check 6: Graph Connectivity Analysis
  Nodes with at least one edge: 3,989 / 10,275
  Connected components: 1
  Exit goal nodes with zero edges: 105 / 105
```

Only 3,989 of 10,275 nodes have any edges (38%). The rest are isolated. The single connected component contains 3,989 nodes — all reached exclusively via jump-to-stop or jump-to-wall transitions.

Full state transition sequence from the basics replay (526 frames):

```
frame   state         x       y    vx     vy    transition
    7   immobile   396.0  158.0  0.00   0.00    falling -> immobile   [spawn landing, state 0 ✓]
   13   running    396.0  158.0 -0.07   0.00    immobile -> running
  149   jumping     46.6  150.6  0.47  -3.03    running -> jumping
  187   falling     91.1   58.0  1.86  -0.00    jumping -> falling
  217   running    159.6   86.0  2.76   0.00    falling -> running     [jump 1 landing, state 1 ✗]
  304   jumping    441.4   84.0  3.33  -2.00    running -> jumping
  307   falling    451.3   78.1  3.31  -1.93    jumping -> falling
  353   running    602.9   86.0  3.31   0.00    falling -> running     [jump 2 landing, state 1 ✗]
  458   groundslide 947.1  86.0  3.26   0.00    running -> groundslide
  480   immobile   977.1   86.0  0.05   0.00    groundslide -> immobile [state 0 ✓]
  525   (won)      919.5   86.0 -2.49   0.00    running -> state8
```

The two real jumps (frames 149–217 and 304–353) **both land in state 1 (running)**. With `_SURFACE_STATES = {0, 5}`, both landings are completely invisible to `_simulate_jump_arc`'s detection loop. These are the exact edges that would connect the left/mid floor to the right floor where the exit lives — and both are silently dropped.

The final approach to the exit (frames 353–525) is pure running from x=602 to x=919. Run arcs are also broken so none of this corridor is represented in the graph either.

## Required Fix

### 1. Correct the state code constants

```python
# Correct state codes matching nclone-native/src/ninja.rs
_STATE_IMMOBILE    = 0   # ground, not moving
_STATE_RUNNING     = 1   # ground, running
_STATE_GROUNDSLIDE = 2   # ground, decelerating
_STATE_JUMPING     = 3   # airborne, jump held
_STATE_FALLING     = 4   # airborne, falling
_STATE_WALL_SLIDING = 5
_STATE_DEAD         = 6
_STATE_AWAITING_DEATH = 7

_DEATH_STATES   = {_STATE_DEAD, _STATE_AWAITING_DEATH}
_AIRBORNE_STATES = {_STATE_JUMPING, _STATE_FALLING}     # = {3, 4}
_SURFACE_STATES  = {_STATE_IMMOBILE, _STATE_RUNNING, _STATE_GROUNDSLIDE, _STATE_WALL_SLIDING}  # = {0,1,2,5}
```

### 2. Fix `_simulate_arc` for run arcs

With the corrected `_SURFACE_STATES = {0, 1, 2, 5}`, the current fallback:
```python
if not was_airborne and state_code in _SURFACE_STATES and frame > 0:
    return ...
```
would fire on frame 1 (ninja is running = state 1 ∈ {0,1,2,5}), returning a 1-frame trivial edge every time. The fallback must be removed and replaced with a post-loop return:

```python
def _simulate_arc(sim, h_input, jump_input, max_frames, collect_waypoints=True):
    was_airborne = False
    for frame in range(1, max_frames + 1):
        sim.tick_physics_only(h_input, jump_input)
        state_code = sim.get_ninja_state_code()
        if state_code in _DEATH_STATES:
            return None, frame, waypoints
        pos = sim.get_ninja_pos(); vel = sim.get_ninja_velocity()
        if collect_waypoints and frame % ARC_WAYPOINT_INTERVAL == 0:
            waypoints.append((frame, pos[0], pos[1], vel[0], vel[1]))
        if state_code in _AIRBORNE_STATES:
            was_airborne = True
        elif was_airborne and state_code in _SURFACE_STATES:
            return (pos[0], pos[1], vel[0], vel[1], state_code), frame, waypoints
    # For run arcs: return final position after max_frames
    state_code = sim.get_ninja_state_code()
    pos = sim.get_ninja_pos(); vel = sim.get_ninja_velocity()
    if state_code in _SURFACE_STATES:
        return (pos[0], pos[1], vel[0], vel[1], state_code), max_frames, waypoints
    return None, max_frames, waypoints
```

### 3. Fix `_simulate_jump_arc` to detect running landings

In the Phase 2 fall loop, change:
```python
if state_code in _SURFACE_STATES:   # was {0,5}, now {0,1,2,5}
    return ...
```
With `_SURFACE_STATES` corrected to `{0, 1, 2, 5}`, this is automatic.

## Expected Impact After Fix

- Run arcs on flat floor will generate edges between floor nodes at distances corresponding to `RUN_DURATIONS` (1–60 frames of running)
- Jump arcs will generate edges for all landings, including the common case of landing with nonzero horizontal velocity (state 1)
- Exit and switch goal nodes will acquire both incoming and outgoing edges
- `d_max_exit` should become nonzero for the basics level

## Files to Modify

- `nclone/value_function/edge_enumeration.py` — fix state code constants and `_simulate_arc`
- Rerun `nclone/value_function/precompute_cli.py` for all levels after the fix
