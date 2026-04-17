## 'basics' Level Description: Hairpin Corridor

### Overall Geometry

The level occupies the standard 42×23 tile grid (1056×600 pixels at 24px per cell). The playable geometry forms a **hairpin / U-shaped corridor** consisting of two horizontal corridors stacked vertically, connected by open space on the left side of the map.

### Platform Structure

The **upper (right) corridor** is a long horizontal platform spanning from roughly the left-center of the map to the far-right edge, approximately 2 cells (48px) tall in interior clearance. Its right end is capped by a solid wall. This corridor sits at roughly rows 3–6 from the top of the level.

The **lower corridor** runs underneath and to the left, also 2 cells (48px) tall in interior clearance, spanning from the far-left edge to roughly the center-right of the map. The ninja spawns at the right end of this corridor. The floor of the lower corridor and the ceiling of the upper corridor share a solid tile divider between them — this is the "middle platform" that forms the ceiling of the lower path and the floor of the upper path.

On the **left side of the map**, there is a gap of 3 empty tiles (72px) of vertical clearance between the left end of the lower corridor's ceiling and the left end of the upper corridor's floor. This open space is the transition zone where the ninja must jump from the lower level up to the upper level. The left wall of the level boundary provides a surface for wall-jumping.

### Ninja Spawn

The ninja spawns at the **right end of the lower corridor**, standing on the floor — approximately at the point where the lower corridor meets open air to the right.

### Entity Placement

**Exit switch (type 4, blue circle):** Located on the upper corridor near its far-right end, close to the right wall cap. This is the Phase 0 objective — the ninja must touch it to activate the exit door.

**Exit door (type 3, gray circle):** Located on the upper corridor slightly to the left of the exit switch. This is the Phase 1 objective.

**Three toggle mines (type 1, dark red dots):**

- **Mine 1:** Positioned on the **left wall of the leftmost solid tile of the middle platform** (the divider between the two corridors). This mine sits at the left edge of the shared solid structure, in the transition zone. The agent ascending from the lower corridor must navigate around it when jumping up and to the right.

- **Mine 2:** Positioned **on the surface in the middle of the middle platform**, directly on the floor of the upper corridor. Because it sits on the continuous running surface of the upper corridor, the agent must make **vertical clearance over this mine with a jump** while traversing the upper corridor toward the exit switch. There is enough ceiling clearance in the 48px-tall corridor for the ninja (radius 10px) to hop over the mine's collision radius.

- **Mine 3:** Positioned on the left wall of the rightmost wall at the end of the upper corridor, already in a toggled (deadly) state. Toggled-on mines cannot have their state changed back — this mine is a permanent hazard. It is not on the optimal goal path; the exit switch and exit door are both located to its left. The danger is that an agent running rightward at high speed along the upper corridor may overshoot past the exit objectives and contact Mine 3, resulting in death. The agent must brake if it has rightward momentum and reverse direction leftward toward the exit switch/door before reaching this mine.

### Required Gameplay Sequence (Optimal Route)

1. **Phase 0 — Spawn to exit switch:**
   - From spawn, the ninja runs **left** along the lower corridor toward the left side of the map.
   - At the left edge, the ninja uses the **left boundary wall to wall-jump**, launching upward and to the right into the 3-tile vertical gap between the corridors.
   - The ninja lands on the upper corridor floor, navigating past **Mine 1** at the left edge of the middle platform during the ascent.
   - The ninja then runs **right along the entire upper corridor**.
   - Mid-way, the ninja must **jump to clear Mine 2** on the floor surface, maintaining forward momentum.
   - After clearing Mine 2, the ninja continues rightward along the upper corridor, approaching and activating the exit switch on the ground.

2. **Phase 1 — Exit switch to exit door:**
   - Near the right end of the upper corridor, the ninja must decelerate and stop before reaching Mine 3 at the far-right wall. The exit switch is located to the left of Mine 3, so the agent collects the switch without needing to reach the wall. Overshooting into Mine 3 is lethal and unrecoverable.
   - After collecting the switch, the ninja moves a short distance **back to the left** to reach the now-active **exit door**. This phase is trivially short.

### Key Characteristics for Validation

**Hairpin routing with non-trivial A\* path.** Despite the exit being to the right, the ninja must first travel left from spawn, ascend via wall-jump, and then travel right. The `path_requires_detour` reachability feature should be active at spawn, and `next_hop_direction` should point left initially. This makes the level a strong test for graph-based path guidance over naive Euclidean direction-following.

**Mine interaction complexity.** The three mines create distinct challenges: Mine 1 tests navigation during the vertical transition, Mine 2 tests mid-run jumping on a flat surface, and Mine 3 tests wall-adjacent hazard management where jumping over is not an option. The `deadly_mine_ratio` observation will change dynamically based on how the agent contacts these mines.

**Low platforming difficulty, moderate hazard planning.** The geometry itself is simple (flat corridors, one wall-jump), but the mines require deliberate routing decisions. An agent that blindly runs at max speed will likely toggle mines into deadly states.

**Expected completion time.** The total path length covers the lower corridor (~15 tiles left), the wall-jump transition, and the upper corridor (~25+ tiles right), plus the short backtrack to the exit door. At near-max horizontal speed (~3.333 px/frame) with acceleration and the jump maneuvers, an efficient agent should complete the level in roughly 400–600 frames (7–10 seconds real time).

**Simple surface graph.** Two main horizontal floor surfaces (upper and lower corridors), wall surfaces on the left boundary and right cap, and the middle platform edges. The precomputed graph should have a low node count, and Dijkstra should complete near-instantly.

**Mine 3 as a braking test.** Mine 3 is permanently deadly and acts as a hard boundary that punishes agents running at full speed without deceleration planning. Because the drag coefficient is high (~0.993) and ground friction alone decelerates slowly, an agent approaching at max horizontal speed (~3.333 px/frame) needs to begin active braking (holding the opposite input direction) several tiles in advance. This tests whether the agent has learned momentum management — a core N++ skill. The mine SDF gradient features should provide a strong avoidance signal as the agent approaches.