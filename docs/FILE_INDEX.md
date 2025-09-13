# File Index

Brief purpose notes for key files in `nclone/` to aid navigation during Phase 1.

- `nclone/constants.py`: Shared numeric constants and enums used across modules.
- `nclone/sim_config.py`: Simulation configuration flags and defaults.
- `nclone/physics.py`: Physics helpers and formulations used by the simulator.
- `nclone/nsim.py`: Core simulation loop, step function, and world state containers.
- `nclone/nsim_renderer.py`: Frame composition from simulation state for display.
- `nclone/render_utils.py`: Utility routines for drawing primitives and text.
- `nclone/tile_definitions.py`: Collision segments and geometry per tile type.
- `nclone/map_loader.py`: Load and parse map files into simulation structures.
- `nclone/maps/`: Static maps content; large but data-only.
- `nclone/map_generation/`: Procedural map generation modules.
- `nclone/map_augmentation/`: Map transforms (e.g., mirroring) for augmentation.
- `nclone/ninja.py`: Player character state machine, input handling, and physics.
- `nclone/entities.py`: Entity classes and interactions (hazards, doors, etc.).
- `nclone/entity_renderer.py`: Sprite- and primitive-based rendering for entities.
- `nclone/tile_renderer.py`: Rendering for tiles and terrain.
- `nclone/nplay.py`: Interactive runner with windowed rendering.
- `nclone/nplay_headless.py`: Headless simulation entrypoint that returns RGB arrays.
- `nclone/run_multiple_headless.py`: Multi-process launcher for headless simulations.
- `nclone/debug_overlay_renderer.py`: On-screen diagnostics overlay controls.
- `nclone/ntrace.py`: Lightweight tracing/profiling helpers.
- `nclone/test_environment.py`: Manual environment smoke test and demo.

Environments:

- `nclone/gym_environment/npp_environment.py`: Gymnasium-compatible environment and main entrypoint.
- `nclone/gym_environment/observation_processor.py`: Build observation dict (images + state vector).
- `nclone/gym_environment/constants.py`: Env-specific constants.
- `nclone/gym_environment/reward_calculation/*`: Reward shaping modules.
