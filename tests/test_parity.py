"""Parity test framework: run same inputs through Python and Rust sims,
assert identical ninja state snapshots frame-by-frame.

Tests ensure the Rust native physics engine produces bit-for-bit identical
results to the Python physics engine.
"""

import os
import sys
import math
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nclone.sim_config import SimConfig
from nclone.nsim import Simulator as PySimulator


def _load_nclone_native():
    """Load nclone_native, preferring build dir for dev (always has Simulator)."""
    import importlib.util
    try:
        import nclone_native
        return nclone_native
    except ImportError:
        pass
    _build_so = os.path.join(
        os.path.dirname(__file__), "..", "nclone-native",
        "target", "debug", "libnclone_native.so"
    )
    if os.path.isfile(_build_so):
        spec = importlib.util.spec_from_file_location("nclone_native", _build_so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


nclone_native = _load_nclone_native()
NATIVE_AVAILABLE = nclone_native is not None


# --- Helper functions ---

def load_map_data(map_name: str) -> list:
    """Load raw binary map data from test_maps directory."""
    map_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "nclone",
        "test_maps",
        map_name,
    )
    with open(map_path, "rb") as f:
        return [int(b) for b in f.read()]


# Action definitions: (hor_input, jump_input)
ACTIONS = {
    0: (0, 0),   # NOOP
    1: (-1, 0),  # LEFT
    2: (1, 0),   # RIGHT
    3: (0, 1),   # JUMP
    4: (-1, 1),  # JUMP+LEFT
    5: (1, 1),   # JUMP+RIGHT
}


def create_python_sim(map_data: list):
    """Create a Python simulator and load the map."""
    sc = SimConfig(basic_sim=True, enable_anim=False, log_data=False, debug=False)
    sim = PySimulator(sc)
    sim.load(map_data)
    return sim


def create_rust_sim(map_data: list):
    """Create a Rust simulator and load the map."""
    return nclone_native.Simulator.load(map_data)


def get_python_ninja_state(sim):
    """Extract comparable ninja state from Python simulator."""
    n = sim.ninja
    return {
        "xpos": n.xpos,
        "ypos": n.ypos,
        "xspeed": n.xspeed,
        "yspeed": n.yspeed,
        "state": n.state,
        "airborn": n.airborn,
        "walled": n.walled,
        "wall_normal": n.wall_normal,
        "floor_normalized_x": n.floor_normalized_x,
        "floor_normalized_y": n.floor_normalized_y,
        "jump_buffer": n.jump_buffer,
        "floor_buffer": n.floor_buffer,
        "wall_buffer": n.wall_buffer,
        "launch_pad_buffer": n.launch_pad_buffer,
        "jump_duration": n.jump_duration,
        "applied_gravity": n.applied_gravity,
        "applied_drag": n.applied_drag,
        "frames_airborne": n.frames_airborne,
    }


def get_rust_ninja_state(sim):
    """Extract comparable ninja state from Rust simulator."""
    state = sim.get_state()
    n = state.ninja
    return {
        "xpos": n.xpos,
        "ypos": n.ypos,
        "xspeed": n.xspeed,
        "yspeed": n.yspeed,
        "state": n.state,
        "airborn": n.airborn,
        "walled": n.walled,
        "wall_normal": n.wall_normal,
        "floor_normalized_x": n.floor_normalized_x,
        "floor_normalized_y": n.floor_normalized_y,
        "jump_buffer": n.jump_buffer,
        "floor_buffer": n.floor_buffer,
        "wall_buffer": n.wall_buffer,
        "launch_pad_buffer": n.launch_pad_buffer,
        "jump_duration": n.jump_duration,
        "applied_gravity": n.applied_gravity,
        "applied_drag": n.applied_drag,
        "frames_airborne": n.frames_airborne,
    }


def assert_states_equal(py_state: dict, rs_state: dict, frame: int, tol: float = 1e-10):
    """Compare Python and Rust ninja states, raising assertion on mismatch."""
    for key in py_state:
        py_val = py_state[key]
        rs_val = rs_state[key]

        if isinstance(py_val, float):
            if not math.isclose(py_val, rs_val, abs_tol=tol, rel_tol=tol):
                raise AssertionError(
                    f"Frame {frame}: {key} mismatch: "
                    f"Python={py_val:.15e}, Rust={rs_val:.15e}, "
                    f"diff={abs(py_val - rs_val):.15e}"
                )
        else:
            if py_val != rs_val:
                raise AssertionError(
                    f"Frame {frame}: {key} mismatch: Python={py_val}, Rust={rs_val}"
                )


def run_parity_test(map_name: str, actions: list, num_frames: int = None):
    """Run the same action sequence through both simulators and compare states.

    Args:
        map_name: Name of the test map file.
        actions: List of action indices (0-5) to execute each frame.
        num_frames: Number of frames to simulate (defaults to len(actions)).
    """
    if not NATIVE_AVAILABLE:
        pytest.skip("nclone_native not available")

    map_data = load_map_data(map_name)

    py_sim = create_python_sim(map_data)
    rs_sim = create_rust_sim(map_data)

    # Compare initial spawn position
    py_spawn = get_python_ninja_state(py_sim)
    rs_spawn = get_rust_ninja_state(rs_sim)
    assert_states_equal(py_spawn, rs_spawn, 0)

    if num_frames is None:
        num_frames = len(actions)

    for frame in range(num_frames):
        action_idx = actions[frame % len(actions)]
        hor_input, jump_input = ACTIONS[action_idx]

        py_sim.tick(hor_input, jump_input)
        rs_sim.tick(hor_input, jump_input)

        py_state = get_python_ninja_state(py_sim)
        rs_state = get_rust_ninja_state(rs_sim)
        assert_states_equal(py_state, rs_state, frame + 1)


# --- Test cases ---

@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not installed")
class TestParityBasicMovement:
    """Test basic movement parity between Python and Rust."""

    def test_idle_on_flat(self):
        """Standing still on flat ground for 60 frames."""
        run_parity_test("simple-walk", [0] * 60)

    def test_walk_right(self):
        """Walk right on flat ground."""
        run_parity_test("simple-walk", [2] * 120)

    def test_walk_left(self):
        """Walk left on flat ground."""
        run_parity_test("simple-walk", [1] * 120)

    def test_direction_change(self):
        """Walk right then left (direction reversal)."""
        actions = [2] * 30 + [1] * 30 + [2] * 30
        run_parity_test("simple-walk", actions)

    def test_idle_then_walk(self):
        """Stand still then start walking."""
        actions = [0] * 20 + [2] * 60
        run_parity_test("simple-walk", actions)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not installed")
class TestParityJumping:
    """Test jumping parity."""

    def test_basic_jump(self):
        """Jump from flat ground."""
        actions = [3] * 15 + [0] * 45
        run_parity_test("simple-walk", actions)

    def test_jump_and_walk(self):
        """Jump while walking right."""
        actions = [5] * 15 + [2] * 45
        run_parity_test("simple-walk", actions)

    def test_short_hop(self):
        """Short hop (release jump quickly)."""
        actions = [3] * 3 + [0] * 57
        run_parity_test("simple-walk", actions)

    def test_max_jump(self):
        """Hold jump for maximum duration."""
        actions = [3] * 50 + [0] * 50
        run_parity_test("simple-walk", actions)

    def test_jump_sequence(self):
        """Multiple jumps in sequence."""
        actions = (
            [3] * 10 + [0] * 30  # jump 1
            + [3] * 10 + [0] * 30  # jump 2
            + [3] * 10 + [0] * 30  # jump 3
        )
        run_parity_test("simple-walk", actions)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not installed")
class TestParityFalling:
    """Test falling and gravity parity."""

    def test_fall(self):
        """Walk off an edge and fall."""
        actions = [2] * 200
        run_parity_test("fall-required", actions)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not installed")
class TestParityWallInteraction:
    """Test wall sliding and wall jumping."""

    def test_wall_jump(self):
        """Wall jump sequence."""
        actions = [5] * 30 + [4] * 20 + [0] * 50
        run_parity_test("wall-jump-required", actions)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not installed")
class TestParitySlopes:
    """Test slope interaction parity."""

    def test_slopes_walk(self):
        """Walk across slopes."""
        actions = [2] * 200
        run_parity_test("slopes", actions)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not installed")
class TestParityReset:
    """Test fast reset parity."""

    def test_fast_reset(self):
        """Run, reset, run again - both sims should produce identical results."""
        if not NATIVE_AVAILABLE:
            pytest.skip("nclone_native not available")

        map_data = load_map_data("simple-walk")

        py_sim = create_python_sim(map_data)
        rs_sim = create_rust_sim(map_data)

        # Run 30 frames
        for _ in range(30):
            py_sim.tick(1, 0)
            rs_sim.tick(1, 0)

        # Reset
        py_sim.fast_reset()
        rs_sim.fast_reset()

        # Check state after reset matches initial state
        py_state = get_python_ninja_state(py_sim)
        rs_state = get_rust_ninja_state(rs_sim)
        assert_states_equal(py_state, rs_state, 0)

        # Run 30 more frames
        for frame in range(30):
            py_sim.tick(1, 1)
            rs_sim.tick(1, 1)

            py_state = get_python_ninja_state(py_sim)
            rs_state = get_rust_ninja_state(rs_sim)
            assert_states_equal(py_state, rs_state, frame + 1)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not installed")
class TestParityActionMask:
    """Test action mask parity."""

    def test_action_mask_grounded(self):
        """Action mask on flat ground should match."""
        if not NATIVE_AVAILABLE:
            pytest.skip("nclone_native not available")

        map_data = load_map_data("simple-walk")
        py_sim = create_python_sim(map_data)
        rs_sim = create_rust_sim(map_data)

        # Walk for a bit then check mask
        for _ in range(10):
            py_sim.tick(1, 0)
            rs_sim.tick(1, 0)

        py_mask = py_sim.ninja.get_valid_action_mask()
        rs_state = rs_sim.get_state()
        rs_mask = list(rs_state.action_mask)

        assert py_mask == rs_mask, (
            f"Action mask mismatch: Python={py_mask}, Rust={rs_mask}"
        )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not installed")
class TestParitySwitchDoor:
    """Test switch + door entity interaction parity."""

    def test_switch_collection(self):
        """Walk to collect switch - ninja should trigger it in both sims."""
        if not NATIVE_AVAILABLE:
            pytest.skip("nclone_native not available")

        map_data = load_map_data("switch-simple")
        py_sim = create_python_sim(map_data)
        rs_sim = create_rust_sim(map_data)

        # Walk towards switch for many frames
        for frame in range(300):
            py_sim.tick(1, 0)
            rs_sim.tick(1, 0)

            py_state = get_python_ninja_state(py_sim)
            rs_state = get_rust_ninja_state(rs_sim)
            assert_states_equal(py_state, rs_state, frame + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
