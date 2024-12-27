import cProfile
import pstats
import io
import os
from nsim import Simulator
from nsim_renderer import NSimRenderer
from sim_config import init_sim_config
from map import MapGenerator
import random
from datetime import datetime


def ensure_benchmark_dir():
    """Ensure benchmark directories exist."""
    os.makedirs('benchmark', exist_ok=True)
    os.makedirs('benchmark/renderer', exist_ok=True)


def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_stats(stats, filename):
    """Save stats to a file."""
    with open(f'benchmark/{filename}', 'w') as f:
        s = io.StringIO()
        stats.stream = s
        stats.print_stats()
        f.write(s.getvalue())


def benchmark_map_generation(num_maps=100):
    """Benchmark the map generation process."""
    print("Running map generation benchmark...")
    map_generator = MapGenerator()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(num_maps):
        map_generator.generate_random_map("SIMPLE_HORIZONTAL_NO_BACKTRACK")

    profiler.disable()

    timestamp = get_timestamp()
    stats = pstats.Stats(profiler)

    # Save cumulative stats
    save_stats(
        stats.sort_stats('cumulative'),
        f'map_generation_cumulative_{timestamp}.txt'
    )

    return stats


def run_simulation_benchmark(num_frames=1000):
    """Run a benchmark of the simulation for a specified number of frames."""
    print("Running simulation benchmark...")
    sim_config = init_sim_config()
    simulator = Simulator(sim_config)
    map_generator = MapGenerator()

    map_generator.generate_random_map("SIMPLE_HORIZONTAL_NO_BACKTRACK")
    test_map_data = map_generator.generate()
    simulator.load(test_map_data)

    profiler = cProfile.Profile()
    profiler.enable()

    for frame_num in range(num_frames):
        hor_input = random.choice([-1, 0, 1])
        jump_input = random.choice([0, 1])
        simulator.tick(hor_input, jump_input)

    profiler.disable()

    timestamp = get_timestamp()
    stats = pstats.Stats(profiler)

    # Save cumulative and time stats
    save_stats(
        stats.sort_stats('cumulative'),
        f'simulation_cumulative_{timestamp}.txt'
    )
    save_stats(
        stats.sort_stats('time'),
        f'simulation_time_{timestamp}.txt'
    )

    return stats


def analyze_specific_method(method_name, num_frames=1000):
    """Analyze a specific method's performance in detail."""
    print(f"Analyzing method: {method_name}...")
    sim_config = init_sim_config()
    simulator = Simulator(sim_config)

    map_generator = MapGenerator()
    map_generator.generate_random_map("SIMPLE_HORIZONTAL_NO_BACKTRACK")
    test_map_data = map_generator.generate()
    simulator.load(test_map_data)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(num_frames):
        hor_input = random.choice([-1, 0, 1])
        jump_input = random.choice([0, 1])
        simulator.tick(hor_input, jump_input)

    profiler.disable()

    timestamp = get_timestamp()
    stats = pstats.Stats(profiler)

    # Save method-specific analysis
    with open(f'benchmark/method_{method_name}_{timestamp}.txt', 'w') as f:
        s = io.StringIO()
        stats.stream = s
        stats.print_callees(method_name)
        f.write(s.getvalue())


def benchmark_renderer(num_frames=1000):
    """Run detailed benchmarks of the rendering pipeline."""
    print("Running renderer benchmarks...")
    sim_config = init_sim_config()
    simulator = Simulator(sim_config)
    renderer = NSimRenderer(simulator)

    map_generator = MapGenerator()
    map_generator.generate_random_map("SIMPLE_HORIZONTAL_NO_BACKTRACK")
    test_map_data = map_generator.generate()
    simulator.load(test_map_data)

    # Profile full render pipeline
    profiler = cProfile.Profile()
    profiler.enable()

    for frame_num in range(num_frames):
        hor_input = random.choice([-1, 0, 1])
        jump_input = random.choice([0, 1])
        simulator.tick(hor_input, jump_input)
        renderer.draw(frame_num == 0)  # Only init on first frame

    profiler.disable()
    timestamp = get_timestamp()
    stats = pstats.Stats(profiler)

    # Save detailed rendering stats
    save_stats(
        stats.sort_stats('cumulative'),
        f'renderer/full_render_cumulative_{timestamp}.txt'
    )
    save_stats(
        stats.sort_stats('time'),
        f'renderer/full_render_time_{timestamp}.txt'
    )

    # Profile collision map rendering
    profiler = cProfile.Profile()
    profiler.enable()

    for frame_num in range(num_frames):
        renderer.draw_collision_map(frame_num == 0)

    profiler.disable()
    stats = pstats.Stats(profiler)

    save_stats(
        stats.sort_stats('cumulative'),
        f'renderer/collision_map_cumulative_{timestamp}.txt'
    )
    save_stats(
        stats.sort_stats('time'),
        f'renderer/collision_map_time_{timestamp}.txt'
    )

    # Profile individual render components
    render_methods = [
        '_update_screen_size',
        '_update_tile_offsets',
        '_draw_tiles',
        '_draw_entities'
    ]

    for method in render_methods:
        print(f"Profiling render method: {method}")
        profiler = cProfile.Profile()
        profiler.enable()

        for frame_num in range(num_frames):
            if method == '_draw_tiles':
                getattr(renderer, method)(frame_num == 0)
            elif method == '_draw_entities':
                getattr(renderer, method)(frame_num == 0)
            else:
                getattr(renderer, method)()

        profiler.disable()
        stats = pstats.Stats(profiler)

        save_stats(
            stats.sort_stats('cumulative'),
            f'renderer/method_{method}_{timestamp}.txt'
        )

    return stats


def main():
    ensure_benchmark_dir()
    print("Starting benchmarks...")

    benchmark_map_generation(100)
    run_simulation_benchmark(1000)
    benchmark_renderer(1000)  # Add renderer benchmark

    important_methods = [
        'tick',
        'move',
        'think',
        'integrate',
        'pre_collision',
        'collide_vs_objects',
        'collide_vs_tiles',
        'post_collision',
        'update_graphics'
    ]

    for method in important_methods:
        analyze_specific_method(method)

    print("Benchmarks complete. Results saved in benchmark/ directory.")


if __name__ == "__main__":
    main()
