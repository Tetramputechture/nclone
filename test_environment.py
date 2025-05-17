import pygame
from .nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
import argparse
import time

# Initialize pygame
pygame.init()
pygame.display.set_caption("N++ Environment Test")

# Argument parser
parser = argparse.ArgumentParser(description="Test N++ environment with frametime logging.")
parser.add_argument('--log-frametimes', action='store_true', help='Enable frametime logging to stdout.')
args = parser.parse_args()

# Create environment
env = BasicLevelNoGold(render_mode='human',
                       enable_frame_stack=False, enable_debug_overlay=True, eval_mode=False)

# Initialize clock for 60 FPS
clock = pygame.time.Clock()
running = True
last_time = time.perf_counter()

# Main game loop
while running:
    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                # Reset environment
                observation, info = env.reset()

    # Get keyboard state
    keys = pygame.key.get_pressed()
    # observation, info = env.reset()
    # Map keyboard inputs to environment actions
    action = 0  # Default to NOOP
    if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = 4  # Jump + Left
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = 5  # Jump + Right
        else:
            action = 3  # Jump only
    else:
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = 1  # Left
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = 2  # Right

    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Reset if episode is done
    if terminated or truncated:
        observation, info = env.reset()

    # Print observation shape
    # print(observation['game_state'].shape)

    # print(f'Gold collected: {env.get_gold_collected()}')

    current_time = time.perf_counter()
    if args.log_frametimes:
        frame_time_ms = (current_time - last_time) * 1000
        print(f"Frametime: {frame_time_ms:.2f} ms")
    last_time = current_time

# Cleanup
pygame.quit()
env.close()
