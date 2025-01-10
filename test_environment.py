import pygame
import numpy as np
from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

# Initialize pygame
pygame.init()
pygame.display.set_caption("N++ Environment Test")

# Create environment
env = BasicLevelNoGold(render_mode='human',
                       enable_frame_stack=False, enable_debug_overlay=True)

# Initialize clock for 60 FPS
clock = pygame.time.Clock()
running = True

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

    # print(f'Gold collected: {env.get_gold_collected()}')

    # Maintain 60 FPS
    clock.tick(60)

# Cleanup
pygame.quit()
env.close()
