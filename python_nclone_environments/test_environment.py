import sys
import os
import pygame

# Add the project root (the directory containing this script and nclone_environments)
# to sys.path. This allows the script to find the nclone_environments package
# when run directly, regardless of the current working directory.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

# Placeholder for the rest of the script
def main():
    print("Initializing Pygame...")
    pygame.init()
    print("Pygame initialized.")
    
    print("Creating environment...")
    try:
        env = BasicLevelNoGold()
        print("Environment created successfully.")
        # Add your environment interaction logic here
        # For example:
        # obs, info = env.reset()
        # for _ in range(100):
        #     action = env.action_space.sample() # Replace with your agent's action
        #     obs, reward, terminated, truncated, info = env.step(action)
        #     env.render()
        #     if terminated or truncated:
        #         break
    except Exception as e:
        print(f"Error creating or using environment: {e}")
    finally:
        print("Quitting Pygame...")
        pygame.quit()
        print("Pygame quit.")

if __name__ == "__main__":
    main()
