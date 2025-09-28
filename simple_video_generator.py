#!/usr/bin/env python3
"""
Simple Video Generator for nclone JSONL Replay Files

This tool generates video output from JSONL replay files created by the binary_replay_parser.
It uses the nclone simulator and pygame to render frames and create an MP4 video.
"""

import argparse
import json
import logging
import pygame
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from nclone.nsim import Simulator
from nclone.sim_config import SimConfig
from nclone.nsim_renderer import NSimRenderer

logger = logging.getLogger(__name__)


class SimpleVideoGenerator:
    """Generate video from JSONL replay files."""
    
    def __init__(self, width: int = 1056, height: int = 600, fps: int = 60):
        """
        Initialize the video generator.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels  
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("nclone Replay")
        
        # Initialize simulator and renderer
        sim_config = SimConfig(enable_anim=True)
        self.sim = Simulator(sim_config)
        self.renderer = NSimRenderer(self.sim, self.screen)
        
    def load_jsonl_frames(self, jsonl_file: Path) -> List[Dict[str, Any]]:
        """
        Load frames from JSONL file.
        
        Args:
            jsonl_file: Path to JSONL file
            
        Returns:
            List of frame dictionaries
        """
        frames = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                frame = json.loads(line.strip())
                frames.append(frame)
        
        logger.info(f"Loaded {len(frames)} frames from {jsonl_file}")
        return frames
        
    def setup_simulation(self, first_frame: Dict[str, Any]) -> bool:
        """
        Set up the simulation based on the first frame.
        
        Args:
            first_frame: First frame data
            
        Returns:
            True if setup successful
        """
        try:
            # Create a simple default map for visualization
            # This is a basic map that allows the ninja to move around
            map_data = self._create_simple_map()
            self.sim.load(map_data)
            
            # Set initial ninja position from frame data
            player_state = first_frame.get('player_state', {})
            position = player_state.get('position', {'x': 100, 'y': 400})
            
            self.sim.ninja.xpos = float(position['x'])
            self.sim.ninja.ypos = float(position['y'])
            
            logger.info(f"Simulation setup complete. Ninja at ({self.sim.ninja.xpos}, {self.sim.ninja.ypos})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup simulation: {e}")
            return False
    
    def _create_simple_map(self) -> List[int]:
        """Create a simple map for visualization."""
        # Create basic map structure (similar to the parser's default map)
        map_data = [0] * 184  # Header
        
        # Basic metadata
        map_data[0:4] = [6, 0, 0, 0]
        map_data[4:8] = [221, 4, 0, 0]
        
        # Tile data (42x23 = 966 bytes)
        tile_data = []
        for y in range(23):
            for x in range(42):
                if x == 0 or x == 41 or y == 0 or y == 22:
                    tile_data.append(1)  # Wall
                elif y == 21:
                    tile_data.append(1)  # Floor
                else:
                    tile_data.append(0)  # Empty
        
        map_data.extend(tile_data)
        
        # Pad to minimum size
        while len(map_data) < 1245:
            map_data.append(0)
            
        return map_data
        
    def generate_video(self, frames: List[Dict[str, Any]], output_file: Path) -> bool:
        """
        Generate video from frame data.
        
        Args:
            frames: List of frame dictionaries
            output_file: Output video file path
            
        Returns:
            True if video generation successful
        """
        if not frames:
            logger.error("No frames to process")
            return False
            
        # Setup simulation
        if not self.setup_simulation(frames[0]):
            return False
            
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_file), fourcc, self.fps, (self.width, self.height)
        )
        
        logger.info(f"Generating video with {len(frames)} frames...")
        
        try:
            for i, frame in enumerate(frames):
                if i % 60 == 0:  # Log progress every second
                    logger.info(f"Processing frame {i}/{len(frames)}")
                
                # Update ninja state from frame data
                self._update_ninja_from_frame(frame)
                
                # Render frame
                self.screen.fill((0, 0, 0))  # Clear screen
                self.renderer.render()
                pygame.display.flip()
                
                # Convert pygame surface to numpy array
                frame_array = pygame.surfarray.array3d(self.screen)
                frame_array = np.transpose(frame_array, (1, 0, 2))  # Correct orientation
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)  # Convert to BGR
                
                # Write frame to video
                video_writer.write(frame_array)
                
        except Exception as e:
            logger.error(f"Error during video generation: {e}")
            return False
        finally:
            video_writer.release()
            
        logger.info(f"Video saved to {output_file}")
        return True
        
    def _update_ninja_from_frame(self, frame: Dict[str, Any]):
        """Update ninja state from frame data."""
        player_state = frame.get('player_state', {})
        
        # Update position
        position = player_state.get('position', {})
        if 'x' in position:
            self.sim.ninja.xpos = float(position['x'])
        if 'y' in position:
            self.sim.ninja.ypos = float(position['y'])
            
        # Update velocity
        velocity = player_state.get('velocity', {})
        if 'x' in velocity:
            self.sim.ninja.xspeed = float(velocity['x'])
        if 'y' in velocity:
            self.sim.ninja.yspeed = float(velocity['y'])
            
        # Update state based on ground/wall info
        on_ground = player_state.get('on_ground', False)
        wall_sliding = player_state.get('wall_sliding', False)
        
        if wall_sliding:
            self.sim.ninja.state = 5  # Wall sliding
        elif on_ground:
            self.sim.ninja.state = 0  # On ground
        else:
            self.sim.ninja.state = 3  # In air
            
    def cleanup(self):
        """Clean up resources."""
        pygame.quit()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate video from nclone JSONL replay files"
    )
    
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output MP4 video file"
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="Video framerate (default: 60)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Validate input
    if not args.input.exists():
        logger.error(f"Input file does not exist: {args.input}")
        return 1
        
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate video
    generator = SimpleVideoGenerator(fps=args.fps)
    
    try:
        frames = generator.load_jsonl_frames(args.input)
        success = generator.generate_video(frames, args.output)
        
        if success:
            logger.info("Video generation completed successfully!")
            return 0
        else:
            logger.error("Video generation failed")
            return 1
            
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return 1
    finally:
        generator.cleanup()


if __name__ == "__main__":
    exit(main())