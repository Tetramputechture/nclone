"""
Gameplay environment integration for the N++ MCP server.

This module provides tools for playing and testing N++ levels using
the integrated gameplay environment.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from PIL import Image

from fastmcp import FastMCP
from ..map_generation.map import Map

from . import map_operations

logger = logging.getLogger(__name__)

# Global variable to hold current gameplay environment
current_gameplay_env = None


def register_gameplay_tools(mcp: FastMCP) -> None:
    """Register all gameplay environment tools with the FastMCP server."""

    @mcp.tool()
    async def initialize_gameplay(
        headless: bool = True,
        enable_debug_overlay: bool = False,
        seed: int = 42,
        use_current_map: bool = True,
        custom_map_path: Optional[str] = None,
    ) -> str:
        """Initialize the N++ gameplay environment for playing a level.

        Args:
            headless: Run in headless mode (no GUI display)
            enable_debug_overlay: Enable debug visualization overlays
            seed: Random seed for environment initialization
            use_current_map: Use the currently loaded map (if available)
            custom_map_path: Path to custom map file to load instead

        Returns:
            Status of gameplay initialization
        """
        try:
            global current_gameplay_env
            current_map = map_operations.current_map

            # Import here to avoid circular dependencies
            try:
                from ..gym_environment.npp_environment import NppEnvironment
            except ImportError:
                return "✗ Failed to import NppEnvironment. Make sure gym_environment is available."

            # Determine map to use
            map_to_use = None
            if custom_map_path:
                # Load custom map
                try:
                    file_path = Path(custom_map_path)
                    if not file_path.exists():
                        return f"✗ Custom map file not found: {custom_map_path}"

                    with open(file_path, "rb") as f:
                        data = f.read()

                    if len(data) % 4 != 0:
                        return f"✗ Invalid map file format. File size {len(data)} is not divisible by 4"

                    map_data = []
                    for i in range(0, len(data), 4):
                        value = int.from_bytes(
                            data[i : i + 4], byteorder="little", signed=True
                        )
                        map_data.append(value)

                    map_operations.current_map = Map.from_map_data(map_data)
                    map_to_use = custom_map_path

                except Exception as e:
                    return f"✗ Failed to load custom map: {str(e)}"

            elif use_current_map and current_map is not None:
                # Use current map - save it temporarily
                with tempfile.NamedTemporaryFile(
                    suffix=".map", delete=False
                ) as tmp_file:
                    map_data = current_map.map_data()
                    for value in map_data:
                        tmp_file.write(
                            value.to_bytes(4, byteorder="little", signed=True)
                        )
                    map_to_use = tmp_file.name
            else:
                return "✗ No map available. Either load a map first or provide custom_map_path."

            # Close existing environment if any
            if current_gameplay_env is not None:
                try:
                    current_gameplay_env.close()
                except Exception:
                    pass

            # Initialize environment
            render_mode = "rgb_array" if headless else "human"
            current_gameplay_env = NppEnvironment(
                render_mode=render_mode,
                enable_frame_stack=False,
                enable_debug_overlay=enable_debug_overlay and not headless,
                eval_mode=False,
                seed=seed,
                custom_map_path=map_to_use,
            )

            # Reset environment to initialize
            observation, info = current_gameplay_env.reset()

            # Get basic info about initialized environment
            ninja_pos = "unknown"
            try:
                if hasattr(current_gameplay_env, "nplay_headless") and hasattr(
                    current_gameplay_env.nplay_headless, "ninja_position"
                ):
                    pos = current_gameplay_env.nplay_headless.ninja_position()
                    ninja_pos = f"({pos[0]:.1f}, {pos[1]:.1f})"
                elif hasattr(current_gameplay_env, "sim") and hasattr(
                    current_gameplay_env.sim, "ninja"
                ):
                    ninja_pos = f"({current_gameplay_env.sim.ninja.x:.1f}, {current_gameplay_env.sim.ninja.y:.1f})"
            except Exception:
                pass

            # Clean up temporary file if created
            if (
                use_current_map
                and current_map is not None
                and map_to_use
                and "/tmp/" in map_to_use
            ):
                try:
                    os.unlink(map_to_use)
                except Exception:
                    pass

            mode_str = "headless" if headless else "windowed"
            debug_str = " with debug overlay" if enable_debug_overlay else ""
            map_source = (
                "current map" if use_current_map else f"custom map ({custom_map_path})"
            )

            return f"✓ Initialized N++ gameplay environment in {mode_str} mode{debug_str}. Using {map_source}. Ninja at {ninja_pos}. Use step_environment() to play."

        except Exception as e:
            logger.error(f"Error initializing gameplay: {e}")
            return f"✗ Failed to initialize gameplay: {str(e)}"

    @mcp.tool()
    async def step_environment(
        action: Literal[
            "noop", "left", "right", "jump", "jump_left", "jump_right"
        ] = "noop",
        num_steps: int = 1,
    ) -> str:
        """Step the gameplay environment with the specified action.

        Args:
            action: Action to perform in the environment
                - "noop": No action (0)
                - "left": Move left (1)
                - "right": Move right (2)
                - "jump": Jump only (3)
                - "jump_left": Jump + Move left (4)
                - "jump_right": Jump + Move right (5)
            num_steps: Number of steps to execute (1-60, represents frames)

        Returns:
            Status and information about the environment step
        """
        try:
            global current_gameplay_env

            if current_gameplay_env is None:
                return "✗ No gameplay environment initialized. Call initialize_gameplay() first."

            if num_steps < 1 or num_steps > 60:
                return "✗ num_steps must be between 1 and 60"

            # Map action strings to environment action integers
            action_mapping = {
                "noop": 0,
                "left": 1,
                "right": 2,
                "jump": 3,
                "jump_left": 4,
                "jump_right": 5,
            }

            if action not in action_mapping:
                available_actions = ", ".join(action_mapping.keys())
                return f"✗ Invalid action '{action}'. Available actions: {available_actions}"

            action_id = action_mapping[action]

            # Execute the specified number of steps
            total_reward = 0.0
            final_info = {}
            steps_executed = 0

            for step in range(num_steps):
                observation, reward, terminated, truncated, info = (
                    current_gameplay_env.step(action_id)
                )
                total_reward += reward
                final_info = info
                steps_executed += 1

                # If episode ended, break early
                if terminated or truncated:
                    break

            # Get current ninja position
            ninja_pos = "unknown"
            try:
                if hasattr(current_gameplay_env, "nplay_headless") and hasattr(
                    current_gameplay_env.nplay_headless, "ninja_position"
                ):
                    pos = current_gameplay_env.nplay_headless.ninja_position()
                    ninja_pos = f"({pos[0]:.1f}, {pos[1]:.1f})"
                elif hasattr(current_gameplay_env, "sim") and hasattr(
                    current_gameplay_env.sim, "ninja"
                ):
                    ninja_pos = f"({current_gameplay_env.sim.ninja.x:.1f}, {current_gameplay_env.sim.ninja.y:.1f})"
            except Exception:
                pass

            # Parse episode status
            episode_ended = terminated or truncated
            status_info = ""
            if episode_ended:
                if terminated:
                    if total_reward > 0:
                        status_info = " - LEVEL COMPLETED! 🎉"
                    else:
                        status_info = " - Ninja died ☠️"
                elif truncated:
                    status_info = " - Episode truncated (timeout)"

            # Get additional info from environment
            extra_info = []
            if "ninja_state" in final_info:
                extra_info.append(f"ninja_state: {final_info['ninja_state']}")
            if "gold_collected" in final_info:
                extra_info.append(f"gold_collected: {final_info['gold_collected']}")
            if "time_remaining" in final_info:
                extra_info.append(f"time_remaining: {final_info['time_remaining']:.1f}")

            extra_str = f" ({', '.join(extra_info)})" if extra_info else ""

            return (
                f"✓ Executed {steps_executed} step(s) with action '{action}'. "
                f"Ninja at {ninja_pos}. Total reward: {total_reward:.1f}{status_info}{extra_str}"
            )

        except Exception as e:
            logger.error(f"Error stepping environment: {e}")
            return f"✗ Failed to step environment: {str(e)}"

    @mcp.tool()
    async def export_current_frame(
        filepath: str, include_debug_info: bool = False
    ) -> str:
        """Export the current frame from the gameplay environment to an image file.

        Args:
            filepath: Path where to save the frame image (supports .png, .jpg, .jpeg)
            include_debug_info: Whether to include debug overlays in the export

        Returns:
            Status of frame export operation
        """
        try:
            global current_gameplay_env

            if current_gameplay_env is None:
                return "✗ No gameplay environment initialized. Call initialize_gameplay() first."

            # Get frame from environment
            frame = current_gameplay_env.render()

            if frame is None:
                return "✗ Failed to get frame from environment. Make sure environment is in rgb_array mode."

            # Convert numpy array to PIL Image
            if not isinstance(frame, np.ndarray):
                return f"✗ Frame is not a numpy array: {type(frame)}"

            # Handle different frame formats
            image = None
            if len(frame.shape) == 3:
                if frame.shape[2] == 1:
                    # Single channel (grayscale) - squeeze to 2D
                    frame_2d = np.squeeze(frame, axis=2)
                    image = Image.fromarray(frame_2d.astype(np.uint8), mode="L")
                elif frame.shape[2] == 3:
                    # RGB format
                    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
                elif frame.shape[2] == 4:
                    # RGBA format
                    image = Image.fromarray(frame.astype(np.uint8), mode="RGBA")
                else:
                    return f"✗ Unsupported frame format with {frame.shape[2]} channels"
            elif len(frame.shape) == 2:
                # Already 2D grayscale
                image = Image.fromarray(frame.astype(np.uint8), mode="L")
            else:
                return f"✗ Unsupported frame shape {frame.shape}"

            if image is None:
                return "✗ Failed to convert frame to image"

            # Create directory if it doesn't exist
            file_path = Path(filepath)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save image
            image.save(file_path)

            # Get current ninja position for additional info
            ninja_pos = "unknown"
            try:
                if hasattr(current_gameplay_env, "nplay_headless") and hasattr(
                    current_gameplay_env.nplay_headless, "ninja_position"
                ):
                    pos = current_gameplay_env.nplay_headless.ninja_position()
                    ninja_pos = f"({pos[0]:.1f}, {pos[1]:.1f})"
            except Exception:
                pass

            file_size = file_path.stat().st_size
            return (
                f"✓ Exported frame to {filepath} ({file_size} bytes, {frame.shape}). "
                f"Ninja position: {ninja_pos}"
            )

        except Exception as e:
            logger.error(f"Error exporting frame: {e}")
            return f"✗ Failed to export frame: {str(e)}"

    @mcp.tool()
    async def reset_gameplay() -> str:
        """Reset the current gameplay environment to the initial state.

        Returns:
            Status of environment reset
        """
        try:
            global current_gameplay_env

            if current_gameplay_env is None:
                return "✗ No gameplay environment initialized. Call initialize_gameplay() first."

            # Reset environment
            observation, info = current_gameplay_env.reset()

            # Get ninja position after reset
            ninja_pos = "unknown"
            try:
                if hasattr(current_gameplay_env, "nplay_headless") and hasattr(
                    current_gameplay_env.nplay_headless, "ninja_position"
                ):
                    pos = current_gameplay_env.nplay_headless.ninja_position()
                    ninja_pos = f"({pos[0]:.1f}, {pos[1]:.1f})"
                elif hasattr(current_gameplay_env, "sim") and hasattr(
                    current_gameplay_env.sim, "ninja"
                ):
                    ninja_pos = f"({current_gameplay_env.sim.ninja.x:.1f}, {current_gameplay_env.sim.ninja.y:.1f})"
            except Exception:
                pass

            # Get additional reset info
            extra_info = []
            if "ninja_state" in info:
                extra_info.append(f"ninja_state: {info['ninja_state']}")
            if "time_remaining" in info:
                extra_info.append(f"time_remaining: {info['time_remaining']:.1f}")

            extra_str = f" ({', '.join(extra_info)})" if extra_info else ""

            return f"✓ Reset gameplay environment. Ninja at {ninja_pos}{extra_str}. Ready for new episode."

        except Exception as e:
            logger.error(f"Error resetting gameplay: {e}")
            return f"✗ Failed to reset gameplay: {str(e)}"

    @mcp.tool()
    async def get_gameplay_state() -> str:
        """Get current state information from the gameplay environment.

        Returns:
            Detailed information about the current gameplay state
        """
        try:
            global current_gameplay_env

            if current_gameplay_env is None:
                return "✗ No gameplay environment initialized. Call initialize_gameplay() first."

            # Get ninja position and state
            ninja_pos = "unknown"
            ninja_state = "unknown"
            try:
                if hasattr(current_gameplay_env, "nplay_headless"):
                    if hasattr(current_gameplay_env.nplay_headless, "ninja_position"):
                        pos = current_gameplay_env.nplay_headless.ninja_position()
                        ninja_pos = f"({pos[0]:.1f}, {pos[1]:.1f})"
                    if hasattr(current_gameplay_env.nplay_headless, "ninja_state"):
                        ninja_state = current_gameplay_env.nplay_headless.ninja_state()
                elif hasattr(current_gameplay_env, "sim") and hasattr(
                    current_gameplay_env.sim, "ninja"
                ):
                    ninja = current_gameplay_env.sim.ninja
                    ninja_pos = f"({ninja.x:.1f}, {ninja.y:.1f})"
                    ninja_state = getattr(ninja, "state", "unknown")
            except Exception:
                pass

            # Get environment info
            env_info = {}
            try:
                # Try to get last step info if available
                if (
                    hasattr(current_gameplay_env, "_last_info")
                    and current_gameplay_env._last_info
                ):
                    env_info = current_gameplay_env._last_info
            except Exception:
                pass

            # Get render mode and other environment settings
            render_mode = getattr(current_gameplay_env, "render_mode", "unknown")

            # Format the state information
            state_info = f"""🎮 Gameplay State Information:

🥷 Ninja Status:
  Position: {ninja_pos}
  State: {ninja_state}

🎯 Environment:
  Render mode: {render_mode}
  Frame stack: {getattr(current_gameplay_env, "enable_frame_stack", "unknown")}
  Debug overlay: {getattr(current_gameplay_env, "enable_debug_overlay", "unknown")}"""

            # Add additional environment info if available
            if env_info:
                state_info += "\n\n📊 Last Step Info:"
                for key, value in env_info.items():
                    state_info += f"\n  {key}: {value}"

            # Add action space information
            try:
                action_space = current_gameplay_env.action_space
                state_info += f"\n\n🕹️  Action Space: {action_space.n} actions (0=noop, 1=left, 2=right, 3=jump, 4=jump_left, 5=jump_right)"
            except Exception:
                state_info += "\n\n🕹️  Action Space: Unable to retrieve"

            # Add observation space information
            try:
                obs_space = current_gameplay_env.observation_space
                state_info += f"\n\n👁️  Observation Space: {obs_space.shape if hasattr(obs_space, 'shape') else obs_space}"
            except Exception:
                state_info += "\n\n👁️  Observation Space: Unable to retrieve"

            state_info += "\n\n💡 Available actions: step_environment(), export_current_frame(), reset_gameplay()"

            return state_info

        except Exception as e:
            logger.error(f"Error getting gameplay state: {e}")
            return f"✗ Failed to get gameplay state: {str(e)}"

    @mcp.tool()
    async def toggle_debug_overlay(
        overlay_type: Literal["exploration", "grid"] = "exploration",
    ) -> str:
        """Toggle debug overlays in the gameplay environment.

        Args:
            overlay_type: Type of debug overlay to toggle
                - "exploration": Toggle exploration debug overlay
                - "grid": Toggle grid debug overlay

        Returns:
            Status of debug overlay toggle
        """
        try:
            global current_gameplay_env

            if current_gameplay_env is None:
                return "✗ No gameplay environment initialized. Call initialize_gameplay() first."

            success = False
            message = ""

            try:
                if overlay_type == "exploration":
                    # Try to toggle exploration debug
                    if hasattr(current_gameplay_env, "set_exploration_debug_enabled"):
                        # Toggle by checking current state (if possible) or just enable/disable
                        current_gameplay_env.set_exploration_debug_enabled(
                            True
                        )  # Enable for now
                        success = True
                        message = "exploration debug overlay enabled"
                    else:
                        message = "exploration debug overlay not available in this environment"

                elif overlay_type == "grid":
                    # Try to toggle grid debug
                    if hasattr(current_gameplay_env, "set_grid_debug_enabled"):
                        current_gameplay_env.set_grid_debug_enabled(
                            True
                        )  # Enable for now
                        success = True
                        message = "grid debug overlay enabled"
                    else:
                        message = "grid debug overlay not available in this environment"

            except Exception as e:
                message = f"failed to toggle {overlay_type} overlay: {str(e)}"

            if success:
                return f"✓ Debug overlay toggled - {message}"
            else:
                return f"⚠️  {message}. Note: Debug overlays may only work in windowed (non-headless) mode."

        except Exception as e:
            logger.error(f"Error toggling debug overlay: {e}")
            return f"✗ Failed to toggle debug overlay: {str(e)}"

    @mcp.tool()
    async def close_gameplay() -> str:
        """Close the current gameplay environment and free resources.

        Returns:
            Status of environment closure
        """
        try:
            global current_gameplay_env

            if current_gameplay_env is None:
                return "✓ No gameplay environment was running."

            try:
                current_gameplay_env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")

            current_gameplay_env = None

            return "✓ Gameplay environment closed and resources freed."

        except Exception as e:
            logger.error(f"Error closing gameplay: {e}")
            return f"✗ Failed to close gameplay: {str(e)}"
