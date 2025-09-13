#!/usr/bin/env python3
"""
N++ Map Creation MCP Server

This server provides tools for creating, importing, and exporting N++ levels using the MCP protocol.
It integrates with the existing nclone map generation system to provide comprehensive level creation
capabilities for LLM-assisted game development.

The server has been refactored into focused modules for improved maintainability:
- map_operations: Basic map creation and manipulation
- file_operations: Import/export functionality
- building_tools: Advanced building operations
- templates: Pre-designed structures and patterns
- analysis: Map analysis and validation
- gameplay: Gameplay environment integration
"""

import logging

from fastmcp import FastMCP

# Import all registration functions from our modules
from .map_operations import register_map_operations
from .file_operations import register_file_operations
from .building_tools import register_building_tools
from .templates import register_template_tools
from .analysis import register_analysis_tools
from .gameplay import register_gameplay_tools

# Configure logging to stderr (required for MCP servers using stdio)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # stderr by default
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("nclone-level-creator")


def register_all_tools():
    """Register all MCP tools from all modules."""
    logger.info("Registering MCP tools...")

    # Register tools from each module
    register_map_operations(mcp)
    logger.info("✓ Registered map operations tools")

    register_file_operations(mcp)
    logger.info("✓ Registered file operations tools")

    register_building_tools(mcp)
    logger.info("✓ Registered building tools")

    register_template_tools(mcp)
    logger.info("✓ Registered template tools")

    register_analysis_tools(mcp)
    logger.info("✓ Registered analysis tools")

    register_gameplay_tools(mcp)
    logger.info("✓ Registered gameplay tools")

    logger.info("All tools registered successfully")


def run_server():
    """Run the MCP server."""
    logger.info("Starting N++ Level Creator MCP Server")

    # Register all tools
    register_all_tools()

    # Run the server
    logger.info("Server ready - listening on stdio transport")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
