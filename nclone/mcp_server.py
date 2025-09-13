#!/usr/bin/env python3
"""
N++ Map Creation MCP Server

This server provides tools for creating, importing, and exporting N++ levels using the MCP protocol.
It integrates with the existing nclone map generation system to provide comprehensive level creation
capabilities for LLM-assisted game development.

REFACTORED: This file now serves as a backward compatibility shim that redirects to the new
modular mcp_server package structure for improved maintainability and code organization.

For the actual implementation, see the mcp_server package:
- mcp_server/constants.py: Type mappings and constants
- mcp_server/map_operations.py: Basic map operations
- mcp_server/file_operations.py: Import/export functionality
- mcp_server/building_tools.py: Advanced building operations
- mcp_server/templates.py: Template and pattern creation
- mcp_server/analysis.py: Map analysis and validation
- mcp_server/gameplay.py: Gameplay environment integration
- mcp_server/server.py: Main server implementation
"""

import logging

# Redirect to the new modular implementation
from .mcp_server.server import run_server

# Configure logging to stderr (required for MCP servers using stdio)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # stderr by default
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Run the MCP server via the new modular implementation."""
    logger.info("Starting MCP server via backward compatibility shim")
    logger.info("Redirecting to modular mcp_server package...")
    run_server()
