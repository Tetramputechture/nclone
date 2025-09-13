"""
N++ Map Creation MCP Server Package

This package provides a comprehensive MCP server for creating, editing, and testing N++ levels.
It has been organized into focused modules for better maintainability and code organization.

Modules:
- constants: Type mappings and constants
- map_operations: Basic map creation and manipulation tools
- file_operations: Import/export functionality
- building_tools: Advanced building operations (slopes, corridors, platforms, etc.)
- templates: Pre-designed structures and pattern creation
- analysis: Map connectivity analysis and validation
- gameplay: Gameplay environment integration for testing levels
- server: Main server entry point

Usage:
    To run the server, use:
    python -m nclone.mcp_server.server

    Or import and run programmatically:
    from nclone.mcp_server.server import run_server
    run_server()
"""

from .server import run_server

__version__ = "1.0.0"
__author__ = "nclone team"

__all__ = ["run_server"]
