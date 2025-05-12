"""MCP server package initialization"""

from agentic_diffusion_mcp.config import load_config
from agentic_diffusion_mcp.server.app import create_mcp_server

# Create server instance with default configuration
server = create_mcp_server(load_config())

__all__ = ["server", "create_mcp_server"]
