"""MCP server implementation with Echo tool"""

import asyncio
import click
import sys
from typing import Optional, AsyncIterator

from mcp import types
from mcp.server.fastmcp import FastMCP

from agentic_diffusion_mcp.config import ServerConfig, load_config
from agentic_diffusion_mcp.logging_config import setup_logging, logger
from agentic_diffusion_mcp.tools.echo import echo


def create_mcp_server(config: Optional[ServerConfig] = None) -> FastMCP:
    """Create and configure the MCP server instance"""
    if config is None:
        config = load_config()
    
    # Set up logging first
    setup_logging(config)
    
    server = FastMCP(config.name)

    # Register all tools with the server
    register_tools(server)

    return server


def register_tools(mcp_server: FastMCP) -> None:
    """Register all MCP tools with the server"""

    @mcp_server.tool(
        name="echo",
        description="Echo back the input text with optional case transformation",
    )
    def echo_tool(text: str, transform: Optional[str] = None) -> types.TextContent:
        """Wrapper around the echo tool implementation"""
        return echo(text, transform)



# Create a server instance that can be imported by the MCP CLI
server = create_mcp_server()


@click.command()
@click.option("--port", default=3001, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type (stdio or sse)",
)
def main(port: int, transport: str) -> int:
    """Run the server with specified transport."""
    try:
        if transport == "stdio":
            asyncio.run(server.run_stdio_async())
        else:
            server.settings.port = port
            asyncio.run(server.run_sse_async())
        return 0
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())