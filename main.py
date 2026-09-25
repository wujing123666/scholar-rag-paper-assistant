"""Compatibility entry point for the Modular RAG MCP server."""

import sys
from src.mcp_server.server import main


if __name__ == "__main__":
    sys.exit(main())
