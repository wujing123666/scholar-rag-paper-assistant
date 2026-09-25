"""Process-wide synchronization for ChromaDB client creation."""

from threading import RLock


# ChromaDB's shared-system registry can race when several MCP tool calls create
# PersistentClient instances for the same path at once.
CHROMA_CLIENT_LOCK = RLock()
