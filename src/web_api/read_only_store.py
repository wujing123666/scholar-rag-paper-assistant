"""A Chroma adapter that never creates or modifies collections."""

from typing import Any

import chromadb
from chromadb.config import Settings

from src.core.settings import resolve_path
from src.libs.vector_store.chroma_lock import CHROMA_CLIENT_LOCK
from src.libs.vector_store.chroma_store import ChromaStore


class ReadOnlyChromaStore(ChromaStore):
    def __init__(self, **kwargs: Any) -> None:
        path = resolve_path(kwargs.get("persist_directory", ""))
        if kwargs.get("mode", "local") != "local" or not (path / "chroma.sqlite3").is_file():
            raise ValueError("Web queries require an existing local index")
        super().__init__(**kwargs)

    def _ensure_open(self) -> None:
        if self.client is not None and self.collection is not None:
            return
        with CHROMA_CLIENT_LOCK:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        self.collection = self.client.get_collection(self.collection_name)

    def _deny_write(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Index writes are disabled in the web query service")

    upsert = _deny_write
    delete = _deny_write
    clear = _deny_write
    delete_by_metadata = _deny_write
    begin_rebuild = _deny_write
    upsert_rebuild = _deny_write
    commit_rebuild = _deny_write
    abort_rebuild = _deny_write
