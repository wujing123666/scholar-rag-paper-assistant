"""Paper-level dense retrieval for fuzzy semantic descriptions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.paper_assistant.catalog import PaperCatalog, PaperProfile
from src.paper_assistant.retriever import PaperSearchResult

PAPER_RECORD_PREFIX = "paper:"
DEFAULT_SYNC_BATCH_SIZE = 64


def profile_text(profile: PaperProfile) -> str:
    """Build one readable embedding document for one logical paper."""
    fields = (
        ("英文标题", profile.canonical_title),
        ("中文标题", profile.title_zh),
        ("主题标签", "；".join(profile.tags)),
        ("方法", "；".join(profile.method_summaries)),
        ("数据集", "；".join(profile.datasets)),
        ("记忆线索", "；".join(profile.memory_cues)),
        ("作者", "；".join(profile.authors)),
        ("年份", profile.year),
        ("期刊或会议", profile.venue),
    )
    return "\n".join(f"{label}：{value}" for label, value in fields if value)


@dataclass(frozen=True)
class PaperIndexSync:
    """Summary of changes made while aligning Chroma with the catalog."""

    total: int
    embedded: int
    reused: int
    deleted: int
    rebuilt: bool


class PaperDenseRetriever:
    """Rank complete paper profiles using cosine similarity."""

    name = "paper_dense"

    def __init__(
        self,
        catalog: PaperCatalog,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
    ) -> None:
        self.catalog = catalog
        self.embedding = embedding
        self.vector_store = vector_store
        self.embedding_model = str(
            getattr(
                embedding,
                "index_identity",
                getattr(embedding, "model", embedding.__class__.__qualname__),
            )
        )
        self.embedding_dimension = embedding.get_dimension()
        self.index_sync = self._sync_index()

    @staticmethod
    def _record_id(paper_id: str) -> str:
        return f"{PAPER_RECORD_PREFIX}{paper_id}"

    @staticmethod
    def _profile_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _records_for_profiles(
        self,
        items: list[tuple[str, PaperProfile, str, str]],
        vectors: list[list[float]],
    ) -> list[dict]:
        if len(vectors) != len(items):
            raise ValueError("Embedding provider returned an invalid document matrix")
        records = []
        for (record_id, profile, text, content_hash), vector in zip(items, vectors):
            if len(vector) != self.embedding_dimension:
                raise ValueError(
                    "Embedding provider returned a document vector with the wrong dimension"
                )
            records.append(
                {
                    "id": record_id,
                    "vector": vector,
                    "document": text,
                    "metadata": {
                        "record_type": "paper_profile",
                        "paper_id": profile.paper_id,
                        "profile_hash": content_hash,
                        "embedding_model": self.embedding_model,
                        "embedding_dimension": self.embedding_dimension,
                    },
                }
            )
        return records

    def _embed_profiles(
        self, items: list[tuple[str, PaperProfile, str, str]]
    ) -> list[dict]:
        vectors = self.embedding.embed([item[2] for item in items])
        return self._records_for_profiles(items, vectors)

    def _sync_index(self) -> PaperIndexSync:
        profiles = {self._record_id(item.paper_id): item for item in self.catalog.profiles}
        current_ids = set(self.vector_store.list_ids())
        current_records = self.vector_store.get_by_ids(sorted(current_ids)) if current_ids else []
        current = {
            record["id"]: record
            for record in current_records
            if record and record.get("id")
        }

        dimension_rebuild = any(
            record.get("metadata", {}).get("embedding_dimension")
            != self.embedding_dimension
            for record in current.values()
        )
        rebuilt = dimension_rebuild or any(
            record.get("metadata", {}).get("embedding_model") != self.embedding_model
            for record in current.values()
        )
        expected_ids = set(profiles)
        stale_ids = sorted(current_ids - expected_ids)

        changed: list[tuple[str, PaperProfile, str, str]] = []
        for record_id, profile in profiles.items():
            text = profile_text(profile)
            content_hash = self._profile_hash(text)
            metadata = current.get(record_id, {}).get("metadata", {})
            if (
                dimension_rebuild
                or metadata.get("profile_hash") != content_hash
                or metadata.get("embedding_model") != self.embedding_model
                or metadata.get("embedding_dimension") != self.embedding_dimension
            ):
                changed.append((record_id, profile, text, content_hash))

        deleted = 0
        if dimension_rebuild:
            # Chroma collections have a fixed vector dimension. Do not remove the
            # last usable index until every replacement vector has succeeded.
            replacement_records = self._embed_profiles(changed) if changed else []
            self.vector_store.clear()
            deleted = len(current_ids)
            if replacement_records:
                self.vector_store.upsert(replacement_records)
        else:
            # Keep the provider's own request batch size as the persistence
            # checkpoint. A later failure leaves completed batches available for
            # reuse when the same synchronization job is restarted.
            sync_batch_size = int(
                getattr(self.embedding, "batch_size", DEFAULT_SYNC_BATCH_SIZE)
            )
            if sync_batch_size < 1:
                sync_batch_size = DEFAULT_SYNC_BATCH_SIZE
            for start in range(0, len(changed), sync_batch_size):
                batch = changed[start : start + sync_batch_size]
                self.vector_store.upsert(self._embed_profiles(batch))
            if stale_ids:
                self.vector_store.delete(stale_ids)
                deleted += len(stale_ids)

        return PaperIndexSync(
            total=len(profiles),
            embedded=len(changed),
            reused=len(profiles) - len(changed),
            deleted=deleted,
            rebuilt=rebuilt,
        )

    def search(self, query: str, top_k: int = 3) -> list[PaperSearchResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")

        vectors = self.embedding.embed([query], is_query=True)
        if len(vectors) != 1 or len(vectors[0]) != self.embedding_dimension:
            raise ValueError("Embedding provider returned an invalid query vector")
        matches = self.vector_store.query(
            vectors[0], top_k=min(top_k, len(self.catalog))
        )
        results = []
        for match in matches:
            paper_id = str(match.get("metadata", {}).get("paper_id", ""))
            if not paper_id:
                record_id = str(match.get("id", ""))
                if record_id.startswith(PAPER_RECORD_PREFIX):
                    paper_id = record_id[len(PAPER_RECORD_PREFIX) :]
            try:
                paper = self.catalog.get(paper_id)
            except KeyError:
                continue
            results.append(
                PaperSearchResult(
                    paper=paper,
                    score=float(match["score"]),
                    matched_terms=(),
                )
            )
        return results
