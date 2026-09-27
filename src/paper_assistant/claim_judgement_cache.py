"""Content-addressed cache for Claim-Evidence judge verdicts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.paper_assistant.chunk_retriever import ChunkSearchResult
from src.paper_assistant.grounded_answer import GroundedCitation, GroundedClaim

CLAIM_JUDGE_PROMPT_VERSION = "claim-evidence-v1"


@dataclass(frozen=True)
class CachedClaimJudgement:
    supported: bool
    actual_model: str | None


class ClaimJudgementCache:
    """Store verdicts without persisting claim or paper text."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS claim_judgements (
                    cache_key TEXT PRIMARY KEY,
                    requested_model TEXT NOT NULL,
                    actual_model TEXT,
                    supported INTEGER NOT NULL CHECK (supported IN (0, 1)),
                    prompt_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0)

    @staticmethod
    def build_key(
        claim: GroundedClaim,
        citations_by_id: dict[str, GroundedCitation],
        evidence_by_chunk: dict[str, ChunkSearchResult],
        *,
        requested_model: str,
        service_namespace: str = "default",
        disable_thinking: bool = False,
        prompt_version: str = CLAIM_JUDGE_PROMPT_VERSION,
    ) -> str:
        canonical_by_chunk: dict[str, dict[str, str | int]] = {}
        for citation_id in set(claim.citations):
            citation = citations_by_id[citation_id]
            result = evidence_by_chunk[citation.chunk_id]
            canonical_by_chunk[citation.chunk_id] = {
                "chunk_id": citation.chunk_id,
                "paper_id": citation.paper_id,
                "page_number": citation.page_number,
                "section": citation.section,
                "text_sha256": hashlib.sha256(
                    result.text.encode("utf-8")
                ).hexdigest(),
            }
        evidence = [canonical_by_chunk[key] for key in sorted(canonical_by_chunk)]
        canonical = json.dumps(
            {
                "claim": " ".join(claim.text.split()),
                "evidence": evidence,
                "requested_model": requested_model,
                "service_namespace": service_namespace,
                "thinking_mode": (
                    "disabled" if disable_thinking else "provider_default"
                ),
                "prompt_version": prompt_version,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> CachedClaimJudgement | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT supported, actual_model FROM claim_judgements "
                "WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return CachedClaimJudgement(bool(row[0]), row[1])

    def put(
        self,
        cache_key: str,
        *,
        requested_model: str,
        actual_model: str | None,
        supported: bool,
        prompt_version: str = CLAIM_JUDGE_PROMPT_VERSION,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO claim_judgements (
                    cache_key, requested_model, actual_model, supported,
                    prompt_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    actual_model = excluded.actual_model,
                    supported = excluded.supported,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    requested_model,
                    actual_model,
                    int(supported),
                    prompt_version,
                    datetime.now(UTC).isoformat(),
                ),
            )


def open_claim_judgement_cache(
    path: str | Path,
) -> tuple[ClaimJudgementCache | None, str | None]:
    """Open the optional cache, falling back to live judging on local failures."""
    try:
        return ClaimJudgementCache(path), None
    except (OSError, sqlite3.Error) as error:
        return None, f"cache_unavailable:{type(error).__name__}"
