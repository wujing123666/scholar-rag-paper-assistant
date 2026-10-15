"""Server-only settings; clients cannot supply index or provider paths."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.core.settings import REPO_ROOT
from src.paper_assistant.service import PaperAssistantConfig


def path_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


@dataclass(frozen=True)
class WebSettings:
    rag: PaperAssistantConfig = field(default_factory=PaperAssistantConfig)
    state_path: Path = REPO_ROOT / "data/web"
    label: str = "论文资料库"
    answer_workers: int = 3
    search_workers: int = 8
    queue_size: int = 20
    queue_wait: float = 180
    search_queue_wait: float = 60
    job_limit: int = 2000
    job_ttl_seconds: float = 86400
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:18602", "http://localhost:18602",
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://127.0.0.1:18603", "http://localhost:18603",
    )

    @classmethod
    def from_environment(cls) -> "WebSettings":
        default_llm = ("config/settings.deepseek.local.yaml"
                       if (REPO_ROOT / "config/settings.deepseek.local.yaml").is_file()
                       else "config/settings.yaml")
        rag = PaperAssistantConfig(
            catalog_path=path_env("SCHOLARRAG_CATALOG", "data/papers/paper_catalog.csv"),
            inbox_path=path_env("SCHOLARRAG_INBOX", "data/papers/inbox"),
            chroma_path=path_env("SCHOLARRAG_CHROMA_PATH", "data/db/chroma"),
            settings_path=path_env("SCHOLARRAG_SETTINGS", default_llm),
            embedding_settings_path=path_env("SCHOLARRAG_EMBEDDING_SETTINGS", os.getenv("SCHOLARRAG_SETTINGS", default_llm)),
            claim_cache_path=path_env("SCHOLARRAG_CLAIM_CACHE", "data/web/claims.sqlite3"),
            paper_reranker=os.getenv("SCHOLARRAG_PAPER_RERANKER", "fastembed"),
            chunk_reranker=os.getenv("SCHOLARRAG_CHUNK_RERANKER", "fastembed"),
            verify_claims=os.getenv("SCHOLARRAG_CLAIM_VERIFICATION", "consensus"),
            read_only_indexes=True,
        )
        return cls(
            rag=rag,
            state_path=path_env("SCHOLARRAG_WEB_STATE", "data/web"),
            label=os.getenv("SCHOLARRAG_DATASET_LABEL", "个人论文资料库"),
            answer_workers=int(os.getenv("SCHOLARRAG_ANSWER_WORKERS", "3")),
            search_workers=int(os.getenv("SCHOLARRAG_SEARCH_WORKERS", "8")),
            queue_size=int(os.getenv("SCHOLARRAG_MAX_QUEUE", "20")),
            queue_wait=float(os.getenv("SCHOLARRAG_ANSWER_QUEUE_WAIT_SECONDS", "180")),
            search_queue_wait=float(os.getenv("SCHOLARRAG_SEARCH_QUEUE_WAIT_SECONDS", "60")),
            job_limit=int(os.getenv("SCHOLARRAG_JOB_LIMIT", "2000")),
            job_ttl_seconds=float(os.getenv("SCHOLARRAG_JOB_TTL_SECONDS", "86400")),
            allowed_origins=tuple(filter(None, os.getenv("SCHOLARRAG_WEB_ORIGINS", ",".join(cls.allowed_origins)).split(","))),
        )
