"""Public, credential-free HTTP contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryRequest(Model):
    query: str = Field(min_length=1, max_length=2000)
    paper_ids: list[str] = Field(default_factory=list, max_length=3)
    dataset_id: str = Field(min_length=1, max_length=64)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()

    @field_validator("paper_ids")
    @classmethod
    def unique_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class PaperFile(Model):
    file_id: str
    name: str
    url: str


class Paper(Model):
    paper_id: str
    title: str
    canonical_title: str
    authors: list[str]
    year: str | None
    venue: str
    tags: list[str]
    summary: str
    files: list[PaperFile]


class Citation(Model):
    citation_id: str
    paper_id: str
    file_id: str
    paper_title: str
    page_number: int = Field(ge=1)
    section: str
    chunk_id: str
    excerpt: str


class Claim(Model):
    text: str
    citations: list[str]


class Verification(Model):
    status: Literal["verified", "skipped", "unavailable"]


class AnswerResult(Model):
    status: Literal["answered", "refused", "evidence_only"]
    claims: list[Claim]
    citations: list[Citation]
    reason: str | None
    verification: Verification
    source: Literal["api"] = "api"


class SearchResult(Model):
    papers: list[Paper]
    rejected: bool
    reason: str | None


class ApiError(Model):
    code: str
    message: str
    request_id: str


class Job(Model):
    job_id: str
    dataset_id: str
    kind: Literal["answer", "search"]
    state: Literal["queued", "running", "completed", "failed", "queue_timeout", "interrupted"]
    result: AnswerResult | SearchResult | None = None
    error: ApiError | None = None


class Health(Model):
    ready: bool
    dataset_id: str
    label: str
    papers_count: int
    verification_mode: str
    session_id: str


class PaperList(Model):
    dataset_id: str
    papers: list[Paper]
