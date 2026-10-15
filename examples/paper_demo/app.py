"""Run from the repository root: python -m streamlit run examples/paper_demo/app.py."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "paper_demo"
# Always isolate data and write paths, even if a production shell has env vars set.
os.environ.update(
    SCHOLARRAG_CATALOG=str(DEMO / "paper_catalog.csv"),
    SCHOLARRAG_INBOX=str(DEMO / "pdfs"),
    SCHOLARRAG_CHROMA_PATH=str(ROOT / "tmp" / "paper_demo" / "chroma"),
    SCHOLARRAG_CLAIM_CACHE=str(ROOT / "tmp" / "paper_demo" / "claims.sqlite3"),
    SCHOLARRAG_REQUEST_LOG=str(ROOT / "tmp" / "paper_demo" / "requests.jsonl"),
    SCHOLARRAG_PAPER_RERANKER="none",
    SCHOLARRAG_CHUNK_RERANKER="none",
    SCHOLARRAG_CLAIM_VERIFICATION="single",
)
os.environ.setdefault("SCHOLARRAG_SETTINGS", str(DEMO / "settings.yaml"))

import streamlit as st  # noqa: E402

from src.observability.dashboard.pages.paper_assistant import render  # noqa: E402

st.set_page_config(page_title="ScholarRAG | Public demo", page_icon="📚", layout="wide")
st.info(
    "公开演示库：3 份人工合成文档，内容不代表真实论文。问答会调用你配置的 API；离线检查请运行 scripts.check_paper_demo。"
)
render()
