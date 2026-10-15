"""Export schemas without starting RAG, opening Chroma or reading provider keys."""

import json
from pathlib import Path

from src.web_api.app import create_app
from src.web_api.settings import WebSettings

destination = Path("frontend/openapi.json")
destination.write_text(json.dumps(create_app(WebSettings()).openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
