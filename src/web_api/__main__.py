"""python -m src.web_api --port 18603 (one worker only)."""

import argparse

import uvicorn

from src.web_api.app import create_app

parser = argparse.ArgumentParser(description="ScholarRAG single-process HTTP service")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=18603)
arguments = parser.parse_args()
uvicorn.run(create_app(), host=arguments.host, port=arguments.port, workers=1)
