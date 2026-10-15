"""Pytest configuration and shared fixtures.

This module contains pytest configuration and fixtures that are shared
across all test modules.
"""

import os
import socket
import sys
from pathlib import Path

import pytest

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def offline_network_guard(monkeypatch):
    """Opt-in CI guard: a unit test must mock every outbound connection."""
    if os.getenv("SCHOLARRAG_TEST_OFFLINE") != "1":
        return

    def guarded(method):
        def connect(sock, address):
            # Windows asyncio uses a loopback TCP socketpair for its wake-up pipe.
            local = (
                sock.family == getattr(socket, "AF_UNIX", None)
                or isinstance(address, tuple)
                and address[0] in {"127.0.0.1", "::1", "localhost"}
            )
            if not local:
                raise RuntimeError("External network disabled for offline unit tests")
            return method(sock, address)
        return connect

    monkeypatch.setattr(socket.socket, "connect", guarded(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guarded(socket.socket.connect_ex))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory path.
    
    Returns:
        Path to the project root directory.
    """
    return PROJECT_ROOT


@pytest.fixture
def sample_documents_dir(project_root: Path) -> Path:
    """Return the sample documents directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the sample documents directory.
    """
    return project_root / "tests" / "fixtures" / "sample_documents"


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the config directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the config directory.
    """
    return project_root / "config"
