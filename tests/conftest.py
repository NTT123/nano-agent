"""Pytest configuration for local test runs."""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest


def _add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_repo_root_to_path()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "asyncio: mark async tests to run in an event loop"
    )


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool:
    test_func = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_func):
        funcargs = {
            name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames
        }
        asyncio.run(test_func(**funcargs))
        return True
    return False
