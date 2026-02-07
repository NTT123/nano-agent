"""Configuration loading/saving for nano-cli."""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.nano-cli.json")


def load_cli_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load CLI config from disk. Returns empty dict if not found or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_cli_config(config: dict[str, Any], path: str = DEFAULT_CONFIG_PATH) -> None:
    """Persist CLI config to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
