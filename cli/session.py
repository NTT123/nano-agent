"""Session persistence for the CLI application."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from nano_agent import DAG

# Auto-save session file in current directory
SESSION_FILE = ".nano-cli-session.json"


class SessionStore:
    """Save/load session DAGs with optional auto-save."""

    def __init__(self, session_file: str = SESSION_FILE) -> None:
        self.session_file = session_file

    async def auto_save(self, dag: DAG | None) -> None:
        """Silently auto-save the current DAG if available."""
        if not dag or not dag._heads:
            return
        try:
            filepath = Path.cwd() / self.session_file
            loop = asyncio.get_event_loop()
            session_id = datetime.now().isoformat()
            await loop.run_in_executor(
                None, lambda: dag.save(filepath, session_id=session_id)
            )
        except Exception:
            # Silently ignore save errors to not interrupt the user
            pass

    async def save(self, dag: DAG | None, filename: str) -> Path | None:
        """Save a DAG to disk and return the resolved filepath."""
        if not dag or not dag._heads:
            return None
        if not filename.endswith(".json"):
            filename += ".json"
        filepath = Path(filename).resolve()
        loop = asyncio.get_event_loop()
        session_id = datetime.now().isoformat()
        await loop.run_in_executor(
            None, lambda: dag.save(filepath, session_id=session_id)
        )
        return filepath

    async def load(self, filename: str) -> tuple[DAG, dict[str, Any], Path]:
        """Load a DAG from disk and return (dag, metadata, filepath)."""
        if not filename.endswith(".json"):
            filename += ".json"
        filepath = Path(filename).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        loop = asyncio.get_event_loop()
        loaded_dag, metadata = await loop.run_in_executor(
            None, lambda: DAG.load(filepath)
        )
        return loaded_dag, metadata, filepath
