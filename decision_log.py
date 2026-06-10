"""Local JSON persistence for revenue-management decision logs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DECISION_DIR = Path("decisions")
DECISION_FILE = DECISION_DIR / "log.json"


def load_decisions(path: Path = DECISION_FILE) -> list[dict[str, Any]]:
    """Load logged decisions.

    Args:
        path: JSON log file path.

    Returns:
        List of decision entries.
    """
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def append_decision(entry: dict[str, Any], path: Path = DECISION_FILE) -> None:
    """Append a decision entry to the local JSON log.

    Args:
        entry: Decision fields to persist.
        path: JSON log file path.

    Returns:
        None.
    """
    decisions = load_decisions(path)
    clean = dict(entry)
    clean.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    clean.setdefault("outcome", "")
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions.append(clean)
    path.write_text(json.dumps(decisions, indent=2, default=str), encoding="utf-8")


def update_decision_outcome(index: int, outcome: str, path: Path = DECISION_FILE) -> None:
    """Update the outcome text for one decision.

    Args:
        index: Zero-based entry index in the loaded log.
        outcome: Outcome text to save.
        path: JSON log file path.

    Returns:
        None.
    """
    decisions = load_decisions(path)
    if 0 <= index < len(decisions):
        decisions[index]["outcome"] = outcome
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decisions, indent=2, default=str), encoding="utf-8")


def decisions_dataframe(path: Path = DECISION_FILE) -> pd.DataFrame:
    """Return decisions as a DataFrame for display and export.

    Args:
        path: JSON log file path.

    Returns:
        Decision log DataFrame.
    """
    decisions = load_decisions(path)
    if not decisions:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "arrival_date",
                "action_taken",
                "rate_before",
                "rate_after",
                "demand_at_time",
                "compset_at_time",
                "notes",
                "outcome",
            ]
        )
    return pd.DataFrame(decisions)


def decision_summary(path: Path = DECISION_FILE) -> dict[str, int]:
    """Summarize the decision log.

    Args:
        path: JSON log file path.

    Returns:
        Dictionary containing the number of decisions logged.
    """
    return {"count": len(load_decisions(path))}
