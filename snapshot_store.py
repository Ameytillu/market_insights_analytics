"""Local JSON persistence for Lighthouse snapshot trend tracking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from comparator import DEMAND_NUM


SNAPSHOT_DIR = Path("snapshots")


def save_snapshot(df: pd.DataFrame, snapshot_date: datetime | None = None, directory: Path = SNAPSHOT_DIR) -> Path | None:
    """Persist a daily detail snapshot as JSON.

    Args:
        df: Lighthouse daily detail DataFrame.
        snapshot_date: Upload timestamp. Defaults to current local time.
        directory: Directory where JSON files are stored.

    Returns:
        Path to the saved file, or ``None`` when the input has no dates.
    """
    if df.empty or "Date" not in df.columns:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    snap_time = snapshot_date or datetime.now()
    clean = df.copy()
    clean["Date"] = pd.to_datetime(clean["Date"]).dt.strftime("%Y-%m-%d")
    start = clean["Date"].min()
    payload = {
        "snapshot_date": snap_time.isoformat(timespec="seconds"),
        "date_range_start": start,
        "records": clean.where(pd.notna(clean), None).to_dict(orient="records"),
    }
    path = directory / f"snapshot_{start}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_snapshots(directory: Path = SNAPSHOT_DIR) -> list[dict[str, Any]]:
    """Load persisted snapshot payloads.

    Args:
        directory: Directory containing snapshot JSON files.

    Returns:
        List of snapshot payload dictionaries sorted by snapshot date.
    """
    if not directory.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for path in sorted(directory.glob("snapshot_*.json")):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(payloads, key=lambda item: str(item.get("snapshot_date", "")))


def build_trend_dataframe(directory: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    """Build a long-form historical demand trend DataFrame.

    Args:
        directory: Directory containing snapshot JSON files.

    Returns:
        DataFrame with snapshot date, arrival date, demand ordinal, demand string, rate, and compset level.
    """
    rows: list[dict[str, Any]] = []
    for payload in load_snapshots(directory):
        snapshot_date = pd.to_datetime(payload.get("snapshot_date"), errors="coerce")
        for record in payload.get("records", []):
            arrival_date = pd.to_datetime(record.get("Date"), errors="coerce")
            demand = record.get("Demand level")
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "arrival_date": arrival_date,
                    "demand_level_num": DEMAND_NUM.get(str(demand)),
                    "demand_level_str": demand,
                    "my_price": _price_number(record.get("My price")),
                    "compset_level": record.get("Smart Compset price level"),
                }
            )
    columns = [
        "snapshot_date",
        "arrival_date",
        "demand_level_num",
        "demand_level_str",
        "my_price",
        "compset_level",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).dropna(subset=["snapshot_date", "arrival_date"]).sort_values(
        ["snapshot_date", "arrival_date"]
    )


def snapshot_summary(directory: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    """Summarize stored snapshots for sidebar display.

    Args:
        directory: Directory containing snapshot JSON files.

    Returns:
        Dictionary with count and optional date range labels.
    """
    trend = build_trend_dataframe(directory)
    if trend.empty:
        return {"count": 0, "start": None, "end": None}
    return {
        "count": int(trend["snapshot_date"].nunique()),
        "start": trend["snapshot_date"].min(),
        "end": trend["snapshot_date"].max(),
    }


def _price_number(value: Any) -> float | None:
    if isinstance(value, str) and value.strip().lower() == "sold out":
        return None
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)
