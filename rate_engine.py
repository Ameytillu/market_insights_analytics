"""Transient rate recommendation engine for market and DPU signals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from comparator import LEVEL_NUM


DEMAND_MULTIPLIERS = {
    "Low": 0.92,
    "Normal": 1.00,
    "Elevated": 1.06,
    "High": 1.13,
    "Very high": 1.20,
    "Sold out": 1.00,
}


def recommend_rate(date_row: pd.Series, dpu_row: pd.Series | None, config: dict[str, Any]) -> dict[str, Any]:
    """Recommend a transient rate for one arrival date.

    Args:
        date_row: One row from the Lighthouse daily detail DataFrame.
        dpu_row: Matching row from the DPU DataFrame, or ``None`` when unavailable.
        config: Property-level settings including room count, target occupancy, ADR growth, and rate limits.

    Returns:
        Dictionary containing recommendation values and plain-English calculation steps.
    """
    min_rate = float(config.get("min_rate", 150.0))
    max_rate = float(config.get("max_rate", 600.0))
    growth = float(config.get("adr_growth_target", 0.03))
    current_rate = date_row.get("My price")
    current_numeric = _price_number(current_rate)
    step_by_step: list[str] = []

    stly_adr = _dpu_number(dpu_row, "STLY ADR")
    if stly_adr is not None:
        base = stly_adr * (1 + growth)
        step_by_step.append(f"Step 1: STLY ADR {_money(stly_adr)} x {1 + growth:.2f} growth = {_money(base)} base")
    else:
        base = current_numeric if current_numeric is not None else min_rate
        step_by_step.append(f"Step 1: Current published rate {_money(base)} used as the base")

    demand = str(date_row.get("Demand level", "Normal"))
    demand_multiplier = DEMAND_MULTIPLIERS.get(demand, 1.0)
    adjusted = base * demand_multiplier
    step_by_step.append(f"Step 2: {demand} demand x {demand_multiplier:.2f} = {_money(adjusted)}")

    rooms_variance = _rooms_variance(dpu_row)
    pace_adjustment = 0.0
    if rooms_variance is not None:
        if rooms_variance > 10:
            pace_adjustment = 0.04
            pace_text = f"Ahead of pace ({rooms_variance:+.0f} rooms)"
        elif rooms_variance < -10:
            pace_adjustment = -0.04
            pace_text = f"Behind pace ({rooms_variance:+.0f} rooms)"
        else:
            pace_text = f"On pace ({rooms_variance:+.0f} rooms)"
        adjusted *= 1 + pace_adjustment
        step_by_step.append(f"Step 3: {pace_text} x {1 + pace_adjustment:.2f} = {_money(adjusted)}")
    else:
        step_by_step.append("Step 3: No DPU pace data available, no pace adjustment applied")

    compset_level = date_row.get("Smart Compset price level")
    my_level = date_row.get("My price level")
    gap = LEVEL_NUM.get(str(compset_level), np.nan) - LEVEL_NUM.get(str(my_level), np.nan)
    compset_adjustment = 0.0
    if pd.notna(gap) and gap >= 2:
        compset_adjustment = 0.05
        compset_text = "Compset 2+ levels above"
    elif pd.notna(gap) and gap <= -2:
        compset_adjustment = -0.03
        compset_text = "My rate level 2+ levels above compset"
    else:
        compset_text = "Compset positioning close"
    adjusted *= 1 + compset_adjustment
    step_by_step.append(f"Step 4: {compset_text} x {1 + compset_adjustment:.2f} = {_money(adjusted)}")

    bounded = max(min_rate, min(max_rate, adjusted))
    recommended = round(bounded / 5) * 5
    step_by_step.append(f"Step 5: Floor/ceiling then round to nearest $5 = {_money(recommended)}")

    return {
        "recommended_rate": float(recommended),
        "current_rate": "Sold out" if _is_sold_out(current_rate) else current_numeric,
        "rate_delta": None if current_numeric is None else float(recommended - current_numeric),
        "base_used": float(base),
        "demand_multiplier": float(demand_multiplier),
        "pace_adjustment": float(pace_adjustment),
        "compset_adjustment": float(compset_adjustment),
        "step_by_step": step_by_step,
    }


def _dpu_number(dpu_row: pd.Series | None, column: str) -> float | None:
    if dpu_row is None or column not in dpu_row:
        return None
    value = pd.to_numeric(dpu_row.get(column), errors="coerce")
    return None if pd.isna(value) else float(value)


def _rooms_variance(dpu_row: pd.Series | None) -> float | None:
    variance = _dpu_number(dpu_row, "Rooms Variance")
    if variance is not None:
        return variance
    rooms = _dpu_number(dpu_row, "Rooms on Books")
    stly = _dpu_number(dpu_row, "STLY Rooms")
    if rooms is None or stly is None:
        return None
    return rooms - stly


def _price_number(value: Any) -> float | None:
    if _is_sold_out(value):
        return None
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _is_sold_out(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "sold out"


def _money(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "N/A"
    return f"${float(number):,.0f}"
