"""Generate prioritized pricing and demand alerts from Lighthouse snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from comparator import DEMAND_ORDER, LEVEL_NUM, compare_snapshots


SEVERITY_RANK = {"critical": 0, "warning": 1, "opportunity": 2, "info": 3}
UNDERPRICED_LEVELS = {"Very low", "Low", "Normal"}
HIGH_DEMAND_LEVELS = {"High", "Very high"}
ELEVATED_DEMAND_LEVELS = {"Elevated", "High", "Very high"}


def generate_alerts(df_today: pd.DataFrame, df_yesterday: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate alert dictionaries for future dates.

    Args:
        df_today: Parsed Daily details dataframe for the current snapshot.
        df_yesterday: Parsed Daily details dataframe for the prior snapshot.

    Returns:
        Prioritized list of alert dictionaries with severity, date, title, and body.
    """
    comparison = compare_snapshots(df_today, df_yesterday)
    today = pd.Timestamp(datetime.today().date())
    future = comparison[comparison["Date"] >= today].copy()
    alerts: list[dict[str, Any]] = []

    for _, row in future.iterrows():
        demand = row.get("Demand_today")
        prior_demand = row.get("Demand_yest")
        demand_delta = row.get("Demand_delta")
        my_level = row.get("MyLevel_today")
        compset_level = row.get("CompsetLevel_today")
        compset_gap = row.get("Compset_gap")
        compset_delta = row.get("CompsetLevel_delta")
        price_today = row.get("Price_today")
        price_yest = row.get("Price_yest")
        price_delta = row.get("Price_delta")
        flight_signal = row.get("FlightMeta_today")
        events = int(row.get("Events", 0))

        if pd.notna(demand_delta) and demand_delta >= 2:
            alerts.append(
                _alert(
                    "critical",
                    row,
                    f"Demand surged {int(demand_delta)} levels overnight",
                    f"Demand moved from {prior_demand} to {demand}. Current rate is {_money(price_today)} at a {my_level} price level.",
                )
            )

        if (
            pd.notna(compset_gap)
            and compset_gap >= 2
            and demand in ELEVATED_DEMAND_LEVELS
            and not _is_sold_out(price_today)
        ):
            alerts.append(
                _alert(
                    "critical",
                    row,
                    "Compset is priced well above you",
                    f"Your level is {my_level} while the compset is {compset_level} on a {demand} demand date.",
                )
            )

        if demand in HIGH_DEMAND_LEVELS and my_level in UNDERPRICED_LEVELS and not _is_sold_out(price_today):
            alerts.append(
                _alert(
                    "warning",
                    row,
                    f"{demand} demand with {my_level} rate level",
                    f"Published rate is {_money(price_today)} and compset level is {compset_level}. Review whether the rate should move up.",
                )
            )

        if pd.notna(demand_delta) and demand_delta <= -2:
            alerts.append(
                _alert(
                    "warning",
                    row,
                    f"Demand fell {abs(int(demand_delta))} levels overnight",
                    f"Demand moved from {prior_demand} to {demand}. Current rate is {_money(price_today)} at a {my_level} price level.",
                )
            )

        if _is_sold_out(price_today) and not _is_sold_out(price_yest):
            alerts.append(
                _alert(
                    "warning",
                    row,
                    "Date newly flipped to Sold Out",
                    f"The date moved from {_money(price_yest)} to Sold out. Validate whether the date closed at the right rate.",
                )
            )

        if (
            pd.notna(compset_delta)
            and compset_delta >= 1
            and demand in ELEVATED_DEMAND_LEVELS
            and not _is_sold_out(price_today)
        ):
            alerts.append(
                _alert(
                    "opportunity",
                    row,
                    "Compset raised rates",
                    f"Compset moved up {int(compset_delta)} level(s) to {compset_level}. Your rate is {_money(price_today)} at {my_level}.",
                )
            )

        if flight_signal == "Higher" and my_level in UNDERPRICED_LEVELS and not _is_sold_out(price_today):
            alerts.append(
                _alert(
                    "opportunity",
                    row,
                    "Strong flight search with underpriced rate",
                    f"Flight search is above benchmark while your price level is {my_level}. Current rate is {_money(price_today)}.",
                )
            )

        if events >= 3 and my_level in UNDERPRICED_LEVELS and not _is_sold_out(price_today):
            alerts.append(
                _alert(
                    "opportunity",
                    row,
                    f"{events} events on an underpriced date",
                    f"Demand is {demand}, rate is {_money(price_today)}, and your price level is {my_level}.",
                )
            )

        if pd.notna(price_delta) and abs(float(price_delta)) >= 10:
            direction = "increased" if price_delta > 0 else "decreased"
            alerts.append(
                _alert(
                    "info",
                    row,
                    f"Published rate {direction} {_money(abs(float(price_delta)))}",
                    f"Rate changed from {_money(price_yest)} to {_money(price_today)}. Demand is {demand}; compset level is {compset_level}.",
                )
            )

    return sorted(_dedupe_alerts(alerts), key=lambda alert: (SEVERITY_RANK[alert["severity"]], alert["date"]))


def _alert(severity: str, row: pd.Series, title: str, body: str) -> dict[str, Any]:
    """Create a normalized alert dictionary."""
    date_value = pd.Timestamp(row["Date"])
    return {
        "severity": severity,
        "date": date_value,
        "date_str": date_value.strftime("%a, %b %d"),
        "title": title,
        "body": body,
    }


def _dedupe_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove exact duplicate alert titles per date."""
    seen: set[tuple[pd.Timestamp, str]] = set()
    deduped: list[dict[str, Any]] = []
    for alert in alerts:
        key = (alert["date"], alert["title"])
        if key not in seen:
            seen.add(key)
            deduped.append(alert)
    return deduped


def _money(value: Any) -> str:
    """Format a rate as whole dollars."""
    if _is_sold_out(value):
        return "Sold out"
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "unavailable"
    return f"${float(number):,.0f}"


def _is_sold_out(value: Any) -> bool:
    """Return whether a cell carries the Sold out sentinel."""
    return isinstance(value, str) and value.strip().lower() == "sold out"
