"""Generate prioritized pricing and demand alerts from Lighthouse snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from comparator import LEVEL_NUM, compare_snapshots


SEVERITY_RANK = {"critical": 0, "warning": 1, "opportunity": 2, "info": 3}
UNDERPRICED_LEVELS = {"Very low", "Low", "Normal"}
HIGH_DEMAND_LEVELS = {"High", "Very high"}
ELEVATED_DEMAND_LEVELS = {"Elevated", "High", "Very high"}
DEFAULT_SUPPRESSIONS = {
    "low_demand_weekday_flight": True,
    "compset_min_demand": True,
    "rate_change_threshold_20": True,
    "dedupe_consecutive": True,
    "sold_out_pricing": True,
}


def generate_alerts(
    df_today: pd.DataFrame,
    df_yesterday: pd.DataFrame,
    dpu_df: pd.DataFrame | None = None,
    suppressions: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Generate alert dictionaries for future dates.

    Args:
        df_today: Parsed Daily details dataframe for the current snapshot.
        df_yesterday: Parsed Daily details dataframe for the prior snapshot.
        dpu_df: Optional DPU dataframe indexed by arrival date.
        suppressions: Optional toggle dictionary for property-tuned suppression rules.

    Returns:
        Prioritized list of alert dictionaries with severity, date, title, and body.
    """
    active_suppressions = {**DEFAULT_SUPPRESSIONS, **(suppressions or {})}
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
        dpu_row = _matching_dpu_row(dpu_df, row["Date"])
        sold_out = _is_sold_out(price_today)
        suppress_pricing = bool(active_suppressions["sold_out_pricing"] and sold_out)

        if dpu_row is not None:
            rooms = _number(dpu_row.get("Rooms on Books"))
            stly_rooms = _number(dpu_row.get("STLY Rooms"))
            rooms_variance = _number(dpu_row.get("Rooms Variance"))
            adr = _number(dpu_row.get("ADR on Books"))
            stly_adr = _number(dpu_row.get("STLY ADR"))
            if rooms_variance is None and rooms is not None and stly_rooms is not None:
                rooms_variance = rooms - stly_rooms
            if demand == "Very high" and rooms_variance is not None and rooms_variance < -10:
                alerts.append(
                    _alert(
                        "critical",
                        row,
                        "High demand day but behind pace",
                        (
                            f"Demand is {demand}. Rooms on books are {_whole(rooms)}, STLY rooms are "
                            f"{_whole(stly_rooms)}, variance {_signed_whole(rooms_variance)}."
                        ),
                        context=(demand, compset_level, my_level),
                    )
                )
            if sold_out and adr is not None and stly_adr is not None and adr < stly_adr:
                alerts.append(
                    _alert(
                        "warning",
                        row,
                        "Sold out but ADR below STLY",
                        f"ADR on books is {_money(adr)} versus STLY ADR {_money(stly_adr)}, a difference of {_money(adr - stly_adr)}.",
                        context=(demand, compset_level, my_level),
                    )
                )

        if pd.notna(demand_delta) and demand_delta >= 2:
            alerts.append(
                _alert(
                    "critical",
                    row,
                    f"Demand surged {int(demand_delta)} levels overnight",
                    f"Demand moved from {prior_demand} to {demand}. Current rate is {_money(price_today)} at a {my_level} price level.",
                    context=(demand, compset_level, my_level),
                )
            )

        if (
            pd.notna(compset_gap)
            and compset_gap >= 2
            and demand in ELEVATED_DEMAND_LEVELS
            and not suppress_pricing
            and _allow_compset_alert(demand, active_suppressions)
        ):
            alerts.append(
                _alert(
                    "critical",
                    row,
                    "Compset is priced well above you",
                    f"Your level is {my_level} while the compset is {compset_level} on a {demand} demand date.",
                    context=(demand, compset_level, my_level),
                )
            )

        if demand in HIGH_DEMAND_LEVELS and my_level in UNDERPRICED_LEVELS and not suppress_pricing:
            alerts.append(
                _alert(
                    "warning",
                    row,
                    f"{demand} demand with {my_level} rate level",
                    f"Published rate is {_money(price_today)} and compset level is {compset_level}. Review whether the rate should move up.",
                    context=(demand, compset_level, my_level),
                )
            )

        if pd.notna(demand_delta) and demand_delta <= -2:
            alerts.append(
                _alert(
                    "warning",
                    row,
                    f"Demand fell {abs(int(demand_delta))} levels overnight",
                    f"Demand moved from {prior_demand} to {demand}. Current rate is {_money(price_today)} at a {my_level} price level.",
                    context=(demand, compset_level, my_level),
                )
            )

        if sold_out and not _is_sold_out(price_yest) and not suppress_pricing:
            alerts.append(
                _alert(
                    "warning",
                    row,
                    "Date newly flipped to Sold Out",
                    f"The date moved from {_money(price_yest)} to Sold out. Validate whether the date closed at the right rate.",
                    context=(demand, compset_level, my_level),
                )
            )

        if (
            pd.notna(compset_delta)
            and compset_delta >= 1
            and demand in ELEVATED_DEMAND_LEVELS
            and not suppress_pricing
            and _allow_compset_alert(demand, active_suppressions)
        ):
            alerts.append(
                _alert(
                    "opportunity",
                    row,
                    "Compset raised rates",
                    f"Compset moved up {int(compset_delta)} level(s) to {compset_level}. Your rate is {_money(price_today)} at {my_level}.",
                    context=(demand, compset_level, my_level),
                )
            )

        if (
            flight_signal == "Higher"
            and my_level in UNDERPRICED_LEVELS
            and not suppress_pricing
            and _allow_flight_alert(row, demand, active_suppressions)
        ):
            alerts.append(
                _alert(
                    "opportunity",
                    row,
                    "Strong flight search with underpriced rate",
                    f"Flight search is above benchmark while your price level is {my_level}. Current rate is {_money(price_today)}.",
                    context=(demand, compset_level, my_level),
                )
            )

        if events >= 3 and my_level in UNDERPRICED_LEVELS and not suppress_pricing:
            alerts.append(
                _alert(
                    "opportunity",
                    row,
                    f"{events} events on an underpriced date",
                    f"Demand is {demand}, rate is {_money(price_today)}, and your price level is {my_level}.",
                    context=(demand, compset_level, my_level),
                )
            )

        rate_threshold = 20 if active_suppressions["rate_change_threshold_20"] else 10
        if pd.notna(price_delta) and abs(float(price_delta)) >= rate_threshold and not suppress_pricing:
            direction = "increased" if price_delta > 0 else "decreased"
            alerts.append(
                _alert(
                    "info",
                    row,
                    f"Published rate {direction} {_money(abs(float(price_delta)))}",
                    f"Rate changed from {_money(price_yest)} to {_money(price_today)}. Demand is {demand}; compset level is {compset_level}.",
                    context=(demand, compset_level, my_level),
                )
            )

    deduped = _dedupe_alerts(alerts)
    if active_suppressions["dedupe_consecutive"]:
        deduped = _collapse_consecutive_alerts(deduped)
    return sorted(deduped, key=lambda alert: (SEVERITY_RANK[alert["severity"]], alert["date"]))


def _alert(
    severity: str,
    row: pd.Series,
    title: str,
    body: str,
    context: tuple[Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """Create a normalized alert dictionary."""
    date_value = pd.Timestamp(row["Date"])
    return {
        "severity": severity,
        "date": date_value,
        "date_str": date_value.strftime("%a, %b %d"),
        "title": title,
        "body": body,
        "context": context,
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


def _collapse_consecutive_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse 4+ consecutive same-title alerts with matching context."""
    remaining = sorted(alerts, key=lambda alert: (alert["title"], str(alert.get("context")), alert["date"]))
    collapsed: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, alert in enumerate(remaining):
        if index in used:
            continue
        group = [index]
        prior_date = alert["date"]
        for next_index in range(index + 1, len(remaining)):
            other = remaining[next_index]
            if other["title"] != alert["title"] or other.get("context") != alert.get("context"):
                continue
            if (other["date"] - prior_date).days == 1:
                group.append(next_index)
                prior_date = other["date"]
        if len(group) >= 4:
            dates = [remaining[item]["date"].strftime("%a %b %d") for item in group]
            first = dict(alert)
            first["title"] = f"{alert['title']} - {len(group)} dates affected"
            first["date_str"] = f"{dates[0]} to {dates[-1]}"
            first["body"] = f"{alert['body']} Dates affected: {', '.join(dates)}."
            collapsed.append(first)
            used.update(group)
        else:
            collapsed.append(alert)
            used.add(index)
    return collapsed


def _allow_flight_alert(row: pd.Series, demand: Any, suppressions: dict[str, bool]) -> bool:
    """Return whether the flight-search alert is actionable."""
    if not suppressions["low_demand_weekday_flight"]:
        return True
    weekday = pd.Timestamp(row["Date"]).day_name()
    return not (demand == "Low" and weekday in {"Monday", "Tuesday", "Wednesday", "Thursday"})


def _allow_compset_alert(demand: Any, suppressions: dict[str, bool]) -> bool:
    """Return whether compset divergence should alert at this demand level."""
    if not suppressions["compset_min_demand"]:
        return True
    return demand not in {"Low", "Normal"}


def _matching_dpu_row(dpu_df: pd.DataFrame | None, date_value: Any) -> pd.Series | None:
    """Return matching DPU row for an arrival date."""
    if dpu_df is None or dpu_df.empty:
        return None
    date_key = pd.Timestamp(date_value).normalize()
    if date_key in dpu_df.index:
        row = dpu_df.loc[date_key]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row
    return None


def _money(value: Any) -> str:
    """Format a rate as whole dollars."""
    if _is_sold_out(value):
        return "Sold out"
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "unavailable"
    return f"${float(number):,.0f}"


def _number(value: Any) -> float | None:
    """Return a float or ``None`` for unavailable values."""
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _whole(value: Any) -> str:
    """Format a whole number or unavailable placeholder."""
    number = _number(value)
    return "unavailable" if number is None else f"{number:,.0f}"


def _signed_whole(value: Any) -> str:
    """Format a signed whole number or unavailable placeholder."""
    number = _number(value)
    return "unavailable" if number is None else f"{number:+,.0f}"


def _is_sold_out(value: Any) -> bool:
    """Return whether a cell carries the Sold out sentinel."""
    return isinstance(value, str) and value.strip().lower() == "sold out"
