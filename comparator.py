"""Compare Lighthouse daily snapshots using ordinal demand and price logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEMAND_ORDER = ["Low", "Normal", "Elevated", "High", "Very high", "Sold out"]
PRICE_LEVEL_ORDER = ["Very low", "Low", "Normal", "Elevated", "High", "Very high", "Sold out"]
BENCHMARK_ORDER = ["Lower", "Normal", "Higher"]

DEMAND_NUM = {value: index for index, value in enumerate(DEMAND_ORDER)}
LEVEL_NUM = {value: index for index, value in enumerate(PRICE_LEVEL_ORDER)}
BENCHMARK_NUM = {value: index - 1 for index, value in enumerate(BENCHMARK_ORDER)}


def compare_snapshots(df_today: pd.DataFrame, df_yesterday: pd.DataFrame) -> pd.DataFrame:
    """Merge two daily detail dataframes and compute overnight changes.

    Args:
        df_today: Parsed Daily details dataframe for the current snapshot.
        df_yesterday: Parsed Daily details dataframe for the prior snapshot.

    Returns:
        Dataframe merged on ``Date`` with demand, price, compset, search-signal,
        and event deltas.
    """
    today = df_today.copy()
    yesterday = df_yesterday.copy()
    today["Date"] = pd.to_datetime(today["Date"]).dt.normalize()
    yesterday["Date"] = pd.to_datetime(yesterday["Date"]).dt.normalize()

    merged = today.merge(yesterday, on="Date", how="inner", suffixes=("_today", "_yest"))
    out = pd.DataFrame({"Date": merged["Date"]})
    out["Day"] = _first_existing(merged, ["Day_today", "Day"]).fillna(out["Date"].dt.day_name().str[:3])

    out["Demand_today"] = _clean_series(_first_existing(merged, ["Demand level_today", "Demand level"]))
    out["Demand_yest"] = _clean_series(_first_existing(merged, ["Demand level_yest"]))
    out["Demand_today_num"] = out["Demand_today"].map(DEMAND_NUM)
    out["Demand_yest_num"] = out["Demand_yest"].map(DEMAND_NUM)
    out["Demand_delta"] = out["Demand_today_num"] - out["Demand_yest_num"]

    out["Price_today"] = _first_existing(merged, ["My price_today", "My price"])
    out["Price_yest"] = _first_existing(merged, ["My price_yest"])
    out["Price_today_numeric"] = out["Price_today"].apply(_price_to_float)
    out["Price_yest_numeric"] = out["Price_yest"].apply(_price_to_float)
    out["Price_delta"] = out["Price_today_numeric"] - out["Price_yest_numeric"]

    out["MyLevel_today"] = _clean_series(_first_existing(merged, ["My price level_today", "My price level"]))
    out["MyLevel_yest"] = _clean_series(_first_existing(merged, ["My price level_yest"]))
    out["MyLevel_today_num"] = out["MyLevel_today"].map(LEVEL_NUM)
    out["MyLevel_yest_num"] = out["MyLevel_yest"].map(LEVEL_NUM)
    out["MyLevel_delta"] = out["MyLevel_today_num"] - out["MyLevel_yest_num"]

    out["CompsetLevel_today"] = _clean_series(
        _first_existing(merged, ["Smart Compset price level_today", "Smart Compset price level"])
    )
    out["CompsetLevel_yest"] = _clean_series(_first_existing(merged, ["Smart Compset price level_yest"]))
    out["CompsetLevel_today_num"] = out["CompsetLevel_today"].map(LEVEL_NUM)
    out["CompsetLevel_yest_num"] = out["CompsetLevel_yest"].map(LEVEL_NUM)
    out["CompsetLevel_delta"] = out["CompsetLevel_today_num"] - out["CompsetLevel_yest_num"]
    out["Compset_gap"] = out["CompsetLevel_today_num"] - out["MyLevel_today_num"]

    for source, prefix in [
        ("Flight level (Meta/OTA) vs. benchmark", "FlightMeta"),
        ("Hotel level (Meta/OTA) vs. benchmark", "HotelMeta"),
        ("Flight level (GDS) vs. benchmark", "FlightGDS"),
        ("Hotel level (GDS) vs. benchmark", "HotelGDS"),
    ]:
        out[f"{prefix}_today"] = _clean_series(_first_existing(merged, [f"{source}_today", source]))
        out[f"{prefix}_yest"] = _clean_series(_first_existing(merged, [f"{source}_yest"]))
        out[f"{prefix}_today_num"] = out[f"{prefix}_today"].map(BENCHMARK_NUM)
        out[f"{prefix}_yest_num"] = out[f"{prefix}_yest"].map(BENCHMARK_NUM)
        out[f"{prefix}_delta"] = out[f"{prefix}_today_num"] - out[f"{prefix}_yest_num"]

    out["Events"] = pd.to_numeric(
        _first_existing(merged, ["Nr. of events and holidays_today", "Nr. of events and holidays"]),
        errors="coerce",
    ).fillna(0).astype(int)
    out["Unavailable_hotels"] = pd.to_numeric(
        _first_existing(merged, ["Unavailable hotels_today", "Unavailable hotels"]),
        errors="coerce",
    )

    return out.sort_values("Date").reset_index(drop=True)


def ordinal_value(value: Any, scale: str) -> float:
    """Return the ordinal value for a demand, price-level, or benchmark signal.

    Args:
        value: Categorical value to map.
        scale: One of ``demand``, ``price``, or ``benchmark``.

    Returns:
        Numeric ordinal value, or NaN when unknown.
    """
    maps = {"demand": DEMAND_NUM, "price": LEVEL_NUM, "benchmark": BENCHMARK_NUM}
    mapping = maps[scale]
    return float(mapping.get(_clean_value(value), np.nan))


def _first_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first existing dataframe column from a candidate list."""
    for column in columns:
        if column in df.columns:
            return df[column]
    return pd.Series(np.nan, index=df.index)


def _clean_series(series: pd.Series) -> pd.Series:
    """Normalize categorical string series."""
    return series.map(_clean_value)


def _clean_value(value: Any) -> Any:
    """Normalize blank and categorical values."""
    if pd.isna(value):
        return np.nan
    text = " ".join(str(value).strip().split())
    if text in {"", "--"}:
        return np.nan
    if text.lower() == "sold out":
        return "Sold out"
    if text.lower() == "very high":
        return "Very high"
    if text.lower() == "very low":
        return "Very low"
    return text.title() if text.lower() in {"low", "normal", "elevated", "high", "higher", "lower"} else text


def _price_to_float(value: Any) -> float:
    """Convert a rate to float while treating Sold out as unavailable."""
    if pd.isna(value) or str(value).strip().lower() == "sold out":
        return float("nan")
    return float(pd.to_numeric(value, errors="coerce"))
