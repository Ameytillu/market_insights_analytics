"""Parse Lighthouse Market Insights Excel exports into analysis-ready dataframes."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


EXCEL_EPOCH = datetime(1899, 12, 30)
NULL_SENTINELS = {"--", "", "nan", "NaN", "None"}


def excel_serial_to_date(serial: Any) -> pd.Timestamp:
    """Convert an Excel serial date to a pandas timestamp.

    Args:
        serial: Excel serial date value.

    Returns:
        Parsed timestamp, or ``pd.NaT`` when the value is not a valid serial.
    """
    try:
        value = float(serial)
    except (TypeError, ValueError):
        return pd.NaT
    return pd.Timestamp(EXCEL_EPOCH + timedelta(days=value))


def parse_lighthouse_export(file: Any) -> dict[str, pd.DataFrame | None]:
    """Parse a Lighthouse Market Insights workbook.

    Args:
        file: Uploaded file object, file-like object, or filesystem path accepted by pandas.

    Returns:
        Dictionary containing ``daily``, ``geo``, and ``los`` dataframes.
    """
    workbook = pd.ExcelFile(file, engine="openpyxl")
    return {
        "daily": _parse_daily_details(workbook),
        "geo": _parse_geo_breakdown(workbook),
        "los": _parse_stay_pattern_breakdown(workbook),
    }


def _parse_daily_details(workbook: pd.ExcelFile) -> pd.DataFrame:
    """Parse and normalize the Daily details sheet."""
    sheet_name = _find_sheet(workbook, ["daily"])
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(raw, ["Date", "Demand level"])
    df = pd.read_excel(workbook, sheet_name=sheet_name, header=header_row)
    df = _clean_frame(df)
    df.columns = [_normalize_column_name(column) for column in df.columns]

    if "Date" not in df.columns:
        raise ValueError("Daily details sheet is missing a Date column.")

    df["Date"] = df["Date"].apply(_parse_date_value)
    df = df[df["Date"].notna()].copy()
    if df.empty:
        raise ValueError("Daily details sheet did not contain any parseable dates.")

    for column in df.columns:
        if column not in {"My price", "My price level"}:
            df[column] = df[column].map(_null_if_sentinel)

    for column in ["My price", "My price level", "Demand level", "Smart Compset price level"]:
        if column in df.columns:
            df[column] = df[column].apply(_clean_level_value)

    if "My price" in df.columns:
        df["My price numeric"] = df["My price"].apply(_price_to_number)

    for column in ["Unavailable hotels"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Nr. of events and holidays" in df.columns:
        df["Nr. of events and holidays"] = (
            pd.to_numeric(df["Nr. of events and holidays"], errors="coerce").fillna(0).astype(int)
        )

    for column in [col for col in df.columns if "Most searched LOS" in col]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Day" not in df.columns:
        df["Day"] = df["Date"].dt.day_name().str[:3]

    return df.sort_values("Date").reset_index(drop=True)


def _parse_geo_breakdown(workbook: pd.ExcelFile) -> pd.DataFrame | None:
    """Parse the Geo breakdown sheet into a compact wide dataframe."""
    sheet_name = _find_sheet(workbook, ["geo"], required=False)
    if sheet_name is None:
        return None

    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(raw, ["Top 10 countries"], required=False)
    if header_row is None:
        return None

    df = pd.read_excel(workbook, sheet_name=sheet_name, header=header_row)
    df = _clean_frame(df)
    if df.shape[1] < 9:
        return df.replace("--", np.nan)

    out = pd.DataFrame(
        {
            "country_flight_meta": df.iloc[:, 0].map(_clean_text),
            "pct_flight_meta": pd.to_numeric(df.iloc[:, 1].map(_null_if_sentinel), errors="coerce"),
            "los_flight_meta": pd.to_numeric(df.iloc[:, 2].map(_null_if_sentinel), errors="coerce"),
            "country_flight_gds": df.iloc[:, 3].map(_clean_text),
            "pct_flight_gds": pd.to_numeric(df.iloc[:, 4].map(_null_if_sentinel), errors="coerce"),
            "los_flight_gds": pd.to_numeric(df.iloc[:, 5].map(_null_if_sentinel), errors="coerce"),
            "country_hotel_meta": df.iloc[:, 6].map(_clean_text),
            "pct_hotel_meta": pd.to_numeric(df.iloc[:, 7].map(_null_if_sentinel), errors="coerce"),
            "los_hotel_meta": pd.to_numeric(df.iloc[:, 8].map(_null_if_sentinel), errors="coerce"),
        }
    )
    return out.dropna(how="all").reset_index(drop=True)


def _parse_stay_pattern_breakdown(workbook: pd.ExcelFile) -> pd.DataFrame | None:
    """Parse the Stay pattern breakdown sheet."""
    sheet_name = _find_sheet(workbook, ["stay", "pattern", "los"], required=False)
    if sheet_name is None:
        return None

    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(raw, ["LOS"], required=False)
    if header_row is None:
        return None

    df = pd.read_excel(workbook, sheet_name=sheet_name, header=header_row)
    df = _clean_frame(df)
    rename: dict[Any, str] = {}
    for column in df.columns:
        text = str(column).strip().lower()
        if "bucket" in text or text == "los":
            rename[column] = "LOS bucket"
        elif "flight" in text and "meta" in text:
            rename[column] = "los_flight_meta"
        elif "flight" in text and "gds" in text:
            rename[column] = "los_flight_gds"
        elif "hotel" in text and "meta" in text:
            rename[column] = "los_hotel_meta"
        elif "hotel" in text and "gds" in text:
            rename[column] = "los_hotel_gds"
    df = df.rename(columns=rename)

    for column in ["los_flight_meta", "los_flight_gds", "los_hotel_meta", "los_hotel_gds"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column].map(_null_if_sentinel), errors="coerce")
    return df.reset_index(drop=True)


def _find_sheet(workbook: pd.ExcelFile, keywords: list[str], required: bool = True) -> str | None:
    """Find a worksheet by keyword."""
    for sheet_name in workbook.sheet_names:
        lowered = sheet_name.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            return sheet_name
    if required:
        raise ValueError(f"Workbook is missing a sheet matching: {', '.join(keywords)}.")
    return None


def _find_header_row(raw: pd.DataFrame, required_values: list[str], required: bool = True) -> int | None:
    """Find the first row containing all required header tokens."""
    for index, row in raw.iterrows():
        row_text = " ".join(str(value) for value in row.dropna().tolist()).lower()
        if all(value.lower() in row_text for value in required_values):
            return int(index)
    if required:
        raise ValueError(f"Could not locate header row containing: {', '.join(required_values)}.")
    return None


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully empty rows and columns."""
    return df.dropna(how="all", axis=0).dropna(how="all", axis=1).reset_index(drop=True)


def _normalize_column_name(column: Any) -> str:
    """Normalize workbook column names while preserving business wording."""
    return " ".join(str(column).strip().split())


def _parse_date_value(value: Any) -> pd.Timestamp:
    """Parse a Lighthouse date cell without breaking openpyxl datetime values."""
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    if isinstance(value, date):
        return pd.Timestamp(value).normalize()
    if isinstance(value, (int, float, np.integer, np.floating)) and 30000 <= float(value) <= 80000:
        return excel_serial_to_date(value).normalize()
    return pd.to_datetime(value, errors="coerce")


def _null_if_sentinel(value: Any) -> Any:
    """Convert Lighthouse null sentinels to NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str) and value.strip() in NULL_SENTINELS:
        return np.nan
    return value


def _clean_level_value(value: Any) -> Any:
    """Clean categorical values while preserving the Sold out state."""
    value = _null_if_sentinel(value)
    if pd.isna(value):
        return np.nan
    text = " ".join(str(value).strip().split())
    if text.lower() == "sold out":
        return "Sold out"
    if text.lower() == "very high":
        return "Very high"
    if text.lower() == "very low":
        return "Very low"
    return text.title() if text.lower() in {"low", "normal", "elevated", "high", "higher", "lower"} else text


def _clean_text(value: Any) -> Any:
    """Clean display text and null sentinels."""
    value = _null_if_sentinel(value)
    if pd.isna(value):
        return np.nan
    return " ".join(str(value).strip().split())


def _price_to_number(value: Any) -> float:
    """Convert numeric prices to floats while leaving Sold out unavailable."""
    value = _null_if_sentinel(value)
    if pd.isna(value) or str(value).strip().lower() == "sold out":
        return float("nan")
    return float(pd.to_numeric(value, errors="coerce"))
