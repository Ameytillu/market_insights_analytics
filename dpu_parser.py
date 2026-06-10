"""Parse Daily Pick-Up Excel reports into arrival-date operational data."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "Arrival Date",
    "Rooms on Books",
    "ADR on Books",
    "STLY Rooms",
    "STLY ADR",
]


def parse_dpu_report(file: Any) -> pd.DataFrame:
    """Parse a DPU workbook and return one clean row per arrival date.

    Args:
        file: Uploaded file object, file-like object, or filesystem path accepted by pandas.

    Returns:
        DataFrame indexed by ``Arrival Date`` as normalized ``pd.Timestamp`` values.
        Missing optional columns are present with null values.
    """
    workbook = pd.ExcelFile(file, engine="openpyxl")
    frames: list[pd.DataFrame] = []
    for sheet_order, sheet_name in enumerate(workbook.sheet_names):
        parsed = _parse_sheet(workbook, sheet_name, sheet_order)
        if parsed is not None and not parsed.empty:
            frames.append(parsed)

    if not frames:
        return _empty_dpu_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Arrival Date"]).copy()
    combined["Arrival Date"] = pd.to_datetime(combined["Arrival Date"]).dt.normalize()
    combined = combined.sort_values(["Arrival Date", "_sheet_order"])
    combined = combined.groupby("Arrival Date", as_index=False).agg(
        {
            "Rooms on Books": _last_valid,
            "ADR on Books": _last_valid,
            "STLY Rooms": _last_valid,
            "STLY ADR": _last_valid,
        }
    )
    combined["Rooms Variance"] = combined["Rooms on Books"] - combined["STLY Rooms"]
    combined["ADR Variance"] = combined["ADR on Books"] - combined["STLY ADR"]
    return combined.set_index("Arrival Date").sort_index()


def _parse_sheet(workbook: pd.ExcelFile, sheet_name: str, sheet_order: int) -> pd.DataFrame | None:
    """Parse one DPU sheet when a usable header row can be found."""
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(raw)
    if header_row is None:
        return None

    standard = _parse_standard_dpu_table(raw, header_row, sheet_order)
    if standard is not None and not standard.empty:
        return standard

    df = pd.read_excel(workbook, sheet_name=sheet_name, header=header_row)
    df = df.dropna(how="all").dropna(how="all", axis=1)
    if df.empty:
        return None

    mapping = _detect_columns(df.columns)
    if "Arrival Date" not in mapping.values():
        return None

    out = pd.DataFrame(index=df.index)
    for source, target in mapping.items():
        out[target] = df[source]
    for column in CANONICAL_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    out["Arrival Date"] = out["Arrival Date"].map(_parse_date)
    for column in ["Rooms on Books", "ADR on Books", "STLY Rooms", "STLY ADR"]:
        out[column] = pd.to_numeric(out[column].map(_null_if_blank), errors="coerce")
    out["_sheet_order"] = sheet_order
    return out[CANONICAL_COLUMNS + ["_sheet_order"]]


def _parse_standard_dpu_table(raw: pd.DataFrame, header_row: int, sheet_order: int) -> pd.DataFrame | None:
    """Parse the common DPU monthly table with Date, Day, Rooms, Occ, Revenue, ADR."""
    headers = [_normalize(value) for value in raw.iloc[header_row].tolist()]
    date_columns = [index for index, value in enumerate(headers) if _matches_arrival_date(value)]
    if not date_columns:
        return None

    date_col = date_columns[0]
    next_date_col = next((column for column in date_columns[1:] if column > date_col), len(headers))
    block_headers = headers[date_col:next_date_col]
    rooms_col = _relative_column(block_headers, "rooms", date_col)
    adr_col = _relative_column(block_headers, "adr", date_col)
    if rooms_col is None and adr_col is None:
        return None

    rows: list[dict[str, Any]] = []
    saw_data = False
    for row_index in range(header_row + 1, len(raw.index)):
        arrival_date = _parse_date(raw.iat[row_index, date_col])
        if pd.isna(arrival_date):
            if saw_data:
                break
            continue
        saw_data = True
        rows.append(
            {
                "Arrival Date": arrival_date,
                "Rooms on Books": raw.iat[row_index, rooms_col] if rooms_col is not None else np.nan,
                "ADR on Books": raw.iat[row_index, adr_col] if adr_col is not None else np.nan,
                "STLY Rooms": np.nan,
                "STLY ADR": np.nan,
                "_sheet_order": sheet_order,
            }
        )

    if not rows:
        return None
    out = pd.DataFrame(rows)
    for column in ["Rooms on Books", "ADR on Books", "STLY Rooms", "STLY ADR"]:
        out[column] = pd.to_numeric(out[column].map(_null_if_blank), errors="coerce")
    return out[CANONICAL_COLUMNS + ["_sheet_order"]]


def _relative_column(headers: list[str], target: str, offset: int) -> int | None:
    """Return the absolute column index of a target header inside a table block."""
    for relative_index, value in enumerate(headers):
        if value == target or target in value.split():
            return offset + relative_index
    return None


def _detect_columns(columns: pd.Index) -> dict[Any, str]:
    """Map messy property-specific DPU column names to canonical names."""
    detected: dict[Any, str] = {}
    for column in columns:
        text = _normalize(column)
        compact = text.replace(" ", "")
        if _matches_arrival_date(text):
            detected[column] = "Arrival Date"
        elif _matches_rooms_on_books(text, compact):
            detected[column] = "Rooms on Books"
        elif _matches_adr_on_books(text, compact):
            detected[column] = "ADR on Books"
        elif _matches_stly_rooms(text, compact):
            detected[column] = "STLY Rooms"
        elif _matches_stly_adr(text, compact):
            detected[column] = "STLY ADR"
    return detected


def _find_header_row(raw: pd.DataFrame) -> int | None:
    """Find the first row that looks like DPU headers."""
    for index, row in raw.iterrows():
        values = [_normalize(value) for value in row.dropna().tolist()]
        text = " ".join(values)
        has_date = any(_matches_arrival_date(value) for value in values)
        has_operational_metric = any(token in text for token in ["rooms", "rms", "rob", "otb", "adr", "stly", "ly"])
        if has_date and has_operational_metric:
            return int(index)
    return None


def _matches_arrival_date(text: str) -> bool:
    return ("arrival" in text or "arrive" in text or text in {"date", "stay date", "business date"}) and "date" in text


def _matches_rooms_on_books(text: str, compact: str) -> bool:
    if "stly" in compact or "lastyear" in compact or "ly" == compact:
        return False
    return ("rooms" in text or "rms" in text or "rob" in compact) and (
        "book" in text or "otb" in compact or "on books" in text or "rob" in compact
    )


def _matches_adr_on_books(text: str, compact: str) -> bool:
    if "stly" in compact or "lastyear" in compact:
        return False
    return "adr" in text and ("book" in text or "otb" in compact or "on books" in text or "current" in text)


def _matches_stly_rooms(text: str, compact: str) -> bool:
    has_stly = "stly" in compact or "same time last year" in text or "last year" in text or "ly" in text
    return has_stly and ("rooms" in text or "rms" in text or "rob" in compact)


def _matches_stly_adr(text: str, compact: str) -> bool:
    has_stly = "stly" in compact or "same time last year" in text or "last year" in text or "ly" in text
    return has_stly and "adr" in text


def _parse_date(value: Any) -> pd.Timestamp:
    """Parse a DPU arrival date cell."""
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    if isinstance(value, date):
        return pd.Timestamp(value).normalize()
    return pd.to_datetime(value, errors="coerce")


def _normalize(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def _null_if_blank(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str) and value.strip() in {"", "--", "nan", "None"}:
        return np.nan
    return value


def _empty_dpu_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex([], name="Arrival Date")
    return pd.DataFrame(columns=CANONICAL_COLUMNS[1:] + ["Rooms Variance", "ADR Variance"], index=index)


def _last_valid(series: pd.Series) -> Any:
    """Return the latest non-null value from a sheet-ordered series."""
    valid = series.dropna()
    return np.nan if valid.empty else valid.iloc[-1]
