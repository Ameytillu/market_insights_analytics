"""Streamlit UI for the Market Insights Analytics revenue intelligence app."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from plotly.subplots import make_subplots

from alerts import DEFAULT_SUPPRESSIONS, generate_alerts
from comparator import DEMAND_NUM, DEMAND_ORDER, LEVEL_NUM, compare_snapshots
from decision_log import append_decision, decision_summary, decisions_dataframe, update_decision_outcome
from dpu_parser import parse_dpu_report
from parser import parse_lighthouse_export
from rate_engine import recommend_rate
from snapshot_store import build_trend_dataframe, save_snapshot, snapshot_summary


APP_TITLE = "Market Insights Analytics"
APP_TIMEZONE = "America/Chicago"
DEMAND_COLORS = {
    "Low": "#94a3b8",
    "Normal": "#60a5fa",
    "Elevated": "#fbbf24",
    "High": "#f97316",
    "Very high": "#ef4444",
    "Sold out": "#7c3aed",
}
SEVERITY_COLORS = {
    "critical": "#ef4444",
    "warning": "#f59e0b",
    "opportunity": "#16a34a",
    "info": "#3b82f6",
}
SIGNAL_NUM = {"Lower": -1, "Normal": 0, "Higher": 1}


def main() -> None:
    """Run the Streamlit application.

    Args:
        None.

    Returns:
        None.
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    inject_css()
    require_login()
    render_header()

    uploaded_yesterday, uploaded_today, uploaded_dpu = render_uploaders()
    sidebar_config = render_config_sidebar()
    if uploaded_today is None:
        render_empty_state()
        render_ai_sidebar(None)
        return

    today_data, yesterday_data = parse_uploads(uploaded_today, uploaded_yesterday)
    dpu_df = parse_dpu_upload(uploaded_dpu)
    df_today = today_data["daily"]
    df_yesterday = yesterday_data["daily"] if yesterday_data else None
    if df_today is None or df_today.empty:
        st.error("The uploaded workbook did not contain usable Daily details rows.")
        return

    df_today = df_today.copy()
    df_today["Date"] = pd.to_datetime(df_today["Date"]).dt.normalize()
    current_day = current_business_day()
    df_future = df_today[df_today["Date"] >= current_day].copy()
    if df_future.empty:
        df_future = df_today.copy()

    render_snapshot_summary(df_today, df_yesterday is not None)
    save_snapshot(df_today)
    render_ai_sidebar(df_future)
    render_tabs(today_data, df_today, df_yesterday, df_future, current_day, dpu_df, sidebar_config)


def inject_css() -> None:
    """Add app-level styles.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { max-width: 1440px; padding: 1.25rem 2rem 3rem; }
.app-header { background:#07172f; color:white; padding:1.5rem 1.75rem; border-radius:8px; margin-bottom:1rem; }
.app-header h1 { margin:0; font-size:1.9rem; letter-spacing:0; }
.app-header p { color:#bfdbfe; margin:.35rem 0 0; }
.section-label { color:#475569; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin:.5rem 0 .65rem; }
.kpi-tile { background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1rem; min-height:96px; }
.kpi-value { color:#0f172a; font-size:1.65rem; font-weight:700; line-height:1.1; }
.kpi-label { color:#64748b; font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; margin-top:.35rem; }
.alert-card { background:#fff; border:1px solid #e2e8f0; border-left-width:5px; border-radius:8px; padding:.85rem 1rem; margin:.6rem 0; }
.alert-title { color:#0f172a; font-weight:700; font-size:.95rem; }
.alert-meta { color:#64748b; font-size:.78rem; margin-top:.1rem; }
.alert-body { color:#334155; font-size:.88rem; margin-top:.45rem; }
.callout { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:1rem; color:#334155; }
.login-card { max-width:420px; margin:5rem auto 0; background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1.25rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:.8rem; }
</style>
""",
        unsafe_allow_html=True,
    )


def require_login() -> None:
    """Gate all application content behind a password.

    Args:
        None.

    Returns:
        None.
    """
    if st.session_state.get("authenticated"):
        return

    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.subheader(APP_TITLE)
    st.caption("Sign in to access market snapshots, alerts, and data exports.")
    password = st.text_input("Password", type="password")
    submitted = st.button("Sign in", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    expected = st.secrets.get("PASSWORD")
    if not expected:
        st.error('Missing Streamlit secret: `PASSWORD`.')
        st.stop()
    if submitted:
        if password == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


def render_header() -> None:
    """Render the page header.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(
        f"""
<div class="app-header">
  <h1>{APP_TITLE}</h1>
  <p>Daily revenue intelligence for market insights snapshots.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def current_business_day() -> pd.Timestamp:
    """Return today's hotel-local business date.

    Args:
        None.

    Returns:
        Normalized date in the app's configured hotel timezone.
    """
    return pd.Timestamp(datetime.now(ZoneInfo(APP_TIMEZONE)).date())


def render_uploaders() -> tuple[Any | None, Any | None, Any | None]:
    """Render the two snapshot uploaders.

    Args:
        None.

    Returns:
        Tuple of yesterday upload, today upload, and optional DPU upload.
    """
    st.markdown('<div class="section-label">Load Snapshots</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        yesterday = st.file_uploader("Yesterday's market insights export", type=["xlsx"], key="yesterday_upload")
    with right:
        today = st.file_uploader("Today's market insights export", type=["xlsx"], key="today_upload")
    dpu = st.file_uploader("DPU Report (optional)", type=["xlsx"], key="dpu_upload")
    return yesterday, today, dpu


def render_config_sidebar() -> dict[str, Any]:
    """Render persistent sidebar settings and summaries.

    Args:
        None.

    Returns:
        Dictionary of property settings and alert suppression toggles.
    """
    snap_summary = snapshot_summary()
    dec_summary = decision_summary()
    with st.sidebar:
        st.header("Property Settings")
        with st.expander("Property settings", expanded=False):
            total_rooms = st.number_input("Total rooms", min_value=1, value=433, step=1)
            min_rate = st.number_input("Minimum rate", min_value=0.0, value=150.0, step=5.0)
            max_rate = st.number_input("Maximum rate", min_value=0.0, value=600.0, step=5.0)
            adr_growth_pct = st.number_input("ADR growth target %", value=3.0, step=0.5)
            occupancy_target_pct = st.number_input("Occupancy target %", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

        st.caption("Snapshot history")
        if snap_summary["count"]:
            start = pd.Timestamp(snap_summary["start"]).strftime("%b %d, %Y")
            end = pd.Timestamp(snap_summary["end"]).strftime("%b %d, %Y")
            st.write(f"{snap_summary['count']} snapshots stored ({start} to {end})")
        else:
            st.write("No snapshots stored yet")

        st.caption("Decision log")
        st.write(f"{dec_summary['count']} decisions logged")

        with st.expander("Alert suppression", expanded=False):
            labels = {
                "low_demand_weekday_flight": "Suppress low-demand weekday flight signals",
                "compset_min_demand": "Require Elevated+ demand for compset gaps",
                "rate_change_threshold_20": "Use $20 threshold for rate-change info",
                "dedupe_consecutive": "Collapse repeated consecutive alerts",
                "sold_out_pricing": "Suppress pricing alerts on sold-out dates",
            }
            suppressions = {
                key: st.checkbox(labels[key], value=DEFAULT_SUPPRESSIONS[key], key=f"suppress_{key}")
                for key in DEFAULT_SUPPRESSIONS
            }

    return {
        "total_rooms": int(total_rooms),
        "occupancy_target": float(occupancy_target_pct) / 100,
        "adr_growth_target": float(adr_growth_pct) / 100,
        "min_rate": float(min_rate),
        "max_rate": float(max_rate),
        "suppressions": suppressions,
    }


@st.cache_data(show_spinner=False)
def parse_cached(file_bytes: bytes, filename: str) -> dict[str, pd.DataFrame | None]:
    """Parse an uploaded workbook with Streamlit caching.

    Args:
        file_bytes: Uploaded workbook bytes.
        filename: Uploaded filename used only as a cache key component.

    Returns:
        Parsed workbook dictionary.
    """
    from io import BytesIO

    _ = filename
    return parse_lighthouse_export(BytesIO(file_bytes))


def parse_uploads(today_file: Any, yesterday_file: Any | None) -> tuple[dict[str, pd.DataFrame | None], dict[str, pd.DataFrame | None] | None]:
    """Parse today and optional yesterday uploads.

    Args:
        today_file: Today's uploaded workbook.
        yesterday_file: Optional yesterday uploaded workbook.

    Returns:
        Parsed today data and optional parsed yesterday data.
    """
    with st.spinner("Parsing market insights exports..."):
        try:
            today_data = parse_cached(today_file.getvalue(), today_file.name)
            yesterday_data = (
                parse_cached(yesterday_file.getvalue(), yesterday_file.name) if yesterday_file is not None else None
            )
        except Exception as exc:
            st.error(f"Parse error: {exc}")
            st.stop()
    return today_data, yesterday_data


@st.cache_data(show_spinner=False)
def parse_dpu_cached(file_bytes: bytes, filename: str, parser_version: str) -> pd.DataFrame:
    """Parse a DPU workbook with Streamlit caching.

    Args:
        file_bytes: Uploaded workbook bytes.
        filename: Uploaded filename used only as a cache key component.
        parser_version: Manual cache-busting version for parser dependency changes.

    Returns:
        Parsed DPU DataFrame.
    """
    from io import BytesIO

    _ = filename
    _ = parser_version
    return parse_dpu_report(BytesIO(file_bytes))


def parse_dpu_upload(dpu_file: Any | None) -> pd.DataFrame | None:
    """Parse an optional DPU upload.

    Args:
        dpu_file: Optional uploaded DPU workbook.

    Returns:
        Parsed DPU DataFrame or ``None``.
    """
    if dpu_file is None:
        return None
    with st.spinner("Parsing DPU report..."):
        try:
            parsed = parse_dpu_cached(dpu_file.getvalue(), dpu_file.name, "central_time_pickup_v5")
        except Exception as exc:
            st.warning(f"DPU report could not be parsed: {exc}")
            return None
    if parsed.empty:
        st.warning("DPU report did not contain recognizable arrival-date rows.")
        return None
    return parsed


def render_snapshot_summary(df_today: pd.DataFrame, comparison_active: bool) -> None:
    """Show snapshot date range and mode.

    Args:
        df_today: Current daily detail dataframe.
        comparison_active: Whether yesterday was loaded.

    Returns:
        None.
    """
    min_date = df_today["Date"].min().strftime("%b %d, %Y")
    max_date = df_today["Date"].max().strftime("%b %d, %Y")
    mode = "Comparison mode" if comparison_active else "Single-file exploration"
    st.caption(f"{mode} | Snapshot window: {min_date} to {max_date} | {len(df_today):,} daily rows")


def render_tabs(
    today_data: dict[str, pd.DataFrame | None],
    df_today: pd.DataFrame,
    df_yesterday: pd.DataFrame | None,
    df_future: pd.DataFrame,
    current_day: pd.Timestamp,
    dpu_df: pd.DataFrame | None,
    config: dict[str, Any],
) -> None:
    """Render the full tab set in the required order.

    Args:
        today_data: Parsed current workbook dictionary.
        df_today: Current daily detail dataframe.
        df_yesterday: Optional prior daily detail dataframe.
        df_future: Forward-period daily detail dataframe.
        current_day: Today's date normalized to midnight.
        dpu_df: Optional parsed DPU dataframe.
        config: Property settings and alert suppression toggles.

    Returns:
        None.
    """
    tabs = st.tabs(
        [
            "🚨 Action Center",
            "📈 Demand & Rate",
            "⚖️ Overnight Changes",
            "📊 Operational View",
            "💡 Rate Recommendations",
            "📈 Demand Trends",
            "🌍 Market Intelligence",
            "📝 Decision Log",
            "📋 Full Data",
        ]
    )
    with tabs[0]:
        if df_yesterday is None:
            st.info("Upload yesterday's Lighthouse export to unlock action alerts.")
        else:
            render_action_center(df_today, df_yesterday, current_day, dpu_df, config["suppressions"])
    with tabs[1]:
        render_demand_rate(df_today, df_future)
    with tabs[2]:
        if df_yesterday is None:
            st.info("Upload yesterday's Lighthouse export to compare overnight changes.")
        else:
            render_overnight_changes(df_today, df_yesterday, current_day)
    with tabs[3]:
        render_operational_view(df_future, dpu_df, config)
    with tabs[4]:
        render_rate_recommendations(df_future, dpu_df, config)
    with tabs[5]:
        render_demand_trends()
    with tabs[6]:
        render_market_intelligence(today_data, df_future)
    with tabs[7]:
        render_decision_log(df_future)
    with tabs[8]:
        render_full_data(df_today, df_future)


def render_empty_state() -> None:
    """Render the upload prompt.

    Args:
        None.

    Returns:
        None.
    """
    st.info("Upload today's market insights `.xlsx` export to begin. Add yesterday's export to unlock alerts and overnight changes.")


def render_ai_sidebar(df_context: pd.DataFrame | None) -> None:
    """Render the sidebar AI assistant and enforce the session call cap.

    Args:
        df_context: Forward-period daily dataframe used as AI grounding context.

    Returns:
        None.
    """
    if "ai_calls" not in st.session_state:
        st.session_state["ai_calls"] = 0

    with st.sidebar:
        st.header("AI Assistant")
        remaining = max(0, 15 - int(st.session_state["ai_calls"]))
        st.caption(f"{remaining} OpenAI calls remaining this session")
        if df_context is None or df_context.empty:
            st.info("Load a snapshot to ask questions about the forward period.")
            return

        question = st.text_area("Ask about the loaded snapshot", height=90, placeholder="What dates need pricing attention?")
        submitted = st.button("Ask", type="primary", use_container_width=True)
        if not submitted:
            return
        if remaining <= 0:
            st.error("Session AI call limit reached. Refresh the app to start a new session.")
            return
        if not st.secrets.get("OPENAI_API_KEY"):
            st.error("Missing Streamlit secret: `OPENAI_API_KEY`.")
            return
        if not question.strip():
            st.warning("Enter a question first.")
            return

        context = df_context.head(30).copy()
        context["Date"] = context["Date"].dt.strftime("%Y-%m-%d")
        context_csv = context.to_csv(index=False)
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        st.session_state["ai_calls"] += 1
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                stream=True,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a hotel revenue management analyst. The user has loaded a market "
                            "insights snapshot. Answer questions about the data concisely and in plain language. "
                            "Always ground answers in the actual numbers from the snapshot."
                        ),
                    },
                    {"role": "user", "content": f"Forward-period data as CSV, truncated to 30 rows:\n\n{context_csv}\n\nQuestion: {question}"},
                ],
            )
            st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)


def render_comparison_tabs(
    today_data: dict[str, pd.DataFrame | None],
    df_today: pd.DataFrame,
    df_yesterday: pd.DataFrame,
    df_future: pd.DataFrame,
    current_day: pd.Timestamp,
) -> None:
    """Render tabs for two-file comparison mode.

    Args:
        today_data: Parsed current workbook dictionary.
        df_today: Current daily detail dataframe.
        df_yesterday: Prior daily detail dataframe.
        df_future: Forward-period current daily detail dataframe.
        current_day: Today's date normalized to midnight.

    Returns:
        None.
    """
    tab_action, tab_demand, tab_changes, tab_market, tab_data = st.tabs(
        ["Action Center", "Demand & Rate", "Overnight Changes", "Market Intelligence", "Full Data"]
    )
    with tab_action:
        render_action_center(df_today, df_yesterday, current_day)
    with tab_demand:
        render_demand_rate(df_today, df_future)
    with tab_changes:
        render_overnight_changes(df_today, df_yesterday, current_day)
    with tab_market:
        render_market_intelligence(today_data, df_future)
    with tab_data:
        render_full_data(df_today, df_future)


def render_single_file_tabs(
    today_data: dict[str, pd.DataFrame | None],
    df_today: pd.DataFrame,
    df_future: pd.DataFrame,
) -> None:
    """Render tabs for single-file exploration mode.

    Args:
        today_data: Parsed current workbook dictionary.
        df_today: Current daily detail dataframe.
        df_future: Forward-period daily detail dataframe.

    Returns:
        None.
    """
    tab_demand, tab_market, tab_signals = st.tabs(["Demand & Rate", "Market Intelligence", "Pricing Signals & Full Data"])
    with tab_demand:
        render_demand_rate(df_today, df_future)
    with tab_market:
        render_market_intelligence(today_data, df_future)
    with tab_signals:
        render_pricing_scan(df_future)
        st.divider()
        render_full_data(df_today, df_future)


def render_action_center(
    df_today: pd.DataFrame,
    df_yesterday: pd.DataFrame,
    current_day: pd.Timestamp,
    dpu_df: pd.DataFrame | None = None,
    suppressions: dict[str, bool] | None = None,
) -> None:
    """Render alert summary, filterable alerts, and future overnight table.

    Args:
        df_today: Current daily detail dataframe.
        df_yesterday: Prior daily detail dataframe.
        current_day: Today's date normalized to midnight.
        dpu_df: Optional DPU dataframe.
        suppressions: Optional alert suppression toggles.

    Returns:
        None.
    """
    alerts = generate_alerts(df_today, df_yesterday, dpu_df=dpu_df, suppressions=suppressions)
    counts = {severity: sum(1 for alert in alerts if alert["severity"] == severity) for severity in SEVERITY_COLORS}
    cols = st.columns(4)
    for col, severity in zip(cols, ["critical", "warning", "opportunity", "info"]):
        with col:
            render_kpi(severity.title(), str(counts[severity]), SEVERITY_COLORS[severity])

    selected = st.multiselect(
        "Filter alerts",
        ["critical", "warning", "opportunity", "info"],
        default=["critical", "warning", "opportunity", "info"],
    )
    for alert in [item for item in alerts if item["severity"] in selected]:
        render_alert_card(alert)
    if not alerts:
        st.success("No pricing-action alerts were detected for future dates.")

    st.markdown('<div class="section-label">Overnight Changes - All Future Dates</div>', unsafe_allow_html=True)
    comparison = compare_snapshots(df_today, df_yesterday)
    future = comparison[comparison["Date"] >= current_day].copy()
    show = future[
        [
            "Date",
            "Day",
            "Demand_yest",
            "Demand_today",
            "Demand_delta",
            "Price_yest",
            "Price_today",
            "Price_delta",
            "MyLevel_today",
            "CompsetLevel_today",
        ]
    ].copy()
    show["Date"] = show["Date"].dt.strftime("%a %b %d")
    show["Price_yest"] = show["Price_yest"].map(money)
    show["Price_today"] = show["Price_today"].map(money)
    show["Price_delta"] = show["Price_delta"].map(lambda value: "" if pd.isna(value) else f"${value:+,.0f}")
    st.dataframe(show, use_container_width=True, hide_index=True)


def render_demand_rate(df_today: pd.DataFrame, df_future: pd.DataFrame) -> None:
    """Render demand, rate, heatmap, positioning, and KPI views.

    Args:
        df_today: Current daily detail dataframe.
        df_future: Forward-period daily detail dataframe.

    Returns:
        None.
    """
    plot_df = df_future.sort_values("Date").head(60).copy()
    plot_df["demand_num"] = plot_df["Demand level"].map(DEMAND_NUM)
    plot_df["price_num"] = plot_df["My price"].apply(price_number)

    sold_out_count = int((plot_df["My price"].astype(str).str.lower() == "sold out").sum())
    very_high_count = int((plot_df["Demand level"] == "Very high").sum())
    avg_rate = plot_df["price_num"].mean()
    underpriced_count = len(pricing_scan_rows(plot_df))

    cols = st.columns(4)
    values = [
        ("Sold-Out Count", str(sold_out_count), "#7c3aed"),
        ("Very-High Demand Days", str(very_high_count), "#ef4444"),
        ("Avg Rate", money(avg_rate), "#0f172a"),
        ("Underpriced Dates", str(underpriced_count), "#f97316"),
    ]
    for col, (label, value, color) in zip(cols, values):
        with col:
            render_kpi(label, value, color)

    st.markdown('<div class="section-label">Demand Level and Published Rate - Forward 60 Days</div>', unsafe_allow_html=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=plot_df["Date"],
            y=plot_df["demand_num"],
            name="Demand level",
            marker_color=[DEMAND_COLORS.get(level, "#94a3b8") for level in plot_df["Demand level"]],
            text=plot_df["Demand level"],
            hovertemplate="%{x|%b %d}<br>Demand: %{text}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["price_num"],
            name="My rate",
            mode="lines+markers",
            line=dict(color="#0f172a", width=2),
            hovertemplate="%{x|%b %d}<br>Rate: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    style_plot(fig, height=360)
    fig.update_yaxes(tickvals=list(DEMAND_NUM.values()), ticktext=DEMAND_ORDER, secondary_y=False, title_text="Demand")
    fig.update_yaxes(tickprefix="$", secondary_y=True, title_text="Rate")
    st.plotly_chart(fig, use_container_width=True)

    render_demand_heatmap(plot_df)
    render_rate_positioning(df_future)


def render_demand_heatmap(df: pd.DataFrame) -> None:
    """Render a day-of-week heatmap.

    Args:
        df: Forward-period daily dataframe.

    Returns:
        None.
    """
    st.markdown('<div class="section-label">Day-of-Week Demand Heatmap</div>', unsafe_allow_html=True)
    data = df.copy()
    data["Week"] = data["Date"].dt.strftime("Wk %m/%d")
    data["DayName"] = data["Date"].dt.day_name()
    data["DemandNum"] = data["Demand level"].map(DEMAND_NUM)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    values = data.pivot_table(index="DayName", columns="Week", values="DemandNum", aggfunc="mean").reindex(order)
    labels = data.pivot_table(index="DayName", columns="Week", values="Demand level", aggfunc="first").reindex(order)
    if values.dropna(how="all").empty:
        st.info("Demand heatmap data is unavailable.")
        return
    fig = go.Figure(
        go.Heatmap(
            z=values.values,
            x=values.columns,
            y=values.index,
            text=labels.values,
            texttemplate="%{text}",
            colorscale=[[0, "#94a3b8"], [0.2, "#60a5fa"], [0.45, "#fbbf24"], [0.7, "#f97316"], [1, "#ef4444"]],
            showscale=False,
            hovertemplate="%{y} %{x}<br>%{text}<extra></extra>",
        )
    )
    style_plot(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)


def render_rate_positioning(df_future: pd.DataFrame) -> None:
    """Render rate positioning table.

    Args:
        df_future: Forward-period daily dataframe.

    Returns:
        None.
    """
    st.markdown('<div class="section-label">Rate Positioning</div>', unsafe_allow_html=True)
    columns = ["Date", "Day", "Demand level", "My price", "My price level", "Smart Compset price level"]
    available = [column for column in columns if column in df_future.columns]
    table = df_future[available].copy()
    table["Position"] = table.apply(position_vs_compset, axis=1)
    table["Date"] = table["Date"].dt.strftime("%a %b %d")
    if "My price" in table.columns:
        table["My price"] = table["My price"].map(money)
    st.dataframe(table, use_container_width=True, hide_index=True, height=330)


def render_overnight_changes(df_today: pd.DataFrame, df_yesterday: pd.DataFrame, current_day: pd.Timestamp) -> None:
    """Render overnight demand and price changes.

    Args:
        df_today: Current daily detail dataframe.
        df_yesterday: Prior daily detail dataframe.
        current_day: Today's date normalized to midnight.

    Returns:
        None.
    """
    comparison = compare_snapshots(df_today, df_yesterday)
    future = comparison[comparison["Date"] >= current_day].copy()

    st.markdown('<div class="section-label">Demand Level Shifts</div>', unsafe_allow_html=True)
    fig = go.Figure(
        go.Bar(
            x=future["Date"],
            y=future["Demand_delta"],
            marker_color=["#ef4444" if value > 0 else "#16a34a" if value < 0 else "#cbd5e1" for value in future["Demand_delta"].fillna(0)],
            hovertemplate="%{x|%b %d}<br>Demand shift: %{y:+.0f} steps<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="#cbd5e1")
    style_plot(fig, height=300)
    fig.update_yaxes(title="Ordinal steps")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Rate Changes</div>', unsafe_allow_html=True)
    price_changes = future[future["Price_delta"].notna()].copy()
    fig_price = go.Figure(
        go.Bar(
            x=price_changes["Date"],
            y=price_changes["Price_delta"],
            marker_color=["#16a34a" if value > 0 else "#ef4444" if value < 0 else "#cbd5e1" for value in price_changes["Price_delta"]],
            hovertemplate="%{x|%b %d}<br>Rate change: $%{y:+,.0f}<extra></extra>",
        )
    )
    fig_price.add_hline(y=0, line_color="#cbd5e1")
    style_plot(fig_price, height=300)
    fig_price.update_yaxes(title="Rate change", tickprefix="$")
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown('<div class="section-label">Newly Sold-Out Dates</div>', unsafe_allow_html=True)
    sold = future[
        future["Price_today"].astype(str).str.lower().eq("sold out")
        & ~future["Price_yest"].astype(str).str.lower().eq("sold out")
    ]
    if sold.empty:
        st.info("No future dates newly flipped to Sold out.")
    for _, row in sold.iterrows():
        render_alert_card(
            {
                "severity": "warning",
                "date_str": row["Date"].strftime("%a, %b %d"),
                "title": "Newly Sold Out",
                "body": f"Rate moved from {money(row['Price_yest'])} to Sold out. Demand is {row['Demand_today']}.",
            }
        )


def render_operational_view(df_future: pd.DataFrame, dpu_df: pd.DataFrame | None, config: dict[str, Any]) -> None:
    """Render combined Lighthouse and DPU operational context.

    Args:
        df_future: Forward-period daily detail dataframe.
        dpu_df: Optional parsed DPU dataframe.
        config: Property settings used for pickup forecast limits.

    Returns:
        None.
    """
    if dpu_df is None or dpu_df.empty:
        st.info("Upload a DPU report to unlock this view.")
        return

    dpu_start = pd.Timestamp(dpu_df.index.min()).strftime("%b %d, %Y")
    dpu_end = pd.Timestamp(dpu_df.index.max()).strftime("%b %d, %Y")
    st.caption(f"DPU parsed {len(dpu_df):,} arrival dates from {dpu_start} to {dpu_end}.")

    combined = build_operational_table(df_future, dpu_df, config)
    urgent = combined[(combined["Demand level"] == "Very high") & (combined["Rooms Pickup"] < 0)]
    sold_cheap = combined[
        combined["My price"].astype(str).str.lower().eq("sold out")
        & combined["ADR on Books"].notna()
    ]
    soft_ahead = combined[(combined["Rooms Pickup"] > 10) & (combined["Demand level"].isin(["Low", "Normal"]))]

    matched = combined["Rooms on Books"].notna()
    total_pickup = combined["Rooms Pickup"].dropna().sum()
    avg_forecast = combined["Forecast Pickup"].dropna().mean()
    cols = st.columns(4)
    with cols[0]:
        render_kpi("DPU Matches", f"{int(matched.sum())}/{len(combined)}", "#0f172a")
    with cols[1]:
        render_kpi("Total Pickup", f"{total_pickup:+,.0f}", "#0f766e" if total_pickup >= 0 else "#ef4444")
    with cols[2]:
        render_kpi("Avg Forecast Pickup", "" if pd.isna(avg_forecast) else f"{avg_forecast:+,.0f}", "#3b82f6")
    with cols[3]:
        render_kpi("Strong Pickup Dates", str(int((combined["Rooms Pickup"] > 10).sum())), "#7c3aed")

    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            f'<div class="callout"><b>{len(urgent)}</b> high-demand date(s) have negative room pickup. '
            f'{_date_list(urgent)}</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="callout"><b>{len(sold_cheap)}</b> sold-out date(s) have ADR on books to review. '
            f'{_date_list(sold_cheap)}</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div class="callout"><b>{len(soft_ahead)}</b> soft-demand date(s) have strong room pickup. '
            f'{_date_list(soft_ahead)}</div>',
            unsafe_allow_html=True,
        )

    display = combined.copy()
    display["Date"] = display["Date"].dt.strftime("%a %b %d")
    for column in ["ADR on Books", "My price"]:
        if column in display.columns:
            display[column] = display[column].map(money)
    for column in [
        "Rooms on Books",
        "Transient Rooms",
        "Group Rooms",
        "Rooms Pickup",
        "Transient Pickup",
        "Group Pickup",
        "Cumulative Rooms Pickup",
        "Pickup Per Day",
        "Forecast Pickup",
        "Forecast Rooms OTB",
    ]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):,.0f}")
    st.dataframe(display, use_container_width=True, hide_index=True, height=460)

    st.markdown('<div class="section-label">Pickup Detail</div>', unsafe_allow_html=True)
    detail = combined[combined["Rooms on Books"].notna()].copy()
    if detail.empty:
        st.info("No DPU rows matched the Lighthouse dates. Check whether the DPU month matches the Lighthouse arrival month.")
    else:
        options = detail["Date"].dt.strftime("%Y-%m-%d").tolist()
        selected = st.selectbox("Arrival date pickup detail", options=options)
        row = detail[detail["Date"].dt.strftime("%Y-%m-%d") == selected].iloc[0]
        explanation = pickup_explanation(row)
        st.markdown(
            f'<div class="callout">For <b>{row["Date"].strftime("%a %b %d")}</b>, DPU pickup is '
            f'<b>{_signed_number(row.get("Rooms Pickup"))}</b> rooms: transient '
            f'<b>{_signed_number(row.get("Transient Pickup"))}</b>, group '
            f'<b>{_signed_number(row.get("Group Pickup"))}</b>. <b>{html.escape(explanation)}</b> '
            f'Cumulative rooms moved from <b>{_whole_number(row.get("Pickup Start Rooms"))}</b> to '
            f'<b>{_whole_number(row.get("Rooms on Books"))}</b>, a total change of '
            f'<b>{_signed_number(row.get("Cumulative Rooms Pickup"))}</b> across '
            f'<b>{_whole_number(row.get("Pickup Snapshot Count"))}</b> DPU snapshots. Current velocity is '
            f'<b>{_signed_number(row.get("Pickup Per Day"))}</b> rooms/day. Forecast additional pickup is '
            f'<b>{_signed_number(row.get("Forecast Pickup"))}</b>, putting projected rooms OTB near '
            f'<b>{_whole_number(row.get("Forecast Rooms OTB"))}</b>.</div>',
            unsafe_allow_html=True,
        )


def render_rate_recommendations(df_future: pd.DataFrame, dpu_df: pd.DataFrame | None, config: dict[str, Any]) -> None:
    """Render transient rate recommendations and exports.

    Args:
        df_future: Forward-period daily detail dataframe.
        dpu_df: Optional parsed DPU dataframe.
        config: Property-level settings.

    Returns:
        None.
    """
    if dpu_df is None or dpu_df.empty:
        st.info("Upload a DPU report to unlock this view. Showing reduced recommendations from Lighthouse data only.")

    rows: list[dict[str, Any]] = []
    details: dict[str, list[str]] = {}
    for _, row in df_future.sort_values("Date").iterrows():
        dpu_row = _lookup_dpu(dpu_df, row["Date"])
        recommendation = recommend_rate(row, dpu_row, config)
        date_key = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
        rows.append(
            {
                "Date": pd.Timestamp(row["Date"]),
                "Demand": row.get("Demand level"),
                "Rooms OTB": None if dpu_row is None else dpu_row.get("Rooms on Books"),
                "ADR OTB": None if dpu_row is None else dpu_row.get("ADR on Books"),
                "STLY ADR": None if dpu_row is None else dpu_row.get("STLY ADR"),
                "Recommended Rate": recommendation["recommended_rate"],
                "Current Rate": recommendation["current_rate"],
                "Delta ($)": recommendation["rate_delta"],
            }
        )
        details[date_key] = recommendation["step_by_step"]

    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No future dates are available for recommendations.")
        return

    display = table.copy()
    display["Date Label"] = display["Date"].dt.strftime("%a %b %d")
    styled = display[
        ["Date Label", "Demand", "Rooms OTB", "ADR OTB", "STLY ADR", "Recommended Rate", "Current Rate", "Delta ($)"]
    ].style.map(_delta_color, subset=["Delta ($)"]).format(
        {
            "Rooms OTB": lambda value: "" if pd.isna(value) else f"{float(value):,.0f}",
            "ADR OTB": money,
            "STLY ADR": money,
            "Recommended Rate": money,
            "Current Rate": money,
            "Delta ($)": lambda value: "" if pd.isna(value) else f"${float(value):+,.0f}",
        }
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    selected = st.selectbox(
        "Recommendation walkthrough",
        options=table["Date"].dt.strftime("%Y-%m-%d").tolist(),
    )
    with st.expander(f"Calculation for {selected}", expanded=True):
        for step in details[selected]:
            st.write(step)

    export = table.copy()
    export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
    st.download_button(
        "Download recommendations CSV",
        data=export.to_csv(index=False),
        file_name="rate_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_demand_trends() -> None:
    """Render persisted demand trend analysis.

    Args:
        None.

    Returns:
        None.
    """
    st.warning("Snapshot history is stored locally in ./snapshots/. On Streamlit Cloud this resets on redeploy; export CSV to preserve it locally.")
    trend = build_trend_dataframe()
    if trend.empty:
        st.info("No stored snapshots are available yet. Upload a Lighthouse snapshot to begin building trend history.")
        return

    arrival_dates = sorted(trend["arrival_date"].dt.date.unique())
    selected_date = st.date_input("Arrival date to analyze", value=arrival_dates[0], min_value=min(arrival_dates), max_value=max(arrival_dates))
    selected = trend[trend["arrival_date"].dt.date == selected_date].sort_values("snapshot_date")
    if selected.empty:
        st.info("No trend records exist for the selected arrival date.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=selected["snapshot_date"],
                y=selected["demand_level_num"],
                text=selected["demand_level_str"],
                mode="lines+markers",
                name="Demand forecast",
                hovertemplate="%{x|%b %d}<br>%{text}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=selected["snapshot_date"],
                y=selected["my_price"],
                mode="lines+markers",
                name="My rate",
                line=dict(color="#0f172a"),
                hovertemplate="%{x|%b %d}<br>Rate: $%{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )
        style_plot(fig, height=330)
        fig.update_yaxes(tickvals=list(DEMAND_NUM.values()), ticktext=DEMAND_ORDER, secondary_y=False, title_text="Demand")
        fig.update_yaxes(tickprefix="$", secondary_y=True, title_text="Rate")
        st.plotly_chart(fig, use_container_width=True)

        first = selected.iloc[0]
        last = selected.iloc[-1]
        st.markdown(
            f'<div class="callout">Demand for <b>{selected_date}</b> has moved from '
            f'<b>{html.escape(str(first["demand_level_str"]))}</b> to <b>{html.escape(str(last["demand_level_str"]))}</b> '
            f'over <b>{len(selected)}</b> snapshots. Rate moved from <b>{money(first["my_price"])}</b> to '
            f'<b>{money(last["my_price"])}</b>.</div>',
            unsafe_allow_html=True,
        )

    heatmap = trend.copy()
    heatmap["Snapshot"] = heatmap["snapshot_date"].dt.strftime("%m/%d")
    heatmap["Arrival"] = heatmap["arrival_date"].dt.strftime("%m/%d")
    values = heatmap.pivot_table(index="Snapshot", columns="Arrival", values="demand_level_num", aggfunc="last")
    labels = heatmap.pivot_table(index="Snapshot", columns="Arrival", values="demand_level_str", aggfunc="last")
    fig_heat = go.Figure(
        go.Heatmap(
            z=values.values,
            x=values.columns,
            y=values.index,
            text=labels.values,
            texttemplate="%{text}",
            colorscale=[[0, "#94a3b8"], [0.2, "#60a5fa"], [0.45, "#fbbf24"], [0.7, "#f97316"], [1, "#ef4444"]],
            showscale=False,
        )
    )
    style_plot(fig_heat, height=360)
    st.plotly_chart(fig_heat, use_container_width=True)

    export = trend.copy()
    export["snapshot_date"] = export["snapshot_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    export["arrival_date"] = export["arrival_date"].dt.strftime("%Y-%m-%d")
    st.download_button(
        "Download trend CSV",
        data=export.to_csv(index=False),
        file_name="demand_trends.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_decision_log(df_future: pd.DataFrame) -> None:
    """Render decision entry and history management.

    Args:
        df_future: Forward-period daily detail dataframe.

    Returns:
        None.
    """
    st.warning("Decision history is stored locally in ./decisions/log.json. On Streamlit Cloud this resets on redeploy; export CSV to preserve it locally.")
    st.markdown('<div class="section-label">Log a Decision</div>', unsafe_allow_html=True)
    min_date = df_future["Date"].min().date()
    max_date = df_future["Date"].max().date()
    selected_date = st.date_input("Arrival date", value=min_date, min_value=min_date, max_value=max_date, key="decision_date")
    selected_rows = df_future[df_future["Date"].dt.date == selected_date]
    selected = selected_rows.iloc[0] if not selected_rows.empty else pd.Series(dtype=object)
    current_rate = selected.get("My price", np.nan)
    current_numeric = pd.to_numeric(current_rate, errors="coerce")
    rate_before_default = 0.0 if str(current_rate).lower() == "sold out" or pd.isna(current_numeric) else float(current_numeric)

    cols = st.columns(2)
    with cols[0]:
        st.write(f"Demand at time: {selected.get('Demand level', 'N/A')}")
        st.write(f"Compset level: {selected.get('Smart Compset price level', 'N/A')}")
        action = st.selectbox(
            "Action",
            ["Raised rate", "Lowered rate", "Closed channel", "Added restriction", "Opened availability", "No action"],
        )
    with cols[1]:
        rate_before = st.number_input("Rate before", value=rate_before_default, step=5.0)
        rate_after = st.number_input("Rate after", value=rate_before_default, step=5.0)
    notes = st.text_area("Notes")
    if st.button("Save decision", type="primary"):
        append_decision(
            {
                "arrival_date": selected_date.isoformat(),
                "action_taken": action,
                "rate_before": rate_before,
                "rate_after": rate_after,
                "demand_at_time": selected.get("Demand level", ""),
                "compset_at_time": selected.get("Smart Compset price level", ""),
                "notes": notes,
                "outcome": "",
            }
        )
        st.success("Decision saved.")

    st.markdown('<div class="section-label">Decision History</div>', unsafe_allow_html=True)
    history = decisions_dataframe()
    if history.empty:
        st.info("No decisions have been logged yet.")
        return

    history = history.copy()
    history["Delta"] = pd.to_numeric(history["rate_after"], errors="coerce") - pd.to_numeric(history["rate_before"], errors="coerce")
    history = history.sort_values("arrival_date", ascending=False).reset_index(drop=False).rename(columns={"index": "Log Index"})
    display = history.rename(
        columns={
            "arrival_date": "Date",
            "action_taken": "Action",
            "rate_before": "Rate Before",
            "rate_after": "Rate After",
            "demand_at_time": "Demand at Time",
            "compset_at_time": "Compset at Time",
            "notes": "Notes",
            "outcome": "Outcome",
        }
    )
    st.dataframe(
        display[["Date", "Action", "Rate Before", "Rate After", "Delta", "Demand at Time", "Compset at Time", "Notes", "Outcome"]],
        use_container_width=True,
        hide_index=True,
    )
    selected_index = st.selectbox("Update outcome for log entry", options=history["Log Index"].tolist())
    current_outcome = str(history.loc[history["Log Index"] == selected_index, "outcome"].iloc[0])
    outcome = st.text_input("Outcome", value=current_outcome)
    if st.button("Save outcome"):
        update_decision_outcome(int(selected_index), outcome)
        st.success("Outcome updated.")

    st.download_button(
        "Download decision log CSV",
        data=display.to_csv(index=False),
        file_name="decision_log.csv",
        mime="text/csv",
        use_container_width=True,
    )


def build_operational_table(df_future: pd.DataFrame, dpu_df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Merge Lighthouse daily detail rows with DPU rows.

    Args:
        df_future: Forward-period Lighthouse dataframe.
        dpu_df: Parsed DPU dataframe indexed by arrival date.
        config: Optional property settings for forecast limits.

    Returns:
        Combined operational dataframe.
    """
    left = df_future[
        ["Date", "Day", "Demand level", "Smart Compset price level", "My price", "My price level"]
    ].copy()
    left["Date"] = pd.to_datetime(left["Date"]).dt.normalize()
    left["MonthDay"] = left["Date"].dt.strftime("%m-%d")
    left["DayOfMonth"] = left["Date"].dt.day
    right = dpu_df.reset_index().copy()
    if "Arrival Date" in right.columns:
        right = right.rename(columns={"Arrival Date": "Date"})
    elif "index" in right.columns:
        right = right.rename(columns={"index": "Date"})
    right["Date"] = pd.to_datetime(right["Date"]).dt.normalize()
    right["DPU Date"] = right["Date"]
    right["MonthDay"] = right["Date"].dt.strftime("%m-%d")
    right["DayOfMonth"] = right["Date"].dt.day
    dpu_columns = [column for column in right.columns if column not in {"Date", "MonthDay", "DayOfMonth"}]
    combined = left.merge(right[["Date", "MonthDay", "DayOfMonth"] + dpu_columns], on=["Date", "MonthDay", "DayOfMonth"], how="left")
    combined["DPU Match"] = np.where(combined["Rooms on Books"].notna(), "Exact date", "No match")

    missing = combined["Rooms on Books"].isna()
    if missing.any():
        fallback = right.sort_values("Date").drop_duplicates("MonthDay", keep="last").set_index("MonthDay")
        for column in dpu_columns:
            combined.loc[missing, column] = combined.loc[missing, "MonthDay"].map(fallback[column])
        filled = missing & combined["Rooms on Books"].notna()
        combined.loc[filled, "DPU Match"] = "Month/day"

    missing = combined["Rooms on Books"].isna()
    if missing.any():
        day_fallback = right.sort_values("Date").drop_duplicates("DayOfMonth", keep="last").set_index("DayOfMonth")
        for column in dpu_columns:
            combined.loc[missing, column] = combined.loc[missing, "DayOfMonth"].map(day_fallback[column])
        filled = missing & combined["Rooms on Books"].notna()
        combined.loc[filled, "DPU Match"] = "Day-of-month"

    if "Rooms Pickup" not in combined.columns:
        combined["Rooms Pickup"] = np.nan
    if "Pickup Start Rooms" not in combined.columns:
        combined["Pickup Start Rooms"] = np.nan
    if "Pickup Per Day" not in combined.columns:
        combined["Pickup Per Day"] = np.nan
    if "Pickup Snapshot Count" not in combined.columns:
        combined["Pickup Snapshot Count"] = np.nan
    for column in ["Transient Rooms", "Group Rooms", "Transient Pickup", "Group Pickup", "Cumulative Rooms Pickup"]:
        if column not in combined.columns:
            combined[column] = np.nan
    if "Rooms Variance" not in combined.columns:
        combined["Rooms Variance"] = combined["Rooms on Books"] - combined["STLY Rooms"]
    if "ADR Variance" not in combined.columns:
        combined["ADR Variance"] = combined["ADR on Books"] - combined["STLY ADR"]
    combined["Forecast Pickup"] = combined.apply(lambda row: forecast_room_pickup(row, config or {}), axis=1)
    combined["Forecast Rooms OTB"] = combined["Rooms on Books"] + combined["Forecast Pickup"]
    combined["Pickup Explanation"] = combined.apply(pickup_explanation, axis=1)
    combined["Pace Status"] = np.select(
        [combined["Rooms Variance"].isna(), combined["Rooms Variance"] > 5, combined["Rooms Variance"] < -5],
        ["N/A", "Ahead", "Behind"],
        default="On Pace",
    )
    return combined[
        [
            "Date",
            "Day",
            "Demand level",
            "Smart Compset price level",
            "My price",
            "My price level",
            "Rooms on Books",
            "ADR on Books",
            "Transient Rooms",
            "Group Rooms",
            "Pickup Start Rooms",
            "Rooms Pickup",
            "Transient Pickup",
            "Group Pickup",
            "Cumulative Rooms Pickup",
            "Pickup Per Day",
            "Pickup Snapshot Count",
            "Forecast Pickup",
            "Forecast Rooms OTB",
            "Pickup Explanation",
            "DPU Match",
        ]
    ]


def forecast_room_pickup(row: pd.Series, config: dict[str, Any]) -> float:
    """Forecast additional room pickup for one arrival date.

    Args:
        row: Combined Lighthouse/DPU operational row.
        config: Property settings containing total room count.

    Returns:
        Forecast additional room pickup, capped by remaining physical room supply.
    """
    rooms_on_books = pd.to_numeric(row.get("Rooms on Books"), errors="coerce")
    pickup_per_day = pd.to_numeric(row.get("Pickup Per Day"), errors="coerce")
    if pd.isna(rooms_on_books) or pd.isna(pickup_per_day):
        return float("nan")

    forecast_date = row.get("DPU Date") if pd.notna(row.get("DPU Date")) else row.get("Date")
    arrival_date = pd.Timestamp(forecast_date).normalize()
    days_to_arrival = max((arrival_date - current_business_day()).days, 0)
    if days_to_arrival == 0:
        return 0.0

    demand_multiplier = {
        "Low": 0.70,
        "Normal": 0.90,
        "Elevated": 1.05,
        "High": 1.20,
        "Very high": 1.35,
        "Sold out": 0.0,
    }.get(str(row.get("Demand level")), 1.0)
    forecast = pickup_per_day * min(days_to_arrival, 21) * demand_multiplier
    total_rooms = float(config.get("total_rooms", 433))
    remaining_supply = max(total_rooms - rooms_on_books, 0)
    return float(max(-rooms_on_books, min(remaining_supply, forecast)))


def pickup_explanation(row: pd.Series) -> str:
    """Explain the likely driver of room pickup for an operational row.

    Args:
        row: Combined Lighthouse/DPU operational row.

    Returns:
        Plain-English explanation for positive, flat, or negative pickup.
    """
    total = pd.to_numeric(row.get("Rooms Pickup"), errors="coerce")
    transient = pd.to_numeric(row.get("Transient Pickup"), errors="coerce")
    group = pd.to_numeric(row.get("Group Pickup"), errors="coerce")
    if pd.isna(total):
        return "Pickup detail is unavailable for this date."
    if total < 0:
        if pd.notna(group) and group < 0 and (pd.isna(transient) or abs(group) >= abs(transient)):
            return f"Pickup is negative mainly because group rooms were released ({_signed_number(group)} group rooms)."
        if pd.notna(transient) and transient < 0:
            return f"Pickup is negative mainly because transient rooms washed or cancelled ({_signed_number(transient)} transient rooms)."
        return "Pickup is negative, so inventory moved backward versus the prior DPU snapshot."
    if total > 0:
        if pd.notna(group) and group > 0 and pd.notna(transient) and transient > 0:
            return "Pickup is positive from both transient and group segments."
        if pd.notna(group) and group > 0 and (pd.isna(transient) or group >= transient):
            return f"Pickup is positive mainly from group rooms ({_signed_number(group)} group rooms)."
        if pd.notna(transient) and transient > 0:
            return f"Pickup is positive mainly from transient rooms ({_signed_number(transient)} transient rooms)."
        return "Pickup is positive versus the prior DPU snapshot."
    return "Pickup is flat versus the prior DPU snapshot."


def _lookup_dpu(dpu_df: pd.DataFrame | None, date_value: Any) -> pd.Series | None:
    """Find one DPU row by normalized arrival date."""
    if dpu_df is None or dpu_df.empty:
        return None
    key = pd.Timestamp(date_value).normalize()
    if key not in dpu_df.index:
        month_day = key.strftime("%m-%d")
        matches = dpu_df[dpu_df.index.strftime("%m-%d") == month_day]
        if matches.empty:
            return None
        return matches.iloc[-1]
    row = dpu_df.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _date_list(df: pd.DataFrame) -> str:
    """Return a compact list of dates for a callout."""
    if df.empty:
        return "No dates currently flagged."
    dates = df["Date"].dt.strftime("%b %d").head(5).tolist()
    suffix = "..." if len(df) > 5 else ""
    return f"Dates: {', '.join(dates)}{suffix}"


def _delta_color(value: Any) -> str:
    """Return CSS color for recommendation deltas."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or number == 0:
        return "color: #64748b"
    if number > 0:
        return "color: #16a34a; font-weight: 700"
    return "color: #ef4444; font-weight: 700"


def _whole_number(value: Any) -> str:
    """Format a whole numeric value for operational callouts."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "N/A"
    return f"{float(number):,.0f}"


def _signed_number(value: Any) -> str:
    """Format a signed numeric value for operational callouts."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "N/A"
    return f"{float(number):+,.0f}"


def render_market_intelligence(today_data: dict[str, pd.DataFrame | None], df_future: pd.DataFrame) -> None:
    """Render geo, LOS, and search-signal market intelligence.

    Args:
        today_data: Parsed current workbook dictionary.
        df_future: Forward-period daily detail dataframe.

    Returns:
        None.
    """
    left, right = st.columns(2)
    geo = today_data.get("geo")
    los = today_data.get("los")

    with left:
        st.markdown('<div class="section-label">Top Origin Countries - Hotel Meta/OTA</div>', unsafe_allow_html=True)
        if geo is not None and not geo.empty and {"country_hotel_meta", "pct_hotel_meta"}.issubset(geo.columns):
            geo_plot = geo.dropna(subset=["country_hotel_meta", "pct_hotel_meta"]).sort_values("pct_hotel_meta").tail(10)
            render_horizontal_bar(geo_plot, "pct_hotel_meta", "country_hotel_meta", "Search Mix", "#3b82f6", percent=True)
        else:
            st.info("Hotel Meta/OTA origin data is unavailable.")

        st.markdown('<div class="section-label">Avg LOS by Origin - Flight Meta/OTA</div>', unsafe_allow_html=True)
        if geo is not None and not geo.empty and {"country_flight_meta", "los_flight_meta"}.issubset(geo.columns):
            los_geo = geo.dropna(subset=["country_flight_meta", "los_flight_meta"]).sort_values("los_flight_meta")
            render_horizontal_bar(los_geo, "los_flight_meta", "country_flight_meta", "Avg LOS", "#7c3aed", percent=False)
        else:
            st.info("Flight Meta/OTA LOS by origin is unavailable.")

    with right:
        st.markdown('<div class="section-label">LOS Distribution by Channel</div>', unsafe_allow_html=True)
        if los is not None and not los.empty:
            render_los_distribution(los)
            render_los_callout(los)
        else:
            st.info("Stay pattern breakdown data is unavailable.")

        st.markdown('<div class="section-label">Search Signals vs Benchmark - Forward 30 Days</div>', unsafe_allow_html=True)
        render_search_signals(df_future)


def render_pricing_scan(df_future: pd.DataFrame) -> None:
    """Render the single-file pricing opportunity scan.

    Args:
        df_future: Forward-period daily detail dataframe.

    Returns:
        None.
    """
    st.markdown('<div class="section-label">Pricing Opportunity Scan</div>', unsafe_allow_html=True)
    rows = pricing_scan_rows(df_future)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("No high-demand underpricing flags were found in the forward period.")


def render_full_data(df_today: pd.DataFrame, df_future: pd.DataFrame) -> None:
    """Render full daily detail data and CSV export.

    Args:
        df_today: Current daily detail dataframe.
        df_future: Forward-period daily detail dataframe.

    Returns:
        None.
    """
    st.markdown('<div class="section-label">Full Daily Details</div>', unsafe_allow_html=True)
    future_only = st.toggle("Future dates only", value=True)
    data = (df_future if future_only else df_today).copy()
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    if "My price" in data.columns:
        data["My price"] = data["My price"].map(money)
    if "My price numeric" in data.columns:
        data = data.drop(columns=["My price numeric"])
    if "Unavailable hotels" in data.columns:
        data["Unavailable hotels"] = pd.to_numeric(data["Unavailable hotels"], errors="coerce").map(
            lambda value: "" if pd.isna(value) else f"{value:.1%}"
        )
    st.dataframe(data, use_container_width=True, hide_index=True, height=420)
    st.download_button(
        "Download CSV",
        data=data.to_csv(index=False),
        file_name="market_insights_daily_details.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_los_distribution(los: pd.DataFrame) -> None:
    """Render grouped LOS distribution bars.

    Args:
        los: Parsed stay pattern dataframe.

    Returns:
        None.
    """
    fig = go.Figure()
    channels = {
        "Flight Meta/OTA": "los_flight_meta",
        "Flight GDS": "los_flight_gds",
        "Hotel Meta/OTA": "los_hotel_meta",
        "Hotel GDS": "los_hotel_gds",
    }
    colors = ["#0284c7", "#0f766e", "#f97316", "#7c3aed"]
    for (label, column), color in zip(channels.items(), colors):
        if column in los.columns:
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=los["LOS bucket"],
                    y=pd.to_numeric(los[column], errors="coerce") * 100,
                    marker_color=color,
                    hovertemplate=f"%{{x}}<br>{label}: %{{y:.1f}}%<extra></extra>",
                )
            )
    style_plot(fig, height=320)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(ticksuffix="%", title="Share")
    st.plotly_chart(fig, use_container_width=True)


def render_los_callout(los: pd.DataFrame) -> None:
    """Render the hotel Meta/OTA LOS insight callout.

    Args:
        los: Parsed stay pattern dataframe.

    Returns:
        None.
    """
    if "los_hotel_meta" not in los.columns or "LOS bucket" not in los.columns:
        return
    values = pd.to_numeric(los["los_hotel_meta"], errors="coerce")
    if values.dropna().empty:
        return
    index = values.idxmax()
    bucket = los.loc[index, "LOS bucket"]
    pct = values.loc[index] * 100
    st.markdown(
        f'<div class="callout">Hotel Meta/OTA demand is most concentrated in <b>{html.escape(str(bucket))}</b> stays '
        f'at <b>{pct:.1f}%</b>. Prioritize rate fences and stay controls around that stay pattern.</div>',
        unsafe_allow_html=True,
    )


def render_search_signals(df_future: pd.DataFrame) -> None:
    """Render flight and hotel search signals.

    Args:
        df_future: Forward-period daily detail dataframe.

    Returns:
        None.
    """
    columns = ["Flight level (Meta/OTA) vs. benchmark", "Hotel level (Meta/OTA) vs. benchmark"]
    if not all(column in df_future.columns for column in columns):
        st.info("Search signal columns are unavailable.")
        return
    data = df_future.sort_values("Date").head(30).copy()
    fig = go.Figure()
    for label, column, color in [
        ("Flight Meta/OTA", columns[0], "#0284c7"),
        ("Hotel Meta/OTA", columns[1], "#f97316"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data[column].map(SIGNAL_NUM),
                text=data[column],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate="%{x|%b %d}<br>%{text}<extra></extra>",
            )
        )
    style_plot(fig, height=260)
    fig.update_yaxes(tickvals=[-1, 0, 1], ticktext=["Lower", "Normal", "Higher"])
    st.plotly_chart(fig, use_container_width=True)


def render_horizontal_bar(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    label: str,
    color: str,
    percent: bool,
) -> None:
    """Render a horizontal bar chart.

    Args:
        df: Source dataframe.
        x_column: Numeric x-axis column.
        y_column: Category y-axis column.
        label: Axis label.
        color: Bar color.
        percent: Whether source values are decimal percentages.

    Returns:
        None.
    """
    plot = df.copy()
    x_values = pd.to_numeric(plot[x_column], errors="coerce")
    if percent:
        x_values = x_values * 100
    fig = go.Figure(
        go.Bar(
            x=x_values,
            y=plot[y_column],
            orientation="h",
            marker_color=color,
            text=[f"{value:.1f}%" if percent else f"{value:.1f}" for value in x_values],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.1f}" + ("%" if percent else "") + "<extra></extra>",
        )
    )
    style_plot(fig, height=310)
    fig.update_xaxes(title=label, ticksuffix="%" if percent else "")
    st.plotly_chart(fig, use_container_width=True)


def pricing_scan_rows(df_future: pd.DataFrame) -> list[dict[str, Any]]:
    """Build pricing scan rows for high-demand underpriced dates.

    Args:
        df_future: Forward-period daily detail dataframe.

    Returns:
        List of issue rows for display.
    """
    rows: list[dict[str, Any]] = []
    for _, row in df_future.iterrows():
        demand = row.get("Demand level")
        my_level = row.get("My price level")
        compset_level = row.get("Smart Compset price level")
        if demand in {"High", "Very high"} and my_level in {"Very low", "Low", "Normal"}:
            rows.append(
                {
                    "Date": row["Date"].strftime("%a %b %d"),
                    "Demand": demand,
                    "My Rate": money(row.get("My price")),
                    "My Level": my_level,
                    "Compset Level": compset_level,
                    "Signal": "High demand with low or normal rate level",
                }
            )
        gap = LEVEL_NUM.get(str(compset_level), np.nan) - LEVEL_NUM.get(str(my_level), np.nan)
        if pd.notna(gap) and gap >= 2 and str(row.get("My price")).lower() != "sold out":
            rows.append(
                {
                    "Date": row["Date"].strftime("%a %b %d"),
                    "Demand": demand,
                    "My Rate": money(row.get("My price")),
                    "My Level": my_level,
                    "Compset Level": compset_level,
                    "Signal": "Compset is two or more levels above my rate",
                }
            )
    return rows


def position_vs_compset(row: pd.Series) -> str:
    """Compare my price level to the smart compset level.

    Args:
        row: Daily detail row.

    Returns:
        ``Above compset``, ``At compset``, ``Below compset``, or ``Sold out``.
    """
    if str(row.get("My price")).strip().lower() == "sold out":
        return "Sold out"
    my_level = LEVEL_NUM.get(str(row.get("My price level")))
    compset = LEVEL_NUM.get(str(row.get("Smart Compset price level")))
    if my_level is None or compset is None:
        return "Unknown"
    if my_level > compset:
        return "Above compset"
    if my_level < compset:
        return "Below compset"
    return "At compset"


def render_kpi(label: str, value: str, color: str) -> None:
    """Render a KPI tile.

    Args:
        label: KPI label.
        value: KPI value.
        color: Value color.

    Returns:
        None.
    """
    st.markdown(
        f'<div class="kpi-tile"><div class="kpi-value" style="color:{color}">{html.escape(value)}</div>'
        f'<div class="kpi-label">{html.escape(label)}</div></div>',
        unsafe_allow_html=True,
    )


def render_alert_card(alert: dict[str, Any]) -> None:
    """Render one alert card.

    Args:
        alert: Alert dictionary.

    Returns:
        None.
    """
    color = SEVERITY_COLORS.get(str(alert.get("severity")), "#3b82f6")
    st.markdown(
        f"""
<div class="alert-card" style="border-left-color:{color}">
  <div class="alert-title">{html.escape(str(alert.get("title", "")))}</div>
  <div class="alert-meta">{html.escape(str(alert.get("severity", "")).title())} | {html.escape(str(alert.get("date_str", "")))}</div>
  <div class="alert-body">{html.escape(str(alert.get("body", "")))}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def style_plot(fig: go.Figure, height: int) -> None:
    """Apply consistent Plotly styling.

    Args:
        fig: Plotly figure to mutate.
        height: Figure height in pixels.

    Returns:
        None.
    """
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=8, r=8, t=20, b=8),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")


def price_number(value: Any) -> float:
    """Convert a published price to a number.

    Args:
        value: Price cell value.

    Returns:
        Numeric price or NaN.
    """
    if pd.isna(value) or str(value).strip().lower() == "sold out":
        return float("nan")
    return float(pd.to_numeric(value, errors="coerce"))


def money(value: Any) -> str:
    """Format a value as whole-dollar currency.

    Args:
        value: Numeric or string price value.

    Returns:
        Whole-dollar formatted string.
    """
    if isinstance(value, str) and value.strip().lower() == "sold out":
        return "Sold out"
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "N/A"
    return f"${float(number):,.0f}"


if __name__ == "__main__":
    main()
