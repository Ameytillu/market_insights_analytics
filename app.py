"""Streamlit UI for the Market Insights Analytics revenue intelligence app."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from plotly.subplots import make_subplots

from alerts import generate_alerts
from comparator import DEMAND_NUM, DEMAND_ORDER, LEVEL_NUM, compare_snapshots
from parser import parse_lighthouse_export


APP_TITLE = "Market Insights Analytics"
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

    uploaded_yesterday, uploaded_today = render_uploaders()
    if uploaded_today is None:
        render_empty_state()
        render_ai_sidebar(None)
        return

    today_data, yesterday_data = parse_uploads(uploaded_today, uploaded_yesterday)
    df_today = today_data["daily"]
    df_yesterday = yesterday_data["daily"] if yesterday_data else None
    if df_today is None or df_today.empty:
        st.error("The uploaded workbook did not contain usable Daily details rows.")
        return

    df_today = df_today.copy()
    df_today["Date"] = pd.to_datetime(df_today["Date"]).dt.normalize()
    current_day = pd.Timestamp(datetime.today().date())
    df_future = df_today[df_today["Date"] >= current_day].copy()
    if df_future.empty:
        df_future = df_today.copy()

    render_snapshot_summary(df_today, df_yesterday is not None)
    render_ai_sidebar(df_future)

    if df_yesterday is not None:
        render_comparison_tabs(today_data, df_today, df_yesterday, df_future, current_day)
    else:
        render_single_file_tabs(today_data, df_today, df_future)


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


def render_uploaders() -> tuple[Any | None, Any | None]:
    """Render the two snapshot uploaders.

    Args:
        None.

    Returns:
        Tuple of yesterday upload and today upload.
    """
    st.markdown('<div class="section-label">Load Snapshots</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        yesterday = st.file_uploader("Yesterday's market insights export", type=["xlsx"], key="yesterday_upload")
    with right:
        today = st.file_uploader("Today's market insights export", type=["xlsx"], key="today_upload")
    return yesterday, today


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


def render_action_center(df_today: pd.DataFrame, df_yesterday: pd.DataFrame, current_day: pd.Timestamp) -> None:
    """Render alert summary, filterable alerts, and future overnight table.

    Args:
        df_today: Current daily detail dataframe.
        df_yesterday: Prior daily detail dataframe.
        current_day: Today's date normalized to midnight.

    Returns:
        None.
    """
    alerts = generate_alerts(df_today, df_yesterday)
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
