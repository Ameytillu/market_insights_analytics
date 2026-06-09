# Lighthouse Market Monitor

Lighthouse Market Monitor is a Streamlit revenue intelligence app for hotel revenue management teams using Lighthouse Market Insights Excel exports. Upload one daily snapshot for exploration, or upload yesterday and today together to unlock overnight pricing alerts and market-change analysis.

## Features

- Password-protected Streamlit app using `st.secrets["PASSWORD"]`
- Two uploaders for yesterday and today Lighthouse `.xlsx` exports
- Single-file snapshot mode for demand, rate, geo, LOS, and pricing-signal review
- Two-file comparison mode with overnight demand and rate deltas
- Prioritized alerts for demand surges, underpricing, compset gaps, sold-out flips, event flags, flight-search signals, and published rate changes
- Demand and rate dual-axis chart for the forward 60-day window
- Day-of-week demand heatmap
- Rate positioning table comparing my level with smart compset level
- Market intelligence charts for origin country mix, average LOS, LOS distribution, and benchmark search signals
- Full daily-detail dataframe with future-date toggle and CSV download
- Sidebar AI assistant using `gpt-4o-mini`, grounded in the loaded forward-period data and capped at 15 calls per session

## Data Source

Export **Market Insights** from Lighthouse as an `.xlsx` workbook. The app expects these sheets:

- **Daily details**: one row per stay date, usually around 60 rows, with fields such as `Date`, `Day`, `Demand level`, `Unavailable hotels`, `Nr. of events and holidays`, `My price`, `My price level`, `Smart Compset price level`, flight and hotel benchmark signals, origin columns, and most-searched LOS columns.
- **Geo breakdown**: top origin countries by channel with search mix percentages and average LOS.
- **Stay pattern breakdown**: LOS buckets such as `LOS 1`, `LOS 2`, `LOS 3`, `LOS 4-7`, and `LOS 8-14`, with channel shares stored as decimals.

Important parsing rules:

- `Sold out` is preserved as a string in `My price` and `My price level`.
- `--` is treated as null.
- Openpyxl datetime objects are accepted directly, with Excel serial dates handled as a fallback.
- Demand and price levels are compared ordinally, not alphabetically.
- Pricing-action alerts only evaluate future dates.

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secrets Configuration

Create `.streamlit/secrets.toml` locally:

```toml
PASSWORD = "choose-a-dashboard-password"
OPENAI_API_KEY = "your-openai-api-key"
```

`OPENAI_API_KEY` is only required for the sidebar AI assistant. The rest of the app works without making OpenAI calls, but the assistant will show a configuration error if the key is missing.

## Streamlit Cloud Setup

1. Push the project to GitHub.
2. Create a Streamlit Cloud app and set the main file path to `app.py`.
3. Open **App settings**.
4. Go to **Secrets**.
5. Add:

```toml
PASSWORD = "choose-a-dashboard-password"
OPENAI_API_KEY = "your-openai-api-key"
```

6. Save secrets and redeploy the app.

## Project Files

```text
app.py            Streamlit UI, tabs, charts, upload flow, auth, and AI assistant
parser.py         Excel parser for Daily details, Geo breakdown, and Stay pattern breakdown
comparator.py     Today-vs-yesterday merge and ordinal delta calculations
alerts.py         Prioritized pricing and demand alert generation
requirements.txt  Python package requirements
README.md         Setup, secrets, feature, and data-source documentation
```
