# ib_chart

A lightweight Flask server + HTML UI for Interactive Brokers (IBKR) charts.

## What it does

- Serves `ib_chart.html` (single chart) and `ib_multichart.html` (multi panel).
- Provides HTTP APIs for:
  - daily OHLCV (`/api/pricehistory`) from local CSVs or IB historical
  - intraday 1m/5m bars (`/api/intraday`, `/api/intraday/stream`) from IB realtime
  - quote (`/api/quote`)
  - EPS/Revenue endpoint (`/api/eps-revenue`)

## Requirements

- Python 3.10+ recommended
- TWS or IB Gateway running and API enabled

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configure (optional)

- **IB connection**
  - `IB_HOST` (default `127.0.0.1`)
  - `IB_PORT` (default `4001`)
  - `IB_CLIENT_ID` (default `1001`)

- **Local daily CSV directory**
  - `IB_CHART_LOCAL_DAILY_DIR`
  - Default: `D:\US_stocks_daily_data\listed stocks from 2000_cleaned`

## Run

```bash
python ib_server.py
```

Open:

- `http://127.0.0.1:5001/` (serves the chart UI)
- `http://127.0.0.1:5001/ib_chart.html`
- `http://127.0.0.1:5001/ib_multichart.html`

## License

MIT. See `LICENSE`.

