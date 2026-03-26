import sys
import threading
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import yfinance as yf
from finvizfinance.quote import Statements
from finvizfinance.util import web_scrap


# For debugging period selection between yfinance 0q / +1q vs finviz actuals
LAST_SELECT_INFO = {}


def _safe_isna(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None


def _fmt_num(x, decimals: int = 2) -> str:
    if x is None or _safe_isna(x):
        return ""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    # EPS can be a very small decimal number.
    if abs(xf) >= 1000:
        return f"{xf:,.0f}"
    return f"{xf:,.{decimals}f}".rstrip("0").rstrip(".")


def _fmt_money(x) -> str:
    if x is None or _safe_isna(x):
        return ""
    try:
        # finvizfinance "Total Revenue" is in $ millions.
        # Example: 1900 => $1.9B. Convert to dollars for K/M/B formatting.
        xf = float(x) * 1e6
    except Exception:
        return str(x)
    # Format in dollars with K/M/B: K=1e3, M=1e6, B=1e9.
    sign = "-" if xf < 0 else ""
    v = abs(xf)

    if v >= 1e9:
        return f"{sign}{v / 1e9:.2f}B".rstrip("0").rstrip(".")
    if v >= 1e6:
        return f"{sign}{v / 1e6:.2f}M".rstrip("0").rstrip(".")
    if v >= 1e3:
        return f"{sign}{v / 1e3:.2f}K".rstrip("0").rstrip(".")
    return f"{xf:,.0f}"


def _fmt_pct(x: Optional[float], decimals: int = 2) -> str:
    if x is None or _safe_isna(x):
        return "N/A"
    try:
        xf = float(x) * 100.0
    except Exception:
        return "N/A"
    s = f"{xf:+.{decimals}f}%"
    return s.rstrip("0").rstrip(".")


def fetch_yfinance_eps_next_q_est_and_prev(
    ticker: str, period: str = "0q"
) -> Tuple[Optional[float], Optional[float]]:
    """
    Use yfinance `earnings_estimate` to read the specified period (e.g. "0q" or "+1q")
    EPS estimate (avg), and return `yearAgoEps` as the YoY baseline for that quarter.
    Units: EPS per share.
    """
    try:
        t = yf.Ticker(ticker)
        df = getattr(t, "earnings_estimate", None)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None, None
        if period not in df.index:
            return None, None
        if "avg" not in df.columns:
            return None, None
        if "yearAgoEps" not in df.columns:
            return None, None

        curr = df.loc[period, "avg"]
        prev = df.loc[period, "yearAgoEps"]

        curr_f = None if curr is None or _safe_isna(curr) else float(curr)
        prev_f = None if prev is None or _safe_isna(prev) else float(prev)
        return curr_f, prev_f
    except Exception:
        return None, None


def fetch_yfinance_rev_next_q_est_and_prev_millions(
    ticker: str, period: str = "0q"
) -> Tuple[Optional[float], Optional[float]]:
    """
    Use yfinance `revenue_estimate` to read the specified period (e.g. "0q" or "+1q")
    revenue estimate (avg), and return `yearAgoRevenue` as the YoY baseline.
    Units: finviz-style "Total Revenue (millions)" (divide by 1e6).
    """
    try:
        t = yf.Ticker(ticker)
        df = getattr(t, "revenue_estimate", None)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None, None
        if period not in df.index:
            return None, None
        if "avg" not in df.columns:
            return None, None
        if "yearAgoRevenue" not in df.columns:
            return None, None

        curr = df.loc[period, "avg"]
        prev = df.loc[period, "yearAgoRevenue"]

        curr_m = None if curr is None or _safe_isna(curr) else float(curr) / 1e6
        prev_m = None if prev is None or _safe_isna(prev) else float(prev) / 1e6
        return curr_m, prev_m
    except Exception:
        return None, None


def _pick_row_index(income_df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    `income_df.index` contains metric names (e.g. TotalRevenue / Total Revenue / Diluted EPS).
    Perform a fuzzy match and return the best row label.
    """
    if income_df is None or income_df.empty:
        return None
    idx_str = [str(i) for i in income_df.index]

    # Exact match first.
    for c in candidates:
        if c in income_df.index:
            return c

    # Substring match next (case-insensitive).
    for c in candidates:
        c_low = c.lower()
        for i, i_str in zip(income_df.index, idx_str):
            if c_low in i_str.lower():
                return i
    return None


def _quarter_columns_sorted(df: pd.DataFrame, n: int) -> List[str]:
    """
    Columns in `quarterly_*` are typically date strings (e.g. 2024-03-31).
    Try sorting by time and return the last `n`.
    """
    cols = list(df.columns)
    if not cols:
        return []

    parsed: List[Tuple[str, Optional[pd.Timestamp]]] = []
    for c in cols:
        try:
            parsed.append((c, pd.to_datetime(str(c), errors="coerce")))
        except Exception:
            parsed.append((c, None))

    if all(p is not None for _, p in parsed) and any(p is not None for _, p in parsed):
        cols_sorted = [c for c, _ in sorted(parsed, key=lambda x: x[1])]
        return cols_sorted[-n:]

    # Fallback: keep original order and take the last `n`.
    return cols[-n:]


@dataclass
class QuarterEpsRevenue:
    quarter_end: str
    eps: Optional[float]
    revenue: Optional[float]
    eps_yoy: Optional[float]
    revenue_yoy: Optional[float]


def fetch_recent_eps_revenue(ticker: str, quarters: int = 3) -> List[QuarterEpsRevenue]:
    """
    Fetch finviz quarterly Income Statement via finvizfinance:
    - Total Revenue
    - EPS (Diluted)
    """
    statements = Statements()
    # Next-quarter estimates: yfinance can lag, so fetch both 0q and +1q.
    eps0_est, eps0_prev = fetch_yfinance_eps_next_q_est_and_prev(ticker, period="0q")
    eps1_est, eps1_prev = fetch_yfinance_eps_next_q_est_and_prev(ticker, period="+1q")
    rev0_est, rev0_prev = fetch_yfinance_rev_next_q_est_and_prev_millions(
        ticker, period="0q"
    )
    rev1_est, rev1_prev = fetch_yfinance_rev_next_q_est_and_prev_millions(
        ticker, period="+1q"
    )
    # YoY needs the same quarter one year earlier; take 4 extra quarters.
    q_total = quarters + 4

    df = statements.get_statements(ticker=ticker, statement="I", timeframe="Q")

    def parse_float(x) -> Optional[float]:
        if x is None or _safe_isna(x):
            return None
        s = str(x).strip()
        if not s or s == "-" or s.lower() == "n/a":
            return None
        # finviz can return "143,756.00" or "-0.16"
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return None

    def parse_date(x) -> str:
        # Period End Date example: 12/31/2025
        try:
            dt = pd.to_datetime(str(x), errors="coerce")
            if pd.isna(dt):
                return str(x)
            return dt.date().isoformat()
        except Exception:
            return str(x)

    required_rows = ["Period End Date", "Total Revenue", "EPS (Diluted)"]
    missing = [r for r in required_rows if r not in df.index]
    if missing:
        raise RuntimeError(f"finviz missing required rows: {missing}")

    # finviz columns are usually ordered newest -> older (e.g. 0 is the latest).
    cols_recent_to_old = list(df.columns)[:q_total]
    cols_chrono = cols_recent_to_old[::-1]  # old -> new

    quarter_ends = [parse_date(df.loc["Period End Date", c]) for c in cols_chrono]
    eps_vals = [parse_float(df.loc["EPS (Diluted)", c]) for c in cols_chrono]
    revenue_vals = [parse_float(df.loc["Total Revenue", c]) for c in cols_chrono]

    def pct_change(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
        if curr is None or prev is None:
            return None
        try:
            prev_f = float(prev)
            if prev_f == 0:
                return None
            # Intuitive YoY: (curr - prev) / abs(prev).
            # This avoids confusing results when both are negative.
            return (float(curr) - prev_f) / abs(prev_f)
        except Exception:
            return None

    eps_yoy_vals: List[Optional[float]] = [None] * len(eps_vals)
    rev_yoy_vals: List[Optional[float]] = [None] * len(revenue_vals)
    for i in range(len(eps_vals)):
        if i - 4 >= 0:
            eps_yoy_vals[i] = pct_change(eps_vals[i], eps_vals[i - 4])
            rev_yoy_vals[i] = pct_change(revenue_vals[i], revenue_vals[i - 4])

    eps_latest_actual = eps_vals[-1] if eps_vals else None
    rev_latest_actual = revenue_vals[-1] if revenue_vals else None
    eps_prev_actual = eps_vals[-2] if len(eps_vals) >= 2 else None
    rev_prev_actual = revenue_vals[-2] if len(revenue_vals) >= 2 else None

    def approximately_equal(a: Optional[float], b: Optional[float], decimals: int = 1) -> bool:
        """
        Compare after rounding to improve cross-source tolerance.
        """
        if a is None or b is None:
            return False
        try:
            return round(float(a), decimals) == round(float(b), decimals)
        except Exception:
            return False

    # Detect whether yfinance actuals lag by one quarter:
    # - If yfinance latest actual ≈ finviz latest actual => yfinance is up-to-date => use 0q for next quarter.
    # - If yfinance latest actual ≈ finviz previous actual => yfinance lags => use +1q for next quarter.
    use_eps_plus1 = False
    use_rev_plus1 = False
    yfin_eps_latest = None
    yfin_rev_latest_m = None
    yfin_col_latest = None
    try:
        t_yf = yf.Ticker(ticker)
        q_income = getattr(t_yf, "quarterly_income_stmt", None)
        if isinstance(q_income, pd.DataFrame) and not q_income.empty:
            cols = list(q_income.columns)
            parsed_cols: List[Tuple[str, Optional[pd.Timestamp]]] = []
            for c in cols:
                try:
                    parsed_cols.append((c, pd.to_datetime(str(c), errors="coerce")))
                except Exception:
                    parsed_cols.append((c, None))
            # Pick latest parseable date column.
            dated = [(c, dt) for c, dt in parsed_cols if dt is not None and not pd.isna(dt)]
            if dated:
                dated_sorted = sorted(dated, key=lambda x: x[1])
                col_latest = dated_sorted[-1][0]
                yfin_col_latest = str(col_latest)

                # EPS (per share)
                if "Diluted EPS" in q_income.index:
                    yfin_eps_latest_val = q_income.loc["Diluted EPS", col_latest]
                    yfin_eps_latest = float(yfin_eps_latest_val) if yfin_eps_latest_val is not None and not _safe_isna(yfin_eps_latest_val) else None

                # Revenue (yfinance dollars -> /1e6 -> millions)
                if "Total Revenue" in q_income.index:
                    yfin_rev_latest_val = q_income.loc["Total Revenue", col_latest]
                    yfin_rev_latest_m = (
                        float(yfin_rev_latest_val) / 1e6
                        if yfin_rev_latest_val is not None and not _safe_isna(yfin_rev_latest_val)
                        else None
                    )

    except Exception:
        pass

    # Revenue selection
    rev_matches_latest = approximately_equal(yfin_rev_latest_m, rev_latest_actual, decimals=1)
    rev_matches_prev = approximately_equal(yfin_rev_latest_m, rev_prev_actual, decimals=1)
    if rev_matches_latest:
        use_rev_plus1 = False
    elif rev_matches_prev:
        use_rev_plus1 = True
    else:
        # Fallback: choose the closer quarter (based on 1-decimal distance).
        try:
            if rev_prev_actual is None or yfin_rev_latest_m is None:
                use_rev_plus1 = False
            else:
                d_latest = abs(round(yfin_rev_latest_m, 1) - round(rev_latest_actual, 1))
                d_prev = abs(round(yfin_rev_latest_m, 1) - round(rev_prev_actual, 1))
                use_rev_plus1 = d_prev < d_latest
        except Exception:
            use_rev_plus1 = False

    # EPS selection
    eps_matches_latest = approximately_equal(yfin_eps_latest, eps_latest_actual, decimals=1)
    eps_matches_prev = approximately_equal(yfin_eps_latest, eps_prev_actual, decimals=1)
    if eps_matches_latest:
        use_eps_plus1 = False
    elif eps_matches_prev:
        use_eps_plus1 = True
    else:
        try:
            if eps_prev_actual is None or yfin_eps_latest is None:
                use_eps_plus1 = False
            else:
                d_latest = abs(round(yfin_eps_latest, 1) - round(eps_latest_actual, 1))
                d_prev = abs(round(yfin_eps_latest, 1) - round(eps_prev_actual, 1))
                use_eps_plus1 = d_prev < d_latest
        except Exception:
            use_eps_plus1 = False

    # Select the more reliable period; if +1q is missing, fall back to 0q.
    eps_next_q_est = eps1_est if use_eps_plus1 and eps1_est is not None else eps0_est
    eps_next_q_prev = eps1_prev if use_eps_plus1 and eps1_prev is not None else eps0_prev
    rev_next_q_est = rev1_est if use_rev_plus1 and rev1_est is not None else rev0_est
    rev_next_q_prev = rev1_prev if use_rev_plus1 and rev1_prev is not None else rev0_prev

    eps_next_q_yoy = pct_change(eps_next_q_est, eps_next_q_prev)
    rev_next_q_yoy = pct_change(rev_next_q_est, rev_next_q_prev)

    global LAST_SELECT_INFO
    LAST_SELECT_INFO = {
        "EPS_use": "+1q" if use_eps_plus1 else "0q",
        "EPS_latest_actual": eps_latest_actual,
        "EPS_prev_actual": eps_prev_actual,
        "YF_eps_latest": yfin_eps_latest,
        "YF_col_latest": yfin_col_latest,
        "EPS_matches_latest": eps_matches_latest,
        "EPS_matches_prev": eps_matches_prev,
        "EPS_eps0_est": eps0_est,
        "EPS_eps1_est": eps1_est,
        "REV_use": "+1q" if use_rev_plus1 else "0q",
        "REV_latest_actual": rev_latest_actual,
        "REV_prev_actual": rev_prev_actual,
        "YF_rev_latest_m": yfin_rev_latest_m,
        "REV_matches_latest": rev_matches_latest,
        "REV_matches_prev": rev_matches_prev,
        "REV_rev0_est": rev0_est,
        "REV_rev1_est": rev1_est,
    }

    # Return the most recent `quarters` (quarter_ends is old -> new).
    start = max(0, len(quarter_ends) - quarters)
    results: List[QuarterEpsRevenue] = []
    for i in range(start, len(quarter_ends)):
        results.append(
            QuarterEpsRevenue(
                quarter_end=quarter_ends[i],
                eps=eps_vals[i],
                revenue=revenue_vals[i],
                eps_yoy=eps_yoy_vals[i],
                revenue_yoy=rev_yoy_vals[i],
            )
        )
    # Append: next-quarter estimates.
    next_label = "Next Quarter (Est)"
    nextnext_label = "Next Next Quarter (Est)"
    try:
        last_dt = pd.to_datetime(results[-1].quarter_end, errors="coerce")
        if not pd.isna(last_dt):
            next_end = (last_dt + pd.DateOffset(months=3)).date().isoformat()
            next_label = f"{next_end} (Est)"
            nextnext_end = (last_dt + pd.DateOffset(months=6)).date().isoformat()
            nextnext_label = f"{nextnext_end} (Est)"
    except Exception:
        pass

    # Only append when at least one estimate is available.
    if eps_next_q_est is not None or rev_next_q_est is not None:
        results.append(
            QuarterEpsRevenue(
                quarter_end=next_label,
                eps=eps_next_q_est,
                revenue=rev_next_q_est,
                eps_yoy=eps_next_q_yoy,
                revenue_yoy=rev_next_q_yoy,
            )
        )

    # Extra behavior:
    # If next quarter uses 0q (both EPS/REV use 0q), add an extra row for +1q
    # (i.e. the following quarter's estimate).
    if (not use_eps_plus1) and (not use_rev_plus1):
        if eps1_est is not None or rev1_est is not None:
            eps1_yoy = pct_change(eps1_est, eps1_prev)
            rev1_yoy = pct_change(rev1_est, rev1_prev)
            results.append(
                QuarterEpsRevenue(
                    quarter_end=nextnext_label,
                    eps=eps1_est,
                    revenue=rev1_est,
                    eps_yoy=eps1_yoy,
                    revenue_yoy=rev1_yoy,
                )
            )
    return results


def main():
    # Simple CLI mode (useful without a GUI).
    if "--nogui" in sys.argv:
        ticker = "AAPL"
        for i, a in enumerate(sys.argv):
            if a == "--ticker" and i + 1 < len(sys.argv):
                ticker = sys.argv[i + 1]
        rows = fetch_recent_eps_revenue(ticker, quarters=5)
        print(f"{ticker}: last 5 quarters EPS/REV (with YoY) + Next Quarter (Est)")
        if LAST_SELECT_INFO:
            print(
                f"[Select] EPS: {LAST_SELECT_INFO.get('EPS_use')} | "
                f"REV: {LAST_SELECT_INFO.get('REV_use')}"
            )
            # Detailed comparison numbers
            def fmt(x):
                if x is None:
                    return "N/A"
                return f"{x:.4f}"
            def fmt1(x):
                if x is None:
                    return "N/A"
                return f"{round(float(x), 1):.1f}"
            print(
                f"[Compare EPS Actuals] finviz_latest={fmt1(LAST_SELECT_INFO.get('EPS_latest_actual'))} "
                f"finviz_prev={fmt1(LAST_SELECT_INFO.get('EPS_prev_actual'))} | "
                f"yfinance_latest={fmt1(LAST_SELECT_INFO.get('YF_eps_latest'))} "
                f"(col={LAST_SELECT_INFO.get('YF_col_latest')}) | "
                f"match_latest={LAST_SELECT_INFO.get('EPS_matches_latest')} "
                f"match_prev={LAST_SELECT_INFO.get('EPS_matches_prev')}"
            )
            print(
                f"[Compare REV Actuals] finviz_latest={fmt1(LAST_SELECT_INFO.get('REV_latest_actual'))}M "
                f"finviz_prev={fmt1(LAST_SELECT_INFO.get('REV_prev_actual'))}M | "
                f"yfinance_latest={fmt1(LAST_SELECT_INFO.get('YF_rev_latest_m'))}M | "
                f"match_latest={LAST_SELECT_INFO.get('REV_matches_latest')} "
                f"match_prev={LAST_SELECT_INFO.get('REV_matches_prev')}"
            )
            print(
                f"[Estimates used] "
                f"EPS +0q={fmt1(LAST_SELECT_INFO.get('EPS_eps0_est'))} | "
                f"EPS +1q={fmt1(LAST_SELECT_INFO.get('EPS_eps1_est'))} | "
                f"REV +0q={fmt1(LAST_SELECT_INFO.get('REV_rev0_est'))}M | "
                f"REV +1q={fmt1(LAST_SELECT_INFO.get('REV_rev1_est'))}M"
            )
        for r in rows:
            print(
                f"{r.quarter_end}  EPS={_fmt_num(r.eps)}  EPS YoY={_fmt_pct(r.eps_yoy)}"
                f"  Revenue={_fmt_money(r.revenue)}  REV YoY={_fmt_pct(r.revenue_yoy)}"
            )
        return

    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as e:
        # If tkinter is unavailable, fall back to console output.
        ticker = "AAPL"
        for i, a in enumerate(sys.argv):
            if a == "--ticker" and i + 1 < len(sys.argv):
                ticker = sys.argv[i + 1]
        rows = fetch_recent_eps_revenue(ticker, quarters=5)
        print(f"{ticker}: last 5 quarters EPS/REV (with YoY) + Next Quarter (Est)")
        if LAST_SELECT_INFO:
            print(
                f"[Select] EPS: {LAST_SELECT_INFO.get('EPS_use')} | "
                f"REV: {LAST_SELECT_INFO.get('REV_use')}"
            )
            def fmt1(x):
                if x is None:
                    return "N/A"
                return f"{round(float(x), 1):.1f}"

            print(
                f"[Compare EPS Actuals] finviz_latest={fmt1(LAST_SELECT_INFO.get('EPS_latest_actual'))} "
                f"finviz_prev={fmt1(LAST_SELECT_INFO.get('EPS_prev_actual'))} | "
                f"yfinance_latest={fmt1(LAST_SELECT_INFO.get('YF_eps_latest'))} "
                f"(col={LAST_SELECT_INFO.get('YF_col_latest')}) | "
                f"match_latest={LAST_SELECT_INFO.get('EPS_matches_latest')} "
                f"match_prev={LAST_SELECT_INFO.get('EPS_matches_prev')}"
            )
            print(
                f"[Compare REV Actuals] finviz_latest={fmt1(LAST_SELECT_INFO.get('REV_latest_actual'))}M "
                f"finviz_prev={fmt1(LAST_SELECT_INFO.get('REV_prev_actual'))}M | "
                f"yfinance_latest={fmt1(LAST_SELECT_INFO.get('YF_rev_latest_m'))}M | "
                f"match_latest={LAST_SELECT_INFO.get('REV_matches_latest')} "
                f"match_prev={LAST_SELECT_INFO.get('REV_matches_prev')}"
            )
            print(
                f"[Estimates used] "
                f"EPS +0q={fmt1(LAST_SELECT_INFO.get('EPS_eps0_est'))} | "
                f"EPS +1q={fmt1(LAST_SELECT_INFO.get('EPS_eps1_est'))} | "
                f"REV +0q={fmt1(LAST_SELECT_INFO.get('REV_rev0_est'))}M | "
                f"REV +1q={fmt1(LAST_SELECT_INFO.get('REV_rev1_est'))}M"
            )
        for r in rows:
            print(
                f"{r.quarter_end}  EPS={_fmt_num(r.eps)}  EPS YoY={_fmt_pct(r.eps_yoy)}"
                f"  Revenue={_fmt_money(r.revenue)}  REV YoY={_fmt_pct(r.revenue_yoy)}"
            )
        return

    root = tk.Tk()
    root.title("EPS & Revenue (Last 5 Quarters)")
    root.geometry("560x360")

    # Top input section
    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    ttk.Label(top, text="Ticker:").grid(row=0, column=0, sticky="w")
    ticker_var = tk.StringVar(value="AAPL")
    entry = ttk.Entry(top, textvariable=ticker_var, width=15)
    entry.grid(row=0, column=1, padx=(6, 12), sticky="w")

    status_var = tk.StringVar(value="Ready")
    status_label = ttk.Label(top, textvariable=status_var)
    status_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

    # Table section
    table_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
    table_frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(
        table_frame,
        columns=("quarter_end", "eps", "eps_yoy", "revenue", "revenue_yoy"),
        show="headings",
        height=7,
    )
    tree.heading("quarter_end", text="Quarter End")
    tree.heading("eps", text="EPS")
    tree.heading("eps_yoy", text="EPS YoY")
    tree.heading("revenue", text="Revenue")
    tree.heading("revenue_yoy", text="REV YoY")
    tree.column("quarter_end", width=180)
    tree.column("eps", width=120, anchor="e")
    tree.column("eps_yoy", width=110, anchor="e")
    tree.column("revenue", width=200, anchor="e")
    tree.column("revenue_yoy", width=110, anchor="e")
    tree.pack(fill="both", expand=True)

    def clear_tree():
        for item in tree.get_children():
            tree.delete(item)

    def set_loading(loading: bool):
        status_var.set("Loading..." if loading else "Ready")

    def worker():
        tck = ticker_var.get().strip().upper()
        if not tck:
            messagebox.showerror("Error", "Ticker cannot be empty.")
            set_loading(False)
            return
        set_loading(True)
        try:
            rows = fetch_recent_eps_revenue(tck, quarters=5)
            if not rows:
                raise RuntimeError("Failed to fetch EPS/Revenue data from finvizfinance (fields may have changed or network failed).")
        except Exception as e:
            rows = None
            err = str(e)
        else:
            err = None

        def update_ui():
            set_loading(False)
            clear_tree()
            if err:
                messagebox.showerror("Fetch Failed", f"{tck}: {err}")
                return
            for r in rows:
                tree.insert(
                    "",
                    "end",
                    values=(
                        r.quarter_end,
                        _fmt_num(r.eps),
                        _fmt_pct(r.eps_yoy),
                        _fmt_money(r.revenue),
                        _fmt_pct(r.revenue_yoy),
                    ),
                )

        root.after(0, update_ui)

    btn = ttk.Button(top, text="Fetch Data", command=lambda: threading.Thread(target=worker, daemon=True).start())
    btn.grid(row=0, column=2, sticky="w")

    root.mainloop()


if __name__ == "__main__":
    main()

