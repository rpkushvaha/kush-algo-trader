import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

IST = ZoneInfo("Asia/Kolkata")

# Nifty 50 list as understood in Nov 2026. Verify against niftyindices.com and edit in the sidebar if needed.
DEFAULT_SYMBOLS = (
    "ADANIENT, ADANIPORTS, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJFINANCE, BAJAJFINSV, "
    "BEL, BHARTIARTL, BSE, CIPLA, COALINDIA, DRREDDY, EICHERMOT, ETERNAL, GRASIM, HCLTECH, HDFCBANK, "
    "HDFCLIFE, HINDALCO, HINDUNILVR, ICICIBANK, INDIGO, INFY, ITC, JIOFIN, JSWSTEEL, KOTAKBANK, LT, "
    "M&M, MARUTI, MAXHEALTH, NESTLEIND, NTPC, ONGC, POWERGRID, RELIANCE, SBILIFE, SBIN, SHRIRAMFIN, "
    "SUNPHARMA, TATACONSUM, TATASTEEL, TCS, TECHM, TITAN, TMPV, TRENT, ULTRACEMCO"
)

# ============================== ENGINE START ==============================

def parse_symbols(text):
    out = []
    for s in text.replace("\n", ",").split(","):
        s = s.strip().upper().replace(" ", "")
        if s and s not in out:
            out.append(s)
    return out


def to_yf(sym):
    if sym.startswith("^") or sym.endswith((".NS", ".BO")):
        return sym
    return sym + ".NS"


def short_name(yf_sym):
    return yf_sym.replace(".NS", "").replace(".BO", "")


def clean_frame(frame):
    frame = frame[["Open", "High", "Low", "Close"]].copy()
    idx = pd.to_datetime(frame.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    frame.index = idx.normalize()
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame[(frame > 0).all(axis=1)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def add_indicators(df, breakout_days):
    d = df.copy()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    prev_close = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev_close).abs(), (d["Low"] - prev_close).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["HH"] = d["High"].rolling(int(breakout_days)).max().shift(1)
    d["MOM63"] = d["Close"] / d["Close"].shift(63) - 1
    d["MOM126"] = d["Close"] / d["Close"].shift(126) - 1
    d["Above200"] = d["Close"] > d["SMA200"]
    d["Trend_OK"] = (d["Close"] > d["SMA200"]) & (d["EMA20"] > d["EMA50"])
    d["Breakout"] = d["Close"] > d["HH"]
    d["Entry"] = d["Trend_OK"] & d["Breakout"] & d["ATR"].notna()
    d["Exit"] = d["Close"] < d["EMA50"]
    return d


def market_filter(nifty_df):
    sma = nifty_df["Close"].rolling(200).mean()
    return nifty_df["Close"] > sma


def run_backtest(data, market_ok, capital, risk, max_pos, daily_limit, atr_mult, rr, cost_side, warmup=200,
                 exit_mode="target", trail_mult=3.0):
    """Daily portfolio simulation for the breakout strategy.
    Signals are read at the close; entries/trend-exits happen at the next day's open.
    exit_mode="target": fixed stop-loss + fixed target.
    exit_mode="trail": no target; the stop rises behind the highest close (trail_mult x ATR at entry).
    Stop-loss and target are checked with each day's High/Low (stop assumed first if both are touched)."""
    if not data:
        return None
    dates = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in data.values()])))
    n = len(dates)
    if n < warmup + 20:
        return None
    A = {}
    for s, d in data.items():
        A[s] = {c: d[c].reindex(dates).to_numpy(dtype=float) for c in ["Open", "High", "Low", "Close", "ATR", "MOM63"]}
        A[s]["Entry"] = d["Entry"].astype(float).reindex(dates).fillna(0).to_numpy() > 0.5
        A[s]["Exit"] = d["Exit"].astype(float).reindex(dates).fillna(0).to_numpy() > 0.5
    if market_ok is not None:
        mk = market_ok.astype(float).reindex(dates).ffill().fillna(0).to_numpy() > 0.5
    else:
        mk = np.ones(n, dtype=bool)

    positions, pending_entries, pending_exits, last_close, trades = {}, {}, set(), {}, []
    state = {"cash": float(capital), "day_pnl": 0.0}
    per_trade_cap = float(capital) / max_pos
    eq_vals, blocked_days = [], 0

    def close_pos(s, day, raw_px, reason):
        pos = positions.pop(s)
        exit_px = raw_px * (1 - cost_side)
        proceeds = pos["qty"] * exit_px
        state["cash"] += proceeds
        pnl = proceeds - pos["qty"] * pos["entry"]
        state["day_pnl"] += pnl
        trades.append({
            "Symbol": short_name(s), "Entry date": pos["entry_date"].date(), "Entry ₹": round(pos["entry"], 2),
            "Exit date": day.date(), "Exit ₹": round(exit_px, 2), "Qty": pos["qty"], "Exit reason": reason,
            "P/L (₹)": round(pnl, 2), "Return %": round((exit_px / pos["entry"] - 1) * 100, 2),
            "Days held": (day - pos["entry_date"]).days,
        })

    for i in range(warmup, n):
        day = dates[i]
        state["day_pnl"] = 0.0

        # 1. Trend exits at today's open (signal came yesterday)
        for s in list(pending_exits):
            if s not in positions:
                pending_exits.discard(s)
                continue
            o = A[s]["Open"][i]
            if np.isnan(o):
                continue
            close_pos(s, day, o, "Trend exit")
            pending_exits.discard(s)

        # 2. New entries at today's open (signal came yesterday)
        for s in list(pending_entries):
            atr = pending_entries.pop(s)
            o = A[s]["Open"][i]
            if np.isnan(o) or s in positions or len(positions) >= max_pos:
                continue
            dist = atr_mult * atr
            if not dist > 0:
                continue
            entry_px = o * (1 + cost_side)
            qty = min(int(risk // dist), int(per_trade_cap // entry_px), int(state["cash"] // entry_px))
            if qty < 1:
                continue
            state["cash"] -= qty * entry_px
            target = o + rr * dist if exit_mode == "target" else float("inf")
            positions[s] = {"qty": qty, "entry": entry_px, "stop": o - dist, "target": target, "entry_date": day,
                            "atr": atr, "high_close": o}

        # 3. Stop-loss / target during the day
        for s in list(positions):
            o, h, l = A[s]["Open"][i], A[s]["High"][i], A[s]["Low"][i]
            if np.isnan(l) or np.isnan(h):
                continue
            pos = positions[s]
            if np.isnan(o):
                o = A[s]["Close"][i]
            if l <= pos["stop"]:
                close_pos(s, day, min(o, pos["stop"]), "Stop-loss")
            elif h >= pos["target"]:
                close_pos(s, day, max(o, pos["target"]), "Target")

        # 4. Trend-exit signals at today's close; raise trailing stops
        for s, p in positions.items():
            if A[s]["Exit"][i]:
                pending_exits.add(s)
            if exit_mode == "trail":
                c = A[s]["Close"][i]
                if not np.isnan(c) and c > p["high_close"]:
                    p["high_close"] = c
                p["stop"] = max(p["stop"], p["high_close"] - trail_mult * p["atr"])

        # 5. Mark to market
        for s in positions:
            c = A[s]["Close"][i]
            if not np.isnan(c):
                last_close[s] = c
        equity = state["cash"] + sum(p["qty"] * last_close.get(s, p["entry"]) for s, p in positions.items())
        eq_vals.append(equity)

        # 6. New buy signals for tomorrow
        blocked = state["day_pnl"] <= -daily_limit
        if blocked:
            blocked_days += 1
        slots = max_pos - len(positions) - len(pending_entries)
        if not blocked and mk[i] and slots > 0:
            cands = []
            for s, a in A.items():
                if s in positions or s in pending_entries:
                    continue
                if a["Entry"][i] and a["ATR"][i] > 0:
                    m = a["MOM63"][i]
                    cands.append((m if not np.isnan(m) else -1e9, s, a["ATR"][i]))
            cands.sort(reverse=True)
            for _, s, atr in cands[:slots]:
                pending_entries[s] = atr

    open_pos = [{
        "Symbol": short_name(s), "Entry date": p["entry_date"].date(), "Entry ₹": round(p["entry"], 2), "Qty": p["qty"],
        "Last close ₹": round(last_close.get(s, p["entry"]), 2), "Stop-loss ₹": round(p["stop"], 2),
        "Target ₹": round(p["target"], 2) if np.isfinite(p["target"]) else None,
        "Open P/L (₹)": round(p["qty"] * (last_close.get(s, p["entry"]) - p["entry"]), 2),
    } for s, p in positions.items()]

    return {
        "equity": pd.Series(eq_vals, index=dates[warmup:]),
        "trades": pd.DataFrame(trades),
        "open": pd.DataFrame(open_pos),
        "blocked_days": blocked_days,
    }


def run_momentum(data, market_ok, capital, top_n, cost_side, warmup=200):
    """Monthly momentum rotation.
    On the last trading day of each month (at the close), rank stocks that are above their 200-day average
    by 6-month return. Hold the top N in equal amounts. A holding is kept while it stays in the top 2N
    (reduces churn and costs). If Nifty 50 is below its 200-day average, move to cash.
    Orders are executed at the next day's open."""
    if not data:
        return None
    dates = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in data.values()])))
    n = len(dates)
    if n < warmup + 30:
        return None
    A = {}
    for s, d in data.items():
        A[s] = {c: d[c].reindex(dates).to_numpy(dtype=float) for c in ["Open", "Close", "MOM126"]}
        A[s]["Above200"] = d["Above200"].astype(float).reindex(dates).fillna(0).to_numpy() > 0.5
    mk = (market_ok.astype(float).reindex(dates).ffill().fillna(0).to_numpy() > 0.5) if market_ok is not None else np.ones(n, dtype=bool)

    cash = float(capital)
    holdings, last_close, trades, eq_vals = {}, {}, [], []
    pending_target = None

    def value_now(i, use_open=False):
        total = cash
        for s, h in holdings.items():
            px = A[s]["Open"][i] if use_open else A[s]["Close"][i]
            if np.isnan(px):
                px = last_close.get(s, h["entry"])
            total += h["qty"] * px
        return total

    for i in range(warmup, n):
        day = dates[i]
        # Execute last month-end decision at today's open
        if pending_target is not None:
            for s in list(holdings):
                if s not in pending_target:
                    o = A[s]["Open"][i]
                    if np.isnan(o):
                        continue
                    h = holdings.pop(s)
                    exit_px = o * (1 - cost_side)
                    cash += h["qty"] * exit_px
                    pnl = h["qty"] * (exit_px - h["entry"])
                    trades.append({
                        "Symbol": short_name(s), "Entry date": h["entry_date"].date(), "Entry ₹": round(h["entry"], 2),
                        "Exit date": day.date(), "Exit ₹": round(exit_px, 2), "Qty": h["qty"],
                        "Exit reason": "Monthly rebalance" if mk[i - 1] else "Market filter (cash)",
                        "P/L (₹)": round(pnl, 2), "Return %": round((exit_px / h["entry"] - 1) * 100, 2),
                        "Days held": (day - h["entry_date"]).days,
                    })
            slot_value = value_now(i, use_open=True) / top_n
            for s in pending_target:
                if s in holdings:
                    continue
                o = A[s]["Open"][i]
                if np.isnan(o):
                    continue
                entry_px = o * (1 + cost_side)
                qty = int(min(slot_value, cash) // entry_px)
                if qty < 1:
                    continue
                cash -= qty * entry_px
                holdings[s] = {"qty": qty, "entry": entry_px, "entry_date": day}
            pending_target = None

        for s in holdings:
            c = A[s]["Close"][i]
            if not np.isnan(c):
                last_close[s] = c
        eq_vals.append(value_now(i))

        # Month-end decision
        is_month_end = (i == n - 1) or (dates[i + 1].month != day.month)
        if is_month_end and i < n - 1:
            if not mk[i]:
                pending_target = set()
            else:
                ranked = sorted(
                    [(A[s]["MOM126"][i], s) for s in A
                     if A[s]["Above200"][i] and not np.isnan(A[s]["MOM126"][i]) and A[s]["MOM126"][i] > 0],
                    reverse=True)
                order = [s for _, s in ranked]
                keep = [s for s in holdings if s in order[: 2 * top_n]]
                target = list(keep)
                for s in order:
                    if len(target) >= top_n:
                        break
                    if s not in target:
                        target.append(s)
                pending_target = set(target)

    open_pos = [{
        "Symbol": short_name(s), "Entry date": h["entry_date"].date(), "Entry ₹": round(h["entry"], 2), "Qty": h["qty"],
        "Last close ₹": round(last_close.get(s, h["entry"]), 2),
        "Open P/L (₹)": round(h["qty"] * (last_close.get(s, h["entry"]) - h["entry"]), 2),
    } for s, h in holdings.items()]
    return {"equity": pd.Series(eq_vals, index=dates[warmup:]), "trades": pd.DataFrame(trades),
            "open": pd.DataFrame(open_pos), "blocked_days": 0}


def momentum_picks(data, market_ok_now, top_n):
    rows = []
    for s, d in data.items():
        last = d.iloc[-1]
        if pd.notna(last["MOM126"]):
            rows.append({"Symbol": short_name(s), "6M return %": round(float(last["MOM126"]) * 100, 1),
                         "Above 200-day avg": "Yes" if bool(last["Above200"]) else "No",
                         "Close ₹": round(float(last["Close"]), 2)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("6M return %", ascending=False).reset_index(drop=True)
    eligible = (df["Above 200-day avg"] == "Yes") & (df["6M return %"] > 0)
    df["Rank"] = np.where(eligible, eligible.cumsum(), None)
    df["Pick"] = np.where(eligible & (df["Rank"].fillna(999).astype(int) <= top_n) & market_ok_now, "BUY / HOLD", "")
    return df[["Rank", "Symbol", "Pick", "6M return %", "Above 200-day avg", "Close ₹"]]


def summarize(eq, trades, capital):
    final = float(eq.iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    out = {
        "final": final,
        "total": (final / capital - 1) * 100,
        "cagr": ((final / capital) ** (1 / years) - 1) * 100 if final > 0 else -100.0,
        "max_dd": float((eq / eq.cummax() - 1).min() * 100),
        "n": len(trades), "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "pf": 0.0,
    }
    if len(trades):
        pl = trades["P/L (₹)"]
        wins, losses = pl[pl > 0], pl[pl <= 0]
        out["win_rate"] = len(wins) / len(pl) * 100
        out["avg_win"] = wins.mean() if len(wins) else 0.0
        out["avg_loss"] = losses.mean() if len(losses) else 0.0
        gl = abs(losses.sum())
        out["pf"] = wins.sum() / gl if gl > 0 else float("inf")
    return out


def scan(data, capital, risk, max_pos, atr_mult, rr):
    rows = []
    for s, d in data.items():
        if d.empty:
            continue
        last = d.iloc[-1]
        close, atr = float(last["Close"]), float(last["ATR"]) if pd.notna(last["ATR"]) else np.nan
        dist = atr_mult * atr if atr == atr else np.nan
        qty = int(min(risk // dist, (capital / max_pos) // close)) if dist == dist and dist > 0 else 0
        if bool(last["Entry"]):
            signal = "BUY"
        elif bool(last["Trend_OK"]):
            signal = "Uptrend, wait"
        else:
            signal = "No trade"
        rows.append({
            "Symbol": short_name(s), "Signal": signal, "Candle date": d.index[-1].date(),
            "Close ₹": round(close, 2),
            "Stop-loss ₹": round(close - dist, 2) if dist == dist else None,
            "Target ₹": round(close + rr * dist, 2) if dist == dist else None,
            "Qty": qty,
            "3M momentum %": round(float(last["MOM63"]) * 100, 1) if pd.notna(last["MOM63"]) else None,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {"BUY": 0, "Uptrend, wait": 1, "No trade": 2}
    df["_o"] = df["Signal"].map(order)
    df["_m"] = df["3M momentum %"].fillna(-1e9)
    return df.sort_values(["_o", "_m"], ascending=[True, False]).drop(columns=["_o", "_m"]).reset_index(drop=True)

# =============================== ENGINE END ===============================


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_many(yf_symbols, period):
    result, failed = {}, []
    raw = None
    try:
        raw = yf.download(list(yf_symbols), period=period, interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
    except Exception:
        raw = None
    for s in yf_symbols:
        try:
            frame = None
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    if s in raw.columns.get_level_values(0):
                        frame = raw[s]
                elif len(yf_symbols) == 1:
                    frame = raw
            if frame is None or frame.dropna(how="all").empty:
                frame = yf.Ticker(s).history(period=period, interval="1d", auto_adjust=True)
            frame = clean_frame(frame)
            if len(frame) < 60:
                failed.append(s)
                continue
            result[s] = frame
        except Exception:
            failed.append(s)
    return result, failed


# ================================== UI ==================================

st.set_page_config(page_title="Kush Algo Trader V4", page_icon="📈", layout="wide")
st.title("📈 Kush Algo Trader — Version 4")
st.caption("Swing trading research • Nifty 50 scanner • Portfolio backtest • Paper trades • No live broker orders")
st.warning(
    "PAPER / RESEARCH ONLY. Prices come from Yahoo Finance (unofficial, may be delayed or incomplete). "
    "Past results do not guarantee future results. This app is not connected to Zerodha and cannot place real orders."
)

if not YF_AVAILABLE:
    st.error("yfinance is not installed. Add `yfinance` to requirements.txt and redeploy.")
    st.stop()

with st.sidebar:
    st.header("Money rules")
    capital = st.number_input("Capital (₹)", min_value=10000, max_value=10000000, value=50000, step=5000)
    risk = st.number_input("Risk per trade (₹)", min_value=100, max_value=100000, value=500, step=100,
                           help="Maximum planned loss if the stop-loss is hit.")
    max_pos = int(st.number_input("Max open trades", min_value=1, max_value=10, value=3))
    daily_limit = st.number_input("Max loss per day (₹)", min_value=100, max_value=1000000, value=1000, step=100,
                                  help="If losses in a day reach this, no new trade is taken the next day.")
    st.divider()
    st.header("Strategy")
    breakout_days = int(st.number_input("Breakout lookback (days)", min_value=10, max_value=100, value=20))
    atr_mult = st.slider("Stop-loss distance (× ATR)", 1.0, 4.0, 2.0, 0.5)
    rr = st.slider("Target (× risk), fixed-target style", 1.0, 4.0, 2.0, 0.5)
    trail_mult = st.slider("Trailing stop distance (× ATR), trailing style", 2.0, 6.0, 3.0, 0.5)
    top_n = int(st.number_input("Momentum: number of stocks to hold", min_value=3, max_value=15, value=5))
    use_mf = st.checkbox("Trade only when Nifty 50 is above its 200-day average", value=True)
    st.divider()
    st.header("Costs")
    cost_pct = st.number_input("Trading cost per side (%)", min_value=0.0, max_value=1.0, value=0.15, step=0.01,
                               help="Rough estimate for delivery trades: STT, stamp duty, exchange, GST, DP charges.")
    slip_pct = st.number_input("Slippage per side (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    with st.expander("Stock list (Nifty 50)"):
        symbols_text = st.text_area("NSE symbols, comma separated", DEFAULT_SYMBOLS, height=220)
        st.caption("Check the latest list on niftyindices.com. Symbols that fail to load are skipped.")

cost_side = (cost_pct + slip_pct) / 100
symbols = parse_symbols(symbols_text)


def load_universe(syms, period):
    yfs = tuple([to_yf(s) for s in syms] + ["^NSEI"])
    raw, failed = fetch_many(yfs, period)
    nifty = raw.pop("^NSEI", None)
    data = {s: add_indicators(df, breakout_days) for s, df in raw.items()}
    mk = market_filter(nifty) if (nifty is not None and use_mf) else None
    return data, nifty, mk, [short_name(f) for f in failed if f != "^NSEI"]


tab_rules, tab_scan, tab_bt, tab_stock, tab_paper = st.tabs(
    ["📋 Rules", "🔍 Scanner", "📊 Backtest", "📈 Single stock", "📝 Paper trades"]
)

# ------------------------------- Rules -------------------------------
STRATS = ["A. Breakout + fixed target", "B. Breakout + trailing stop", "C. Monthly momentum"]

with tab_rules:
    st.subheader("Three strategies, tested side by side")
    st.markdown(f"""
All three trade only Nifty 50 stocks, delivery (no leverage, no options).
**Market filter:** {"new buying only when Nifty 50 is above its 200-day average" if use_mf else "OFF"}.

**A. Breakout + fixed target** (checked daily after 3:30 pm)
- Buy when: close above 200-day average, EMA20 above EMA50, and close breaks the {breakout_days}-day high.
- Stop-loss: entry − {atr_mult:g} × ATR. Target: {rr:g} × risk. Also sell on a close below EMA50.
- Risk ₹{risk:,} per trade, max {max_pos} trades, no new trade after a ₹{daily_limit:,} loss day.

**B. Breakout + trailing stop** (checked daily after 3:30 pm)
- Same buy signal and starting stop as A, but **no fixed target**.
- The stop rises behind the highest close ({trail_mult:g} × ATR below it) and never moves down. Winners are allowed to run.

**C. Monthly momentum** (checked once a month, last trading day)
- Rank stocks above their 200-day average by 6-month return.
- Hold the top {top_n} in equal amounts (₹{capital/top_n:,.0f} each). Keep a stock while it stays in the top {2*top_n}.
- If Nifty 50 falls below its 200-day average, sell everything and wait in cash.
- About 15–30 minutes of work per month.
""")
    st.info("Run the Backtest tab and compare all three against simply holding Nifty 50. Keep the one that wins over 10 years, "
            "then paper trade it for 2 months before using real money.")

# ------------------------------ Scanner ------------------------------
with tab_scan:
    st.subheader("Today's scanner")
    st.caption("Best used after 3:30 pm, when the day's candle is complete. During market hours the last candle is still changing.")
    if st.button("Scan Nifty 50 now", type="primary", width="stretch"):
        with st.spinner("Loading prices for all stocks..."):
            data, nifty, mk, failed = load_universe(symbols, "1y")
        mk_now = bool(mk.iloc[-1]) if mk is not None and len(mk) else True
        st.session_state.scan = {
            "table": scan(data, capital, risk, max_pos, atr_mult, rr),
            "mom": momentum_picks(data, mk_now, top_n),
            "nifty": nifty, "failed": failed,
            "time": datetime.now(IST).strftime("%d %b %Y, %I:%M %p"),
        }
    res = st.session_state.get("scan")
    if res:
        nifty = res["nifty"]
        if nifty is not None and len(nifty) >= 200:
            n_close, n_sma = float(nifty["Close"].iloc[-1]), float(nifty["Close"].rolling(200).mean().iloc[-1])
            if n_close > n_sma:
                st.success(f"Market filter: Nifty 50 ({n_close:,.0f}) is above its 200-day average ({n_sma:,.0f}). Buying allowed.")
            else:
                msg = f"Market filter: Nifty 50 ({n_close:,.0f}) is below its 200-day average ({n_sma:,.0f})."
                if use_mf:
                    st.error(msg + " No new buying by the rules.")
                else:
                    st.warning(msg + " (Filter is OFF in settings.)")
        else:
            st.warning("Could not load Nifty 50 index data, so the market filter could not be checked.")
        st.caption(f"Scanned at {res['time']}")

        st.write("### Strategy A / B: breakout signals")
        table = res["table"]
        buys = table[table["Signal"] == "BUY"] if not table.empty else table
        st.metric("BUY signals today", len(buys))
        if len(buys):
            st.dataframe(buys, width="stretch", hide_index=True)
            st.caption(f"Take at most your free slots (max {max_pos}), from the top. For strategy B, ignore the target column.")
        else:
            st.info("No breakout signal today.")
        with st.expander("All stocks (breakout view)"):
            st.dataframe(table, width="stretch", hide_index=True)

        st.write("### Strategy C: momentum portfolio")
        st.caption(f"Act on this only on the last trading day of the month. Hold the stocks marked BUY / HOLD "
                   f"(about ₹{capital/top_n:,.0f} each). Keep an old holding if its rank is {2*top_n} or better.")
        mom = res["mom"]
        if mom is not None and not mom.empty:
            st.dataframe(mom.head(max(2 * top_n, 10)), width="stretch", hide_index=True)
        if res["failed"]:
            st.caption("Could not load: " + ", ".join(res["failed"]))

# ------------------------------ Backtest ------------------------------
def show_details(result, capital, nifty):
    eq, trades = result["equity"], result["trades"]
    s = summarize(eq, trades, capital)
    c5, c6 = st.columns(2)
    c5.metric("Closed trades", s["n"])
    c6.metric("Win rate", f"{s['win_rate']:.1f}%")
    c7, c8 = st.columns(2)
    c7.metric("Average win / loss", f"₹{s['avg_win']:,.0f} / ₹{s['avg_loss']:,.0f}")
    c8.metric("Profit factor", f"{s['pf']:.2f}" if np.isfinite(s["pf"]) else "∞")
    yearly = eq.groupby(eq.index.year).last()
    prev = yearly.shift(1).fillna(capital)
    ytab = pd.DataFrame({"Year": yearly.index, "Strategy %": ((yearly / prev - 1) * 100).round(1).values})
    if nifty is not None:
        nc = nifty["Close"].reindex(eq.index).ffill().bfill()
        ny = nc.groupby(nc.index.year).last()
        nprev = ny.shift(1).fillna(nc.iloc[0])
        ytab["Nifty 50 %"] = ((ny / nprev - 1) * 100).round(1).values
    st.write("**Return by year**")
    st.dataframe(ytab, width="stretch", hide_index=True)
    if s["n"]:
        st.write("**Exit reasons**")
        st.dataframe(trades["Exit reason"].value_counts().rename_axis("Reason").reset_index(name="Trades"),
                     width="stretch", hide_index=True)
        st.write("**Profit / loss by stock**")
        st.bar_chart(trades.groupby("Symbol")["P/L (₹)"].sum().sort_values(), height=300)
        with st.expander(f"All {s['n']} trades"):
            st.dataframe(trades, width="stretch", hide_index=True)
            st.download_button("Download trades CSV", trades.to_csv(index=False).encode("utf-8"),
                               file_name="kush_backtest_trades.csv", mime="text/csv")
    if not result["open"].empty:
        st.write("**Still open at the end**")
        st.dataframe(result["open"], width="stretch", hide_index=True)
    if result["blocked_days"]:
        st.caption(f"Daily loss limit was hit on {result['blocked_days']} day(s).")


with tab_bt:
    st.subheader("Compare all strategies")
    st.caption("Runs A, B and C on the whole stock list with your money rules and costs, and compares them with simply holding Nifty 50.")
    period_label = st.selectbox("Data period", ["2 years", "5 years", "10 years"], index=2)
    period = {"2 years": "2y", "5 years": "5y", "10 years": "10y"}[period_label]
    if st.button("Run all backtests", type="primary", width="stretch"):
        with st.spinner("Downloading data and simulating... this can take 1–2 minutes"):
            data, nifty, mk, failed = load_universe(symbols, period)
            results = {
                STRATS[0]: run_backtest(data, mk, capital, risk, max_pos, daily_limit, atr_mult, rr, cost_side, exit_mode="target"),
                STRATS[1]: run_backtest(data, mk, capital, risk, max_pos, daily_limit, atr_mult, rr, cost_side,
                                        exit_mode="trail", trail_mult=trail_mult),
                STRATS[2]: run_momentum(data, mk, capital, top_n, cost_side),
            }
        st.session_state.bt = {"results": results, "nifty": nifty, "failed": failed, "capital": capital}
    bt = st.session_state.get("bt")
    if bt and "results" in bt:
        results, cap, nifty = bt["results"], bt["capital"], bt["nifty"]
        valid = {k: v for k, v in results.items() if v is not None}
        if not valid:
            st.error("Not enough data to run the backtest. Try a longer period.")
        else:
            first = next(iter(valid.values()))["equity"]
            st.caption(f"Test period: {first.index[0]:%d %b %Y} to {first.index[-1]:%d %b %Y} "
                       "(the first ~200 trading days are used only to calculate averages).")
            rows, chart = [], pd.DataFrame(index=first.index)
            for name, r in valid.items():
                s = summarize(r["equity"], r["trades"], cap)
                rows.append({"Strategy": name, "Ending ₹": round(s["final"]), "Total %": round(s["total"], 1),
                             "Yearly % (CAGR)": round(s["cagr"], 1), "Worst fall %": round(s["max_dd"], 1),
                             "Trades": s["n"]})
                chart[name] = r["equity"]
            if nifty is not None:
                nc = nifty["Close"].reindex(first.index).ffill().bfill()
                bench = cap * nc / nc.iloc[0]
                bs = summarize(bench, pd.DataFrame(), cap)
                rows.append({"Strategy": "Nifty 50 buy & hold", "Ending ₹": round(bs["final"]), "Total %": round(bs["total"], 1),
                             "Yearly % (CAGR)": round(bs["cagr"], 1), "Worst fall %": round(bs["max_dd"], 1), "Trades": 0})
                chart["Nifty 50 buy & hold"] = bench
            comp = pd.DataFrame(rows).sort_values("Yearly % (CAGR)", ascending=False).reset_index(drop=True)
            st.dataframe(comp, width="stretch", hide_index=True)
            best = comp.iloc[0]["Strategy"]
            if best == "Nifty 50 buy & hold":
                st.warning("Simply holding Nifty 50 beat every strategy in this test.")
            else:
                st.success(f"Best yearly return in this test: {best}. Check the 'Worst fall' column too: "
                           "a big fall is hard to sit through with real money.")
            st.line_chart(chart, height=320)

            pick = st.selectbox("See details of", list(valid.keys()))
            show_details(valid[pick], cap, nifty)
            st.caption("Limitations: uses today's Nifty 50 list for the past (makes results look better than reality), "
                       "dividend-adjusted prices, simple cost estimate, no tax, and orders assumed to fill at the open or "
                       "exactly at the stop/target. Gaps can cause bigger losses than planned. Past results do not "
                       "guarantee future results.")
            if bt["failed"]:
                st.caption("Could not load: " + ", ".join(bt["failed"]))

# ---------------------------- Single stock ----------------------------
with tab_stock:
    st.subheader("Check one stock")
    sym_in = st.text_input("NSE symbol (e.g. TCS, INFY, HDFCBANK)", "TCS")
    sym_period_label = st.selectbox("Data period ", ["2 years", "5 years", "10 years"], index=1)
    sym_period = {"2 years": "2y", "5 years": "5y", "10 years": "10y"}[sym_period_label]
    one = parse_symbols(sym_in)[:1]
    if one:
        with st.spinner("Loading..."):
            data1, nifty1, mk1, failed1 = load_universe(one, sym_period)
        if not data1:
            st.error(f"No data found for {one[0]}. Check the symbol spelling.")
        else:
            key = list(data1.keys())[0]
            d = data1[key]
            last = d.iloc[-1]
            st.line_chart(d[["Close", "EMA20", "EMA50", "SMA200"]].tail(250), height=300)
            if bool(last["Entry"]):
                sig = "BUY"
            elif bool(last["Trend_OK"]):
                sig = "Uptrend, wait for breakout"
            else:
                sig = "No trade"
            a1, a2 = st.columns(2)
            a1.metric("Latest close", f"₹{last['Close']:,.2f}")
            a2.metric("Signal", sig)
            style = st.radio("Exit style", ["Fixed target (A)", "Trailing stop (B)"], horizontal=True)
            r1 = run_backtest(data1, mk1, capital, risk, 1, daily_limit, atr_mult, rr, cost_side,
                              exit_mode="target" if style.startswith("Fixed") else "trail", trail_mult=trail_mult)
            if r1 is None:
                st.info("Not enough history for a backtest on this stock.")
            else:
                s1 = summarize(r1["equity"], r1["trades"], capital)
                b1, b2 = st.columns(2)
                b1.metric("Strategy return", f"{s1['total']:.1f}%")
                b2.metric("Closed trades", s1["n"])
                b3, b4 = st.columns(2)
                b3.metric("Win rate", f"{s1['win_rate']:.1f}%")
                b4.metric("Max drawdown", f"{s1['max_dd']:.1f}%")
                if s1["n"]:
                    st.dataframe(r1["trades"], width="stretch", hide_index=True)
                st.caption("One stock alone is a weak test. The Backtest tab (all stocks) matters more.")

# ---------------------------- Paper trades ----------------------------
with tab_paper:
    st.subheader("Paper trade log")
    st.caption("Virtual trades only. The log lives in this browser session: download the CSV after each update "
               "and load it back next time.")
    if "paper" not in st.session_state:
        st.session_state.paper = []
    paper = st.session_state.paper
    today = datetime.now(IST).date()
    open_trades = [t for t in paper if t["Status"] == "Open"]
    today_pl = sum(float(t["P/L (₹)"] or 0) for t in paper
                   if t["Status"] == "Closed" and str(t["Exit date"]) == str(today))

    p1, p2 = st.columns(2)
    p1.metric("Open trades", f"{len(open_trades)} / {max_pos}")
    p2.metric("Today's closed P/L", f"₹{today_pl:,.0f}")
    if today_pl <= -daily_limit:
        st.error(f"Daily loss limit of ₹{daily_limit:,} reached. No new trades by the rules.")
    elif len(open_trades) >= max_pos:
        st.warning("All trade slots are full. Wait for a trade to close.")

    with st.form("add_paper", clear_on_submit=True):
        st.write("**Add a trade** (copy values from the Scanner)")
        f1, f2 = st.columns(2)
        f_sym = f1.text_input("Symbol")
        f_qty = f2.number_input("Qty", min_value=1, value=1, step=1)
        f3, f4, f5 = st.columns(3)
        f_entry = f3.number_input("Entry ₹", min_value=0.0, value=None, step=0.05)
        f_stop = f4.number_input("Stop-loss ₹", min_value=0.0, value=None, step=0.05)
        f_target = f5.number_input("Target ₹", min_value=0.0, value=None, step=0.05)
        if st.form_submit_button("Add paper trade", type="primary", width="stretch"):
            if not f_sym or f_entry is None or f_stop is None or f_target is None:
                st.error("Fill in symbol, entry, stop-loss and target.")
            elif not (f_stop < f_entry < f_target):
                st.error("For a buy, stop-loss must be below entry and target above entry.")
            else:
                planned_risk = (f_entry - f_stop) * f_qty
                paper.append({
                    "ID": max([t["ID"] for t in paper], default=0) + 1, "Date": str(today),
                    "Symbol": f_sym.strip().upper(), "Qty": int(f_qty), "Entry ₹": round(f_entry, 2),
                    "Stop-loss ₹": round(f_stop, 2), "Target ₹": round(f_target, 2), "Status": "Open",
                    "Exit date": "", "Exit ₹": None, "P/L (₹)": None,
                })
                if planned_risk > risk:
                    st.warning(f"Planned risk ₹{planned_risk:,.0f} is more than your ₹{risk:,} rule.")
                st.success("Paper trade added.")
                st.rerun()

    open_trades = [t for t in paper if t["Status"] == "Open"]
    if open_trades:
        with st.form("close_paper"):
            st.write("**Close a trade**")
            labels = {t["ID"]: f"#{t['ID']} {t['Symbol']} × {t['Qty']} @ ₹{t['Entry ₹']}" for t in open_trades}
            c_id = st.selectbox("Trade", list(labels.keys()), format_func=lambda x: labels[x])
            c_px = st.number_input("Exit price ₹", min_value=0.0, value=None, step=0.05)
            if st.form_submit_button("Close trade", width="stretch"):
                if c_px is None:
                    st.error("Enter the exit price.")
                else:
                    for t in paper:
                        if t["ID"] == c_id:
                            t["Status"], t["Exit date"], t["Exit ₹"] = "Closed", str(today), round(c_px, 2)
                            t["P/L (₹)"] = round((c_px - t["Entry ₹"]) * t["Qty"], 2)
                    st.rerun()

    if paper:
        log = pd.DataFrame(paper)
        st.dataframe(log, width="stretch", hide_index=True)
        closed = log[log["Status"] == "Closed"]
        if len(closed):
            st.metric("Total closed P/L (before costs)", f"₹{closed['P/L (₹)'].astype(float).sum():,.0f}")
        st.download_button("Download paper log CSV", log.to_csv(index=False).encode("utf-8"),
                           file_name="kush_paper_trades.csv", mime="text/csv", width="stretch")

    with st.expander("Load a saved paper log"):
        up = st.file_uploader("Choose the CSV you downloaded earlier", type=["csv"])
        if up is not None and st.button("Load this file"):
            try:
                df_up = pd.read_csv(up, keep_default_na=False)
                need = {"ID", "Date", "Symbol", "Qty", "Entry ₹", "Stop-loss ₹", "Target ₹", "Status", "Exit date", "Exit ₹", "P/L (₹)"}
                if not need.issubset(df_up.columns):
                    st.error("This is not a Kush paper log file.")
                else:
                    recs = df_up[list(need)].to_dict("records")
                    for r in recs:
                        r["ID"], r["Qty"] = int(r["ID"]), int(r["Qty"])
                        r["Exit ₹"] = float(r["Exit ₹"]) if str(r["Exit ₹"]).strip() not in ("", "None", "nan") else None
                        r["P/L (₹)"] = float(r["P/L (₹)"]) if str(r["P/L (₹)"]).strip() not in ("", "None", "nan") else None
                        r["Exit date"] = "" if str(r["Exit date"]) in ("None", "nan") else str(r["Exit date"])
                    st.session_state.paper = recs
                    st.success(f"Loaded {len(recs)} trades.")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not read the file: {e}")
