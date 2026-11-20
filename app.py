import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
import json, zlib, base64

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

IST = ZoneInfo("Asia/Kolkata")
APP_VERSION = "1.1"
APP_DATE = "20 Nov 2026"

# Nifty 50 list as understood in Nov 2026. Verify against niftyindices.com and edit in the sidebar if needed.
DEFAULT_SYMBOLS = (
    "ADANIENT, ADANIPORTS, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJFINANCE, BAJAJFINSV, "
    "BEL, BHARTIARTL, BSE, CIPLA, COALINDIA, DRREDDY, EICHERMOT, ETERNAL, GRASIM, HCLTECH, HDFCBANK, "
    "HDFCLIFE, HINDALCO, HINDUNILVR, ICICIBANK, INDIGO, INFY, ITC, JIOFIN, JSWSTEEL, KOTAKBANK, LT, "
    "M&M, MARUTI, MAXHEALTH, NESTLEIND, NTPC, ONGC, POWERGRID, RELIANCE, SBILIFE, SBIN, SHRIRAMFIN, "
    "SUNPHARMA, TATACONSUM, TATASTEEL, TCS, TECHM, TITAN, TMPV, TRENT, ULTRACEMCO"
)

# NSE weekday trading holidays (equity). Add next year's list in Settings when NSE publishes it.
DEFAULT_HOLIDAYS = (
    "2026-01-15, 2026-01-26, 2026-03-03, 2026-03-26, 2026-03-31, 2026-04-03, 2026-04-14, 2026-05-01, "
    "2026-05-28, 2026-06-26, 2026-09-14, 2026-10-02, 2026-10-20, 2026-11-10, 2026-11-24, 2026-12-25"
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

st.set_page_config(page_title="Kush Algo Trader", page_icon="📈", layout="centered",
                   initial_sidebar_state="collapsed")

if not YF_AVAILABLE:
    st.error("yfinance is not installed. Add `yfinance` to requirements.txt and redeploy.")
    st.stop()

# ---------- Settings kept in session (survive page switches) ----------
DEFAULTS = {
    "capital": 50000, "risk": 500, "max_pos": 3, "daily_limit": 1000,
    "top_n": 5, "use_mf": True,
    "breakout_days": 20, "atr_mult": 2.0, "rr": 2.0, "trail_mult": 3.0,
    "cost_pct": 0.15, "slip_pct": 0.05, "symbols_text": DEFAULT_SYMBOLS, "holidays_text": DEFAULT_HOLIDAYS, "nav": "Home",
}


def _enc(obj):
    raw = json.dumps(obj, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return base64.urlsafe_b64encode(zlib.compress(raw.encode("utf-8"), 9)).decode("ascii")


def _dec(text):
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(text.encode("ascii"))).decode("utf-8"))


SAVE_KEYS = [k for k in DEFAULTS if k != "nav"]
if not st.session_state.get("_restored"):
    st.session_state["_restored"] = True
    try:
        saved = _dec(st.query_params.get("s", "")) if st.query_params.get("s") else {}
        for k, v in saved.get("set", {}).items():
            if k in SAVE_KEYS:
                st.session_state[k] = v
        if isinstance(saved.get("p"), list):
            st.session_state["paper"] = saved["p"]
    except Exception:
        pass

for _k, _v in DEFAULTS.items():
    st.session_state[_k] = st.session_state.get(_k, _v)

capital = int(st.session_state.capital)
risk = int(st.session_state.risk)
max_pos = int(st.session_state.max_pos)
daily_limit = int(st.session_state.daily_limit)
top_n = int(st.session_state.top_n)
use_mf = bool(st.session_state.use_mf)
breakout_days = int(st.session_state.breakout_days)
atr_mult = float(st.session_state.atr_mult)
rr = float(st.session_state.rr)
trail_mult = float(st.session_state.trail_mult)
cost_side = (float(st.session_state.cost_pct) + float(st.session_state.slip_pct)) / 100
symbols = parse_symbols(st.session_state.symbols_text)
today = datetime.now(IST).date()

if "paper" not in st.session_state:
    st.session_state.paper = []

# ---------------------------------- Look ----------------------------------
NAVY, ACC, GOOD, BAD = "#0B1F3A", "#0E9F6E", "#059669", "#DC2626"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,500,1,0');
:root {{ --navy:{NAVY}; --acc:{ACC}; --good:{GOOD}; --bad:{BAD}; --line:#E6E9EF; --muted:#64748B; }}
html, body, .stApp, .stMarkdown, button, input, textarea, select {{ font-family:'Inter', sans-serif; }}
.stApp {{ background:#F6F7FB; }}
#MainMenu, footer {{ visibility:hidden; }}
header[data-testid="stHeader"] {{ background:#F6F7FB; }}
.block-container {{ padding-top:3.2rem; padding-bottom:4rem; max-width:760px; }}
.ms {{ font-family:'Material Symbols Rounded'; font-weight:normal; font-style:normal; line-height:1;
       letter-spacing:normal; text-transform:none; display:inline-block; white-space:nowrap;
       -webkit-font-feature-settings:'liga'; font-feature-settings:'liga'; vertical-align:middle; }}

/* Header */
.kt-hero {{ display:flex; align-items:center; gap:12px; padding:16px 16px; border-radius:18px;
           background:linear-gradient(135deg, #0B1F3A 0%, #15325C 100%); color:#fff; margin-bottom:12px;
           box-shadow:0 6px 18px rgba(11,31,58,.18); }}
.kt-logo {{ width:42px; height:42px; border-radius:12px; background:var(--acc); display:flex; align-items:center;
           justify-content:center; font-weight:800; font-size:20px; color:#fff; flex:none; }}
.kt-title {{ font-size:18px; font-weight:700; letter-spacing:.2px; }}
.kt-sub {{ font-size:12.5px; color:#B8C4D6; margin-top:1px; }}
.kt-badge {{ margin-left:auto; font-size:11px; font-weight:700; letter-spacing:.8px; color:#0B1F3A;
            background:#D1FAE5; padding:4px 9px; border-radius:999px; }}

/* Navigation: 2 rows of 3 buttons; columns stay side by side on phones */
.st-key-navbox {{ background:#ECEFF4; border-radius:14px; padding:5px; gap:5px !important; }}
.st-key-navbox [data-testid="stHorizontalBlock"] {{ flex-wrap:nowrap !important; gap:5px !important; }}
.st-key-navbox [data-testid="stColumn"], .st-key-navbox [data-testid="column"] {{
    min-width:0 !important; width:auto !important; flex:1 1 0 !important; }}
.st-key-navbox button {{ width:100%; min-height:42px; padding:4px 2px !important; border-radius:10px !important;
    background:transparent !important; border:1.5px solid transparent !important; color:#475569 !important;
    box-shadow:none !important; }}
.st-key-navbox button p {{ font-size:13px !important; font-weight:500; white-space:nowrap; }}
.st-key-navbox button[kind="primary"], .st-key-navbox button[data-testid="stBaseButton-primary"] {{
    background:#fff !important; border-color:var(--acc) !important; color:var(--acc) !important;
    box-shadow:0 1px 3px rgba(15,23,42,.08) !important; }}
.st-key-navbox button[kind="primary"] p, .st-key-navbox button[data-testid="stBaseButton-primary"] p {{
    font-weight:700; color:var(--acc) !important; }}

/* Tiles */
.kt-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:4px 0 14px; }}
.kt-tile {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:12px 14px; }}
.kt-lab {{ font-size:12.5px; color:var(--muted); display:flex; gap:6px; align-items:center; }}
.kt-lab .ms {{ color:var(--acc); font-size:19px; }}
.kt-val {{ font-size:20px; font-weight:700; color:#0F172A; margin-top:6px; }}
.kt-val.good {{ color:var(--good); }} .kt-val.bad {{ color:var(--bad); }}
.kt-subt {{ font-size:12px; color:#94A3B8; margin-top:2px; }}

/* Section title */
.kt-sec {{ display:flex; align-items:center; gap:8px; font-size:16px; font-weight:700; color:#0F172A; margin:18px 0 8px; }}
.kt-sec .ms {{ color:var(--acc); font-size:22px; }}

/* Buy / sell boxes */
.kt-act {{ display:grid; grid-template-columns:1fr; gap:10px; margin:4px 0 12px; }}
.kt-box {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:12px 14px; }}
.kt-box .kt-lab {{ font-weight:600; }}
.kt-box.sell .kt-lab .ms {{ color:var(--bad); }}
.kt-chip {{ display:inline-block; margin:6px 6px 0 0; padding:4px 10px; border-radius:999px; font-size:12.5px;
           font-weight:600; background:#E7F6F0; color:#065F46; }}
.kt-box.sell .kt-chip {{ background:#FEE2E2; color:#991B1B; }}
.kt-none {{ font-size:13px; color:#94A3B8; margin-top:6px; }}

/* Status banner */
.kt-status {{ display:flex; gap:10px; align-items:center; padding:12px 14px; border-radius:14px; margin:4px 0 10px;
             font-size:13.5px; font-weight:500; }}
.kt-status .ms {{ font-size:22px; }}
.kt-status.good {{ background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; }}
.kt-status.bad {{ background:#FEF2F2; color:#991B1B; border:1px solid #FECACA; }}
.kt-status.info {{ background:#EFF6FF; color:#1E3A8A; border:1px solid #BFDBFE; }}

/* List card */
.kt-list {{ background:#fff; border:1px solid var(--line); border-radius:16px; overflow:hidden; margin-bottom:10px; }}
.kt-row {{ display:flex; align-items:center; gap:12px; padding:11px 14px; border-top:1px solid #F1F3F7; }}
.kt-row:first-child {{ border-top:none; }}
.kt-rank {{ width:28px; height:28px; border-radius:50%; background:#E7F6F0; color:var(--acc); font-weight:700;
           font-size:13px; display:flex; align-items:center; justify-content:center; flex:none; }}
.kt-sym {{ font-weight:600; color:#0F172A; font-size:14.5px; }}
.kt-meta {{ font-size:12px; color:var(--muted); }}
.kt-right {{ margin-left:auto; text-align:right; }}
.kt-chg {{ font-weight:700; font-size:14px; }}
.kt-chg.good {{ color:var(--good); }} .kt-chg.bad {{ color:var(--bad); }}

/* Native widgets */
[data-testid="stMetric"] {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:10px 14px; }}
[data-testid="stMetricValue"] {{ font-size:22px; font-weight:700; }}
[data-testid="stExpander"] details {{ background:#fff; border:1px solid var(--line); border-radius:14px; }}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{ border-radius:12px; font-weight:600; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{ background:var(--acc); border-color:var(--acc); }}
[data-testid="stForm"] {{ background:#fff; border:1px solid var(--line); border-radius:16px; }}
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}
.kt-foot {{ font-size:11.5px; color:#94A3B8; text-align:center; margin-top:28px; line-height:1.5; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------- Helpers ---------------------------------
def money(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"₹{abs(x):,.0f}"
    if signed and round(x) != 0:
        return ("+" if x > 0 else "−") + s
    return ("−" if x < 0 else "") + s


def tone(x):
    return "good" if x > 0 else ("bad" if x < 0 else "")


def section(icon, text):
    st.markdown(f'<div class="kt-sec"><span class="ms">{icon}</span>{text}</div>', unsafe_allow_html=True)


def status(kind, icon, text):
    st.markdown(f'<div class="kt-status {kind}"><span class="ms">{icon}</span><div>{text}</div></div>',
                unsafe_allow_html=True)


def tiles(items):
    html = ""
    for icon, label, value, sub, tn in items:
        html += (f'<div class="kt-tile"><div class="kt-lab"><span class="ms">{icon}</span>{label}</div>'
                 f'<div class="kt-val {tn}">{value}</div><div class="kt-subt">{sub}</div></div>')
    st.markdown(f'<div class="kt-grid">{html}</div>', unsafe_allow_html=True)


def price_chart(frame, colors, height=300):
    import altair as alt
    long = frame.reset_index().rename(columns={frame.index.name or "index": "Date"})
    long = long.rename(columns={long.columns[0]: "Date"}).melt("Date", var_name="Line", value_name="₹")
    ch = (alt.Chart(long).mark_line(strokeWidth=1.8)
          .encode(x=alt.X("Date:T", title=None),
                  y=alt.Y("₹:Q", title=None, scale=alt.Scale(zero=False)),
                  color=alt.Color("Line:N", scale=alt.Scale(domain=list(frame.columns), range=colors),
                                  legend=alt.Legend(orient="bottom", title=None)),
                  tooltip=["Date:T", "Line:N", alt.Tooltip("₹:Q", format=",.2f")])
          .properties(height=height))
    st.altair_chart(ch, width="stretch")


def data_date(data):
    dates = [d.index[-1] for d in data.values() if not d.empty]
    return max(dates).date() if dates else None


def footer():
    st.markdown(f'<div class="kt-foot">Version {APP_VERSION} · {APP_DATE}<br>Paper / research only · Not investment advice · Not connected to any broker<br>'
                'Prices from Yahoo Finance may be delayed or incomplete</div>', unsafe_allow_html=True)


def load_universe(syms, period):
    yfs = tuple([to_yf(s) for s in syms] + ["^NSEI"])
    raw, failed = fetch_many(yfs, period)
    nifty = raw.pop("^NSEI", None)
    data = {s: add_indicators(df, breakout_days) for s, df in raw.items()}
    mk = market_filter(nifty) if (nifty is not None and use_mf) else None
    return data, nifty, mk, [short_name(f) for f in failed if f != "^NSEI"]


def market_state(nifty):
    if nifty is None or len(nifty) < 200:
        return None, None, None
    c = float(nifty["Close"].iloc[-1])
    a = float(nifty["Close"].rolling(200).mean().iloc[-1])
    return c > a, c, a


def holiday_set():
    out = set()
    for part in str(st.session_state.get("holidays_text", "")).replace("\n", ",").split(","):
        try:
            out.add(pd.Timestamp(part.strip()).date())
        except Exception:
            pass
    return out


def is_trading_day(d):
    return d.weekday() < 5 and d not in holiday_set()


def last_weekday(y, m):
    """Last NSE trading day of the month (skips weekends and listed holidays)."""
    d = (pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)).date()
    while not is_trading_day(d):
        d -= pd.Timedelta(days=1)
    return d


def first_trading_days(y, m, n=3):
    d, out = pd.Timestamp(y, m, 1).date(), []
    while len(out) < n:
        if is_trading_day(d):
            out.append(d)
        d += pd.Timedelta(days=1)
    return out


def next_rebalance(d):
    lw = last_weekday(d.year, d.month)
    if d > lw:
        nm = pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthBegin(1)
        lw = last_weekday(nm.year, nm.month)
    return lw


def last_price(data, sym):
    key = to_yf(sym)
    if key in data and not data[key].empty:
        return float(data[key]["Close"].iloc[-1])
    return None


def paper_summary(data):
    paper = st.session_state.paper
    invested = open_pl = realized = 0.0
    n_open = 0
    for t in paper:
        if t["Status"] == "Open":
            n_open += 1
            px = last_price(data, t["Symbol"]) if data else None
            px = px if px is not None else t["Entry ₹"]
            invested += t["Qty"] * t["Entry ₹"]
            open_pl += t["Qty"] * (px - t["Entry ₹"])
        elif t["P/L (₹)"] is not None:
            realized += float(t["P/L (₹)"])
    return {"invested": invested, "open_pl": open_pl, "realized": realized, "n_open": n_open}


def add_paper(sym, qty, entry, stop=None, target=None, strategy="Manual"):
    paper = st.session_state.paper
    paper.append({
        "ID": max([t["ID"] for t in paper], default=0) + 1, "Date": str(today), "Strategy": strategy,
        "Symbol": sym.strip().upper(), "Qty": int(qty), "Entry ₹": round(float(entry), 2),
        "Stop-loss ₹": round(float(stop), 2) if stop else None,
        "Target ₹": round(float(target), 2) if target else None,
        "Status": "Open", "Exit date": "", "Exit ₹": None, "P/L (₹)": None,
    })


def picks_html(df):
    rows = ""
    for _, r in df.iterrows():
        ch = r["6M return %"]
        qty = int((capital / top_n) // r["Close ₹"]) if r["Close ₹"] else 0
        buy = f"{qty} sh · {money(qty * r['Close ₹'])}" if qty else "too costly for one slot"
        rows += (f'<div class="kt-row"><div class="kt-rank">{int(r["Rank"])}</div>'
                 f'<div><div class="kt-sym">{r["Symbol"]}</div><div class="kt-meta">₹{r["Close ₹"]:,.2f} · {buy}</div></div>'
                 f'<div class="kt-right"><div class="kt-chg {tone(ch)}">{ch:+.1f}%</div>'
                 f'<div class="kt-meta">6 months</div></div></div>')
    return f'<div class="kt-list">{rows}</div>'


# --------------------------------- Header & nav ---------------------------------
st.markdown(f"""
<div class="kt-hero">
  <div class="kt-logo">K</div>
  <div><div class="kt-title">Kush Algo Trader</div><div class="kt-sub">Momentum · Paper trading · v{APP_VERSION}</div></div>
  <div class="kt-badge">PAPER</div>
</div>
""", unsafe_allow_html=True)

NAV = ["Home", "Scanner", "Backtest", "Paper", "Stock", "Settings"]
NAV_ICON = {"Home": ":material/home:", "Scanner": ":material/radar:", "Backtest": ":material/query_stats:",
            "Paper": ":material/receipt_long:", "Stock": ":material/candlestick_chart:", "Settings": ":material/tune:"}
if st.session_state.nav not in NAV:
    st.session_state.nav = "Home"


def _go(name):
    st.session_state.nav = name


with st.container(key="navbox"):
    for row in (NAV[:3], NAV[3:]):
        cols = st.columns(3)
        for col, name in zip(cols, row):
            col.button(name, key=f"nav_{name}", icon=NAV_ICON[name], width="stretch",
                       type="primary" if st.session_state.nav == name else "secondary",
                       on_click=_go, args=(name,))
page = st.session_state.nav


# =================================== HOME ===================================
def page_home():
    with st.spinner("Loading market data..."):
        data, nifty, mk, failed = load_universe(symbols, "1y")
    ok, n_close, n_avg = market_state(nifty)
    mk_now = True if (ok is None or not use_mf) else ok
    reb = next_rebalance(today)
    days_left = (reb - today).days
    ps = paper_summary(data)

    if ok is None:
        status("info", "info", "Could not load Nifty 50 data, so the market filter could not be checked.")
    elif ok:
        status("good", "trending_up", f"<b>Market OK.</b> Nifty 50 {n_close:,.0f} is above its 200-day average {n_avg:,.0f}. Buying allowed.")
    else:
        status("bad", "trending_down", f"<b>Market weak.</b> Nifty 50 {n_close:,.0f} is below its 200-day average {n_avg:,.0f}. "
               + ("Rules say: stay in cash." if use_mf else "Market filter is OFF in Settings."))

    tiles([
        ("event", "Next rebalance", "Today" if days_left == 0 else f"{days_left} days", reb.strftime("%a, %d %b %Y"), ""),
        ("account_balance_wallet", "Paper invested", money(ps["invested"]), f"{ps['n_open']} open positions", ""),
        ("show_chart", "Open P/L", money(ps["open_pl"], signed=True), "on open positions", tone(ps["open_pl"])),
        ("savings", "Booked P/L", money(ps["realized"], signed=True), "closed paper trades", tone(ps["realized"])),
    ])

    dd = data_date(data)
    if dd:
        st.caption(f"Prices as of close on {dd.strftime('%a, %d %b %Y')}")
    picks = momentum_picks(data, mk_now, top_n)
    now = datetime.now(IST)
    month_end = last_weekday(today.year, today.month)
    in_window = (today == month_end) or (today in first_trading_days(today.year, today.month))
    if today == month_end:
        title, note = f"Top {top_n} for this rebalance", (
            "List is final after 3:30 pm today." if now.hour * 60 + now.minute < 930 else "Final list for this month.")
    elif in_window:
        title, note = f"Top {top_n} for this rebalance", "Month-end rebalance window is open."
    else:
        title, note = f"Current top {top_n}", (
            f"This list can still change. The final list comes after 3:30 pm on {reb.strftime('%a, %d %b')}.")
    section("workspace_premium", title)
    st.caption(note)
    if picks is None or picks.empty:
        st.info("No data for picks.")
        return
    chosen = picks[picks["Pick"] != ""]
    if chosen.empty:
        status("bad", "block", "No stock qualifies right now (market filter or no stock in an uptrend). Stay in cash.")
        return
    st.markdown(picks_html(chosen), unsafe_allow_html=True)
    st.caption(f"Slot size about {money(capital / top_n)} (capital ÷ {top_n}).")

    # What the paper portfolio should change on rebalance day
    open_c = [t for t in st.session_state.paper if t["Status"] == "Open" and t.get("Strategy") == "C"]
    keep_zone = set(picks.loc[picks["Rank"].notna() & (picks["Rank"].fillna(999).astype(int) <= 2 * top_n), "Symbol"])
    to_sell = [t["Symbol"] for t in open_c if (not mk_now) or t["Symbol"] not in keep_zone]
    held = {t["Symbol"] for t in open_c}
    free_slots = max(0, top_n - (len(held) - len(set(to_sell))))
    to_buy = [s for s in chosen["Symbol"] if s not in held][:free_slots]

    def box(kind, icon, label, items):
        inner = "".join(f'<span class="kt-chip">{x}</span>' for x in items) or '<div class="kt-none">Nothing</div>'
        return (f'<div class="kt-box {kind}"><div class="kt-lab"><span class="ms">{icon}</span>{label}</div>{inner}</div>')

    section("swap_horiz", "Rebalance plan" if in_window else f"Plan for {reb.strftime('%d %b')} (preview)")
    st.markdown('<div class="kt-act">' + box("sell", "remove_shopping_cart", "Sell", to_sell)
                + box("buy", "add_shopping_cart", "Buy", to_buy) + '</div>', unsafe_allow_html=True)

    prices_ready = not (today == month_end and (dd is None or dd < today))
    if in_window and not prices_ready:
        status("bad", "hourglass_top", f"Today's closing prices are not in yet (latest: "
               f"{dd.strftime('%d %b') if dd else 'none'}). Yahoo usually adds them by the evening. "
               "Tap Refresh on the Scanner page later, then record.")
    elif not in_window:
        status("info", "schedule", f"Recording opens on <b>{reb.strftime('%a, %d %b')}</b>. Buying now would use "
               "today's prices, not the month-end prices the strategy was tested on.")
    if in_window and prices_ready:
        st.caption("Tip: after 3:30 pm, place these as AMO (after-market) orders in Zerodha; they fill at the next "
                   "morning's open, the same way the backtest assumed.")
        if to_buy and st.button(f"Record {len(to_buy)} buy(s) as paper trades", type="primary", width="stretch",
                                icon=":material/add_task:"):
            added = 0
            for s in to_buy:
                px = last_price(data, s)
                if px:
                    qty = int((capital / top_n) // px)
                    if qty >= 1:
                        add_paper(s, qty, px, strategy="C")
                        added += 1
            st.toast(f"{added} paper trade(s) recorded")
            st.rerun()
        if to_sell:
            st.caption("Close the 'Sell' stocks on the Paper page first.")
    if failed:
        st.caption("Could not load: " + ", ".join(failed))


# ================================== SCANNER ==================================
def page_scanner():
    top = st.columns([3, 1])
    top[0].caption("Prices refresh every hour. Best checked after 3:30 pm.")
    if top[1].button("Refresh", icon=":material/refresh:", width="stretch"):
        fetch_many.clear()
        st.rerun()
    with st.spinner("Loading prices for all stocks..."):
        data, nifty, mk, failed = load_universe(symbols, "1y")
    ok, n_close, n_avg = market_state(nifty)
    mk_now = True if (ok is None or not use_mf) else ok
    if ok is not None:
        status("good" if ok else "bad", "trending_up" if ok else "trending_down",
               f"Nifty 50 {n_close:,.0f} vs 200-day average {n_avg:,.0f}")

    section("workspace_premium", "Momentum ranking (strategy C)")
    st.caption(f"Stocks above their 200-day average, ranked by 6-month return. Hold the top {top_n}; "
               f"keep an old holding while it ranks {2 * top_n} or better.")
    mom = momentum_picks(data, mk_now, top_n)
    if mom is not None and not mom.empty:
        ranked = mom[mom["Rank"].notna()].head(2 * top_n)
        rows = ""
        for _, r in ranked.iterrows():
            rk = int(r["Rank"])
            tag = ('<span class="kt-chip">Hold</span>' if rk <= top_n and r["Pick"]
                   else '<span class="kt-chip" style="background:#F1F5F9;color:#475569">Keep zone</span>')
            rows += (f'<div class="kt-row"><div class="kt-rank">{rk}</div>'
                     f'<div><div class="kt-sym">{r["Symbol"]}</div><div class="kt-meta">₹{r["Close ₹"]:,.2f}</div></div>'
                     f'<div class="kt-right"><div class="kt-chg {tone(r["6M return %"])}">{r["6M return %"]:+.1f}%</div>'
                     f'{tag}</div></div>')
        st.markdown(f'<div class="kt-list">{rows}</div>', unsafe_allow_html=True)
        with st.expander("Full ranking"):
            st.dataframe(mom, width="stretch", hide_index=True)

    section("bolt", "Breakout signals (strategies A / B)")
    st.caption("Kept for comparison only. These strategies did not pass the 10-year backtest.")
    table = scan(data, capital, risk, max_pos, atr_mult, rr)
    buys = table[table["Signal"] == "BUY"] if not table.empty else table
    with st.expander(f"Breakout BUY signals today: {len(buys)}"):
        if len(buys):
            st.dataframe(buys, width="stretch", hide_index=True)
        else:
            st.write("No breakout signal today.")
    if failed:
        st.caption("Could not load: " + ", ".join(failed))


# ================================== BACKTEST ==================================
STRATS = ["A. Breakout + fixed target", "B. Breakout + trailing stop", "C. Monthly momentum"]
SHORT = {STRATS[0]: "A. Breakout", STRATS[1]: "B. Trailing", STRATS[2]: "C. Momentum"}
LINE_COLORS = {STRATS[0]: "#94A3B8", STRATS[1]: "#CBD5E1", STRATS[2]: ACC, "Nifty 50 buy & hold": NAVY}


def show_details(result, cap, nifty):
    eq, trades = result["equity"], result["trades"]
    s = summarize(eq, trades, cap)
    c1, c2 = st.columns(2)
    c1.metric("Closed trades", s["n"])
    c2.metric("Win rate", f"{s['win_rate']:.1f}%")
    c3, c4 = st.columns(2)
    c3.metric("Avg win / loss", f"₹{s['avg_win']:,.0f} / ₹{abs(s['avg_loss']):,.0f}")
    c4.metric("Profit factor", f"{s['pf']:.2f}" if np.isfinite(s["pf"]) else "∞")
    yearly = eq.groupby(eq.index.year).last()
    prev = yearly.shift(1).fillna(cap)
    ytab = pd.DataFrame({"Year": yearly.index.astype(str), "Strategy %": ((yearly / prev - 1) * 100).round(1).values})
    if nifty is not None:
        nc = nifty["Close"].reindex(eq.index).ffill().bfill()
        ny = nc.groupby(nc.index.year).last()
        nprev = ny.shift(1).fillna(nc.iloc[0])
        ytab["Nifty 50 %"] = ((ny / nprev - 1) * 100).round(1).values
    section("calendar_month", "Return by year")
    st.dataframe(ytab, width="stretch", hide_index=True)
    if s["n"]:
        section("logout", "Exit reasons")
        st.dataframe(trades["Exit reason"].value_counts().rename_axis("Reason").reset_index(name="Trades"),
                     width="stretch", hide_index=True)
        section("bar_chart", "Profit / loss by stock")
        st.bar_chart(trades.groupby("Symbol")["P/L (₹)"].sum().sort_values(), height=300, color=ACC)
        with st.expander(f"All {s['n']} trades"):
            st.dataframe(trades, width="stretch", hide_index=True)
            st.download_button("Download trades CSV", trades.to_csv(index=False).encode("utf-8"),
                               file_name="kush_backtest_trades.csv", mime="text/csv", icon=":material/download:")
    if not result["open"].empty:
        section("hourglass_top", "Still open at the end")
        st.dataframe(result["open"], width="stretch", hide_index=True)
    if result["blocked_days"]:
        st.caption(f"Daily loss limit was hit on {result['blocked_days']} day(s).")


def page_backtest():
    st.caption("Runs A, B and C on the whole stock list with your money rules and costs, "
               "and compares them with simply holding Nifty 50.")
    period_label = st.selectbox("Data period", ["2 years", "5 years", "10 years"], index=2)
    period = {"2 years": "2y", "5 years": "5y", "10 years": "10y"}[period_label]
    if st.button("Run all backtests", type="primary", width="stretch", icon=":material/play_arrow:"):
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
    if not (bt and "results" in bt):
        return
    results, cap, nifty = bt["results"], bt["capital"], bt["nifty"]
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        st.error("Not enough data to run the backtest. Try a longer period.")
        return
    first = next(iter(valid.values()))["equity"]
    rows, chart = [], pd.DataFrame(index=first.index)
    for name, r in valid.items():
        s = summarize(r["equity"], r["trades"], cap)
        rows.append({"Strategy": SHORT[name], "Yearly %": round(s["cagr"], 1), "Worst fall %": round(s["max_dd"], 1),
                     "Ending ₹": f"{s['final']:,.0f}", "Trades": s["n"]})
        chart[name] = r["equity"]
    if nifty is not None:
        nc = nifty["Close"].reindex(first.index).ffill().bfill()
        bench = cap * nc / nc.iloc[0]
        bs = summarize(bench, pd.DataFrame(), cap)
        rows.append({"Strategy": "Nifty 50", "Yearly %": round(bs["cagr"], 1), "Worst fall %": round(bs["max_dd"], 1),
                     "Ending ₹": f"{bs['final']:,.0f}", "Trades": 0})
        chart["Nifty 50 buy & hold"] = bench
    comp = pd.DataFrame(rows).sort_values("Yearly %", ascending=False).reset_index(drop=True)
    best = comp.iloc[0]

    section("emoji_events", "Result")
    st.caption(f"Test period: {first.index[0]:%d %b %Y} to {first.index[-1]:%d %b %Y}")
    tiles([
        ("emoji_events", "Best strategy", best["Strategy"].split(". ")[-1], f"{best['Yearly %']}% a year", "good"),
        ("trending_down", "Its worst fall", f"{best['Worst fall %']}%", "from a peak", "bad"),
    ])
    st.dataframe(comp, width="stretch", hide_index=True)
    st.line_chart(chart, height=320, color=[LINE_COLORS.get(c, "#64748B") for c in chart.columns])

    pick = st.selectbox("See details of", list(valid.keys()), index=len(valid) - 1)
    show_details(valid[pick], cap, nifty)
    st.caption("Limitations: uses today's Nifty 50 list for the past (makes results look better than reality), "
               "dividend-adjusted prices, simple cost estimate, no tax, and orders assumed to fill at the open or "
               "exactly at the stop/target. Past results do not guarantee future results.")
    if bt["failed"]:
        st.caption("Could not load: " + ", ".join(bt["failed"]))


# =================================== PAPER ===================================
def page_paper():
    with st.spinner("Loading prices..."):
        data, _, _, _ = load_universe(symbols, "1y")
    paper = st.session_state.paper
    ps = paper_summary(data)
    today_pl = sum(float(t["P/L (₹)"] or 0) for t in paper
                   if t["Status"] == "Closed" and str(t["Exit date"]) == str(today))
    tiles([
        ("account_balance_wallet", "Invested", money(ps["invested"]), f"{ps['n_open']} open", ""),
        ("show_chart", "Open P/L", money(ps["open_pl"], signed=True), "at last close", tone(ps["open_pl"])),
        ("savings", "Booked P/L", money(ps["realized"], signed=True), "all closed trades", tone(ps["realized"])),
        ("today", "Booked today", money(today_pl, signed=True), str(today), tone(today_pl)),
    ])
    if today_pl <= -daily_limit:
        status("bad", "block", f"Daily loss limit of {money(daily_limit)} reached. No new trades today.")

    open_trades = [t for t in paper if t["Status"] == "Open"]
    if open_trades:
        section("inventory_2", "Open positions")
        rows = []
        for t in open_trades:
            px = last_price(data, t["Symbol"])
            pl = t["Qty"] * (px - t["Entry ₹"]) if px else None
            rows.append({"#": t["ID"], "Strategy": t.get("Strategy", "Manual"), "Symbol": t["Symbol"], "Qty": t["Qty"],
                         "Entry ₹": t["Entry ₹"], "Last ₹": round(px, 2) if px else None,
                         "P/L ₹": round(pl, 2) if pl is not None else None,
                         "Stop ₹": t.get("Stop-loss ₹"), "Target ₹": t.get("Target ₹")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        with st.form("close_paper"):
            st.markdown("**Close a position**")
            labels = {t["ID"]: f"#{t['ID']} {t['Symbol']} × {t['Qty']} @ ₹{t['Entry ₹']}" for t in open_trades}
            c_id = st.selectbox("Position", list(labels.keys()), format_func=lambda x: labels[x])
            c_px = st.number_input("Exit price ₹ (leave empty to use last close)", min_value=0.0, value=None, step=0.05)
            if st.form_submit_button("Close position", width="stretch", icon=":material/check_circle:"):
                for t in paper:
                    if t["ID"] == c_id:
                        px = c_px if c_px else last_price(data, t["Symbol"])
                        if not px:
                            st.error("No price found. Enter the exit price.")
                            break
                        t["Status"], t["Exit date"], t["Exit ₹"] = "Closed", str(today), round(px, 2)
                        t["P/L (₹)"] = round((px - t["Entry ₹"]) * t["Qty"], 2)
                        st.rerun()

    with st.form("add_paper", clear_on_submit=True):
        st.markdown("**Add a position**")
        f1, f2 = st.columns(2)
        f_sym = f1.text_input("Symbol")
        f_strat = f2.selectbox("Strategy", ["C", "A", "B", "Manual"])
        f3, f4 = st.columns(2)
        f_qty = f3.number_input("Qty", min_value=1, value=1, step=1)
        f_entry = f4.number_input("Entry ₹ (empty = last close)", min_value=0.0, value=None, step=0.05)
        f5, f6 = st.columns(2)
        f_stop = f5.number_input("Stop-loss ₹ (optional)", min_value=0.0, value=None, step=0.05)
        f_target = f6.number_input("Target ₹ (optional)", min_value=0.0, value=None, step=0.05)
        if st.form_submit_button("Add position", type="primary", width="stretch", icon=":material/add:"):
            entry = f_entry if f_entry else (last_price(data, f_sym) if f_sym else None)
            if not f_sym or not entry:
                st.error("Enter a valid symbol (and entry price if it is not a Nifty 50 stock).")
            elif (f_stop and f_stop >= entry) or (f_target and f_target <= entry):
                st.error("Stop-loss must be below entry and target above entry.")
            else:
                if f_stop and (entry - f_stop) * f_qty > risk:
                    st.warning(f"Planned risk {money((entry - f_stop) * f_qty)} is more than your {money(risk)} rule.")
                add_paper(f_sym, f_qty, entry, f_stop, f_target, f_strat)
                st.toast("Position added")
                st.rerun()

    closed = [t for t in paper if t["Status"] == "Closed"]
    if closed:
        with st.expander(f"Closed trades ({len(closed)})"):
            st.dataframe(pd.DataFrame(closed), width="stretch", hide_index=True)

    section("save", "Save your log")
    st.caption("Your log and settings are kept in this page's web address. Open the app from the same browser tab "
               "(or bookmark it after a change). Keep a CSV backup too.")
    if paper:
        st.download_button("Download paper log", pd.DataFrame(paper).to_csv(index=False).encode("utf-8"),
                           file_name="kush_paper_trades.csv", mime="text/csv", width="stretch", icon=":material/download:")
    up = st.file_uploader("Load a saved log (CSV)", type=["csv"])
    if up is not None and st.button("Load this file", width="stretch", icon=":material/upload:"):
        try:
            df_up = pd.read_csv(up, keep_default_na=False)
            need = ["ID", "Date", "Symbol", "Qty", "Entry ₹", "Stop-loss ₹", "Target ₹", "Status", "Exit date", "Exit ₹", "P/L (₹)"]
            if not set(need).issubset(df_up.columns):
                st.error("This is not a Kush paper log file.")
            else:
                def num(v):
                    v = str(v).strip()
                    return float(v) if v not in ("", "None", "nan") else None
                recs = []
                for r in df_up.to_dict("records"):
                    recs.append({
                        "ID": int(r["ID"]), "Date": str(r["Date"]), "Strategy": str(r.get("Strategy", "") or "Manual"),
                        "Symbol": str(r["Symbol"]), "Qty": int(r["Qty"]), "Entry ₹": num(r["Entry ₹"]),
                        "Stop-loss ₹": num(r["Stop-loss ₹"]), "Target ₹": num(r["Target ₹"]), "Status": str(r["Status"]),
                        "Exit date": "" if str(r["Exit date"]) in ("None", "nan") else str(r["Exit date"]),
                        "Exit ₹": num(r["Exit ₹"]), "P/L (₹)": num(r["P/L (₹)"]),
                    })
                st.session_state.paper = recs
                st.toast(f"Loaded {len(recs)} trades")
                st.rerun()
        except Exception as e:
            st.error(f"Could not read the file: {e}")


# =================================== STOCK ===================================
def page_stock():
    sym_in = st.text_input("NSE symbol", "TCS")
    sym_period_label = st.session_state.get("stock_period", "5 years")
    sym_period = {"2 years": "2y", "5 years": "5y", "10 years": "10y"}[sym_period_label]
    one = parse_symbols(sym_in)[:1]
    if not one:
        return
    with st.spinner("Loading..."):
        data1, nifty1, mk1, failed1 = load_universe(one, sym_period)
    if not data1:
        st.error(f"No data found for {one[0]}. Check the symbol spelling.")
        return
    d = next(iter(data1.values()))
    last = d.iloc[-1]
    m6 = float(last["MOM126"]) * 100 if pd.notna(last["MOM126"]) else None
    trend = "Uptrend" if bool(last["Above200"]) else "Below 200-day avg"
    tiles([
        ("payments", "Last close", f"₹{last['Close']:,.2f}", str(d.index[-1].date()), ""),
        ("trending_up", "6-month return", f"{m6:+.1f}%" if m6 is not None else "—", trend, tone(m6 or 0)),
    ])
    section("show_chart", "Last 12 months")
    price_chart(d[["Close", "EMA20", "EMA50", "SMA200"]].tail(250), [NAVY, "#F59E0B", "#3B82F6", ACC])
    with st.expander("Breakout backtest on this stock (A / B)"):
        st.selectbox("Backtest period", ["2 years", "5 years", "10 years"], key="stock_period",
                     index=["2 years", "5 years", "10 years"].index(sym_period_label))
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
            if s1["n"]:
                st.dataframe(r1["trades"], width="stretch", hide_index=True)
            st.caption("One stock alone is a weak test. The Backtest page (all stocks) matters more.")


# ================================== SETTINGS ==================================
def page_settings():
    section("account_balance", "Money")
    c1, c2 = st.columns(2)
    c1.number_input("Capital (₹)", min_value=10000, max_value=10000000, step=5000, key="capital")
    c2.number_input("Momentum: stocks to hold", min_value=3, max_value=15, key="top_n")
    st.toggle("Market filter: buy only when Nifty 50 is above its 200-day average", key="use_mf")

    section("receipt", "Costs")
    c3, c4 = st.columns(2)
    c3.number_input("Trading cost per side (%)", min_value=0.0, max_value=1.0, step=0.01, key="cost_pct",
                    help="Rough estimate for delivery trades: STT, stamp duty, exchange, GST, DP charges.")
    c4.number_input("Slippage per side (%)", min_value=0.0, max_value=1.0, step=0.01, key="slip_pct")

    with st.expander("Breakout strategies A / B (comparison only)"):
        d1, d2 = st.columns(2)
        d1.number_input("Risk per trade (₹)", min_value=100, max_value=100000, step=100, key="risk")
        d2.number_input("Max open trades", min_value=1, max_value=10, key="max_pos")
        d3, d4 = st.columns(2)
        d3.number_input("Max loss per day (₹)", min_value=100, max_value=1000000, step=100, key="daily_limit")
        d4.number_input("Breakout lookback (days)", min_value=10, max_value=100, key="breakout_days")
        st.slider("Stop-loss distance (× ATR)", 1.0, 4.0, step=0.5, key="atr_mult")
        st.slider("Target (× risk), strategy A", 1.0, 4.0, step=0.5, key="rr")
        st.slider("Trailing stop (× ATR), strategy B", 2.0, 6.0, step=0.5, key="trail_mult")

    with st.expander("Market holidays (NSE)"):
        st.text_area("Weekday holidays, YYYY-MM-DD, comma separated", height=140, key="holidays_text")
        st.caption("Used to find the last trading day of each month. Add next year's dates from nseindia.com "
                   "when they are announced.")

    with st.expander("Stock list (Nifty 50)"):
        st.text_area("NSE symbols, comma separated", height=200, key="symbols_text")
        st.caption("Check the latest list on niftyindices.com. Symbols that fail to load are skipped.")

    section("menu_book", "How the strategies work")
    with st.expander("C. Monthly momentum (main strategy)"):
        st.markdown(f"""
- On the **last trading day of each month**, rank Nifty 50 stocks that are above their 200-day average by their 6-month return.
- Hold the **top {top_n}** in equal amounts (about {money(capital / top_n)} each).
- Keep an old holding while it stays in the top {2 * top_n}; sell it when it drops out.
- If Nifty 50 is below its 200-day average, sell everything and wait in cash.
- About 15–30 minutes of work per month.
""")
    with st.expander("A / B. Breakout (comparison only)"):
        st.markdown(f"""
- Buy when the close is above the 200-day average, EMA20 is above EMA50, and the close breaks the {breakout_days}-day high.
- Stop-loss: entry − {atr_mult:g} × ATR. **A:** target {rr:g} × risk. **B:** no target, stop trails {trail_mult:g} × ATR below the highest close.
- Risk {money(risk)} per trade, max {max_pos} trades, no new trade after a {money(daily_limit)} loss day.
""")


PAGES = {"Home": page_home, "Scanner": page_scanner, "Backtest": page_backtest,
         "Paper": page_paper, "Stock": page_stock, "Settings": page_settings}
PAGES[page]()
footer()

try:
    _state = {"set": {k: st.session_state[k] for k in SAVE_KEYS if st.session_state[k] != DEFAULTS[k]},
              "p": st.session_state.paper}
    if _state["set"] or _state["p"]:
        _code = _enc(_state)
        if st.query_params.get("s") != _code:
            st.query_params["s"] = _code
    elif "s" in st.query_params:
        del st.query_params["s"]
except Exception:
    pass
