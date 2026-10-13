import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Kush Algo Trader", page_icon="📈", layout="wide")
st.title("📈 Kush Algo Trader — Version 1")
st.caption("Paper trading prototype • Intraday & Swing • No live orders are sent")
st.warning(
    "DEMO / PAPER MODE ONLY. Demo prices are synthetic, not market quotes. "
    "This app is not connected to Zerodha and cannot place real orders."
)

with st.sidebar:
    st.header("Strategy Settings")
    mode = st.radio("Trading style", ["Intraday", "Swing"])
    fast_ema = st.number_input("Fast EMA", min_value=2, max_value=50, value=9)
    slow_ema = st.number_input("Slow EMA", min_value=5, max_value=200, value=21)
    rsi_period = st.number_input("RSI period", min_value=2, max_value=50, value=14)
    rsi_buy_max = st.slider("Buy only if RSI is below", 40, 80, 65)
    rsi_sell_min = st.slider("Exit if RSI is above", 50, 90, 75)
    initial_capital = st.number_input(
        "Virtual capital (₹)", min_value=1000,
        max_value=10000000, value=100000, step=5000
    )
    risk_pct = st.slider("Risk per trade (%)", 0.25, 2.0, 1.0, 0.25)
    stop_pct = st.slider("Stop-loss (%)", 0.5, 5.0, 1.5, 0.25)
    target_pct = st.slider("Target (%)", 0.5, 10.0, 3.0, 0.5)

if fast_ema >= slow_ema:
    st.error("Fast EMA must be smaller than Slow EMA.")
    st.stop()

st.subheader("1. Choose data")
data_source = st.radio(
    "Data source",
    ["Synthetic demo data", "Upload historical OHLC CSV"],
    horizontal=True
)


def make_demo_data(n=500, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(), periods=n, freq="D"
    )
    returns = rng.normal(0.0004, 0.018, n)
    close = 500 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (
        1 + rng.normal(0, 0.004, n)
    )
    high = np.maximum(open_, close) * (
        1 + rng.uniform(0.001, 0.012, n)
    )
    low = np.minimum(open_, close) * (
        1 - rng.uniform(0.001, 0.012, n)
    )
    return pd.DataFrame({
        "Date": dates, "Open": open_, "High": high,
        "Low": low, "Close": close
    })


if data_source == "Synthetic demo data":
    df = make_demo_data()
    st.info(
        "Synthetic prices are only for testing the interface; "
        "they do not represent NSE prices."
    )
    ticker = st.text_input("Symbol label", "DEMO")
else:
    uploaded = st.file_uploader(
        "Upload CSV with Date, Open, High, Low, Close columns",
        type=["csv"]
    )
    if uploaded is None:
        st.info("Upload an OHLC CSV to run a backtest.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
        required = {"Date", "Open", "High", "Low", "Close"}
        if not required.issubset(df.columns):
            st.error("CSV must contain: Date, Open, High, Low, Close.")
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = (
            df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if len(df) < slow_ema + rsi_period + 5:
            st.error("Not enough rows for selected EMA/RSI settings.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    ticker = st.text_input("Symbol label", "UPLOADED")


def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


df = df.copy()
df["EMA_fast"] = df["Close"].ewm(
    span=int(fast_ema), adjust=False
).mean()
df["EMA_slow"] = df["Close"].ewm(
    span=int(slow_ema), adjust=False
).mean()
df["RSI"] = calc_rsi(df["Close"], int(rsi_period))
df["Signal"] = ""

cross_up = (
    (df["EMA_fast"] > df["EMA_slow"])
    & (df["EMA_fast"].shift(1) <= df["EMA_slow"].shift(1))
)
cross_down = (
    (df["EMA_fast"] < df["EMA_slow"])
    & (df["EMA_fast"].shift(1) >= df["EMA_slow"].shift(1))
)

df.loc[cross_up & (df["RSI"] <= rsi_buy_max), "Signal"] = "BUY"
df.loc[cross_down | (df["RSI"] >= rsi_sell_min), "Signal"] = "SELL"

st.subheader("2. Price chart & signals")
st.line_chart(
    df.set_index("Date")[["Close", "EMA_fast", "EMA_slow"]],
    height=320
)
st.dataframe(
    df.tail(15)[[
        "Date", "Open", "High", "Low", "Close",
        "EMA_fast", "EMA_slow", "RSI", "Signal"
    ]].sort_values("Date", ascending=False),
    use_container_width=True,
    hide_index=True
)

last = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"₹{last['Close']:,.2f}")
c2.metric("Fast EMA", f"₹{last['EMA_fast']:,.2f}")
c3.metric("Slow EMA", f"₹{last['EMA_slow']:,.2f}")
c4.metric("Latest signal", last["Signal"] if last["Signal"] else "HOLD")

st.subheader("3. Simple historical backtest")
position = 0
positions = []

for _, row in df.iterrows():
    if position == 0 and row["Signal"] == "BUY":
        position = 1
    elif position == 1 and row["Signal"] == "SELL":
        position = 0
    positions.append(position)

df["Position"] = positions
df["Market_Return"] = df["Close"].pct_change().fillna(0)
df["Strategy_Return"] = (
    df["Position"].shift(1).fillna(0) * df["Market_Return"]
)
df["Equity"] = initial_capital * (
    1 + df["Strategy_Return"]
).cumprod()
df["BuyHold"] = initial_capital * (
    1 + df["Market_Return"]
).cumprod()
df["Peak"] = df["Equity"].cummax()
df["Drawdown"] = df["Equity"] / df["Peak"] - 1

total_return = (df["Equity"].iloc[-1] / initial_capital - 1) * 100
buy_hold_return = (df["BuyHold"].iloc[-1] / initial_capital - 1) * 100
max_dd = df["Drawdown"].min() * 100
trade_count = int((df["Signal"] == "BUY").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Strategy return", f"{total_return:.2f}%")
m2.metric("Buy & hold return", f"{buy_hold_return:.2f}%")
m3.metric("Max drawdown", f"{max_dd:.2f}%")
m4.metric("BUY signals", str(trade_count))

st.line_chart(df.set_index("Date")[["Equity", "BuyHold"]], height=260)
st.caption(
    "Educational backtest only; excludes brokerage, taxes, slippage, "
    "rejected orders and realistic intraday execution."
)

st.subheader("4. Paper trade ticket")
st.write(
    "Records a virtual trade in this app session only. "
    "No broker order is sent."
)

a, b, c = st.columns(3)
with a:
    side = st.selectbox("Side", ["BUY", "SELL"])
with b:
    entry_price = st.number_input(
        "Planned entry price (₹)",
        min_value=0.01,
        value=float(round(float(last["Close"]), 2)),
        step=0.05
    )
with c:
    available_capital = st.number_input(
        "Available virtual capital (₹)",
        min_value=1000,
        value=int(initial_capital),
        step=1000
    )

risk_amount = available_capital * risk_pct / 100
risk_per_share = entry_price * stop_pct / 100
qty_risk = (
    int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
)
qty_capital = (
    int(available_capital / entry_price) if entry_price > 0 else 0
)
quantity = max(0, min(qty_risk, qty_capital))

if side == "BUY":
    stop_price = entry_price * (1 - stop_pct / 100)
    target_price = entry_price * (1 + target_pct / 100)
else:
    stop_price = entry_price * (1 + stop_pct / 100)
    target_price = entry_price * (1 - target_pct / 100)

r1, r2, r3 = st.columns(3)
r1.metric("Suggested quantity", f"{quantity:,}")
r2.metric("Stop-loss price", f"₹{stop_price:,.2f}")
r3.metric("Target price", f"₹{target_price:,.2f}")

if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []

if st.button("Record PAPER trade", type="primary", use_container_width=True):
    if quantity < 1:
        st.error(
            "Quantity is zero. Increase virtual capital or adjust risk/stop-loss."
        )
    else:
        st.session_state.paper_trades.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": ticker,
            "Style": mode,
            "Side": side,
            "Qty": quantity,
            "Entry": round(entry_price, 2),
            "Stop Loss": round(stop_price, 2),
            "Target": round(target_price, 2),
            "Virtual Value": round(quantity * entry_price, 2)
        })
        st.success("Paper trade recorded.")

if st.session_state.paper_trades:
    trades = pd.DataFrame(st.session_state.paper_trades)
    st.dataframe(trades, use_container_width=True, hide_index=True)
    st.download_button(
        "Download paper trades CSV",
        trades.to_csv(index=False).encode("utf-8"),
        file_name="kush_algo_trader_paper_trades.csv",
        mime="text/csv"
    )
    if st.button("Clear paper trade history"):
        st.session_state.paper_trades = []
        st.rerun()

st.subheader("5. Before any live trading")
st.markdown("""
- This app is **not connected to Zerodha** and does not place real orders.
- Zerodha's API has no sandbox; paper trading here is simulated.
- Never put API secrets or access tokens in browser code, a public repository, or a mobile app.
- Live API orders require a secure backend and current broker/exchange compliance checks, including applicable static-IP requirements.
- Do not use this prototype for live trading without independent testing, order reconciliation, and fail-safe controls.
""")
