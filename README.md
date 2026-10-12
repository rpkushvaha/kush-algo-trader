# Kush Algo Trader Trader — Version 1

Mobile-friendly Streamlit prototype for learning and testing an EMA/RSI strategy with simulated paper trades.

## Included
- Intraday/Swing mode label
- Synthetic demo OHLC data or CSV upload
- EMA crossover + RSI signals
- Simple historical backtest
- Risk-based paper trade ticket with stop-loss and target
- Downloadable paper-trade CSV

## Important limitations
- No Zerodha integration and no live orders.
- Synthetic demo prices are not market data.
- Backtest excludes brokerage, taxes, slippage, and realistic execution.
- No profit is guaranteed; this is not investment advice.
- Never add API keys/secrets to this project or publish them in source code.

## Run on a computer
1. Install Python 3.10 or newer.
2. Extract the ZIP and open a terminal in this folder.
3. Run `python -m pip install -r requirements.txt`
4. Run `streamlit run app.py`

## Android use
Upload the project files to GitHub, then deploy `app.py` using Streamlit Community Cloud. Open the HTTPS URL in Chrome on Android and select Menu (⋮) > Add to Home screen. Hosting limits and availability may change.

## Historical CSV format
Required headers: `Date,Open,High,Low,Close`
Example: `2026-01-02,100,103,99,102`
Use clean historical data with an interval suitable for the intended strategy. Demo data is daily and not suitable for intraday performance testing.

## Suggested rollout
1. Use synthetic data to learn the interface.
2. Upload historical data and test multiple periods.
3. Paper trade for 4–8 weeks, recording trades and realistic costs.
4. Only then plan secure Zerodha integration and verify current broker/exchange requirements.
