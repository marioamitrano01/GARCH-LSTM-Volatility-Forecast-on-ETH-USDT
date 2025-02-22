import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import scipy.stats as scs
from arch import arch_model


import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout  
from sklearn.preprocessing import MinMaxScaler

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_binance_data_paginated(
    symbol: str = "ETHUSDT",
    interval: str = "1d",
    start_time: pd.Timestamp = pd.Timestamp("2020-01-01"),
    end_time: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance in a paginated manner,
    ensuring I get all daily candles from start_time up to end_time.
    """
    if end_time is None:
        end_time = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    all_candles = []
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": 1000
        }
        resp = requests.get(base_url + endpoint, params=params)
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            break

        all_candles.extend(data)
        last_candle_close_time = data[-1][6]  # close_time in ms
        if last_candle_close_time >= end_ts:
            break
        start_ts = last_candle_close_time + 1

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ]
    df = pd.DataFrame(all_candles, columns=columns)

    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_asset_volume", "num_trades",
                    "taker_buy_base_vol", "taker_buy_quote_vol"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms', utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms', utc=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    df = df[df.index <= end_time]
    return df


def create_sequences(data, seq_length):
    """
    Utility for preparing a time series for LSTM.
    data: 1D or 2D array
    seq_length: # of time steps in each sequence
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)





def plot_residual_diagnostics_plotly(std_resid: np.ndarray):
    """
    Produce a Q-Q plot and histogram of standardized residuals
    using Plotly, referencing a Normal distribution from SciPy.
    """

    res = std_resid.copy()
    res = res[~np.isnan(res)]  # drop any NaN

    # Sort residuals
    res_sorted = np.sort(res)
    n = len(res_sorted)

    # Theoretical quantiles from a standard normal distribution
    # Using plotting positions (i - 0.5)/n
    # A simpler approach is: scs.norm.ppf((rank - 0.5)/n)
    qq_theoretical = scs.norm.ppf((np.arange(n) + 0.5) / n)

    slope, intercept, _, _, _ = scs.linregress(qq_theoretical, res_sorted)
    line_qq = intercept + slope * qq_theoretical

    # For the histogram overlay: I'll do a PDF curve
    # I'll normalize the histogram, so we set 'histnorm="probability density"'
    # Then overlay scs.norm.pdf(...) as a Scatter
    x_min, x_max = res.min(), res.max()
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = scs.norm.pdf(x_vals, loc=0, scale=1)  # standard normal pdf

    # Create subplots with 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Q-Q Plot of Std. Residuals", "Histogram of Std. Residuals")
    )

    # 1) QQ points
    fig.add_trace(
        go.Scatter(
            x=qq_theoretical,
            y=res_sorted,
            mode='markers',
            name='QQ Points'
        ),
        row=1, col=1
    )
    # 2) Best-fit line
    fig.add_trace(
        go.Scatter(
            x=qq_theoretical,
            y=line_qq,
            mode='lines',
            line=dict(color='red'),
            name='Best-Fit Line'
        ),
        row=1, col=1
    )

    # 1) Plotly histogram (normalized)
    fig.add_trace(
        go.Histogram(
            x=res,
            histnorm='probability density',
            marker_color='skyblue',
            name='Residuals Hist'
        ),
        row=1, col=2
    )

    # 2) Normal PDF line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Normal PDF'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=500,
        width=1000,
        showlegend=True,
        title="Diagnostic Plots: Standardized Residuals"
    )

    # Adjust layout for subplots
    fig.update_xaxes(title="Theoretical Quantiles", row=1, col=1)
    fig.update_yaxes(title="Empirical Quantiles", row=1, col=1)

    fig.update_xaxes(title="Residual Value", row=1, col=2)
    fig.update_yaxes(title="Probability Density", row=1, col=2)

    fig.show()






def main():
    # Parameters
    forecast_horizon = 30
    lstm_epochs = 50
    lstm_batch_size = 16
    seq_length = 20
    weight_garch = 0.5
    weight_lstm = 0.5

    start_date = pd.Timestamp("2020-01-01", tz="UTC")
    yesterday = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    try:
        df_eth = get_binance_data_paginated(
            symbol="ETHUSDT",
            interval="1d",
            start_time=start_date,
            end_time=yesterday
        )
        logger.info(f"Fetched {len(df_eth)} daily candles for ETH/USDT from {start_date.date()} to {yesterday.date()}.")
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        sys.exit(1)

    if df_eth.empty:
        logger.error("No data retrieved. Exiting.")
        sys.exit(1)


    df_eth["return"] = np.log(df_eth["close"]).diff()
    rstock = df_eth["return"].dropna()
    if len(rstock) < 10:
        logger.error("Not enough return data. Exiting.")
        sys.exit(1)


    logger.info("\n--- Ljung-Box Test on Returns (Lag=20) ---")
    lb_test = sm.stats.diagnostic.acorr_ljungbox(rstock, lags=[20], return_df=True)
    print(lb_test)

    logger.info("\n--- ARCH LM Test on Returns ---")
    arch_test = sm.stats.diagnostic.het_arch(rstock)
    print(f"LM stat = {arch_test[0]:.3f}, LM p-value = {arch_test[1]:.3f}")
    print(f"F stat  = {arch_test[2]:.3f}, F p-value  = {arch_test[3]:.3f}")




    model = arch_model(
        rstock,
        mean='AR', lags=1,
        vol='Garch', p=1, q=1,
        dist='normal',
        rescale=False
    )
    res = model.fit(update_freq=5, disp='off')
    print("\n--- AR(1)-GARCH(1,1) Model Summary ---")
    print(res.summary())

    # Standardized residuals
    std_resid = res.resid / res.conditional_volatility

    logger.info("\n--- Residual Analysis ---")
    lb_resid = sm.stats.diagnostic.acorr_ljungbox(std_resid.dropna(), lags=[20], return_df=True)
    print("\nLjung-Box test on standardized residuals (lag=20):")
    print(lb_resid)
    arch_test_resid = sm.stats.diagnostic.het_arch(std_resid.dropna())
    print(f"\nARCH LM test on standardized residuals:")
    print(f"LM stat = {arch_test_resid[0]:.3f}, LM p-value = {arch_test_resid[1]:.3f}")
    print(f"F stat  = {arch_test_resid[2]:.3f}, F p-value  = {arch_test_resid[3]:.3f}")

    plot_residual_diagnostics_plotly(std_resid.values)

    # GARCH Forecast
    fcast = res.forecast(horizon=forecast_horizon, reindex=False)
    fc_mean = fcast.mean.values[-1]       # AR(1) predicted mean log-returns
    fc_var  = fcast.variance.values[-1]   # GARCH predicted variance
    fc_vol  = np.sqrt(fc_var)




    last_price = df_eth["close"].iloc[-1]
    forecast_start_date = df_eth.index[-1] + pd.Timedelta(days=1)
    point_forecast_prices = [last_price]
    future_dates = []
    for i in range(forecast_horizon):
        future_date = forecast_start_date + pd.Timedelta(days=i)
        future_dates.append(future_date)
        next_price = point_forecast_prices[-1] * np.exp(fc_mean[i])
        point_forecast_prices.append(next_price)
    point_forecast_prices = point_forecast_prices[1:]

    # LSTM VOLATILITY FORECAST 
    hist_vol = res.conditional_volatility.dropna()  
    vol_data = hist_vol.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    vol_scaled = scaler.fit_transform(vol_data)
    X, y = create_sequences(vol_scaled, seq_length)

    logger.info(f"Training LSTM on {len(X)} sequences (epochs={lstm_epochs}, batch_size={lstm_batch_size})...")

    model_lstm = Sequential()
    model_lstm.add(Input(shape=(seq_length, 1)))
    model_lstm.add(LSTM(50, activation='tanh', return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X, y, epochs=lstm_epochs, batch_size=lstm_batch_size, verbose=0)

    # Iterative LSTM forecast
    lstm_forecast_scaled = []
    current_seq = vol_scaled[-seq_length:].copy()
    for i in range(forecast_horizon):
        pred = model_lstm.predict(current_seq.reshape(1, seq_length, 1), verbose=0)
        lstm_forecast_scaled.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], [[pred[0, 0]]], axis=0)
    lstm_forecast_vol = scaler.inverse_transform(
        np.array(lstm_forecast_scaled).reshape(-1, 1)
    ).flatten()




    combined_vol_forecast = (weight_garch * fc_vol + weight_lstm * lstm_forecast_vol)



    num_sims = 200
    rng = np.random.default_rng(seed=42)
    mc_paths = np.zeros((forecast_horizon, num_sims))
    for sim in range(num_sims):
        sim_price = last_price
        for i in range(forecast_horizon):
            epsilon = rng.normal(0, 1)
            daily_return = fc_mean[i] + combined_vol_forecast[i] * epsilon
            sim_price *= np.exp(daily_return)
            mc_paths[i, sim] = sim_price

    p10 = np.percentile(mc_paths, 10, axis=1)
    p50 = np.percentile(mc_paths, 50, axis=1)
    p90 = np.percentile(mc_paths, 90, axis=1)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "ETH/USDT Price: Historical, Deterministic & Monte Carlo Forecast",
            "GARCH & LSTM Volatility Forecasts (Historical + Future)"
        )
    )

    # Historical price
    fig.add_trace(
        go.Scatter(
            x=df_eth.index,
            y=df_eth["close"],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Deterministic point forecast
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=point_forecast_prices,
            mode='lines+markers',
            name='Point Forecast Price',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )

    # Monte Carlo forecast bands
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=p90,
            mode='lines',
            line=dict(width=0, color='rgba(255,0,0,0.2)'),
            name='90th Percentile',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=p10,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0, color='rgba(255,0,0,0.2)'),
            name='10–90% MC Band',
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=p50,
            mode='lines',
            line=dict(color='magenta', dash='dot'),
            name='Median MC Path'
        ),
        row=1, col=1
    )

    # Historical GARCH volatility
    fig.add_trace(
        go.Scatter(
            x=hist_vol.index,
            y=hist_vol,
            mode='lines',
            name='Historical Vol (GARCH)',
            line=dict(color='green')
        ),
        row=2, col=1
    )

    # GARCH forecast vol
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=fc_vol,
            mode='lines+markers',
            name='GARCH Forecast Vol',
            line=dict(color='red', dash='dot')
        ),
        row=2, col=1
    )

    # LSTM forecast vol
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lstm_forecast_vol,
            mode='lines+markers',
            name='LSTM Forecast Vol',
            line=dict(color='orange', dash='dot')
        ),
        row=2, col=1
    )

    # Combined vol
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=combined_vol_forecast,
            mode='lines+markers',
            name='Combined Forecast Vol',
            line=dict(color='purple', dash='dash')
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="GARCH + LSTM Volatility Forecast on ETH/USDT",
        height=900,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0)"),
        xaxis=dict(title="Date"),
        yaxis=dict(title="ETH Price (USDT)"),
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title="Volatility (Std. Dev.)", row=2, col=1)

    fig.show()

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Mean Log-Return": fc_mean,
        "GARCH Vol": fc_vol,
        "LSTM Vol": lstm_forecast_vol,
        "Combined Vol": combined_vol_forecast,
        "Point Forecast Price": point_forecast_prices,
        "MC Median Price": p50,
        "MC 10%": p10,
        "MC 90%": p90
    }).set_index("Date")

    print("\n--Forecast Summary (First 10 lines)--")
    print(forecast_df.head(10).round(3))
    print("\nDone.")


if __name__ == "__main__":
    main()
