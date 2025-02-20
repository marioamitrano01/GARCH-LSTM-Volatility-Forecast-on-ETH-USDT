
# GARCH + Long Short-Term Memory (LSTM) Driven Volatility Forecasting for ETH/USDT

This repository contains a comprehensive forecasting pipeline for ETH/USDT that combines traditional GARCH modeling with an LSTM-based volatility forecast. The pipeline includes:

- **Data Fetching:**  
  Uses Binance's API (with pagination) to download all daily candles from January 1, 2020 up to yesterday.
  
- **LSTM Volatility Forecast:**  
  Trains an LSTM network (with dropout for regularization) on the GARCH conditional volatility to forecast future volatility. The LSTM forecast is then combined with the GARCH forecast via a weighted average.

- **Monte Carlo Simulation:**  
  Uses the combined volatility and AR(1) mean forecast to generate multiple simulated future price paths and builds uncertainty bands.

## Required Libraries

Ensure you have the following libraries installed (Python 3.x):

- `requests`
- `numpy`
- `pandas`
- `plotly`
- `statsmodels`
- `scipy`
- `arch`
- `tensorflow`
- `scikit-learn`

You can install them using pip:

```bash
pip install requests numpy pandas plotly statsmodels scipy arch tensorflow scikit-learn
```

## How to Run

1. Clone the repository.
2. Ensure the required libraries are installed.
3. Run the main script using:

```bash
python your_script_name.py
```

Happy Forecasting !
