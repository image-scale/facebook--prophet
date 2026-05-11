# Goal

## Project
timeweaver — a Python time series forecasting library.

## Description
TimeWeaver is a time series forecasting library that fits an additive model where non-linear trends are combined with yearly, weekly, and daily seasonality, plus holiday effects. The model decomposes time series into trend (linear, logistic, or flat growth), multiple seasonality components (represented as Fourier series), holiday effects, and extra regressors. It supports uncertainty estimation through sampling, cross-validation for model evaluation, and provides both fitting and prediction capabilities with customizable parameters for changepoints, seasonality strength, and holidays.

## Scope
- Core forecasting module with Prophet-style additive model
- Support for linear, logistic, and flat growth trends
- Fourier-series based seasonality (yearly, weekly, daily, custom)
- Holiday effects with configurable windows
- Extra regressors (additive and multiplicative)
- Cross-validation and performance metrics (MSE, RMSE, MAE, MAPE, MDAPE, SMAPE, coverage)
- Model serialization to/from JSON
- Plotting capabilities for forecasts and components
- Warm-start parameter utilities
- Country holiday integration via holidays library
