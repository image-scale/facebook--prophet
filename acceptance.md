# Acceptance Criteria

## Task 1: Core forecaster with trend modeling

### Acceptance Criteria
- [x] TimeWeaver class can be instantiated with growth type (linear, logistic, flat)
- [x] fit() method accepts a DataFrame with 'ds' and 'y' columns
- [x] fit() validates input data: requires ds and y columns, rejects NaN in ds, handles NaN in y
- [x] Dataframe is prepared: dates are sorted, time scaled to [0,1], y scaled appropriately
- [x] Linear growth initialization computes slope k and offset m from first/last points
- [x] Logistic growth initialization computes k and m using capacity column
- [x] Flat growth initialization sets k=0 and m=mean(y)
- [x] Changepoints are auto-selected within changepoint_range (default 80%) of history
- [x] Custom changepoints can be specified as a list of dates
- [x] n_changepoints parameter controls number of changepoints (default 25)
- [x] piecewise_linear(t, deltas, k, m, changepoint_ts) computes trend correctly
- [x] piecewise_logistic(t, cap, deltas, k, m, changepoint_ts) computes logistic trend
- [x] flat_trend(t, m) returns constant m for all time points
- [x] predict_trend() returns trend values on y's original scale
- [x] make_future_dataframe() generates future dates with correct frequency
- [x] Invalid growth type raises ValueError

## Task 2: Seasonality components with Fourier series

### Acceptance Criteria
- [ ] fourier_series(dates, period, series_order) generates sin/cos features
- [ ] Weekly seasonality auto-detected when history >= 2 weeks and spacing < 7 days
- [ ] Yearly seasonality auto-detected when history >= 2 years
- [ ] Daily seasonality auto-detected when history >= 2 days and spacing < 1 day
- [ ] add_seasonality(name, period, fourier_order) adds custom seasonality
- [ ] Conditional seasonality with condition_name column support
- [ ] Seasonality features integrated into prediction
- [ ] seasonality_mode can be 'additive' or 'multiplicative'
- [ ] Seasonality prior scale controls regularization
- [ ] Duplicate seasonality names raise ValueError
