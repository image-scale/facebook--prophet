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
- [x] fourier_series(dates, period, series_order) generates sin/cos features
- [x] Weekly seasonality auto-detected when history >= 2 weeks and spacing < 7 days
- [x] Yearly seasonality auto-detected when history >= 2 years
- [x] Daily seasonality auto-detected when history >= 2 days and spacing < 1 day
- [x] add_seasonality(name, period, fourier_order) adds custom seasonality
- [x] Conditional seasonality with condition_name column support
- [x] Seasonality features integrated into prediction
- [x] seasonality_mode can be 'additive' or 'multiplicative'
- [x] Seasonality prior scale controls regularization
- [x] Built-in seasonality names can be overridden

## Task 3: Holiday effects with windows and prior scales

### Acceptance Criteria
- [x] Holidays passed as DataFrame with 'ds' and 'holiday' columns
- [x] lower_window and upper_window specify days around holiday
- [x] Holiday features created with one-hot encoding for each holiday+offset
- [x] prior_scale per holiday controls regularization strength
- [x] holidays_prior_scale default applies to all holidays
- [x] Holidays integrated into prediction with separate 'holidays' component
- [x] holidays_mode can be 'additive' or 'multiplicative'
- [x] Validation: holidays must have ds and holiday columns
- [x] Validation: lower_window must be <= 0, upper_window >= 0
