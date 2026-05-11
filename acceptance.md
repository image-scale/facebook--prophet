# Acceptance Criteria

## Task 1: Core forecaster with trend modeling

### Acceptance Criteria
- [ ] TimeWeaver class can be instantiated with growth type (linear, logistic, flat)
- [ ] fit() method accepts a DataFrame with 'ds' and 'y' columns
- [ ] fit() validates input data: requires ds and y columns, rejects NaN in ds, handles NaN in y
- [ ] Dataframe is prepared: dates are sorted, time scaled to [0,1], y scaled appropriately
- [ ] Linear growth initialization computes slope k and offset m from first/last points
- [ ] Logistic growth initialization computes k and m using capacity column
- [ ] Flat growth initialization sets k=0 and m=mean(y)
- [ ] Changepoints are auto-selected within changepoint_range (default 80%) of history
- [ ] Custom changepoints can be specified as a list of dates
- [ ] n_changepoints parameter controls number of changepoints (default 25)
- [ ] piecewise_linear(t, deltas, k, m, changepoint_ts) computes trend correctly
- [ ] piecewise_logistic(t, cap, deltas, k, m, changepoint_ts) computes logistic trend
- [ ] flat_trend(t, m) returns constant m for all time points
- [ ] predict_trend() returns trend values on y's original scale
- [ ] make_future_dataframe() generates future dates with correct frequency
- [ ] Invalid growth type raises ValueError
