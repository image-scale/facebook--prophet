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

## Task 4: Extra regressors support

### Acceptance Criteria
- [x] add_regressor(name, prior_scale, standardize, mode) adds custom regressor
- [x] Regressor column must be present in fit() and predict() dataframes
- [x] Regressors must be added prior to model fitting
- [x] Regressor can be 'additive' or 'multiplicative' mode
- [x] Auto standardization: standardize continuous, not binary regressors
- [x] Forced standardization via standardize=True
- [x] Regressor features integrated into prediction
- [x] extra_regressors_additive and extra_regressors_multiplicative group components
- [x] Multiple regressors supported
- [x] Method chaining supported for add_regressor

## Task 5: Prediction and uncertainty estimation

### Acceptance Criteria
- [x] predict() returns yhat_lower and yhat_upper columns
- [x] predict() returns trend_lower and trend_upper columns
- [x] interval_width parameter controls interval size (default 0.80)
- [x] uncertainty_samples=0 disables uncertainty estimation
- [x] sample_predictive_trend() simulates future trend with new changepoints
- [x] sample_model() samples from generative model with noise
- [x] sample_posterior_predictive() returns samples for computing intervals
- [x] sigma_obs computed from training residuals
- [x] Uncertainty wider for future dates due to extrapolation
- [x] Works with all growth types (linear, logistic, flat)

## Task 6: Cross-validation and performance metrics

### Acceptance Criteria
- [x] generate_cutoffs() creates cutoff dates with proper spacing
- [x] cross_validation() performs time series cross-validation
- [x] cross_validation() returns df with ds, yhat, y, cutoff columns
- [x] Custom cutoffs can be provided to cross_validation()
- [x] mse, rmse, mae, mape, smape metric functions implemented
- [x] coverage metric computes interval coverage
- [x] performance_metrics() computes metrics by horizon
- [x] Unfitted model raises RuntimeError
- [x] Metrics skip coverage when intervals not available
- [x] Validation functions exported from package

## Task 7: Model serialization to/from JSON

### Acceptance Criteria
- [x] model_to_dict() converts fitted model to dictionary
- [x] model_from_dict() reconstructs model from dictionary
- [x] model_to_json() serializes model to JSON string
- [x] model_from_json() deserializes model from JSON string
- [x] save_model() saves model to file
- [x] load_model() loads model from file
- [x] Roundtrip preserves predictions
- [x] Works with regressors and holidays
- [x] Unfitted model raises ValueError
- [x] Storage functions exported from package

## Task 8: Country holidays integration

### Acceptance Criteria
- [x] make_holidays_df() creates holiday dataframe for years and country
- [x] get_holiday_names() returns all holiday names for a country
- [x] add_country_holidays() adds country holidays to model
- [x] Country holidays applied during fit()
- [x] Combines with manual holidays if both present
- [x] Supports multiple countries (US, UK, DE, FR)
- [x] Supports province/state regional holidays
- [x] Invalid country raises ValueError
- [x] Adding after fit raises RuntimeError
- [x] Holiday functions exported from package

## Task 9: Plotting forecasts and components

### Acceptance Criteria
- [x] plot() creates forecast plot with observations and predictions
- [x] plot() supports uncertainty intervals
- [x] plot() supports custom axes, labels, figsize, legend
- [x] plot_components() plots trend, seasonality, holidays
- [x] plot_components() includes extra regressors if present
- [x] plot_cross_validation_metric() plots CV metrics by horizon
- [x] All functions return matplotlib Figure
- [x] Graceful error if matplotlib not installed
- [x] Plotting functions exported from package
- [x] Works with different model configurations

## Task 10: Warm-start utilities

### Acceptance Criteria
- [x] regressor_index() returns column index of regressor in beta matrix
- [x] regressor_coefficients() returns DataFrame with coef, bounds, mode, center
- [x] warm_start_params() extracts k, m, sigma_obs, delta, beta for warm-starting
- [x] seasonality_plot_df() creates DataFrame for plotting seasonality component
- [x] get_changepoint_dates() returns Series of changepoint dates
- [x] get_changepoint_magnitudes() returns DataFrame with dates and delta values
- [x] Unfitted model raises RuntimeError
- [x] Invalid regressor/seasonality names raise ValueError
- [x] Works with different growth types (linear, logistic, flat)
- [x] Utility functions exported from package
