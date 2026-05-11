# Todo

## Plan
Build the forecasting library in layers: first implement the core forecaster with trend modeling and prediction capabilities (can work without Stan backend for simple cases), then add seasonality/holiday features, then cross-validation/diagnostics, and finally serialization and plotting. Each feature will be testable independently with synthetic data.

## Tasks
- [ ] Task 1: Core forecaster with trend modeling (src/timeweaver/forecaster.py + tests/test_forecaster.py)
- [ ] Task 2: Seasonality components with Fourier series (extend forecaster.py + tests)
- [ ] Task 3: Holiday effects with windows and prior scales (extend forecaster.py + tests)
- [ ] Task 4: Extra regressors support (extend forecaster.py + tests)
- [ ] Task 5: Prediction and uncertainty estimation (extend forecaster.py + tests)
- [ ] Task 6: Cross-validation and performance metrics (src/timeweaver/validation.py + tests)
- [ ] Task 7: Model serialization to/from JSON (src/timeweaver/storage.py + tests)
- [ ] Task 8: Country holidays integration (src/timeweaver/holidays.py + tests)
- [ ] Task 9: Plotting forecasts and components (src/timeweaver/visualization.py + tests)
- [ ] Task 10: Warm-start utilities (src/timeweaver/utilities.py + tests)
