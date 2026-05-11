"""Tests for TimeWeaver core forecaster."""

import numpy as np
import pandas as pd
import pytest

from timeweaver import TimeWeaver


def create_daily_data(start: str = "2012-01-01", periods: int = 365 * 2) -> pd.DataFrame:
    """Create synthetic daily time series data."""
    dates = pd.date_range(start=start, periods=periods, freq='D')
    np.random.seed(42)
    trend = np.linspace(10, 50, periods)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    noise = np.random.normal(0, 2, periods)
    y = trend + seasonality + noise
    return pd.DataFrame({'ds': dates, 'y': y})


def create_subdaily_data(start: str = "2017-01-01", periods: int = 24 * 30) -> pd.DataFrame:
    """Create synthetic sub-daily time series data."""
    dates = pd.date_range(start=start, periods=periods, freq='h')  # lowercase 'h' for newer pandas
    np.random.seed(42)
    trend = np.linspace(10, 30, periods)
    daily_pattern = 5 * np.sin(2 * np.pi * np.arange(periods) / 24)
    noise = np.random.normal(0, 1, periods)
    y = trend + daily_pattern + noise
    return pd.DataFrame({'ds': dates, 'y': y})


class TestTimeWeaverInit:
    """Tests for TimeWeaver initialization."""

    def test_default_init(self):
        """Test default initialization."""
        m = TimeWeaver()
        assert m.growth == "linear"
        assert m.n_changepoints == 25
        assert m.changepoint_range == 0.8
        assert m.seasonality_mode == "additive"
        assert m.scaling == "absmax"

    def test_growth_types(self):
        """Test valid growth types."""
        for growth in ["linear", "logistic", "flat"]:
            m = TimeWeaver(growth=growth)
            assert m.growth == growth

    def test_invalid_growth_raises(self):
        """Test that invalid growth raises ValueError."""
        with pytest.raises(ValueError, match='"growth" should be'):
            TimeWeaver(growth="exponential")

    def test_invalid_changepoint_range_raises(self):
        """Test that invalid changepoint_range raises ValueError."""
        with pytest.raises(ValueError):
            TimeWeaver(changepoint_range=-0.1)
        with pytest.raises(ValueError):
            TimeWeaver(changepoint_range=1.5)

    def test_invalid_seasonality_mode_raises(self):
        """Test that invalid seasonality_mode raises ValueError."""
        with pytest.raises(ValueError, match='seasonality_mode'):
            TimeWeaver(seasonality_mode="unknown")

    def test_invalid_holidays_mode_raises(self):
        """Test that invalid holidays_mode raises ValueError."""
        with pytest.raises(ValueError, match='holidays_mode'):
            TimeWeaver(holidays_mode="unknown")

    def test_invalid_scaling_raises(self):
        """Test that invalid scaling raises ValueError."""
        with pytest.raises(ValueError, match="scaling"):
            TimeWeaver(scaling="standard")

    def test_custom_changepoints(self):
        """Test initialization with custom changepoints."""
        changepoints = ["2012-06-01", "2012-12-01"]
        m = TimeWeaver(changepoints=changepoints)
        assert m.specified_changepoints is True
        assert m.n_changepoints == 2

    def test_holidays_mode_defaults_to_seasonality(self):
        """Test holidays_mode defaults to seasonality_mode."""
        m = TimeWeaver(seasonality_mode="multiplicative")
        assert m.holidays_mode == "multiplicative"


class TestTimeWeaverDataPrep:
    """Tests for data preparation."""

    def test_prepare_dataframe_creates_t_column(self):
        """Test that prepare_dataframe adds t column."""
        df = create_daily_data(periods=100)
        m = TimeWeaver()
        prepared = m.prepare_dataframe(df.copy(), initialize_scales=True)
        assert 't' in prepared.columns
        assert prepared['t'].min() == 0.0
        assert prepared['t'].max() == 1.0

    def test_prepare_dataframe_creates_y_scaled(self):
        """Test that prepare_dataframe adds y_scaled column."""
        df = create_daily_data(periods=100)
        m = TimeWeaver()
        prepared = m.prepare_dataframe(df.copy(), initialize_scales=True)
        assert 'y_scaled' in prepared.columns
        assert prepared['y_scaled'].max() == 1.0

    def test_prepare_dataframe_minmax_scaling(self):
        """Test minmax scaling."""
        df = create_daily_data(periods=100)
        m = TimeWeaver(scaling="minmax")
        prepared = m.prepare_dataframe(df.copy(), initialize_scales=True)
        assert 'y_scaled' in prepared.columns
        assert prepared['y_scaled'].min() >= 0.0
        assert prepared['y_scaled'].max() <= 1.0

    def test_prepare_dataframe_rejects_nan_ds(self):
        """Test that NaN in ds raises ValueError."""
        df = create_daily_data(periods=100)
        df.loc[5, 'ds'] = pd.NaT
        m = TimeWeaver()
        with pytest.raises(ValueError, match='Found NaN in column ds'):
            m.prepare_dataframe(df.copy(), initialize_scales=True)

    def test_prepare_dataframe_rejects_inf_y(self):
        """Test that infinity in y raises ValueError."""
        df = create_daily_data(periods=100)
        df.loc[5, 'y'] = np.inf
        m = TimeWeaver()
        with pytest.raises(ValueError, match='Found infinity in column y'):
            m.prepare_dataframe(df.copy(), initialize_scales=True)

    def test_prepare_dataframe_sorts_by_date(self):
        """Test that dataframe is sorted by date."""
        df = create_daily_data(periods=100)
        df = df.sample(frac=1)  # shuffle
        m = TimeWeaver()
        prepared = m.prepare_dataframe(df.copy(), initialize_scales=True)
        assert (prepared['ds'].diff().dropna() >= pd.Timedelta(0)).all()


class TestTimeWeaverChangepoints:
    """Tests for changepoint detection."""

    def test_auto_changepoints(self):
        """Test automatic changepoint selection."""
        df = create_daily_data(periods=200)
        m = TimeWeaver(n_changepoints=10)
        m.history = m.prepare_dataframe(df.copy(), initialize_scales=True)
        m.set_changepoints()
        assert len(m.changepoints) == 10
        assert m.changepoints_t.shape[0] == 10

    def test_zero_changepoints(self):
        """Test with n_changepoints=0."""
        df = create_daily_data(periods=200)
        m = TimeWeaver(n_changepoints=0)
        m.history = m.prepare_dataframe(df.copy(), initialize_scales=True)
        m.set_changepoints()
        assert len(m.changepoints) == 0
        assert m.changepoints_t.shape[0] == 1  # dummy
        assert m.changepoints_t[0] == 0

    def test_changepoint_range(self):
        """Test that changepoints respect changepoint_range."""
        df = create_daily_data(periods=200)
        m = TimeWeaver(changepoint_range=0.5, n_changepoints=10)
        m.history = m.prepare_dataframe(df.copy(), initialize_scales=True)
        m.set_changepoints()
        max_cp_t = m.changepoints_t.max()
        # Changepoints should be in first 50% of data
        assert max_cp_t <= 0.5 + 0.01  # small tolerance

    def test_custom_changepoints_validated(self):
        """Test custom changepoints outside training data raises error."""
        df = create_daily_data(start="2012-01-01", periods=100)
        m = TimeWeaver(changepoints=["2015-01-01"])  # after training data
        m.history = m.prepare_dataframe(df.copy(), initialize_scales=True)
        with pytest.raises(ValueError, match='Changepoints must fall within'):
            m.set_changepoints()


class TestTimeWeaverGrowthInit:
    """Tests for growth initialization."""

    def test_linear_growth_init(self):
        """Test linear growth initialization."""
        df = create_daily_data(periods=200)
        m = TimeWeaver()
        history = m.prepare_dataframe(df.copy(), initialize_scales=True)
        k, offset = m.linear_growth_init(history)
        # k should be positive since trend is increasing
        assert k > 0

    def test_flat_growth_init(self):
        """Test flat growth initialization."""
        df = create_daily_data(periods=200)
        m = TimeWeaver(growth="flat")
        history = m.prepare_dataframe(df.copy(), initialize_scales=True)
        k, offset = m.flat_growth_init(history)
        assert k == 0
        # offset should be close to mean of y_scaled
        assert abs(offset - history['y_scaled'].mean()) < 0.01

    def test_logistic_growth_init(self):
        """Test logistic growth initialization."""
        df = create_daily_data(periods=200)
        df['cap'] = df['y'].max() * 2
        m = TimeWeaver(growth="logistic")
        history = m.prepare_dataframe(df.copy(), initialize_scales=True)
        k, offset = m.logistic_growth_init(history)
        assert isinstance(k, float)
        assert isinstance(offset, float)


class TestTimeWeaverTrend:
    """Tests for trend functions."""

    def test_piecewise_linear_no_changepoints(self):
        """Test piecewise linear with no changepoints."""
        t = np.array([0.0, 0.5, 1.0])
        deltas = np.array([])
        k = 1.0
        m = 0.0
        changepoints = np.array([])
        # With empty changepoints, should just be k*t + m
        result = TimeWeaver.piecewise_linear(t, deltas, k, m, changepoints)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_piecewise_linear_with_changepoint(self):
        """Test piecewise linear with a changepoint."""
        t = np.arange(11.0)
        deltas = np.array([0.5])
        k = 1.0
        m = 0.0
        changepoints = np.array([5.0])
        result = TimeWeaver.piecewise_linear(t, deltas, k, m, changepoints)
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_flat_trend(self):
        """Test flat trend function."""
        t = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        m = 2.5
        result = TimeWeaver.flat_trend(t, m)
        expected = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_piecewise_logistic_basic(self):
        """Test piecewise logistic function."""
        t = np.arange(11.0)
        cap = np.ones(11) * 10
        deltas = np.array([0.5])
        k = 1.0
        m = 0.0
        changepoints = np.array([5.0])
        result = TimeWeaver.piecewise_logistic(t, cap, deltas, k, m, changepoints)
        # Values should be between 0 and cap
        assert (result >= 0).all()
        assert (result <= 10).all()
        # Should be monotonically increasing
        assert (np.diff(result) >= 0).all()


class TestTimeWeaverFit:
    """Tests for model fitting."""

    def test_fit_basic(self):
        """Test basic fitting."""
        df = create_daily_data(periods=200)
        m = TimeWeaver()
        m.fit(df)
        assert m.history is not None
        assert m.params is not None
        assert 'k' in m.params
        assert 'm' in m.params

    def test_fit_requires_ds_and_y(self):
        """Test that fit requires ds and y columns."""
        df = pd.DataFrame({'date': pd.date_range('2012-01-01', periods=10)})
        m = TimeWeaver()
        with pytest.raises(ValueError, match='Dataframe must have columns'):
            m.fit(df)

    def test_fit_requires_min_rows(self):
        """Test that fit requires at least 2 rows."""
        df = pd.DataFrame({'ds': ['2012-01-01'], 'y': [1.0]})
        m = TimeWeaver()
        with pytest.raises(ValueError, match='less than 2 non-NaN'):
            m.fit(df)

    def test_fit_handles_nan_y(self):
        """Test that fit handles NaN in y."""
        df = create_daily_data(periods=100)
        df.loc[5, 'y'] = np.nan
        df.loc[10, 'y'] = np.nan
        m = TimeWeaver()
        m.fit(df)
        # Should have 98 rows in history (100 - 2 NaN)
        assert m.history.shape[0] == 98

    def test_fit_only_once(self):
        """Test that model can only be fit once."""
        df = create_daily_data(periods=100)
        m = TimeWeaver()
        m.fit(df)
        with pytest.raises(RuntimeError, match='can only be fit once'):
            m.fit(df)

    def test_fit_logistic_requires_cap(self):
        """Test that logistic growth requires cap column."""
        df = create_daily_data(periods=100)
        m = TimeWeaver(growth="logistic")
        with pytest.raises(ValueError, match='Capacities must be supplied'):
            m.fit(df)

    def test_fit_logistic_with_cap(self):
        """Test fitting with logistic growth and cap."""
        df = create_daily_data(periods=100)
        df['cap'] = df['y'].max() * 2
        m = TimeWeaver(growth="logistic")
        m.fit(df)
        assert m.history is not None

    def test_fit_flat_growth(self):
        """Test fitting with flat growth."""
        df = create_daily_data(periods=100)
        m = TimeWeaver(growth="flat")
        m.fit(df)
        assert m.params['k'][0, 0] == pytest.approx(0.0, abs=1e-10)


class TestTimeWeaverPredict:
    """Tests for prediction."""

    def test_predict_on_history(self):
        """Test prediction on training data."""
        df = create_daily_data(periods=200)
        m = TimeWeaver()
        m.fit(df)
        forecast = m.predict()
        assert 'ds' in forecast.columns
        assert 'trend' in forecast.columns
        assert 'yhat' in forecast.columns
        assert len(forecast) == len(m.history)

    def test_predict_on_future(self):
        """Test prediction on future data."""
        df = create_daily_data(periods=200)
        m = TimeWeaver()
        m.fit(df)
        future = m.make_future_dataframe(periods=30, include_history=False)
        forecast = m.predict(future)
        assert len(forecast) == 30

    def test_predict_before_fit_raises(self):
        """Test that predicting before fit raises error."""
        m = TimeWeaver()
        with pytest.raises(RuntimeError, match='Model has not been fit'):
            m.predict()

    def test_predict_empty_df_raises(self):
        """Test that predicting on empty df raises error."""
        df = create_daily_data(periods=100)
        m = TimeWeaver()
        m.fit(df)
        empty_df = pd.DataFrame({'ds': []})
        with pytest.raises(ValueError, match='Dataframe has no rows'):
            m.predict(empty_df)

    def test_predict_trend_linear(self):
        """Test that linear trend predictions are reasonable."""
        df = create_daily_data(periods=200)
        m = TimeWeaver()
        m.fit(df)
        forecast = m.predict()
        # Trend should be roughly increasing (like the data)
        trend = forecast['trend'].values
        assert trend[-1] > trend[0]


class TestTimeWeaverMakeFuture:
    """Tests for make_future_dataframe."""

    def test_make_future_basic(self):
        """Test basic future dataframe creation."""
        df = create_daily_data(periods=100)
        m = TimeWeaver()
        m.fit(df)
        future = m.make_future_dataframe(periods=30)
        # Should include history + 30 future periods
        assert len(future) == 100 + 30

    def test_make_future_no_history(self):
        """Test future dataframe without history."""
        df = create_daily_data(periods=100)
        m = TimeWeaver()
        m.fit(df)
        future = m.make_future_dataframe(periods=30, include_history=False)
        assert len(future) == 30

    def test_make_future_before_fit_raises(self):
        """Test that make_future before fit raises error."""
        m = TimeWeaver()
        with pytest.raises(RuntimeError, match='Model has not been fit'):
            m.make_future_dataframe(periods=30)

    def test_make_future_daily_freq(self):
        """Test future dataframe with daily frequency."""
        df = create_daily_data(start="2012-01-01", periods=100)
        m = TimeWeaver()
        m.fit(df)
        future = m.make_future_dataframe(periods=3, include_history=False)
        last_hist = df['ds'].max()
        expected = [
            last_hist + pd.Timedelta(days=1),
            last_hist + pd.Timedelta(days=2),
            last_hist + pd.Timedelta(days=3),
        ]
        assert len(future) == 3
        for i, exp in enumerate(expected):
            assert future['ds'].iloc[i] == exp


class TestTimeWeaverConstantData:
    """Tests for edge cases with constant data."""

    def test_constant_y_linear(self):
        """Test with constant y values in linear mode."""
        dates = pd.date_range('2012-01-01', periods=100, freq='D')
        df = pd.DataFrame({'ds': dates, 'y': 20.0})
        m = TimeWeaver(growth="linear")
        m.fit(df)
        forecast = m.predict()
        # With constant data, predictions should be constant
        assert forecast['yhat'].std() < 0.01

    def test_constant_y_flat(self):
        """Test with constant y values in flat mode."""
        dates = pd.date_range('2012-01-01', periods=100, freq='D')
        df = pd.DataFrame({'ds': dates, 'y': 30.0})
        m = TimeWeaver(growth="flat")
        m.fit(df)
        forecast = m.predict()
        # Predictions should be the constant value
        np.testing.assert_array_almost_equal(forecast['yhat'].values, 30.0, decimal=1)


class TestTimeWeaverFourierSeries:
    """Tests for Fourier series generation."""

    def test_fourier_series_weekly(self):
        """Test weekly Fourier series matches expected values."""
        df = create_daily_data(periods=365)
        mat = TimeWeaver.fourier_series(df['ds'], 7, 3)
        # Shape should be (n_dates, 2 * series_order)
        assert mat.shape == (365, 6)
        # Values should be between -1 and 1
        assert mat.min() >= -1.0
        assert mat.max() <= 1.0

    def test_fourier_series_yearly(self):
        """Test yearly Fourier series."""
        df = create_daily_data(periods=365)
        mat = TimeWeaver.fourier_series(df['ds'], 365.25, 3)
        assert mat.shape == (365, 6)

    def test_fourier_series_order_validation(self):
        """Test that invalid series_order raises error."""
        df = create_daily_data(periods=10)
        with pytest.raises(ValueError, match="series_order must be >= 1"):
            TimeWeaver.fourier_series(df['ds'], 7, 0)

    def test_make_seasonality_features(self):
        """Test seasonality feature dataframe creation."""
        df = create_daily_data(periods=100)
        features = TimeWeaver.make_seasonality_features(df['ds'], 7, 3, 'weekly')
        assert features.shape == (100, 6)
        assert all(col.startswith('weekly_delim_') for col in features.columns)


class TestTimeWeaverAutoSeasonality:
    """Tests for automatic seasonality detection."""

    def test_auto_yearly_enabled(self):
        """Test yearly seasonality enabled with 2+ years data."""
        df = create_daily_data(periods=365 * 2 + 10)
        m = TimeWeaver()
        assert m.yearly_seasonality == "auto"
        m.fit(df)
        assert 'yearly' in m.seasonalities
        assert m.seasonalities['yearly']['period'] == 365.25
        assert m.seasonalities['yearly']['fourier_order'] == 10

    def test_auto_yearly_disabled_short_history(self):
        """Test yearly seasonality disabled with < 2 years data."""
        df = create_daily_data(periods=365)  # Only 1 year
        m = TimeWeaver()
        m.fit(df)
        assert 'yearly' not in m.seasonalities

    def test_yearly_seasonality_forced(self):
        """Test yearly seasonality can be forced on."""
        df = create_daily_data(periods=365)  # Short history
        m = TimeWeaver(yearly_seasonality=True)
        m.fit(df)
        assert 'yearly' in m.seasonalities

    def test_auto_weekly_enabled(self):
        """Test weekly seasonality enabled with 2+ weeks data."""
        df = create_daily_data(periods=20)  # More than 2 weeks
        m = TimeWeaver()
        m.fit(df)
        assert 'weekly' in m.seasonalities
        assert m.seasonalities['weekly']['period'] == 7
        assert m.seasonalities['weekly']['fourier_order'] == 3

    def test_auto_weekly_disabled_short_history(self):
        """Test weekly seasonality disabled with < 2 weeks data."""
        df = create_daily_data(periods=10)  # Less than 2 weeks
        m = TimeWeaver()
        m.fit(df)
        assert 'weekly' not in m.seasonalities

    def test_auto_weekly_disabled_weekly_spacing(self):
        """Test weekly seasonality disabled with weekly spacing."""
        df = create_daily_data(periods=100)
        df = df.iloc[::7]  # Weekly data
        m = TimeWeaver()
        m.fit(df)
        assert 'weekly' not in m.seasonalities

    def test_auto_daily_enabled(self):
        """Test daily seasonality enabled with sub-daily data."""
        df = create_subdaily_data(periods=24 * 5)  # 5 days of hourly data
        m = TimeWeaver()
        m.fit(df)
        assert 'daily' in m.seasonalities
        assert m.seasonalities['daily']['period'] == 1
        assert m.seasonalities['daily']['fourier_order'] == 4

    def test_auto_daily_disabled_daily_data(self):
        """Test daily seasonality disabled with daily data."""
        df = create_daily_data(periods=100)
        m = TimeWeaver()
        m.fit(df)
        assert 'daily' not in m.seasonalities

    def test_custom_fourier_order(self):
        """Test custom Fourier order for seasonality."""
        df = create_daily_data(periods=365 * 2 + 10)
        m = TimeWeaver(yearly_seasonality=7, weekly_seasonality=5)
        m.fit(df)
        assert m.seasonalities['yearly']['fourier_order'] == 7
        assert m.seasonalities['weekly']['fourier_order'] == 5


class TestTimeWeaverCustomSeasonality:
    """Tests for custom seasonality."""

    def test_add_custom_seasonality(self):
        """Test adding a custom seasonality."""
        m = TimeWeaver()
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        assert 'monthly' in m.seasonalities
        assert m.seasonalities['monthly']['period'] == 30.5
        assert m.seasonalities['monthly']['fourier_order'] == 5
        assert m.seasonalities['monthly']['prior_scale'] == 10.0

    def test_add_seasonality_with_prior_scale(self):
        """Test adding seasonality with custom prior scale."""
        m = TimeWeaver()
        m.add_seasonality(name='monthly', period=30, fourier_order=5, prior_scale=2.0)
        assert m.seasonalities['monthly']['prior_scale'] == 2.0

    def test_add_seasonality_multiplicative(self):
        """Test adding multiplicative seasonality."""
        m = TimeWeaver()
        m.add_seasonality(name='monthly', period=30, fourier_order=3, mode='multiplicative')
        assert m.seasonalities['monthly']['mode'] == 'multiplicative'

    def test_add_seasonality_invalid_prior_scale(self):
        """Test that invalid prior_scale raises error."""
        m = TimeWeaver()
        with pytest.raises(ValueError, match='Prior scale must be > 0'):
            m.add_seasonality(name='test', period=30, fourier_order=3, prior_scale=0)

    def test_add_seasonality_invalid_fourier_order(self):
        """Test that invalid fourier_order raises error."""
        m = TimeWeaver()
        with pytest.raises(ValueError, match='Fourier order must be > 0'):
            m.add_seasonality(name='test', period=30, fourier_order=0)

    def test_add_seasonality_after_fit_raises(self):
        """Test that adding seasonality after fit raises error."""
        df = create_daily_data(periods=100)
        m = TimeWeaver()
        m.fit(df)
        with pytest.raises(RuntimeError, match='prior to model fitting'):
            m.add_seasonality(name='monthly', period=30, fourier_order=3)

    def test_duplicate_seasonality_name_allowed_override(self):
        """Test that duplicate seasonality name updates the existing one."""
        m = TimeWeaver()
        m.add_seasonality(name='custom', period=30, fourier_order=3)
        # Overriding with same name should work (updates)
        m.add_seasonality(name='custom', period=15, fourier_order=2)
        assert m.seasonalities['custom']['period'] == 15
        assert m.seasonalities['custom']['fourier_order'] == 2

    def test_reserved_name_raises(self):
        """Test that reserved names raise error."""
        m = TimeWeaver()
        with pytest.raises(ValueError, match='reserved'):
            m.add_seasonality(name='trend', period=30, fourier_order=3)

    def test_can_override_builtin_seasonality(self):
        """Test that built-in seasonality names can be overridden."""
        m = TimeWeaver()
        m.add_seasonality(name='weekly', period=7, fourier_order=5)
        assert m.seasonalities['weekly']['fourier_order'] == 5


class TestTimeWeaverConditionalSeasonality:
    """Tests for conditional seasonality."""

    def test_conditional_seasonality_basic(self):
        """Test basic conditional seasonality."""
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False)
        m.add_seasonality(
            name='conditional_weekly',
            period=7,
            fourier_order=3,
            condition_name='is_condition'
        )
        df = create_daily_data(periods=100)
        df['is_condition'] = [False] * 50 + [True] * 50
        m.fit(df)
        assert m.seasonalities['conditional_weekly']['condition_name'] == 'is_condition'

    def test_conditional_seasonality_missing_column_raises(self):
        """Test that missing condition column raises error."""
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False)
        m.add_seasonality(
            name='cond',
            period=7,
            fourier_order=3,
            condition_name='missing_col'
        )
        df = create_daily_data(periods=100)
        with pytest.raises(ValueError, match='missing from dataframe'):
            m.fit(df)

    def test_conditional_seasonality_non_boolean_raises(self):
        """Test that non-boolean condition column raises error."""
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False)
        m.add_seasonality(
            name='cond',
            period=7,
            fourier_order=3,
            condition_name='bad_col'
        )
        df = create_daily_data(periods=100)
        df['bad_col'] = [2] * 100  # Not boolean
        with pytest.raises(ValueError, match='non-boolean'):
            m.fit(df)


class TestTimeWeaverSeasonalityModes:
    """Tests for seasonality modes."""

    def test_default_seasonality_mode(self):
        """Test default seasonality mode is additive."""
        m = TimeWeaver()
        assert m.seasonality_mode == 'additive'

    def test_multiplicative_mode(self):
        """Test multiplicative seasonality mode."""
        df = create_daily_data(periods=365 * 2 + 10)
        m = TimeWeaver(seasonality_mode='multiplicative')
        m.fit(df)
        assert m.seasonalities['yearly']['mode'] == 'multiplicative'
        assert m.seasonalities['weekly']['mode'] == 'multiplicative'

    def test_mixed_modes(self):
        """Test mixed additive and multiplicative modes."""
        m = TimeWeaver(seasonality_mode='multiplicative')
        m.add_seasonality(name='monthly', period=30, fourier_order=3, mode='additive')
        assert m.seasonalities['monthly']['mode'] == 'additive'

    def test_predict_with_seasonality(self):
        """Test prediction includes seasonality components."""
        df = create_daily_data(periods=365 * 2 + 10)
        m = TimeWeaver()
        m.fit(df)
        forecast = m.predict()
        assert 'weekly' in forecast.columns
        assert 'yearly' in forecast.columns
        assert 'additive_terms' in forecast.columns
        assert 'multiplicative_terms' in forecast.columns
        assert 'yhat' in forecast.columns


class TestTimeWeaverHolidays:
    """Tests for holiday effects."""

    def test_holidays_basic(self):
        """Test basic holiday features."""
        holidays = pd.DataFrame({
            'ds': ['2012-06-15', '2012-12-25'],
            'holiday': ['special_day', 'christmas'],
        })
        m = TimeWeaver(holidays=holidays, weekly_seasonality=False, yearly_seasonality=False)
        df = create_daily_data(start="2012-01-01", periods=365)
        m.fit(df)
        forecast = m.predict()
        assert 'holidays' in forecast.columns

    def test_holidays_with_window(self):
        """Test holidays with lower and upper window."""
        holidays = pd.DataFrame({
            'ds': ['2012-06-15'],
            'holiday': ['special_day'],
            'lower_window': [-2],
            'upper_window': [1],
        })
        m = TimeWeaver(holidays=holidays, weekly_seasonality=False, yearly_seasonality=False)
        df = create_daily_data(start="2012-01-01", periods=365)
        m.fit(df)
        # Should have 4 features: -2, -1, 0, +1
        assert m.train_holiday_names is not None
        assert len(m.train_holiday_names) == 4

    def test_holidays_validation_missing_ds(self):
        """Test that holidays must have ds column."""
        holidays = pd.DataFrame({'holiday': ['test']})
        with pytest.raises(ValueError, match='"ds" and "holiday" columns'):
            TimeWeaver(holidays=holidays)

    def test_holidays_validation_missing_holiday(self):
        """Test that holidays must have holiday column."""
        holidays = pd.DataFrame({'ds': ['2012-01-01']})
        with pytest.raises(ValueError, match='"ds" and "holiday" columns'):
            TimeWeaver(holidays=holidays)

    def test_holidays_validation_nan(self):
        """Test that NaN in holidays raises error."""
        holidays = pd.DataFrame({
            'ds': ['2012-01-01', None],
            'holiday': ['test', 'test2'],
        })
        with pytest.raises(ValueError, match='Found a NaN'):
            TimeWeaver(holidays=holidays)

    def test_holidays_validation_lower_window_positive(self):
        """Test that lower_window > 0 raises error."""
        holidays = pd.DataFrame({
            'ds': ['2012-01-01'],
            'holiday': ['test'],
            'lower_window': [1],
            'upper_window': [2],
        })
        with pytest.raises(ValueError, match='lower_window should be <= 0'):
            TimeWeaver(holidays=holidays)

    def test_holidays_validation_upper_window_negative(self):
        """Test that upper_window < 0 raises error."""
        holidays = pd.DataFrame({
            'ds': ['2012-01-01'],
            'holiday': ['test'],
            'lower_window': [-1],
            'upper_window': [-1],
        })
        with pytest.raises(ValueError, match='upper_window should be >= 0'):
            TimeWeaver(holidays=holidays)

    def test_holidays_validation_window_both_required(self):
        """Test that both window columns must be present or neither."""
        holidays = pd.DataFrame({
            'ds': ['2012-01-01'],
            'holiday': ['test'],
            'lower_window': [-1],
        })
        with pytest.raises(ValueError, match='both lower_window and upper_window'):
            TimeWeaver(holidays=holidays)

    def test_holidays_prior_scale(self):
        """Test per-holiday prior scale."""
        holidays = pd.DataFrame({
            'ds': ['2012-06-15', '2012-12-25'],
            'holiday': ['special', 'christmas'],
            'prior_scale': [5.0, 15.0],
        })
        m = TimeWeaver(holidays=holidays, weekly_seasonality=False, yearly_seasonality=False)
        df = create_daily_data(start="2012-01-01", periods=365)
        m.fit(df)
        assert m.train_holiday_names is not None

    def test_holidays_default_prior_scale(self):
        """Test default holidays_prior_scale applies."""
        holidays = pd.DataFrame({
            'ds': ['2012-06-15'],
            'holiday': ['special'],
        })
        m = TimeWeaver(holidays=holidays, holidays_prior_scale=20.0)
        assert m.holidays_prior_scale == 20.0

    def test_holidays_additive_mode(self):
        """Test holidays with additive mode."""
        holidays = pd.DataFrame({
            'ds': ['2012-06-15'],
            'holiday': ['special'],
        })
        m = TimeWeaver(holidays=holidays, holidays_mode='additive',
                       weekly_seasonality=False, yearly_seasonality=False)
        df = create_daily_data(start="2012-01-01", periods=365)
        m.fit(df)
        assert m.holidays_mode == 'additive'
        assert 'holidays' in m.component_modes['additive']

    def test_holidays_multiplicative_mode(self):
        """Test holidays with multiplicative mode."""
        holidays = pd.DataFrame({
            'ds': ['2012-06-15'],
            'holiday': ['special'],
        })
        m = TimeWeaver(holidays=holidays, holidays_mode='multiplicative',
                       weekly_seasonality=False, yearly_seasonality=False)
        df = create_daily_data(start="2012-01-01", periods=365)
        m.fit(df)
        assert m.holidays_mode == 'multiplicative'
        assert 'holidays' in m.component_modes['multiplicative']

    def test_holidays_in_prediction(self):
        """Test that holidays affect predictions on holiday dates."""
        holidays = pd.DataFrame({
            'ds': ['2012-06-15'],
            'holiday': ['special'],
        })
        df = create_daily_data(start="2012-01-01", periods=365)
        m = TimeWeaver(holidays=holidays, weekly_seasonality=False, yearly_seasonality=False)
        m.fit(df)
        forecast = m.predict()
        # Holiday column should exist
        assert 'holidays' in forecast.columns
        # Holiday effect should be non-zero on holiday date
        holiday_row = forecast[forecast['ds'] == pd.Timestamp('2012-06-15')]
        assert len(holiday_row) == 1

    def test_holidays_multiple_same_name(self):
        """Test multiple occurrences of the same holiday."""
        holidays = pd.DataFrame({
            'ds': ['2012-01-01', '2012-07-04', '2013-01-01'],
            'holiday': ['new_year', 'independence', 'new_year'],
        })
        m = TimeWeaver(holidays=holidays, weekly_seasonality=False, yearly_seasonality=False)
        df = create_daily_data(start="2012-01-01", periods=365 * 2)
        m.fit(df)
        assert m.train_holiday_names is not None

    def test_holidays_future_prediction(self):
        """Test that holidays work in future predictions."""
        holidays = pd.DataFrame({
            'ds': ['2012-06-15', '2013-06-15'],
            'holiday': ['special', 'special'],
        })
        df = create_daily_data(start="2012-01-01", periods=365)
        m = TimeWeaver(holidays=holidays, weekly_seasonality=False, yearly_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=365, include_history=False)
        forecast = m.predict(future)
        assert 'holidays' in forecast.columns
