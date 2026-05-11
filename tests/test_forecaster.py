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
