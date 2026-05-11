"""Tests for TimeWeaver utilities."""

import numpy as np
import pandas as pd
import pytest

from timeweaver import (
    TimeWeaver,
    regressor_index,
    regressor_coefficients,
    warm_start_params,
    seasonality_plot_df,
    get_changepoint_dates,
    get_changepoint_magnitudes,
)


def create_daily_data(start: str = "2020-01-01", periods: int = 365) -> pd.DataFrame:
    """Create synthetic daily time series data."""
    dates = pd.date_range(start=start, periods=periods, freq='D')
    np.random.seed(42)
    trend = np.linspace(10, 50, periods)
    seasonality = 5 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    noise = np.random.normal(0, 1, periods)
    y = trend + seasonality + noise
    return pd.DataFrame({'ds': dates, 'y': y})


class TestRegressorIndex:
    """Tests for regressor_index function."""

    def test_regressor_index_basic(self):
        """Test getting index of a regressor."""
        df = create_daily_data()
        df['temp'] = np.random.randn(len(df))
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.add_regressor('temp')
        m.fit(df)

        idx = regressor_index(m, 'temp')
        assert isinstance(idx, int)
        assert idx >= 0

    def test_regressor_index_multiple_regressors(self):
        """Test getting index with multiple regressors."""
        df = create_daily_data()
        df['temp'] = np.random.randn(len(df))
        df['rain'] = np.random.randn(len(df))
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.add_regressor('temp')
        m.add_regressor('rain')
        m.fit(df)

        idx_temp = regressor_index(m, 'temp')
        idx_rain = regressor_index(m, 'rain')
        assert idx_temp != idx_rain

    def test_regressor_index_unfitted_model(self):
        """Test error when model not fitted."""
        m = TimeWeaver()
        m.add_regressor('temp')
        with pytest.raises(RuntimeError, match='not been fit'):
            regressor_index(m, 'temp')

    def test_regressor_index_invalid_name(self):
        """Test error for non-existent regressor."""
        df = create_daily_data()
        df['temp'] = np.random.randn(len(df))
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.add_regressor('temp')
        m.fit(df)

        with pytest.raises(ValueError, match='not found'):
            regressor_index(m, 'nonexistent')


class TestRegressorCoefficients:
    """Tests for regressor_coefficients function."""

    def test_regressor_coefficients_basic(self):
        """Test getting regressor coefficients."""
        df = create_daily_data()
        df['temp'] = np.random.randn(len(df))
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=50)
        m.add_regressor('temp')
        m.fit(df)

        coefs = regressor_coefficients(m)
        assert isinstance(coefs, pd.DataFrame)
        assert 'regressor' in coefs.columns
        assert 'regressor_mode' in coefs.columns
        assert 'center' in coefs.columns
        assert 'coef_lower' in coefs.columns
        assert 'coef' in coefs.columns
        assert 'coef_upper' in coefs.columns
        assert len(coefs) == 1
        assert coefs.iloc[0]['regressor'] == 'temp'

    def test_regressor_coefficients_multiple(self):
        """Test coefficients with multiple regressors."""
        df = create_daily_data()
        df['temp'] = np.random.randn(len(df))
        df['rain'] = np.random.randn(len(df))
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=50)
        m.add_regressor('temp', mode='additive')
        m.add_regressor('rain', mode='multiplicative')
        m.fit(df)

        coefs = regressor_coefficients(m)
        assert len(coefs) == 2
        assert set(coefs['regressor']) == {'temp', 'rain'}
        assert 'additive' in coefs['regressor_mode'].values
        assert 'multiplicative' in coefs['regressor_mode'].values

    def test_regressor_coefficients_unfitted_model(self):
        """Test error when model not fitted."""
        m = TimeWeaver()
        m.add_regressor('temp')
        with pytest.raises(RuntimeError, match='not been fit'):
            regressor_coefficients(m)

    def test_regressor_coefficients_no_regressors(self):
        """Test error when no regressors present."""
        df = create_daily_data()
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)

        with pytest.raises(ValueError, match='No extra regressors'):
            regressor_coefficients(m)

    def test_regressor_coefficients_interval_bounds(self):
        """Test that coefficient bounds are ordered correctly."""
        df = create_daily_data()
        df['temp'] = np.random.randn(len(df))
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=100)
        m.add_regressor('temp')
        m.fit(df)

        coefs = regressor_coefficients(m)
        row = coefs.iloc[0]
        assert row['coef_lower'] <= row['coef'] <= row['coef_upper']


class TestWarmStartParams:
    """Tests for warm_start_params function."""

    def test_warm_start_params_basic(self):
        """Test extracting warm start parameters."""
        df = create_daily_data()
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=50)
        m.fit(df)

        params = warm_start_params(m)
        assert isinstance(params, dict)
        assert 'k' in params
        assert 'm' in params
        assert 'sigma_obs' in params
        assert 'delta' in params
        assert 'beta' in params

    def test_warm_start_params_types(self):
        """Test that warm start parameters have correct types."""
        df = create_daily_data()
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=50)
        m.fit(df)

        params = warm_start_params(m)
        assert isinstance(params['k'], float)
        assert isinstance(params['m'], float)
        assert isinstance(params['sigma_obs'], float)
        assert isinstance(params['delta'], np.ndarray)
        assert isinstance(params['beta'], np.ndarray)

    def test_warm_start_params_unfitted_model(self):
        """Test error when model not fitted."""
        m = TimeWeaver()
        with pytest.raises(RuntimeError, match='not been fit'):
            warm_start_params(m)

    def test_warm_start_params_logistic(self):
        """Test warm start with logistic growth."""
        df = create_daily_data()
        df['cap'] = 100
        df['floor'] = 0
        m = TimeWeaver(growth='logistic', weekly_seasonality=False,
                       yearly_seasonality=False, uncertainty_samples=50)
        m.fit(df)

        params = warm_start_params(m)
        assert 'k' in params
        assert 'm' in params

    def test_warm_start_params_flat(self):
        """Test warm start with flat growth."""
        df = create_daily_data()
        m = TimeWeaver(growth='flat', weekly_seasonality=False,
                       yearly_seasonality=False, uncertainty_samples=50)
        m.fit(df)

        params = warm_start_params(m)
        assert 'k' in params
        assert 'm' in params


class TestSeasonalityPlotDf:
    """Tests for seasonality_plot_df function."""

    def test_seasonality_plot_df_yearly(self):
        """Test generating yearly seasonality plot data."""
        df = create_daily_data(periods=365 * 2 + 10)
        m = TimeWeaver(weekly_seasonality=False, uncertainty_samples=0)
        m.fit(df)

        plot_df = seasonality_plot_df(m, 'yearly')
        assert isinstance(plot_df, pd.DataFrame)
        assert 'ds' in plot_df.columns
        assert 'yearly' in plot_df.columns
        assert 'yearly_lower' in plot_df.columns
        assert 'yearly_upper' in plot_df.columns

    def test_seasonality_plot_df_weekly(self):
        """Test generating weekly seasonality plot data."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(yearly_seasonality=False, uncertainty_samples=0)
        m.fit(df)

        plot_df = seasonality_plot_df(m, 'weekly')
        assert isinstance(plot_df, pd.DataFrame)
        assert 'weekly' in plot_df.columns

    def test_seasonality_plot_df_custom_dates(self):
        """Test with custom dates."""
        df = create_daily_data(periods=365 * 2 + 10)
        m = TimeWeaver(weekly_seasonality=False, uncertainty_samples=0)
        m.fit(df)

        custom_ds = pd.Series(pd.date_range('2021-01-01', periods=100, freq='D'))
        plot_df = seasonality_plot_df(m, 'yearly', ds=custom_ds)
        assert len(plot_df) == 100

    def test_seasonality_plot_df_unfitted_model(self):
        """Test error when model not fitted."""
        m = TimeWeaver()
        with pytest.raises(RuntimeError, match='not been fit'):
            seasonality_plot_df(m, 'yearly')

    def test_seasonality_plot_df_invalid_name(self):
        """Test error for non-existent seasonality."""
        df = create_daily_data()
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)

        with pytest.raises(ValueError, match='not found'):
            seasonality_plot_df(m, 'yearly')

    def test_seasonality_plot_df_custom_seasonality(self):
        """Test with custom seasonality."""
        df = create_daily_data()
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.add_seasonality('monthly', period=30.5, fourier_order=3)
        m.fit(df)

        plot_df = seasonality_plot_df(m, 'monthly')
        assert 'monthly' in plot_df.columns


class TestGetChangepointDates:
    """Tests for get_changepoint_dates function."""

    def test_get_changepoint_dates_basic(self):
        """Test getting changepoint dates."""
        df = create_daily_data()
        m = TimeWeaver(n_changepoints=10, weekly_seasonality=False,
                       yearly_seasonality=False, uncertainty_samples=0)
        m.fit(df)

        cps = get_changepoint_dates(m)
        assert isinstance(cps, pd.Series)
        assert len(cps) == 10

    def test_get_changepoint_dates_returns_copy(self):
        """Test that returned series is a copy."""
        df = create_daily_data()
        m = TimeWeaver(n_changepoints=5, weekly_seasonality=False,
                       yearly_seasonality=False, uncertainty_samples=0)
        m.fit(df)

        cps1 = get_changepoint_dates(m)
        cps2 = get_changepoint_dates(m)
        assert cps1 is not cps2

    def test_get_changepoint_dates_unfitted_model(self):
        """Test error when model not fitted."""
        m = TimeWeaver()
        with pytest.raises(RuntimeError, match='not been fit'):
            get_changepoint_dates(m)

    def test_get_changepoint_dates_no_changepoints(self):
        """Test with zero changepoints."""
        df = create_daily_data()
        m = TimeWeaver(n_changepoints=0, weekly_seasonality=False,
                       yearly_seasonality=False, uncertainty_samples=0)
        m.fit(df)

        cps = get_changepoint_dates(m)
        assert len(cps) == 0


class TestGetChangepointMagnitudes:
    """Tests for get_changepoint_magnitudes function."""

    def test_get_changepoint_magnitudes_basic(self):
        """Test getting changepoint magnitudes."""
        df = create_daily_data()
        m = TimeWeaver(n_changepoints=10, weekly_seasonality=False,
                       yearly_seasonality=False, uncertainty_samples=50)
        m.fit(df)

        mags = get_changepoint_magnitudes(m)
        assert isinstance(mags, pd.DataFrame)
        assert 'ds' in mags.columns
        assert 'delta' in mags.columns
        assert len(mags) == 10

    def test_get_changepoint_magnitudes_unfitted_model(self):
        """Test error when model not fitted."""
        m = TimeWeaver()
        with pytest.raises(RuntimeError, match='not been fit'):
            get_changepoint_magnitudes(m)

    def test_get_changepoint_magnitudes_no_changepoints(self):
        """Test with zero changepoints."""
        df = create_daily_data()
        m = TimeWeaver(n_changepoints=0, weekly_seasonality=False,
                       yearly_seasonality=False, uncertainty_samples=0)
        m.fit(df)

        mags = get_changepoint_magnitudes(m)
        assert len(mags) == 0
        assert 'ds' in mags.columns
        assert 'delta' in mags.columns

    def test_get_changepoint_magnitudes_values(self):
        """Test that delta values are numeric."""
        df = create_daily_data()
        m = TimeWeaver(n_changepoints=5, weekly_seasonality=False,
                       yearly_seasonality=False, uncertainty_samples=50)
        m.fit(df)

        mags = get_changepoint_magnitudes(m)
        assert mags['delta'].dtype in [np.float64, np.float32]


class TestExports:
    """Test that utility functions are properly exported."""

    def test_imports(self):
        """Test that all utility functions can be imported."""
        from timeweaver import (
            regressor_index,
            regressor_coefficients,
            warm_start_params,
            seasonality_plot_df,
            get_changepoint_dates,
            get_changepoint_magnitudes,
        )
        assert callable(regressor_index)
        assert callable(regressor_coefficients)
        assert callable(warm_start_params)
        assert callable(seasonality_plot_df)
        assert callable(get_changepoint_dates)
        assert callable(get_changepoint_magnitudes)
