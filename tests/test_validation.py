"""Tests for TimeWeaver cross-validation and metrics."""

import numpy as np
import pandas as pd
import pytest

from timeweaver import (
    TimeWeaver,
    cross_validation,
    generate_cutoffs,
    performance_metrics,
    mse,
    rmse,
    mae,
    mape,
    smape,
    coverage,
)


def create_daily_data(start: str = "2012-01-01", periods: int = 365 * 2) -> pd.DataFrame:
    """Create synthetic daily time series data."""
    dates = pd.date_range(start=start, periods=periods, freq='D')
    np.random.seed(42)
    trend = np.linspace(10, 50, periods)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    noise = np.random.normal(0, 2, periods)
    y = trend + seasonality + noise
    return pd.DataFrame({'ds': dates, 'y': y})


class TestGenerateCutoffs:
    """Tests for generate_cutoffs function."""

    def test_generate_cutoffs_basic(self):
        """Test basic cutoff generation."""
        df = create_daily_data(periods=365)
        cutoffs = generate_cutoffs(
            df,
            horizon=pd.Timedelta('30 days'),
            initial=pd.Timedelta('180 days'),
            period=pd.Timedelta('30 days'),
        )
        assert len(cutoffs) > 0
        assert all(isinstance(c, pd.Timestamp) for c in cutoffs)

    def test_generate_cutoffs_sorted(self):
        """Test that cutoffs are sorted chronologically."""
        df = create_daily_data(periods=365)
        cutoffs = generate_cutoffs(
            df,
            horizon=pd.Timedelta('30 days'),
            initial=pd.Timedelta('180 days'),
            period=pd.Timedelta('30 days'),
        )
        assert cutoffs == sorted(cutoffs)

    def test_generate_cutoffs_too_short_raises(self):
        """Test that insufficient data raises error."""
        df = create_daily_data(periods=30)
        with pytest.raises(ValueError, match='Less data than horizon'):
            generate_cutoffs(
                df,
                horizon=pd.Timedelta('60 days'),
                initial=pd.Timedelta('30 days'),
                period=pd.Timedelta('7 days'),
            )

    def test_generate_cutoffs_within_range(self):
        """Test cutoffs are within valid range."""
        df = create_daily_data(periods=365)
        horizon = pd.Timedelta('30 days')
        cutoffs = generate_cutoffs(
            df,
            horizon=horizon,
            initial=pd.Timedelta('90 days'),
            period=pd.Timedelta('30 days'),
        )
        min_ds = df['ds'].min()
        max_ds = df['ds'].max()
        for c in cutoffs:
            assert c > min_ds
            assert c <= max_ds - horizon


class TestCrossValidation:
    """Tests for cross_validation function."""

    def test_cross_validation_basic(self):
        """Test basic cross-validation."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        cv_results = cross_validation(m, horizon='30 days', initial='180 days')
        assert 'ds' in cv_results.columns
        assert 'yhat' in cv_results.columns
        assert 'y' in cv_results.columns
        assert 'cutoff' in cv_results.columns

    def test_cross_validation_multiple_cutoffs(self):
        """Test cross-validation has multiple cutoffs."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        cv_results = cross_validation(m, horizon='30 days', initial='90 days',
                                      period='30 days')
        assert cv_results['cutoff'].nunique() >= 2

    def test_cross_validation_with_intervals(self):
        """Test cross-validation includes uncertainty intervals."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=50)
        m.fit(df)
        cv_results = cross_validation(m, horizon='30 days', initial='180 days')
        assert 'yhat_lower' in cv_results.columns
        assert 'yhat_upper' in cv_results.columns

    def test_cross_validation_custom_cutoffs(self):
        """Test cross-validation with custom cutoffs."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        custom_cutoffs = [
            pd.Timestamp('2012-06-01'),
            pd.Timestamp('2012-09-01'),
        ]
        cv_results = cross_validation(m, horizon='30 days', cutoffs=custom_cutoffs)
        assert cv_results['cutoff'].nunique() == 2

    def test_cross_validation_unfitted_raises(self):
        """Test that unfitted model raises error."""
        m = TimeWeaver()
        with pytest.raises(RuntimeError, match='Model has not been fit'):
            cross_validation(m, horizon='30 days')

    def test_cross_validation_invalid_cutoff_raises(self):
        """Test that invalid cutoff raises error."""
        df = create_daily_data(start='2012-01-01', periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        # Cutoff before data starts
        invalid_cutoffs = [pd.Timestamp('2011-01-01')]
        with pytest.raises(ValueError, match='not strictly greater'):
            cross_validation(m, horizon='30 days', cutoffs=invalid_cutoffs)


class TestMetricFunctions:
    """Tests for individual metric functions."""

    def test_mse(self):
        """Test mean squared error."""
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.5, 2.5, 3.5])
        result = mse(y, yhat)
        expected = np.mean((y - yhat) ** 2)
        assert result == pytest.approx(expected)

    def test_rmse(self):
        """Test root mean squared error."""
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.5, 2.5, 3.5])
        result = rmse(y, yhat)
        expected = np.sqrt(np.mean((y - yhat) ** 2))
        assert result == pytest.approx(expected)

    def test_mae(self):
        """Test mean absolute error."""
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.5, 2.5, 3.5])
        result = mae(y, yhat)
        expected = np.mean(np.abs(y - yhat))
        assert result == pytest.approx(expected)

    def test_mape(self):
        """Test mean absolute percentage error."""
        y = np.array([10.0, 20.0, 30.0])
        yhat = np.array([11.0, 22.0, 27.0])
        result = mape(y, yhat)
        expected = np.mean(np.abs((y - yhat) / y))
        assert result == pytest.approx(expected)

    def test_smape(self):
        """Test symmetric mean absolute percentage error."""
        y = np.array([10.0, 20.0, 30.0])
        yhat = np.array([11.0, 22.0, 27.0])
        result = smape(y, yhat)
        expected = np.mean(np.abs(y - yhat) / ((np.abs(y) + np.abs(yhat)) / 2))
        assert result == pytest.approx(expected)

    def test_coverage(self):
        """Test coverage of prediction intervals."""
        y = np.array([10.0, 20.0, 30.0, 40.0])
        yhat_lower = np.array([8.0, 18.0, 32.0, 35.0])
        yhat_upper = np.array([12.0, 22.0, 28.0, 45.0])
        # y[0]=10 in [8,12] yes, y[1]=20 in [18,22] yes
        # y[2]=30 in [32,28] no, y[3]=40 in [35,45] yes
        result = coverage(y, yhat_lower, yhat_upper)
        assert result == pytest.approx(0.75)

    def test_perfect_prediction_metrics(self):
        """Test metrics with perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.0, 2.0, 3.0])
        assert mse(y, yhat) == pytest.approx(0.0)
        assert rmse(y, yhat) == pytest.approx(0.0)
        assert mae(y, yhat) == pytest.approx(0.0)
        assert mape(y, yhat) == pytest.approx(0.0)
        assert smape(y, yhat) == pytest.approx(0.0)


class TestPerformanceMetrics:
    """Tests for performance_metrics function."""

    def test_performance_metrics_basic(self):
        """Test basic performance metrics computation."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        cv_results = cross_validation(m, horizon='30 days', initial='180 days')
        metrics = performance_metrics(cv_results)
        assert 'horizon' in metrics.columns
        assert 'mse' in metrics.columns
        assert 'rmse' in metrics.columns
        assert 'mae' in metrics.columns

    def test_performance_metrics_specific(self):
        """Test requesting specific metrics."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        cv_results = cross_validation(m, horizon='30 days', initial='180 days')
        metrics = performance_metrics(cv_results, metrics=['mae', 'rmse'])
        assert 'mae' in metrics.columns
        assert 'rmse' in metrics.columns
        assert 'mse' not in metrics.columns

    def test_performance_metrics_with_coverage(self):
        """Test performance metrics with coverage."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=50)
        m.fit(df)
        cv_results = cross_validation(m, horizon='30 days', initial='180 days')
        metrics = performance_metrics(cv_results, metrics=['coverage'])
        assert 'coverage' in metrics.columns
        # Coverage should be between 0 and 1
        assert (metrics['coverage'] >= 0).all()
        assert (metrics['coverage'] <= 1).all()

    def test_performance_metrics_no_intervals_no_coverage(self):
        """Test that coverage is skipped without intervals."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        cv_results = cross_validation(m, horizon='30 days', initial='180 days')
        metrics = performance_metrics(cv_results)
        assert 'coverage' not in metrics.columns

    def test_performance_metrics_by_horizon(self):
        """Test metrics are computed by horizon."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        cv_results = cross_validation(m, horizon='30 days', initial='180 days')
        metrics = performance_metrics(cv_results)
        # Each horizon should have its own row
        assert len(metrics) > 0
        assert metrics['horizon'].nunique() == len(metrics)
