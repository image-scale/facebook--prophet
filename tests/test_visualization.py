"""Tests for TimeWeaver visualization."""

import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from timeweaver import (
    TimeWeaver,
    cross_validation,
    plot,
    plot_components,
    plot_cross_validation_metric,
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


class TestPlot:
    """Tests for plot function."""

    def test_plot_basic(self):
        """Test basic forecast plot."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        fcst = m.predict()
        fig = plot(m, fcst)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_uncertainty(self):
        """Test plot with uncertainty intervals."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=50)
        m.fit(df)
        fcst = m.predict()
        fig = plot(m, fcst, uncertainty=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_without_uncertainty(self):
        """Test plot without uncertainty intervals."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=50)
        m.fit(df)
        fcst = m.predict()
        fig = plot(m, fcst, uncertainty=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_custom_ax(self):
        """Test plot with provided axes."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        fcst = m.predict()

        fig, ax = plt.subplots()
        result = plot(m, fcst, ax=ax)
        assert result is fig
        plt.close(fig)

    def test_plot_with_legend(self):
        """Test plot with legend."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        fcst = m.predict()
        fig = plot(m, fcst, include_legend=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_custom_labels(self):
        """Test plot with custom axis labels."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        fcst = m.predict()
        fig = plot(m, fcst, xlabel="Date", ylabel="Value")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Date"
        assert ax.get_ylabel() == "Value"
        plt.close(fig)

    def test_plot_custom_figsize(self):
        """Test plot with custom figure size."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        fcst = m.predict()
        fig = plot(m, fcst, figsize=(12, 8))
        assert fig.get_size_inches()[0] == pytest.approx(12, abs=0.1)
        assert fig.get_size_inches()[1] == pytest.approx(8, abs=0.1)
        plt.close(fig)


class TestPlotComponents:
    """Tests for plot_components function."""

    def test_plot_components_basic(self):
        """Test basic component plot."""
        df = create_daily_data(periods=365 * 2 + 10)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        fcst = m.predict()
        fig = plot_components(m, fcst)
        assert isinstance(fig, plt.Figure)
        # Should have at least trend panel
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_plot_components_with_seasonality(self):
        """Test component plot with seasonality."""
        df = create_daily_data(periods=365 * 2 + 10)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        fcst = m.predict()
        fig = plot_components(m, fcst)
        # Should have trend, weekly, yearly
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_plot_components_with_holidays(self):
        """Test component plot with holidays."""
        df = create_daily_data(periods=365)
        holidays = pd.DataFrame({
            'ds': ['2012-06-15'],
            'holiday': ['special'],
        })
        m = TimeWeaver(holidays=holidays, weekly_seasonality=False,
                       yearly_seasonality=False, uncertainty_samples=0)
        m.fit(df)
        fcst = m.predict()
        fig = plot_components(m, fcst)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_components_with_regressors(self):
        """Test component plot with extra regressors."""
        df = create_daily_data(periods=365)
        df['temp'] = np.random.randn(365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.add_regressor('temp')
        m.fit(df)
        fcst = m.predict()
        fig = plot_components(m, fcst)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_components_custom_figsize(self):
        """Test component plot with custom figure size."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        fcst = m.predict()
        fig = plot_components(m, fcst, figsize=(12, 6))
        assert fig.get_size_inches()[0] == pytest.approx(12, abs=0.1)
        plt.close(fig)


class TestPlotCrossValidationMetric:
    """Tests for plot_cross_validation_metric function."""

    def test_plot_cv_metric_basic(self):
        """Test basic cross-validation metric plot."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        df_cv = cross_validation(m, horizon='30 days', initial='180 days')
        fig = plot_cross_validation_metric(df_cv, metric='mae')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cv_metric_different_metrics(self):
        """Test plotting different metrics."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        df_cv = cross_validation(m, horizon='30 days', initial='180 days')

        for metric in ['mse', 'rmse', 'mae', 'smape']:
            fig = plot_cross_validation_metric(df_cv, metric=metric)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_plot_cv_metric_with_ax(self):
        """Test plotting on provided axes."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.fit(df)
        df_cv = cross_validation(m, horizon='30 days', initial='180 days')

        fig, ax = plt.subplots()
        result = plot_cross_validation_metric(df_cv, metric='mae', ax=ax)
        assert result is fig
        plt.close(fig)
