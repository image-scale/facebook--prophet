"""Visualization utilities for TimeWeaver forecasts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .forecaster import TimeWeaver

logger = logging.getLogger("timeweaver.visualization")

try:
    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateLocator, AutoDateFormatter
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    logger.warning('Matplotlib not installed. Plotting will not work.')


def _check_matplotlib() -> None:
    """Check that matplotlib is installed."""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            'Matplotlib is required for plotting. '
            'Install it with: pip install matplotlib'
        )


def plot(
    model: TimeWeaver,
    fcst: pd.DataFrame,
    ax: "plt.Axes | None" = None,
    uncertainty: bool = True,
    plot_cap: bool = True,
    xlabel: str = "ds",
    ylabel: str = "y",
    figsize: tuple[int, int] = (10, 6),
    include_legend: bool = False,
) -> "plt.Figure":
    """Plot the TimeWeaver forecast.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.
    fcst : pd.DataFrame
        DataFrame output of model.predict().
    ax : plt.Axes or None
        Optional axes to plot on.
    uncertainty : bool
        Whether to plot uncertainty intervals.
    plot_cap : bool
        Whether to plot capacity for logistic growth.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    figsize : tuple
        Figure size (width, height) in inches.
    include_legend : bool
        Whether to include a legend.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    _check_matplotlib()

    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    fcst_t = fcst['ds']

    if model.history is not None:
        ax.plot(
            model.history['ds'], model.history['y'],
            'k.', label='Observed'
        )

    ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2', label='Forecast')

    if 'cap' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['cap'], ls='--', c='k', label='Capacity')

    if model.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k', label='Floor')

    if uncertainty and model.uncertainty_samples > 0:
        if 'yhat_lower' in fcst and 'yhat_upper' in fcst:
            ax.fill_between(
                fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                color='#0072B2', alpha=0.2, label='Uncertainty'
            )

    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if include_legend:
        ax.legend()

    fig.tight_layout()
    return fig


def plot_components(
    model: TimeWeaver,
    fcst: pd.DataFrame,
    uncertainty: bool = True,
    plot_cap: bool = True,
    figsize: tuple[int, int] | None = None,
) -> "plt.Figure":
    """Plot the forecast components.

    Plots trend, holidays, and seasonality components that are available.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.
    fcst : pd.DataFrame
        DataFrame output of model.predict().
    uncertainty : bool
        Whether to plot uncertainty intervals for trend.
    plot_cap : bool
        Whether to plot capacity for logistic growth.
    figsize : tuple or None
        Figure size (width, height) in inches.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    _check_matplotlib()

    components = ['trend']

    if model.train_holiday_names is not None and 'holidays' in fcst.columns:
        components.append('holidays')

    if 'weekly' in model.seasonalities and 'weekly' in fcst.columns:
        components.append('weekly')

    if 'yearly' in model.seasonalities and 'yearly' in fcst.columns:
        components.append('yearly')

    for name in sorted(model.seasonalities.keys()):
        if name in fcst.columns and name not in ['weekly', 'yearly']:
            components.append(name)

    has_additive = any(
        props['mode'] == 'additive'
        for props in model.extra_regressors.values()
    )
    has_multiplicative = any(
        props['mode'] == 'multiplicative'
        for props in model.extra_regressors.values()
    )

    if has_additive and 'extra_regressors_additive' in fcst.columns:
        components.append('extra_regressors_additive')
    if has_multiplicative and 'extra_regressors_multiplicative' in fcst.columns:
        components.append('extra_regressors_multiplicative')

    npanel = len(components)
    figsize = figsize if figsize else (9, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor='w', figsize=figsize)

    if npanel == 1:
        axes = [axes]

    for ax, name in zip(axes, components):
        _plot_component(model, fcst, name, ax, uncertainty, plot_cap)

    fig.tight_layout()
    return fig


def _plot_component(
    model: TimeWeaver,
    fcst: pd.DataFrame,
    name: str,
    ax: "plt.Axes",
    uncertainty: bool = True,
    plot_cap: bool = True,
) -> None:
    """Plot a single forecast component.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.
    fcst : pd.DataFrame
        Forecast dataframe.
    name : str
        Component name.
    ax : plt.Axes
        Axes to plot on.
    uncertainty : bool
        Whether to plot uncertainty.
    plot_cap : bool
        Whether to plot capacity.
    """
    fcst_t = fcst['ds']
    ax.plot(fcst_t, fcst[name], ls='-', c='#0072B2')

    if name == 'trend':
        if uncertainty and model.uncertainty_samples > 0:
            if 'trend_lower' in fcst and 'trend_upper' in fcst:
                ax.fill_between(
                    fcst_t, fcst['trend_lower'], fcst['trend_upper'],
                    color='#0072B2', alpha=0.2
                )
        if 'cap' in fcst and plot_cap:
            ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
        if model.logistic_floor and 'floor' in fcst and plot_cap:
            ax.plot(fcst_t, fcst['floor'], ls='--', c='k')

    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('ds')
    ax.set_ylabel(name)


def plot_cross_validation_metric(
    df_cv: pd.DataFrame,
    metric: str,
    rolling_window: float = 0.1,
    ax: "plt.Axes | None" = None,
    figsize: tuple[int, int] = (10, 6),
) -> "plt.Figure":
    """Plot a performance metric from cross-validation.

    Parameters
    ----------
    df_cv : pd.DataFrame
        Cross-validation results from cross_validation().
    metric : str
        Metric name (mse, rmse, mae, mape, smape, coverage).
    rolling_window : float
        Rolling window proportion for computing metric.
    ax : plt.Axes or None
        Optional axes to plot on.
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    _check_matplotlib()
    from .validation import performance_metrics

    df_pm = performance_metrics(df_cv, metrics=[metric], rolling_window=rolling_window)

    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    horizons = df_pm['horizon'].dt.total_seconds() / 86400
    ax.plot(horizons, df_pm[metric], 'k-')
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel(metric)

    fig.tight_layout()
    return fig
