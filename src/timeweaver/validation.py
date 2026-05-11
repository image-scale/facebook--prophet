"""Cross-validation and performance metrics for TimeWeaver."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .forecaster import TimeWeaver

logger = logging.getLogger("timeweaver")


def generate_cutoffs(
    df: pd.DataFrame,
    horizon: pd.Timedelta,
    initial: pd.Timedelta,
    period: pd.Timedelta,
) -> list[pd.Timestamp]:
    """Generate cutoff dates for cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with historical data containing 'ds' column.
    horizon : pd.Timedelta
        Forecast horizon.
    initial : pd.Timedelta
        Initial training period.
    period : pd.Timedelta
        Period between cutoff dates.

    Returns
    -------
    list[pd.Timestamp]
        List of cutoff dates.
    """
    cutoff = df['ds'].max() - horizon
    if cutoff < df['ds'].min():
        raise ValueError('Less data than horizon.')

    result = [cutoff]
    while result[-1] >= df['ds'].min() + initial:
        cutoff -= period
        if not (((df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)).any()):
            if cutoff > df['ds'].min():
                closest_date = df[df['ds'] <= cutoff].max()['ds']
                cutoff = closest_date - horizon
        result.append(cutoff)

    result = result[:-1]
    if len(result) == 0:
        raise ValueError(
            'Less data than horizon after initial window. '
            'Make horizon or initial shorter.'
        )

    logger.info(
        f'Making {len(result)} forecasts with cutoffs between '
        f'{result[-1]} and {result[0]}'
    )
    return list(reversed(result))


def _copy_model(model: TimeWeaver) -> TimeWeaver:
    """Create a copy of a TimeWeaver model for refitting.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model to copy.

    Returns
    -------
    TimeWeaver
        New unfitted model with same configuration.
    """
    from .forecaster import TimeWeaver

    m = TimeWeaver(
        growth=model.growth,
        changepoints=None,
        n_changepoints=model.n_changepoints,
        changepoint_range=model.changepoint_range,
        changepoint_prior_scale=model.changepoint_prior_scale,
        yearly_seasonality=model.yearly_seasonality,
        weekly_seasonality=model.weekly_seasonality,
        daily_seasonality=model.daily_seasonality,
        seasonality_mode=model.seasonality_mode,
        seasonality_prior_scale=model.seasonality_prior_scale,
        holidays=model.holidays,
        holidays_prior_scale=model.holidays_prior_scale,
        holidays_mode=model.holidays_mode,
        interval_width=model.interval_width,
        uncertainty_samples=model.uncertainty_samples,
        scaling=model.scaling,
    )

    for name, props in model.seasonalities.items():
        if name not in ['daily', 'weekly', 'yearly']:
            m.add_seasonality(
                name=name,
                period=props['period'],
                fourier_order=props['fourier_order'],
                prior_scale=props['prior_scale'],
                mode=props['mode'],
                condition_name=props['condition_name'],
            )

    for name, props in model.extra_regressors.items():
        m.add_regressor(
            name=name,
            prior_scale=props['prior_scale'],
            standardize=props['standardize'],
            mode=props['mode'],
        )

    return m


def _single_cutoff_forecast(
    model: TimeWeaver,
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: pd.Timedelta,
) -> pd.DataFrame:
    """Make forecast for a single cutoff.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model (will be copied).
    df : pd.DataFrame
        Full historical data.
    cutoff : pd.Timestamp
        Cutoff date.
    horizon : pd.Timedelta
        Forecast horizon.

    Returns
    -------
    pd.DataFrame
        Forecast with 'cutoff' column added.
    """
    m = _copy_model(model)

    train = df[df['ds'] <= cutoff].copy()
    m.fit(train, **model.fit_kwargs)

    index_predicted = (df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)
    future = df[index_predicted][['ds']].copy()

    for name in model.extra_regressors:
        future[name] = df.loc[index_predicted, name]

    if model.growth == 'logistic':
        future['cap'] = df.loc[index_predicted, 'cap']
        if model.logistic_floor:
            future['floor'] = df.loc[index_predicted, 'floor']

    for props in model.seasonalities.values():
        if props['condition_name'] is not None:
            future[props['condition_name']] = df.loc[
                index_predicted, props['condition_name']
            ]

    forecast = m.predict(future)
    forecast['y'] = df.loc[index_predicted, 'y'].values
    forecast['cutoff'] = cutoff
    return forecast


def cross_validation(
    model: TimeWeaver,
    horizon: str | pd.Timedelta,
    period: str | pd.Timedelta | None = None,
    initial: str | pd.Timedelta | None = None,
    cutoffs: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Cross-validation for time series.

    Makes forecasts from historical cutoff points and compares them to
    actual values.

    Parameters
    ----------
    model : TimeWeaver
        Fitted TimeWeaver model.
    horizon : str or pd.Timedelta
        Forecast horizon, e.g., '30 days'.
    period : str or pd.Timedelta or None
        Period between cutoffs. Defaults to 0.5 * horizon.
    initial : str or pd.Timedelta or None
        Initial training period. Defaults to 3 * horizon.
    cutoffs : list[pd.Timestamp] or None
        Custom cutoff dates. If None, automatically generated.

    Returns
    -------
    pd.DataFrame
        DataFrame with forecasts, actual values, and cutoffs.
    """
    if model.history is None:
        raise RuntimeError('Model has not been fit.')

    df = model.history.copy().reset_index(drop=True)
    horizon = pd.Timedelta(horizon)

    if cutoffs is None:
        if period is None:
            period = 0.5 * horizon
        else:
            period = pd.Timedelta(period)

        period_max = 0.0
        for s in model.seasonalities.values():
            period_max = max(period_max, s['period'])
        seasonality_dt = pd.Timedelta(f'{period_max} days')

        if initial is None:
            initial = max(3 * horizon, seasonality_dt)
        else:
            initial = pd.Timedelta(initial)

        cutoffs = generate_cutoffs(df, horizon, initial, period)
    else:
        if min(cutoffs) <= df['ds'].min():
            raise ValueError(
                'Minimum cutoff value is not strictly greater than min date in history'
            )
        end_date_minus_horizon = df['ds'].max() - horizon
        if max(cutoffs) > end_date_minus_horizon:
            raise ValueError(
                'Maximum cutoff value is greater than end date minus horizon'
            )

    results = []
    for cutoff in cutoffs:
        forecast = _single_cutoff_forecast(model, df, cutoff, horizon)
        results.append(forecast)

    result_df = pd.concat(results, ignore_index=True)

    columns = ['ds', 'yhat', 'y', 'cutoff']
    if model.uncertainty_samples > 0:
        columns.extend(['yhat_lower', 'yhat_upper'])

    available_columns = [c for c in columns if c in result_df.columns]
    return result_df[available_columns]


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean squared error.

    Parameters
    ----------
    y : np.ndarray
        Actual values.
    yhat : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return float(np.mean((y - yhat) ** 2))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error.

    Parameters
    ----------
    y : np.ndarray
        Actual values.
    yhat : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(np.sqrt(mse(y, yhat)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error.

    Parameters
    ----------
    y : np.ndarray
        Actual values.
    yhat : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    return float(np.mean(np.abs(y - yhat)))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute percentage error.

    Parameters
    ----------
    y : np.ndarray
        Actual values.
    yhat : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean absolute percentage error.
    """
    return float(np.mean(np.abs((y - yhat) / y)))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric mean absolute percentage error.

    Parameters
    ----------
    y : np.ndarray
        Actual values.
    yhat : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Symmetric mean absolute percentage error.
    """
    return float(np.mean(np.abs(y - yhat) / ((np.abs(y) + np.abs(yhat)) / 2)))


def coverage(
    y: np.ndarray, yhat_lower: np.ndarray, yhat_upper: np.ndarray
) -> float:
    """Coverage of prediction intervals.

    Parameters
    ----------
    y : np.ndarray
        Actual values.
    yhat_lower : np.ndarray
        Lower bound of prediction interval.
    yhat_upper : np.ndarray
        Upper bound of prediction interval.

    Returns
    -------
    float
        Proportion of actuals within prediction intervals.
    """
    return float(np.mean((y >= yhat_lower) & (y <= yhat_upper)))


def performance_metrics(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    rolling_window: float = 0.1,
) -> pd.DataFrame:
    """Compute performance metrics from cross-validation results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by cross_validation().
    metrics : list[str] or None
        List of metrics to compute. Defaults to all available.
    rolling_window : float
        Proportion of data for rolling window. 0 computes per horizon,
        1 computes overall.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics by horizon.
    """
    valid_metrics = ['mse', 'rmse', 'mae', 'mape', 'smape', 'coverage']

    if metrics is None:
        metrics = valid_metrics.copy()

    has_intervals = 'yhat_lower' in df and 'yhat_upper' in df
    if 'coverage' in metrics and not has_intervals:
        metrics.remove('coverage')

    if df['y'].abs().min() < 1e-8 and 'mape' in metrics:
        logger.info('Skipping MAPE because y close to 0')
        metrics.remove('mape')

    if len(metrics) == 0:
        return pd.DataFrame()

    df_m = df.copy()
    df_m['horizon'] = df_m['ds'] - df_m['cutoff']
    df_m = df_m.sort_values('horizon')

    horizons = df_m['horizon'].unique()

    results = []
    for h in horizons:
        mask = df_m['horizon'] == h
        y = df_m.loc[mask, 'y'].values
        yhat = df_m.loc[mask, 'yhat'].values

        row = {'horizon': h}
        if 'mse' in metrics:
            row['mse'] = mse(y, yhat)
        if 'rmse' in metrics:
            row['rmse'] = rmse(y, yhat)
        if 'mae' in metrics:
            row['mae'] = mae(y, yhat)
        if 'mape' in metrics:
            row['mape'] = mape(y, yhat)
        if 'smape' in metrics:
            row['smape'] = smape(y, yhat)
        if 'coverage' in metrics:
            row['coverage'] = coverage(
                y,
                df_m.loc[mask, 'yhat_lower'].values,
                df_m.loc[mask, 'yhat_upper'].values,
            )
        results.append(row)

    return pd.DataFrame(results)
