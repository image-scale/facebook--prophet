"""Utilities for TimeWeaver models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .forecaster import TimeWeaver


def regressor_index(model: TimeWeaver, name: str) -> int:
    """Get the column index of a regressor in the beta matrix.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.
    name : str
        Name of the regressor.

    Returns
    -------
    int
        Column index of the regressor in the beta matrix.
    """
    if model.train_component_cols is None:
        raise RuntimeError('Model has not been fit.')
    if name not in model.train_component_cols.columns:
        raise ValueError(f'Regressor {name!r} not found in model.')

    cols = model.train_component_cols
    return int(np.extract(cols[name] == 1, cols.index)[0])


def regressor_coefficients(model: TimeWeaver) -> pd.DataFrame:
    """Summarize the coefficients of extra regressors.

    For additive regressors, the coefficient represents the incremental impact
    on y of a unit increase in the regressor. For multiplicative regressors,
    the incremental impact equals trend(t) multiplied by the coefficient.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model with extra regressors.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - regressor: Name of the regressor
        - regressor_mode: 'additive' or 'multiplicative'
        - center: Mean of the regressor if standardized, else 0
        - coef_lower: Lower bound of coefficient estimate
        - coef: Expected coefficient value
        - coef_upper: Upper bound of coefficient estimate
    """
    if model.history is None:
        raise RuntimeError('Model has not been fit.')
    if len(model.extra_regressors) == 0:
        raise ValueError('No extra regressors found.')

    coefs = []
    for name, props in model.extra_regressors.items():
        idx = regressor_index(model, name)
        beta = model.params['beta'][:, idx]

        if props['mode'] == 'additive':
            coef = beta * model.y_scale / props['std']
        else:
            coef = beta / props['std']

        percentiles = [
            (1 - model.interval_width) / 2,
            1 - (1 - model.interval_width) / 2,
        ]
        coef_bounds = np.quantile(coef, q=percentiles)

        record = {
            'regressor': name,
            'regressor_mode': props['mode'],
            'center': props['mu'],
            'coef_lower': coef_bounds[0],
            'coef': float(np.mean(coef)),
            'coef_upper': coef_bounds[1],
        }
        coefs.append(record)

    return pd.DataFrame(coefs)


def warm_start_params(model: TimeWeaver) -> dict[str, Any]:
    """Extract parameters for warm-starting a new model.

    The returned parameters can be used to initialize a new model with
    similar characteristics. Note that the new model should have the
    same number of changepoints and seasonality features.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.

    Returns
    -------
    dict
        Dictionary containing model parameters suitable for warm-starting.
    """
    if model.history is None:
        raise RuntimeError('Model has not been fit.')

    res: dict[str, Any] = {}

    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = float(np.mean(model.params[pname]))

    for pname in ['delta', 'beta']:
        res[pname] = np.mean(model.params[pname], axis=0)

    return res


def seasonality_plot_df(
    model: TimeWeaver,
    name: str,
    ds: pd.Series | None = None,
) -> pd.DataFrame:
    """Create a DataFrame for plotting a seasonality component.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.
    name : str
        Name of the seasonality (e.g., 'yearly', 'weekly').
    ds : pd.Series or None
        Optional dates for plotting. If None, uses a standard range.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'ds' and the seasonality values.
    """
    if model.history is None:
        raise RuntimeError('Model has not been fit.')
    if name not in model.seasonalities:
        raise ValueError(f'Seasonality {name!r} not found in model.')

    props = model.seasonalities[name]
    period = props['period']

    if ds is None:
        start = pd.Timestamp('2017-01-01')
        n_points = int(np.ceil(period * 24))
        ds = pd.date_range(start=start, periods=n_points, freq='h')
        ds = pd.Series(ds)

    features = model.make_seasonality_features(
        ds, period, props['fourier_order'], name
    )

    cols = model.train_component_cols
    if cols is None:
        raise RuntimeError('Model component columns not available.')

    idx = np.where(cols[name] == 1)[0]
    beta = model.params['beta'][:, idx]

    seasonal = np.matmul(features.values, beta.T)
    if props['mode'] == 'additive':
        seasonal *= model.y_scale

    return pd.DataFrame({
        'ds': ds.values,
        name: np.mean(seasonal, axis=1),
        f'{name}_lower': np.percentile(
            seasonal, 100 * (1 - model.interval_width) / 2, axis=1
        ),
        f'{name}_upper': np.percentile(
            seasonal, 100 * (1 + model.interval_width) / 2, axis=1
        ),
    })


def get_changepoint_dates(model: TimeWeaver) -> pd.Series:
    """Get the dates of detected changepoints.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.

    Returns
    -------
    pd.Series
        Series of changepoint dates.
    """
    if model.changepoints is None:
        raise RuntimeError('Model has not been fit.')
    return model.changepoints.copy()


def get_changepoint_magnitudes(model: TimeWeaver) -> pd.DataFrame:
    """Get the magnitude of rate changes at each changepoint.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.

    Returns
    -------
    pd.DataFrame
        DataFrame with changepoint dates and delta values.
    """
    if model.history is None:
        raise RuntimeError('Model has not been fit.')

    deltas = np.mean(model.params['delta'], axis=0)

    if len(model.changepoints) == 0:
        return pd.DataFrame(columns=['ds', 'delta'])

    return pd.DataFrame({
        'ds': model.changepoints.values,
        'delta': deltas,
    })
