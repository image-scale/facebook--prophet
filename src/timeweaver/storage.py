"""Serialization utilities for TimeWeaver models."""

from __future__ import annotations

import json
from collections import OrderedDict
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

from .forecaster import TimeWeaver

SIMPLE_ATTRIBUTES = [
    'growth', 'n_changepoints', 'specified_changepoints', 'changepoint_range',
    'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
    'seasonality_mode', 'seasonality_prior_scale', 'changepoint_prior_scale',
    'holidays_prior_scale', 'interval_width', 'uncertainty_samples',
    'y_scale', 'y_min', 'scaling', 'logistic_floor', 'country_holidays',
    'component_modes', 'holidays_mode',
]

PD_SERIES = ['changepoints', 'history_dates']

PD_TIMESTAMP = ['start']

PD_TIMEDELTA = ['t_scale']

PD_DATAFRAME = ['holidays', 'history', 'train_component_cols']

NP_ARRAY = ['changepoints_t']

ORDEREDDICT = ['seasonalities', 'extra_regressors']


def model_to_dict(model: TimeWeaver) -> dict[str, Any]:
    """Convert a TimeWeaver model to a dictionary.

    Model must be fitted. The dictionary can be serialized to JSON.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.

    Returns
    -------
    dict
        Dictionary representation of the model.
    """
    if model.history is None:
        raise ValueError('Model must be fitted before serialization.')

    model_dict = {
        attr: getattr(model, attr) for attr in SIMPLE_ATTRIBUTES
    }

    for attr in PD_SERIES:
        value = getattr(model, attr)
        if value is None:
            model_dict[attr] = None
        else:
            model_dict[attr] = value.to_json(orient='split', date_format='iso')

    for attr in PD_TIMESTAMP:
        model_dict[attr] = getattr(model, attr).timestamp()

    for attr in PD_TIMEDELTA:
        model_dict[attr] = getattr(model, attr).total_seconds()

    for attr in PD_DATAFRAME:
        value = getattr(model, attr)
        if value is None:
            model_dict[attr] = None
        else:
            model_dict[attr] = value.to_json(orient='table', index=False)

    for attr in NP_ARRAY:
        model_dict[attr] = getattr(model, attr).tolist()

    for attr in ORDEREDDICT:
        od = getattr(model, attr)
        model_dict[attr] = [list(od.keys()), dict(od)]

    model_dict['fit_kwargs'] = model.fit_kwargs
    model_dict['params'] = {k: v.tolist() for k, v in model.params.items()}
    model_dict['train_holiday_names'] = model.train_holiday_names
    model_dict['__timeweaver_version'] = '0.1.0'

    return model_dict


def model_from_dict(model_dict: dict[str, Any]) -> TimeWeaver:
    """Recreate a TimeWeaver model from a dictionary.

    Parameters
    ----------
    model_dict : dict
        Dictionary created with model_to_dict.

    Returns
    -------
    TimeWeaver
        Reconstructed model.
    """
    model = TimeWeaver()

    for attr in SIMPLE_ATTRIBUTES:
        if attr in model_dict:
            setattr(model, attr, model_dict[attr])

    for attr in PD_SERIES:
        if model_dict.get(attr) is None:
            setattr(model, attr, None)
        else:
            s = pd.read_json(StringIO(model_dict[attr]), typ='series', orient='split')
            if s.name == 'ds':
                if len(s) == 0:
                    s = pd.to_datetime(s)
                s = s.dt.tz_localize(None)
            setattr(model, attr, s)

    for attr in PD_TIMESTAMP:
        ts = pd.Timestamp.fromtimestamp(model_dict[attr], tz='UTC').tz_localize(None)
        setattr(model, attr, ts)

    for attr in PD_TIMEDELTA:
        setattr(model, attr, pd.Timedelta(seconds=model_dict[attr]))

    for attr in PD_DATAFRAME:
        if model_dict.get(attr) is None:
            setattr(model, attr, None)
        else:
            df = pd.read_json(
                StringIO(model_dict[attr]),
                typ='frame',
                orient='table',
                convert_dates=['ds'],
            )
            if attr == 'train_component_cols':
                df.columns.name = 'component'
                df.index.name = 'col'
            setattr(model, attr, df)

    for attr in NP_ARRAY:
        setattr(model, attr, np.array(model_dict[attr]))

    for attr in ORDEREDDICT:
        key_list, unordered_dict = model_dict[attr]
        od = OrderedDict()
        for key in key_list:
            od[key] = unordered_dict[key]
        setattr(model, attr, od)

    model.fit_kwargs = model_dict.get('fit_kwargs', {})
    model.params = {k: np.array(v) for k, v in model_dict['params'].items()}
    model.train_holiday_names = model_dict.get('train_holiday_names')

    return model


def model_to_json(model: TimeWeaver) -> str:
    """Serialize a TimeWeaver model to JSON string.

    Model must be fitted.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.

    Returns
    -------
    str
        JSON string representation.
    """
    return json.dumps(model_to_dict(model))


def model_from_json(model_json: str) -> TimeWeaver:
    """Deserialize a TimeWeaver model from JSON string.

    Parameters
    ----------
    model_json : str
        JSON string created with model_to_json.

    Returns
    -------
    TimeWeaver
        Reconstructed model.
    """
    return model_from_dict(json.loads(model_json))


def save_model(model: TimeWeaver, path: str) -> None:
    """Save a TimeWeaver model to a JSON file.

    Parameters
    ----------
    model : TimeWeaver
        Fitted model.
    path : str
        File path to save to.
    """
    with open(path, 'w') as f:
        f.write(model_to_json(model))


def load_model(path: str) -> TimeWeaver:
    """Load a TimeWeaver model from a JSON file.

    Parameters
    ----------
    path : str
        File path to load from.

    Returns
    -------
    TimeWeaver
        Loaded model.
    """
    with open(path, 'r') as f:
        return model_from_json(f.read())
