"""Country holidays utilities for TimeWeaver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
    _HAS_HOLIDAYS = True
except ImportError:
    _HAS_HOLIDAYS = False

if TYPE_CHECKING:
    from .forecaster import TimeWeaver


def _check_holidays_installed() -> None:
    """Check that the holidays library is installed."""
    if not _HAS_HOLIDAYS:
        raise ImportError(
            'The holidays library is required for country holidays. '
            'Install it with: pip install holidays'
        )


def get_country_holidays_class(country: str) -> type:
    """Get the holidays class for a country.

    Parameters
    ----------
    country : str
        Country code (e.g., 'US', 'UK', 'DE').

    Returns
    -------
    type
        Holidays class for the country.
    """
    _check_holidays_installed()

    substitutions = {
        'TU': 'TR',
    }
    country = substitutions.get(country, country)

    if not hasattr(holidays_lib, country):
        raise ValueError(f'Holidays in {country} are not currently supported.')

    return getattr(holidays_lib, country)


def get_holiday_names(country: str) -> set[str]:
    """Get all possible holiday names for a country.

    Parameters
    ----------
    country : str
        Country code.

    Returns
    -------
    set[str]
        Set of holiday names.
    """
    _check_holidays_installed()
    country_class = get_country_holidays_class(country)
    years = np.arange(1995, 2045)
    return set(country_class(language='en_US', years=years).values())


def make_holidays_df(
    year_list: list[int],
    country: str,
    province: str | None = None,
) -> pd.DataFrame:
    """Create a holidays DataFrame for given years and country.

    Parameters
    ----------
    year_list : list[int]
        List of years to include.
    country : str
        Country code.
    province : str or None
        Province/state code for regional holidays.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'ds' and 'holiday' columns.
    """
    _check_holidays_installed()
    country_class = get_country_holidays_class(country)

    if province is not None:
        country_holidays = country_class(
            expand=False,
            language='en_US',
            subdiv=province,
            years=year_list,
        )
    else:
        country_holidays = country_class(
            expand=False,
            language='en_US',
            years=year_list,
        )

    rows = []
    for date in country_holidays:
        holiday_names = country_holidays.get_list(date)
        for name in holiday_names:
            rows.append({'ds': date, 'holiday': name})

    if len(rows) == 0:
        return pd.DataFrame(columns=['ds', 'holiday'])

    holidays_df = pd.DataFrame(rows)
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
    return holidays_df


def add_country_holidays(
    model: TimeWeaver,
    country: str,
    province: str | None = None,
) -> TimeWeaver:
    """Add country holidays to a TimeWeaver model.

    This modifies the model's holidays DataFrame to include holidays
    for the specified country.

    Parameters
    ----------
    model : TimeWeaver
        Model to add holidays to.
    country : str
        Country code (e.g., 'US', 'UK', 'DE').
    province : str or None
        Province/state code for regional holidays.

    Returns
    -------
    TimeWeaver
        The model with holidays added.
    """
    _check_holidays_installed()

    if model.history is not None:
        raise RuntimeError('Country holidays must be added prior to model fitting.')

    model.country_holidays = country

    return model


def _apply_country_holidays(
    model: TimeWeaver,
    df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Apply country holidays based on the data's date range.

    Parameters
    ----------
    model : TimeWeaver
        Model with country_holidays set.
    df : pd.DataFrame
        DataFrame with 'ds' column.

    Returns
    -------
    pd.DataFrame or None
        Combined holidays DataFrame.
    """
    if model.country_holidays is None:
        return model.holidays

    year_min = df['ds'].min().year
    year_max = df['ds'].max().year
    year_list = list(range(year_min - 1, year_max + 2))

    country_holidays_df = make_holidays_df(year_list, model.country_holidays)

    if model.holidays is not None:
        return pd.concat([model.holidays, country_holidays_df], ignore_index=True)
    return country_holidays_df
