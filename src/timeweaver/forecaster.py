"""TimeWeaver - Time series forecasting with additive models."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger("timeweaver")
logger.setLevel(logging.INFO)


class TimeWeaver:
    """Time series forecaster with additive/multiplicative components.

    Fits an additive model where trend is combined with seasonality and holidays.
    Supports linear, logistic, and flat growth trends with automatic or manual
    changepoint detection.

    Parameters
    ----------
    growth : str
        Type of growth trend: 'linear', 'logistic', or 'flat'.
    changepoints : list or None
        List of dates at which to include potential changepoints.
    n_changepoints : int
        Number of potential changepoints to include (if changepoints is None).
    changepoint_range : float
        Proportion of history to consider for changepoints (0 to 1).
    changepoint_prior_scale : float
        Flexibility of automatic changepoint selection.
    yearly_seasonality : str, bool, or int
        Fit yearly seasonality. 'auto', True, False, or Fourier order.
    weekly_seasonality : str, bool, or int
        Fit weekly seasonality. 'auto', True, False, or Fourier order.
    daily_seasonality : str, bool, or int
        Fit daily seasonality. 'auto', True, False, or Fourier order.
    seasonality_mode : str
        'additive' or 'multiplicative'.
    seasonality_prior_scale : float
        Strength of seasonality model.
    holidays : pd.DataFrame or None
        DataFrame with columns 'holiday' and 'ds', optionally 'lower_window',
        'upper_window', and 'prior_scale'.
    holidays_prior_scale : float
        Strength of holiday components.
    holidays_mode : str or None
        'additive' or 'multiplicative'. Defaults to seasonality_mode.
    interval_width : float
        Width of uncertainty intervals (0 to 1).
    uncertainty_samples : int
        Number of samples for uncertainty estimation.
    scaling : str
        'absmax' or 'minmax' for y scaling.
    """

    def __init__(
        self,
        growth: Literal["linear", "logistic", "flat"] = "linear",
        changepoints: list | pd.Series | None = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        changepoint_prior_scale: float = 0.05,
        yearly_seasonality: Literal["auto"] | bool | int = "auto",
        weekly_seasonality: Literal["auto"] | bool | int = "auto",
        daily_seasonality: Literal["auto"] | bool | int = "auto",
        seasonality_mode: Literal["additive", "multiplicative"] = "additive",
        seasonality_prior_scale: float = 10.0,
        holidays: pd.DataFrame | None = None,
        holidays_prior_scale: float = 10.0,
        holidays_mode: Literal["additive", "multiplicative"] | None = None,
        interval_width: float = 0.80,
        uncertainty_samples: int = 1000,
        scaling: Literal["absmax", "minmax"] = "absmax",
    ) -> None:
        self.growth = growth
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.holidays = holidays
        self.holidays_prior_scale = float(holidays_prior_scale)
        self.holidays_mode = holidays_mode or seasonality_mode
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.scaling = scaling

        if changepoints is not None:
            self.changepoints: pd.Series | None = pd.Series(
                pd.to_datetime(changepoints), name="ds"
            )
            self.n_changepoints = len(self.changepoints)
            self.specified_changepoints = True
        else:
            self.changepoints = None
            self.specified_changepoints = False

        self.start: pd.Timestamp | None = None
        self.y_min: float | None = None
        self.y_scale: float | None = None
        self.t_scale: pd.Timedelta | None = None
        self.logistic_floor: bool = False
        self.changepoints_t: npt.NDArray[np.float64] | None = None
        self.history: pd.DataFrame | None = None
        self.history_dates: pd.Series | None = None
        self.params: dict[str, Any] = {}
        self.seasonalities: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.extra_regressors: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.country_holidays: str | None = None
        self.train_component_cols: pd.DataFrame | None = None
        self.component_modes: dict[str, list[str]] | None = None
        self.train_holiday_names: list[str] | None = None
        self.fit_kwargs: dict[str, Any] = {}

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate initialization parameters."""
        if self.growth not in ("linear", "logistic", "flat"):
            raise ValueError(
                'Parameter "growth" should be "linear", "logistic" or "flat".'
            )
        if not isinstance(self.changepoint_range, (int, float)):
            raise ValueError("changepoint_range must be a number in [0, 1]")
        if self.changepoint_range < 0 or self.changepoint_range > 1:
            raise ValueError('Parameter "changepoint_range" must be in [0, 1]')
        if self.seasonality_mode not in ("additive", "multiplicative"):
            raise ValueError(
                'seasonality_mode must be "additive" or "multiplicative"'
            )
        if self.holidays_mode not in ("additive", "multiplicative"):
            raise ValueError(
                'holidays_mode must be "additive" or "multiplicative"'
            )
        if self.scaling not in ("absmax", "minmax"):
            raise ValueError("scaling must be one of 'absmax' or 'minmax'")
        if self.holidays is not None:
            self._validate_holidays(self.holidays)

    def _validate_holidays(self, holidays: pd.DataFrame) -> None:
        """Validate holidays DataFrame."""
        if not isinstance(holidays, pd.DataFrame):
            raise ValueError('holidays must be a DataFrame')
        if 'ds' not in holidays or 'holiday' not in holidays:
            raise ValueError('holidays must have "ds" and "holiday" columns.')
        holidays['ds'] = pd.to_datetime(holidays['ds'])
        if holidays['ds'].isnull().any() or holidays['holiday'].isnull().any():
            raise ValueError('Found a NaN in holidays dataframe.')
        has_lower = 'lower_window' in holidays
        has_upper = 'upper_window' in holidays
        if has_lower + has_upper == 1:
            raise ValueError(
                'Holidays must have both lower_window and upper_window, or neither'
            )
        if has_lower:
            if holidays['lower_window'].max() > 0:
                raise ValueError('Holiday lower_window should be <= 0')
            if holidays['upper_window'].min() < 0:
                raise ValueError('Holiday upper_window should be >= 0')

    def prepare_dataframe(
        self, df: pd.DataFrame, initialize_scales: bool = False
    ) -> pd.DataFrame:
        """Prepare dataframe for fitting or predicting.

        Adds time index 't' and scales y. Creates auxiliary columns 't',
        'y_scaled', 'floor', and 'cap_scaled'.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns 'ds', 'y', and 'cap' if logistic growth.
        initialize_scales : bool
            Whether to set scaling factors from this dataframe.

        Returns
        -------
        pd.DataFrame
            Prepared dataframe with additional columns.
        """
        if 'y' in df:
            df['y'] = pd.to_numeric(df['y'])
            if np.isinf(df['y'].values).any():
                raise ValueError('Found infinity in column y.')
        if df['ds'].dtype == np.int64:
            df['ds'] = df['ds'].astype(str)
        df['ds'] = pd.to_datetime(df['ds'])
        if df['ds'].dt.tz is not None:
            raise ValueError(
                'Column ds has timezone specified, which is not supported. '
                'Remove timezone.'
            )
        if df['ds'].isnull().any():
            raise ValueError('Found NaN in column ds.')

        for name in self.extra_regressors:
            if name not in df:
                raise ValueError(f'Regressor {name!r} missing from dataframe')
            df[name] = pd.to_numeric(df[name])
            if df[name].isnull().any():
                raise ValueError(f'Found NaN in column {name!r}')

        for props in self.seasonalities.values():
            condition_name = props['condition_name']
            if condition_name is not None:
                if condition_name not in df:
                    raise ValueError(
                        f'Condition {condition_name!r} missing from dataframe'
                    )
                if not df[condition_name].isin([True, False, 0, 1]).all():
                    raise ValueError(
                        f'Found non-boolean in column {condition_name!r}'
                    )
                df[condition_name] = df[condition_name].astype('bool')

        if df.index.name == 'ds':
            df.index.name = None
        df = df.sort_values('ds', kind='mergesort')
        df = df.reset_index(drop=True)

        self._initialize_scales(initialize_scales, df)

        if self.logistic_floor:
            if 'floor' not in df:
                raise ValueError('Expected column "floor".')
        else:
            if self.scaling == "absmax":
                df['floor'] = 0.0
            elif self.scaling == "minmax":
                df['floor'] = self.y_min

        if self.growth == 'logistic':
            if 'cap' not in df:
                raise ValueError(
                    'Capacities must be supplied for logistic growth in column "cap"'
                )
            if (df['cap'] <= df['floor']).any():
                raise ValueError(
                    'cap must be greater than floor (which defaults to 0).'
                )
            df['cap_scaled'] = (df['cap'] - df['floor']) / self.y_scale

        df['t'] = (df['ds'] - self.start) / self.t_scale
        if 'y' in df:
            df['y_scaled'] = (df['y'] - df['floor']) / self.y_scale

        for name, props in self.extra_regressors.items():
            df[name] = (df[name] - props['mu']) / props['std']

        return df

    def _initialize_scales(
        self, initialize_scales: bool, df: pd.DataFrame
    ) -> None:
        """Initialize model scales from dataframe."""
        if not initialize_scales:
            return

        if self.growth == 'logistic' and 'floor' in df:
            self.logistic_floor = True
            if self.scaling == "absmax":
                self.y_min = float((df['y'] - df['floor']).abs().min())
                self.y_scale = float((df['y'] - df['floor']).abs().max())
            elif self.scaling == "minmax":
                self.y_min = float(df['floor'].min())
                self.y_scale = float(df['cap'].max() - self.y_min)
        else:
            if self.scaling == "absmax":
                self.y_min = 0.0
                self.y_scale = float(df['y'].abs().max())
            elif self.scaling == "minmax":
                self.y_min = float(df['y'].min())
                self.y_scale = float(df['y'].max() - self.y_min)

        if self.y_scale == 0:
            self.y_scale = 1.0

        self.start = df['ds'].min()
        self.t_scale = df['ds'].max() - self.start

        for name, props in self.extra_regressors.items():
            standardize = props['standardize']
            n_vals = len(df[name].unique())
            if n_vals < 2:
                standardize = False
            if standardize == 'auto':
                if set(df[name].unique()) == {1, 0}:
                    standardize = False
                else:
                    standardize = True
            if standardize:
                mu = float(df[name].mean())
                std = float(df[name].std())
                self.extra_regressors[name]['mu'] = mu
                self.extra_regressors[name]['std'] = std

    def set_changepoints(self) -> None:
        """Set changepoints for trend.

        Either uses specified changepoints or generates them evenly through
        the first changepoint_range proportion of the history.
        """
        if self.changepoints is not None:
            if len(self.changepoints) > 0:
                history = self.history
                assert history is not None
                too_low = self.changepoints.min() < history['ds'].min()
                too_high = self.changepoints.max() > history['ds'].max()
                if too_low or too_high:
                    raise ValueError(
                        'Changepoints must fall within training data.'
                    )
        else:
            history = self.history
            assert history is not None
            hist_size = int(np.floor(history.shape[0] * self.changepoint_range))
            if self.n_changepoints + 1 > hist_size:
                self.n_changepoints = hist_size - 1
                logger.info(
                    f'n_changepoints greater than number of observations. '
                    f'Using {self.n_changepoints}.'
                )
            if self.n_changepoints > 0:
                cp_indexes = (
                    np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                    .round()
                    .astype(int)
                )
                self.changepoints = history.iloc[cp_indexes]['ds'].tail(-1)
            else:
                self.changepoints = pd.Series(pd.to_datetime([]), name='ds')

        if len(self.changepoints) > 0:
            self.changepoints_t = np.sort(
                np.array((self.changepoints - self.start) / self.t_scale)
            )
        else:
            self.changepoints_t = np.array([0])  # dummy changepoint

    @staticmethod
    def linear_growth_init(df: pd.DataFrame) -> tuple[float, float]:
        """Initialize linear growth parameters.

        Computes slope and offset by fitting a line through the first
        and last points.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with columns 'ds', 'y_scaled', and 't'.

        Returns
        -------
        tuple
            (k, m) where k is the slope and m is the offset.
        """
        i0, i1 = int(df['ds'].idxmin()), int(df['ds'].idxmax())
        T = df['t'].iloc[i1] - df['t'].iloc[i0]
        k = (df['y_scaled'].iloc[i1] - df['y_scaled'].iloc[i0]) / T
        m = df['y_scaled'].iloc[i0] - k * df['t'].iloc[i0]
        return (k, m)

    @staticmethod
    def logistic_growth_init(df: pd.DataFrame) -> tuple[float, float]:
        """Initialize logistic growth parameters.

        Computes rate and offset for logistic growth through the first
        and last points.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with columns 'ds', 'cap_scaled', 'y_scaled', and 't'.

        Returns
        -------
        tuple
            (k, m) where k is the rate and m is the offset.
        """
        i0, i1 = int(df['ds'].idxmin()), int(df['ds'].idxmax())
        T = df['t'].iloc[i1] - df['t'].iloc[i0]

        C0 = df['cap_scaled'].iloc[i0]
        C1 = df['cap_scaled'].iloc[i1]
        y0 = max(0.01 * C0, min(0.99 * C0, df['y_scaled'].iloc[i0]))
        y1 = max(0.01 * C1, min(0.99 * C1, df['y_scaled'].iloc[i1]))

        r0 = C0 / y0
        r1 = C1 / y1

        if abs(r0 - r1) <= 0.01:
            r0 = 1.05 * r0

        L0 = np.log(r0 - 1)
        L1 = np.log(r1 - 1)

        m = L0 * T / (L0 - L1)
        k = (L0 - L1) / T
        return (k, m)

    @staticmethod
    def flat_growth_init(df: pd.DataFrame) -> tuple[float, float]:
        """Initialize flat growth parameters.

        Sets slope to 0 and offset to mean of y_scaled.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with columns 'ds', 'y_scaled', and 't'.

        Returns
        -------
        tuple
            (k, m) where k=0 and m is the mean.
        """
        k = 0.0
        m = float(df['y_scaled'].mean())
        return k, m

    @staticmethod
    def piecewise_linear(
        t: np.ndarray,
        deltas: np.ndarray,
        k: float,
        m: float,
        changepoint_ts: np.ndarray,
    ) -> np.ndarray:
        """Evaluate piecewise linear function.

        Parameters
        ----------
        t : np.ndarray
            Times at which to evaluate.
        deltas : np.ndarray
            Rate changes at each changepoint.
        k : float
            Initial rate.
        m : float
            Initial offset.
        changepoint_ts : np.ndarray
            Changepoint times.

        Returns
        -------
        np.ndarray
            Trend values y(t).
        """
        deltas_t = (changepoint_ts[None, :] <= t[..., None]) * deltas
        k_t = deltas_t.sum(axis=1) + k
        m_t = (deltas_t * -changepoint_ts).sum(axis=1) + m
        return k_t * t + m_t

    @staticmethod
    def piecewise_logistic(
        t: np.ndarray,
        cap: np.ndarray | pd.Series,
        deltas: np.ndarray,
        k: float,
        m: float,
        changepoint_ts: np.ndarray,
    ) -> np.ndarray:
        """Evaluate piecewise logistic function.

        Parameters
        ----------
        t : np.ndarray
            Times at which to evaluate.
        cap : np.ndarray or pd.Series
            Capacities at each t.
        deltas : np.ndarray
            Rate changes at each changepoint.
        k : float
            Initial rate.
        m : float
            Initial offset.
        changepoint_ts : np.ndarray
            Changepoint times.

        Returns
        -------
        np.ndarray
            Trend values y(t).
        """
        k_scalar = float(np.asarray(k).item()) if np.asarray(k).size == 1 else k
        m_scalar = float(np.asarray(m).item()) if np.asarray(m).size == 1 else m

        k_cum = np.concatenate(
            (np.atleast_1d(k_scalar), np.cumsum(deltas) + k_scalar)
        )
        gammas = np.zeros(len(changepoint_ts))
        for i, t_s in enumerate(changepoint_ts):
            gammas[i] = (t_s - m_scalar - np.sum(gammas)) * (
                1 - k_cum[i] / k_cum[i + 1]
            )

        k_t = k_scalar * np.ones_like(t)
        m_t = m_scalar * np.ones_like(t)
        for s, t_s in enumerate(changepoint_ts):
            indx = t >= t_s
            k_t[indx] += deltas[s]
            m_t[indx] += gammas[s]

        return cap / (1 + np.exp(-k_t * (t - m_t)))

    @staticmethod
    def flat_trend(t: np.ndarray, m: float) -> np.ndarray:
        """Evaluate flat trend function.

        Parameters
        ----------
        t : np.ndarray
            Times at which to evaluate.
        m : float
            Constant offset.

        Returns
        -------
        np.ndarray
            Constant trend values.
        """
        return m * np.ones_like(t)

    @staticmethod
    def fourier_series(
        dates: pd.Series,
        period: float,
        series_order: int,
    ) -> np.ndarray:
        """Generate Fourier series components for seasonality.

        Parameters
        ----------
        dates : pd.Series
            Series of timestamps.
        period : float
            Number of days in the period.
        series_order : int
            Number of Fourier components.

        Returns
        -------
        np.ndarray
            Matrix with shape (len(dates), 2 * series_order) containing
            sin and cos terms.
        """
        if series_order < 1:
            raise ValueError("series_order must be >= 1")

        epoch = pd.Timestamp("1970-01-01", tz=dates.dt.tz)
        t = (dates - epoch).dt.total_seconds() / (24 * 60 * 60)

        x_T = np.pi * 2 * t
        fourier_components = np.empty((dates.shape[0], 2 * series_order))
        for i in range(series_order):
            c = (i + 1) / period * x_T
            fourier_components[:, 2 * i] = np.sin(c)
            fourier_components[:, (2 * i) + 1] = np.cos(c)
        return fourier_components

    @classmethod
    def make_seasonality_features(
        cls,
        dates: pd.Series,
        period: float,
        series_order: int,
        prefix: str,
    ) -> pd.DataFrame:
        """Create a dataframe of seasonality features.

        Parameters
        ----------
        dates : pd.Series
            Series of timestamps.
        period : float
            Number of days in the period.
        series_order : int
            Number of Fourier components.
        prefix : str
            Column name prefix.

        Returns
        -------
        pd.DataFrame
            Dataframe with seasonality feature columns.
        """
        features = cls.fourier_series(dates, period, series_order)
        columns = [
            f'{prefix}_delim_{i + 1}'
            for i in range(features.shape[1])
        ]
        return pd.DataFrame(features, columns=columns)

    def _construct_holiday_dataframe(
        self, dates: pd.Series
    ) -> tuple[pd.DataFrame | None, list[float], list[str]]:
        """Construct expanded holiday dataframe with windows.

        Parameters
        ----------
        dates : pd.Series
            Series of timestamps.

        Returns
        -------
        tuple
            (holidays_df, prior_scales, holiday_names)
            holidays_df has columns for each holiday+offset combination.
        """
        if self.holidays is None:
            return None, [], []

        holidays = self.holidays.copy()
        holidays['ds'] = pd.to_datetime(holidays['ds'])

        all_holidays = []
        prior_scales: list[float] = []
        holiday_names: list[str] = []

        for _, row in holidays.iterrows():
            dt = row['ds']
            holiday_name = row['holiday']
            lower_window = int(row.get('lower_window', 0))
            upper_window = int(row.get('upper_window', 0))
            prior_scale = row.get('prior_scale', self.holidays_prior_scale)

            for offset in range(lower_window, upper_window + 1):
                offset_date = dt + pd.Timedelta(days=offset)
                col_name = f'{holiday_name}_delim_{offset}'
                all_holidays.append({
                    'ds': offset_date,
                    'holiday': holiday_name,
                    'col_name': col_name,
                    'prior_scale': float(prior_scale),
                })
                if col_name not in holiday_names:
                    holiday_names.append(col_name)
                    prior_scales.append(float(prior_scale))

        if len(all_holidays) == 0:
            return None, [], []

        holidays_df = pd.DataFrame(all_holidays)
        return holidays_df, prior_scales, holiday_names

    def _make_holiday_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[float], list[str]]:
        """Create one-hot encoded holiday features.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with dates.

        Returns
        -------
        tuple
            (holiday_features, prior_scales, holiday_names)
        """
        expanded, prior_scales, holiday_names = self._construct_holiday_dataframe(
            df['ds']
        )

        if expanded is None or len(holiday_names) == 0:
            return pd.DataFrame({'zeros': np.zeros(df.shape[0])}), [1.0], []

        n_rows = df.shape[0]
        dates = df['ds'].values

        if self.train_holiday_names is not None:
            holiday_names = self.train_holiday_names
            prior_scales = []
            for col_name in holiday_names:
                matching = expanded[expanded['col_name'] == col_name]
                if len(matching) > 0:
                    prior_scales.append(float(matching['prior_scale'].iloc[0]))
                else:
                    prior_scales.append(self.holidays_prior_scale)

        feature_data = {}
        for col_name in holiday_names:
            feature_data[col_name] = np.zeros(n_rows)

        for _, row in expanded.iterrows():
            col_name = row['col_name']
            if col_name in feature_data:
                match_idx = np.where(dates == row['ds'])[0]
                for idx in match_idx:
                    feature_data[col_name][idx] = 1.0

        holiday_features = pd.DataFrame(feature_data)
        return holiday_features, prior_scales, holiday_names

    def _validate_column_name(
        self,
        name: str,
        check_holidays: bool = True,
        check_seasonalities: bool = True,
        check_regressors: bool = True,
    ) -> None:
        """Validate name for seasonality, holiday, or regressor.

        Parameters
        ----------
        name : str
            Name to validate.
        check_holidays : bool
            Check if name conflicts with holidays.
        check_seasonalities : bool
            Check if name conflicts with seasonalities.
        check_regressors : bool
            Check if name conflicts with regressors.
        """
        if '_delim_' in name:
            raise ValueError('Name cannot contain "_delim_"')
        reserved_names = [
            'trend', 'additive_terms', 'daily', 'weekly', 'yearly',
            'holidays', 'zeros', 'extra_regressors_additive', 'yhat',
            'extra_regressors_multiplicative', 'multiplicative_terms',
        ]
        rn_l = [n + '_lower' for n in reserved_names]
        rn_u = [n + '_upper' for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        reserved_names.extend(['ds', 'y', 'cap', 'floor', 'y_scaled', 'cap_scaled'])
        if name in reserved_names:
            raise ValueError(f'Name {name!r} is reserved.')
        if check_holidays and self.holidays is not None:
            if name in self.holidays['holiday'].unique():
                raise ValueError(f'Name {name!r} already used for a holiday.')
        if check_seasonalities and name in self.seasonalities:
            raise ValueError(f'Name {name!r} already used for a seasonality.')
        if check_regressors and name in self.extra_regressors:
            raise ValueError(f'Name {name!r} already used for an added regressor.')

    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: float | None = None,
        mode: Literal["additive", "multiplicative"] | None = None,
        condition_name: str | None = None,
    ) -> "TimeWeaver":
        """Add a custom seasonal component.

        Parameters
        ----------
        name : str
            Name of the seasonality component.
        period : float
            Number of days in one period.
        fourier_order : int
            Number of Fourier components.
        prior_scale : float or None
            Prior scale for this component. Defaults to seasonality_prior_scale.
        mode : str or None
            'additive' or 'multiplicative'. Defaults to seasonality_mode.
        condition_name : str or None
            Column name for conditional seasonality.

        Returns
        -------
        TimeWeaver
            The model instance for chaining.
        """
        if self.history is not None:
            raise RuntimeError('Seasonality must be added prior to model fitting.')
        if name not in ['daily', 'weekly', 'yearly']:
            self._validate_column_name(name, check_seasonalities=False)
        if prior_scale is None:
            ps = self.seasonality_prior_scale
        else:
            ps = float(prior_scale)
        if ps <= 0:
            raise ValueError('Prior scale must be > 0')
        if fourier_order <= 0:
            raise ValueError('Fourier order must be > 0')
        if mode is None:
            mode = self.seasonality_mode
        if mode not in ('additive', 'multiplicative'):
            raise ValueError('mode must be "additive" or "multiplicative"')
        if condition_name is not None:
            self._validate_column_name(condition_name)
        self.seasonalities[name] = {
            'period': period,
            'fourier_order': fourier_order,
            'prior_scale': ps,
            'mode': mode,
            'condition_name': condition_name,
        }
        return self

    def _parse_seasonality_args(
        self,
        name: str,
        arg: Literal["auto"] | bool | int,
        auto_disable: bool,
        default_order: int,
    ) -> int:
        """Get number of Fourier components for built-in seasonalities.

        Parameters
        ----------
        name : str
            Name of the seasonality component.
        arg : 'auto', True, False, or int
            User-specified value.
        auto_disable : bool
            Whether seasonality should be disabled when 'auto'.
        default_order : int
            Default Fourier order.

        Returns
        -------
        int
            Number of Fourier components, or 0 if disabled.
        """
        if arg == 'auto':
            fourier_order = 0
            if name in self.seasonalities:
                logger.info(
                    f'Found custom seasonality named {name!r}, disabling '
                    f'built-in {name!r} seasonality.'
                )
            elif auto_disable:
                logger.info(
                    f'Disabling {name} seasonality. Run with '
                    f'{name}_seasonality=True to override this.'
                )
            else:
                fourier_order = default_order
        elif arg is True:
            fourier_order = default_order
        elif arg is False:
            fourier_order = 0
        else:
            fourier_order = int(arg)
        return fourier_order

    def _set_auto_seasonalities(self) -> None:
        """Set seasonalities that were left on auto.

        Turns on yearly seasonality if there is >= 2 years of history.
        Turns on weekly seasonality if there is >= 2 weeks and spacing < 7 days.
        Turns on daily seasonality if there is >= 2 days and spacing < 1 day.
        """
        history = self.history
        assert history is not None
        first = history['ds'].min()
        last = history['ds'].max()
        dt = history['ds'].diff()
        nonzero_idx = dt.values.nonzero()[0]
        if len(nonzero_idx) == 0:
            min_dt = pd.Timedelta(days=1)
        else:
            min_dt = dt.iloc[nonzero_idx].min()

        # Yearly seasonality
        yearly_disable = last - first < pd.Timedelta(days=730)
        fourier_order = self._parse_seasonality_args(
            'yearly', self.yearly_seasonality, yearly_disable, 10
        )
        if fourier_order > 0:
            self.seasonalities['yearly'] = {
                'period': 365.25,
                'fourier_order': fourier_order,
                'prior_scale': self.seasonality_prior_scale,
                'mode': self.seasonality_mode,
                'condition_name': None,
            }

        # Weekly seasonality
        weekly_disable = (
            (last - first < pd.Timedelta(weeks=2)) or
            (min_dt >= pd.Timedelta(weeks=1))
        )
        fourier_order = self._parse_seasonality_args(
            'weekly', self.weekly_seasonality, weekly_disable, 3
        )
        if fourier_order > 0:
            self.seasonalities['weekly'] = {
                'period': 7,
                'fourier_order': fourier_order,
                'prior_scale': self.seasonality_prior_scale,
                'mode': self.seasonality_mode,
                'condition_name': None,
            }

        # Daily seasonality
        daily_disable = (
            (last - first < pd.Timedelta(days=2)) or
            (min_dt >= pd.Timedelta(days=1))
        )
        fourier_order = self._parse_seasonality_args(
            'daily', self.daily_seasonality, daily_disable, 4
        )
        if fourier_order > 0:
            self.seasonalities['daily'] = {
                'period': 1,
                'fourier_order': fourier_order,
                'prior_scale': self.seasonality_prior_scale,
                'mode': self.seasonality_mode,
                'condition_name': None,
            }

    def _make_all_seasonality_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[float], pd.DataFrame, dict[str, list[str]]]:
        """Create all seasonality and holiday features.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with dates for computing features.

        Returns
        -------
        tuple
            (seasonal_features, prior_scales, component_cols, modes)
        """
        seasonal_features = []
        prior_scales: list[float] = []
        modes: dict[str, list[str]] = {'additive': [], 'multiplicative': []}

        for name, props in self.seasonalities.items():
            features = self.make_seasonality_features(
                df['ds'],
                props['period'],
                props['fourier_order'],
                name,
            )
            if props['condition_name'] is not None:
                features[~df[props['condition_name']]] = 0
            seasonal_features.append(features)
            prior_scales.extend([props['prior_scale']] * features.shape[1])
            modes[props['mode']].append(name)

        holiday_features, holiday_priors, holiday_names = self._make_holiday_features(df)
        if len(holiday_names) > 0:
            seasonal_features.append(holiday_features)
            prior_scales.extend(holiday_priors)
            if self.holidays_mode == 'multiplicative':
                modes['multiplicative'].append('holidays')
            else:
                modes['additive'].append('holidays')

        # Dummy to prevent empty X
        if len(seasonal_features) == 0:
            seasonal_features.append(
                pd.DataFrame({'zeros': np.zeros(df.shape[0])})
            )
            prior_scales.append(1.0)

        seasonal_features_df = pd.concat(seasonal_features, axis=1)
        component_cols, modes = self._regressor_column_matrix(
            seasonal_features_df, modes
        )
        return seasonal_features_df, prior_scales, component_cols, modes

    def _regressor_column_matrix(
        self,
        seasonal_features: pd.DataFrame,
        modes: dict[str, list[str]],
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Create matrix indicating which columns correspond to which components.

        Parameters
        ----------
        seasonal_features : pd.DataFrame
            Seasonal features dataframe.
        modes : dict
            Dictionary with 'additive' and 'multiplicative' component names.

        Returns
        -------
        tuple
            (component_cols, modes)
        """
        components = pd.DataFrame({
            'col': np.arange(seasonal_features.shape[1]),
            'component': [
                x.split('_delim_')[0] for x in seasonal_features.columns
            ],
        })

        # Identify holiday columns and add 'holidays' group
        holiday_names_in_cols = set()
        if self.train_holiday_names is not None:
            for col in seasonal_features.columns:
                prefix = col.split('_delim_')[0]
                if any(col.startswith(hn.split('_delim_')[0]) for hn in self.train_holiday_names):
                    holiday_names_in_cols.add(prefix)
        if holiday_names_in_cols:
            components = self._add_group_component(
                components, 'holidays', list(holiday_names_in_cols)
            )

        for mode in ('additive', 'multiplicative'):
            components = self._add_group_component(
                components, mode + '_terms', modes[mode]
            )
            modes[mode].append(mode + '_terms')

        component_cols = pd.crosstab(
            components['col'], components['component'],
        ).sort_index(level='col')

        for name in ['additive_terms', 'multiplicative_terms']:
            if name not in component_cols:
                component_cols[name] = 0
        component_cols.drop('zeros', axis=1, inplace=True, errors='ignore')

        if self.train_component_cols is not None:
            component_cols = component_cols[self.train_component_cols.columns]

        return component_cols, modes

    def _add_group_component(
        self,
        components: pd.DataFrame,
        name: str,
        group: list[str],
    ) -> pd.DataFrame:
        """Add a group component containing all components in group.

        Parameters
        ----------
        components : pd.DataFrame
            Components dataframe.
        name : str
            Name of the group component.
        group : list
            List of component names in the group.

        Returns
        -------
        pd.DataFrame
            Updated components dataframe.
        """
        new_comp = components[components['component'].isin(set(group))].copy()
        group_cols = new_comp['col'].unique()
        if len(group_cols) > 0:
            new_comp = pd.DataFrame({'col': group_cols, 'component': name})
            components = pd.concat([components, new_comp], ignore_index=True)
        return components

    def fit(self, df: pd.DataFrame, **kwargs: Any) -> "TimeWeaver":
        """Fit the TimeWeaver model.

        Parameters
        ----------
        df : pd.DataFrame
            Historical data with columns 'ds' and 'y'. If growth='logistic',
            must also have 'cap'.
        **kwargs
            Additional arguments for fitting.

        Returns
        -------
        TimeWeaver
            The fitted model.
        """
        if self.history is not None:
            raise RuntimeError(
                'TimeWeaver object can only be fit once. Instantiate a new object.'
            )

        if 'ds' not in df or 'y' not in df:
            raise ValueError(
                'Dataframe must have columns "ds" and "y" with the dates and '
                'values respectively.'
            )

        history = df[df['y'].notnull()].copy()
        if history.shape[0] < 2:
            raise ValueError('Dataframe has less than 2 non-NaN rows.')

        self.history_dates = pd.to_datetime(
            pd.Series(df['ds'].unique(), name='ds')
        ).sort_values()

        self.history = self.prepare_dataframe(history, initialize_scales=True)
        self._set_auto_seasonalities()

        # Store holiday names before making features
        _, _, holiday_names = self._make_holiday_features(self.history)
        self.train_holiday_names = holiday_names if holiday_names else None

        seasonal_features, prior_scales, component_cols, modes = (
            self._make_all_seasonality_features(self.history)
        )
        self.train_component_cols = component_cols
        self.component_modes = modes
        self.set_changepoints()
        self.fit_kwargs = kwargs.copy()

        if self.growth == 'linear':
            k, m = self.linear_growth_init(self.history)
        elif self.growth == 'flat':
            k, m = self.flat_growth_init(self.history)
        else:
            k, m = self.logistic_growth_init(self.history)

        n_changepoints = len(self.changepoints_t) if self.changepoints_t is not None else 0
        n_features = seasonal_features.shape[1]
        self.params = {
            'k': np.array([[k]]),
            'm': np.array([[m]]),
            'delta': np.zeros((1, n_changepoints)),
            'beta': np.zeros((1, n_features)),
            'sigma_obs': np.array([[1e-9]]),
        }

        if len(self.changepoints) == 0:
            self.params['k'] = self.params['k'] + self.params['delta'].reshape(-1)
            self.params['delta'] = np.zeros(self.params['delta'].shape).reshape((-1, 1))

        return self

    def predict_trend(self, df: pd.DataFrame) -> np.ndarray:
        """Predict trend component.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with 't' column and 'cap_scaled' if logistic.

        Returns
        -------
        np.ndarray
            Trend values on original scale.
        """
        k = np.nanmean(self.params['k'])
        m = np.nanmean(self.params['m'])
        deltas = np.nanmean(self.params['delta'], axis=0)

        t = np.array(df['t'])
        changepoints_t = self.changepoints_t
        assert changepoints_t is not None

        if self.growth == 'linear':
            trend = self.piecewise_linear(t, deltas, k, m, changepoints_t)
        elif self.growth == 'logistic':
            cap = df['cap_scaled']
            trend = self.piecewise_logistic(t, cap, deltas, k, m, changepoints_t)
        else:
            trend = self.flat_trend(t, m)

        return trend * self.y_scale + df['floor'].values

    def predict(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Generate predictions.

        Parameters
        ----------
        df : pd.DataFrame or None
            Dataframe with dates for predictions (column 'ds'). If None,
            predictions are made on the training data.

        Returns
        -------
        pd.DataFrame
            Dataframe with forecast components.
        """
        if self.history is None:
            raise RuntimeError('Model has not been fit.')

        if df is None:
            df = self.history.copy()
        else:
            if df.shape[0] == 0:
                raise ValueError('Dataframe has no rows.')
            df = self.prepare_dataframe(df.copy())

        df['trend'] = self.predict_trend(df)
        seasonal_components = self._predict_seasonal_components(df)

        cols = ['ds', 'trend']
        if 'cap' in df:
            cols.append('cap')
        if self.logistic_floor:
            cols.append('floor')

        result = pd.concat([df[cols], seasonal_components], axis=1)
        result['yhat'] = (
            result['trend'] * (1 + result['multiplicative_terms'])
            + result['additive_terms']
        )
        return result

    def _predict_seasonal_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict seasonality components.

        Parameters
        ----------
        df : pd.DataFrame
            Prediction dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with seasonal components.
        """
        seasonal_features, _, component_cols, _ = (
            self._make_all_seasonality_features(df)
        )

        X = seasonal_features.values
        data = {}
        for component in component_cols.columns:
            beta_c = self.params['beta'] * component_cols[component].values
            comp = np.matmul(X, beta_c.transpose())
            assert self.component_modes is not None
            if component in self.component_modes['additive']:
                comp *= self.y_scale
            data[component] = np.nanmean(comp, axis=1)
        return pd.DataFrame(data)

    def make_future_dataframe(
        self,
        periods: int,
        freq: str | None = 'D',
        include_history: bool = True,
    ) -> pd.DataFrame:
        """Generate future dataframe for predictions.

        Parameters
        ----------
        periods : int
            Number of periods to forecast.
        freq : str or None
            Frequency for pd.date_range (e.g., 'D', 'H', 'M').
        include_history : bool
            Whether to include historical dates.

        Returns
        -------
        pd.DataFrame
            Dataframe with 'ds' column for predictions.
        """
        if self.history_dates is None:
            raise RuntimeError('Model has not been fit.')

        if freq is None:
            freq = pd.infer_freq(self.history_dates.tail(5))
            if freq is None:
                raise ValueError('Unable to infer freq')

        last_date = self.history_dates.max()
        dates = pd.date_range(
            start=last_date,
            periods=periods + 1,
            freq=freq,
        )
        dates = dates[dates > last_date]
        dates = dates[:periods]

        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))

        return pd.DataFrame({'ds': dates})
