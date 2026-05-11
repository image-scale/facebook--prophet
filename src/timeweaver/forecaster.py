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
        self.train_holiday_names: pd.Series | None = None
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
        self.set_changepoints()
        self.fit_kwargs = kwargs.copy()

        if self.growth == 'linear':
            k, m = self.linear_growth_init(self.history)
        elif self.growth == 'flat':
            k, m = self.flat_growth_init(self.history)
        else:
            k, m = self.logistic_growth_init(self.history)

        n_changepoints = len(self.changepoints_t) if self.changepoints_t is not None else 0
        self.params = {
            'k': np.array([[k]]),
            'm': np.array([[m]]),
            'delta': np.zeros((1, n_changepoints)),
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

        cols = ['ds', 'trend']
        if 'cap' in df:
            cols.append('cap')
        if self.logistic_floor:
            cols.append('floor')

        result = df[cols].copy()
        result['additive_terms'] = 0.0
        result['multiplicative_terms'] = 0.0
        result['yhat'] = result['trend'] * (1 + result['multiplicative_terms']) + result['additive_terms']

        return result

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
