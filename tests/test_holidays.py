"""Tests for TimeWeaver country holidays integration."""

import numpy as np
import pandas as pd
import pytest

from timeweaver import TimeWeaver, make_holidays_df, get_holiday_names


def create_daily_data(start: str = "2012-01-01", periods: int = 365 * 2) -> pd.DataFrame:
    """Create synthetic daily time series data."""
    dates = pd.date_range(start=start, periods=periods, freq='D')
    np.random.seed(42)
    trend = np.linspace(10, 50, periods)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    noise = np.random.normal(0, 2, periods)
    y = trend + seasonality + noise
    return pd.DataFrame({'ds': dates, 'y': y})


class TestMakeHolidaysDf:
    """Tests for make_holidays_df function."""

    def test_make_holidays_df_basic(self):
        """Test basic holiday dataframe creation."""
        holidays_df = make_holidays_df([2020, 2021], 'US')
        assert 'ds' in holidays_df.columns
        assert 'holiday' in holidays_df.columns
        assert len(holidays_df) > 0

    def test_make_holidays_df_us_holidays(self):
        """Test US holidays are included."""
        holidays_df = make_holidays_df([2020], 'US')
        holiday_names = set(holidays_df['holiday'].unique())
        # Check for common US holidays
        assert any('Christmas' in h for h in holiday_names)
        assert any('Independence' in h or 'July' in h for h in holiday_names)

    def test_make_holidays_df_multiple_years(self):
        """Test holidays for multiple years."""
        holidays_df = make_holidays_df([2020, 2021, 2022], 'US')
        years = holidays_df['ds'].dt.year.unique()
        assert 2020 in years
        assert 2021 in years
        assert 2022 in years

    def test_make_holidays_df_different_countries(self):
        """Test holidays for different countries."""
        for country in ['US', 'UK', 'DE', 'FR']:
            holidays_df = make_holidays_df([2020], country)
            assert len(holidays_df) > 0

    def test_make_holidays_df_invalid_country(self):
        """Test invalid country raises error."""
        with pytest.raises(ValueError, match='not currently supported'):
            make_holidays_df([2020], 'INVALID')

    def test_make_holidays_df_with_province(self):
        """Test holidays with province/state."""
        holidays_df = make_holidays_df([2020], 'US', province='CA')
        assert len(holidays_df) > 0


class TestGetHolidayNames:
    """Tests for get_holiday_names function."""

    def test_get_holiday_names_basic(self):
        """Test getting holiday names."""
        names = get_holiday_names('US')
        assert isinstance(names, set)
        assert len(names) > 0

    def test_get_holiday_names_contains_common(self):
        """Test common holidays are in names."""
        names = get_holiday_names('US')
        # At least some common holiday should be present
        assert any('Christmas' in name for name in names)


class TestAddCountryHolidays:
    """Tests for add_country_holidays method."""

    def test_add_country_holidays_basic(self):
        """Test adding country holidays to model."""
        m = TimeWeaver()
        m.add_country_holidays('US')
        assert m.country_holidays == 'US'

    def test_add_country_holidays_chaining(self):
        """Test add_country_holidays returns self for chaining."""
        m = TimeWeaver()
        result = m.add_country_holidays('US')
        assert result is m

    def test_add_country_holidays_after_fit_raises(self):
        """Test that adding country holidays after fit raises error."""
        df = create_daily_data(periods=100)
        m = TimeWeaver()
        m.fit(df)
        with pytest.raises(RuntimeError, match='prior to model fitting'):
            m.add_country_holidays('US')

    def test_country_holidays_in_fit(self):
        """Test that country holidays are applied during fit."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.add_country_holidays('US')
        m.fit(df)
        # holidays should now contain country holidays
        assert m.holidays is not None
        assert len(m.holidays) > 0

    def test_country_holidays_with_manual_holidays(self):
        """Test country holidays combined with manual holidays."""
        df = create_daily_data(periods=365)
        manual_holidays = pd.DataFrame({
            'ds': ['2012-06-15'],
            'holiday': ['special_day'],
        })
        m = TimeWeaver(
            holidays=manual_holidays,
            weekly_seasonality=False,
            yearly_seasonality=False,
            uncertainty_samples=0,
        )
        m.add_country_holidays('US')
        m.fit(df)
        # Both manual and country holidays should be present
        assert m.holidays is not None
        holiday_names = set(m.holidays['holiday'].unique())
        assert 'special_day' in holiday_names

    def test_country_holidays_in_prediction(self):
        """Test that country holidays affect predictions."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                       uncertainty_samples=0)
        m.add_country_holidays('US')
        m.fit(df)
        forecast = m.predict()
        assert 'holidays' in forecast.columns

    def test_country_holidays_different_countries(self):
        """Test different country codes."""
        df = create_daily_data(periods=365)
        for country in ['US', 'UK', 'DE']:
            m = TimeWeaver(weekly_seasonality=False, yearly_seasonality=False,
                           uncertainty_samples=0)
            m.add_country_holidays(country)
            m.fit(df)
            assert m.country_holidays == country
            assert m.holidays is not None
