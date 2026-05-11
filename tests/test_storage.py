"""Tests for TimeWeaver model serialization."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from timeweaver import (
    TimeWeaver,
    model_to_dict,
    model_from_dict,
    model_to_json,
    model_from_json,
    save_model,
    load_model,
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


class TestModelToDict:
    """Tests for model_to_dict function."""

    def test_model_to_dict_basic(self):
        """Test basic model serialization to dict."""
        df = create_daily_data(periods=365)
        m = TimeWeaver()
        m.fit(df)
        model_dict = model_to_dict(m)
        assert '__timeweaver_version' in model_dict
        assert 'growth' in model_dict
        assert 'params' in model_dict

    def test_model_to_dict_unfitted_raises(self):
        """Test that unfitted model raises error."""
        m = TimeWeaver()
        with pytest.raises(ValueError, match='Model must be fitted'):
            model_to_dict(m)

    def test_model_to_dict_preserves_growth(self):
        """Test that growth type is preserved."""
        df = create_daily_data(periods=365)
        for growth in ['linear', 'flat']:
            m = TimeWeaver(growth=growth)
            m.fit(df)
            model_dict = model_to_dict(m)
            assert model_dict['growth'] == growth

    def test_model_to_dict_with_seasonalities(self):
        """Test serialization with custom seasonality."""
        df = create_daily_data(periods=365)
        m = TimeWeaver()
        m.add_seasonality(name='monthly', period=30, fourier_order=3)
        m.fit(df)
        model_dict = model_to_dict(m)
        # Check seasonalities are serialized
        assert 'seasonalities' in model_dict

    def test_model_to_dict_with_regressors(self):
        """Test serialization with extra regressors."""
        df = create_daily_data(periods=365)
        df['temp'] = np.random.randn(365)
        m = TimeWeaver()
        m.add_regressor('temp')
        m.fit(df)
        model_dict = model_to_dict(m)
        assert 'extra_regressors' in model_dict


class TestModelFromDict:
    """Tests for model_from_dict function."""

    def test_model_from_dict_basic(self):
        """Test basic model deserialization from dict."""
        df = create_daily_data(periods=365)
        m = TimeWeaver()
        m.fit(df)
        model_dict = model_to_dict(m)
        m2 = model_from_dict(model_dict)
        assert m2.growth == m.growth
        assert m2.n_changepoints == m.n_changepoints

    def test_model_from_dict_can_predict(self):
        """Test that deserialized model can make predictions."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        model_dict = model_to_dict(m)
        m2 = model_from_dict(model_dict)
        forecast = m2.predict()
        assert 'yhat' in forecast.columns
        assert len(forecast) == len(m2.history)

    def test_model_roundtrip_predictions_match(self):
        """Test that predictions match after roundtrip."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        forecast1 = m.predict()

        model_dict = model_to_dict(m)
        m2 = model_from_dict(model_dict)
        forecast2 = m2.predict()

        np.testing.assert_array_almost_equal(
            forecast1['yhat'].values,
            forecast2['yhat'].values,
            decimal=5,
        )


class TestModelJson:
    """Tests for JSON serialization."""

    def test_model_to_json(self):
        """Test model serialization to JSON string."""
        df = create_daily_data(periods=365)
        m = TimeWeaver()
        m.fit(df)
        json_str = model_to_json(m)
        assert isinstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert 'growth' in parsed

    def test_model_from_json(self):
        """Test model deserialization from JSON string."""
        df = create_daily_data(periods=365)
        m = TimeWeaver()
        m.fit(df)
        json_str = model_to_json(m)
        m2 = model_from_json(json_str)
        assert m2.growth == m.growth

    def test_model_json_roundtrip(self):
        """Test JSON roundtrip preserves predictions."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        forecast1 = m.predict()

        json_str = model_to_json(m)
        m2 = model_from_json(json_str)
        forecast2 = m2.predict()

        np.testing.assert_array_almost_equal(
            forecast1['yhat'].values,
            forecast2['yhat'].values,
            decimal=5,
        )


class TestSaveLoadModel:
    """Tests for file-based serialization."""

    def test_save_load_model(self):
        """Test saving and loading model to file."""
        df = create_daily_data(periods=365)
        m = TimeWeaver()
        m.fit(df)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_model(m, path)
            assert os.path.exists(path)

            m2 = load_model(path)
            assert m2.growth == m.growth
            assert m2.history is not None
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_predictions(self):
        """Test that saved/loaded model has same predictions."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(uncertainty_samples=0)
        m.fit(df)
        forecast1 = m.predict()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_model(m, path)
            m2 = load_model(path)
            forecast2 = m2.predict()

            np.testing.assert_array_almost_equal(
                forecast1['yhat'].values,
                forecast2['yhat'].values,
                decimal=5,
            )
        finally:
            os.unlink(path)

    def test_save_load_with_regressors(self):
        """Test save/load preserves regressors."""
        df = create_daily_data(periods=365)
        df['temp'] = np.random.randn(365)
        m = TimeWeaver(uncertainty_samples=0)
        m.add_regressor('temp')
        m.fit(df)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_model(m, path)
            m2 = load_model(path)
            assert 'temp' in m2.extra_regressors

            # Should be able to predict with regressor
            future = m.make_future_dataframe(periods=30, include_history=False)
            future['temp'] = np.random.randn(30)
            forecast = m2.predict(future)
            assert len(forecast) == 30
        finally:
            os.unlink(path)

    def test_save_load_with_holidays(self):
        """Test save/load preserves holidays."""
        df = create_daily_data(periods=365)
        holidays = pd.DataFrame({
            'ds': ['2012-06-15'],
            'holiday': ['special'],
        })
        m = TimeWeaver(holidays=holidays, uncertainty_samples=0)
        m.fit(df)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_model(m, path)
            m2 = load_model(path)
            assert m2.holidays is not None
        finally:
            os.unlink(path)


class TestSerializationEdgeCases:
    """Tests for edge cases in serialization."""

    def test_serialization_with_logistic_growth(self):
        """Test serialization with logistic growth."""
        df = create_daily_data(periods=365)
        df['cap'] = df['y'].max() * 2
        m = TimeWeaver(growth='logistic', uncertainty_samples=0)
        m.fit(df)

        json_str = model_to_json(m)
        m2 = model_from_json(json_str)
        assert m2.growth == 'logistic'

        future = m.make_future_dataframe(periods=30, include_history=False)
        future['cap'] = df['cap'].iloc[0]
        forecast = m2.predict(future)
        assert len(forecast) == 30

    def test_serialization_with_flat_growth(self):
        """Test serialization with flat growth."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(growth='flat', uncertainty_samples=0)
        m.fit(df)

        json_str = model_to_json(m)
        m2 = model_from_json(json_str)
        assert m2.growth == 'flat'

    def test_serialization_preserves_seasonality_mode(self):
        """Test that seasonality mode is preserved."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(seasonality_mode='multiplicative', uncertainty_samples=0)
        m.fit(df)

        m2 = model_from_json(model_to_json(m))
        assert m2.seasonality_mode == 'multiplicative'

    def test_serialization_preserves_interval_width(self):
        """Test that interval_width is preserved."""
        df = create_daily_data(periods=365)
        m = TimeWeaver(interval_width=0.95)
        m.fit(df)

        m2 = model_from_json(model_to_json(m))
        assert m2.interval_width == 0.95
