# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import ForecastInputDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.conformalized_forecaster import ConformalizedForecaster
from openstef_models.models.forecasting.constant_quantile_forecaster import ConstantQuantileForecaster

QUANTILES = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]
HORIZONS = [LeadTime(timedelta(hours=1))]


def _input_data(values: list[float], forecast_start: datetime | None = None) -> ForecastInputDataset:
    """Build a single-target hourly input dataset."""
    return ForecastInputDataset(
        data=pd.DataFrame(
            {"load": values},
            index=pd.date_range("2025-01-01", periods=len(values), freq="h"),
        ),
        sample_interval=timedelta(hours=1),
        forecast_start=forecast_start,
    )


def test_conformalized_forecaster_applies_corrections_after_inner_prediction() -> None:
    """Test reference-style corrections are applied to later inner predictions."""
    training = _input_data([0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    inner = ConstantQuantileForecaster(quantiles=QUANTILES, horizons=HORIZONS)
    forecaster = ConformalizedForecaster(
        inner=inner,
        quantiles=QUANTILES,
        horizons=HORIZONS,
        calibration_length=timedelta(hours=1),
        min_calibration_samples=1,
    )

    forecaster.fit(training)
    result = forecaster.predict(
        _input_data(
            training.data["load"].tolist(),
            forecast_start=pd.Timestamp("2025-01-01 06:00").to_pydatetime(),
        )
    )

    assert result.data.loc[result.data.index[0], "quantile_P50"] == pytest.approx(0.0)
    assert result.data.loc[result.data.index[0], "quantile_P90"] == pytest.approx(100.0)
    assert result.data.loc[result.data.index[0], "quantile_P10"] == pytest.approx(0.0)


def test_conformalized_forecaster_does_not_sort_predictions() -> None:
    """Test ordering remains a downstream responsibility."""
    training = _input_data([0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    inner = ConstantQuantileForecaster(quantiles=QUANTILES, horizons=HORIZONS)
    forecaster = ConformalizedForecaster(
        inner=inner,
        quantiles=QUANTILES,
        horizons=HORIZONS,
        calibration_length=timedelta(hours=1),
        min_calibration_samples=1,
    )
    forecaster.fit(training)
    result = forecaster.predict(
        _input_data(training.data["load"].tolist(), forecast_start=pd.Timestamp("2025-01-01 06:00").to_pydatetime())
    )

    assert np.allclose(result.data["quantile_P10"], 0.0)
    assert np.allclose(result.data["quantile_P50"], 0.0)
    assert np.allclose(result.data["quantile_P90"], 100.0)


def test_conformalized_forecaster_requires_fit() -> None:
    """Test the wrapper follows the project not-fitted contract."""
    inner = ConstantQuantileForecaster(quantiles=QUANTILES, horizons=HORIZONS)
    forecaster = ConformalizedForecaster(inner=inner, quantiles=QUANTILES, horizons=HORIZONS)

    with pytest.raises(NotFittedError):
        forecaster.predict(_input_data([0.0, 0.0]))
