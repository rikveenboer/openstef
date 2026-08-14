# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Reference-style forecaster wrapper for asymmetric conformal calibration."""

from datetime import timedelta
from typing import override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import HyperParams
from openstef_models.models.forecasting.forecaster import Forecaster
from openstef_models.transforms.postprocessing.conformalized_quantile_calibrator import (
    ConformalizedQuantileCalibrator,
)


class ConformalizedForecaster(Forecaster):
    """Wrap a forecaster and calibrate its predictions before ensemble combination.

    The inner forecaster is fitted normally. A recent contiguous slice of the
    training data is then predicted with the fitted inner forecaster and used to
    fit a :class:`ConformalizedQuantileCalibrator`. Subsequent predictions are
    transformed by that calibrator. The wrapper does not sort quantiles.

    Calibration uses the training data, including the calibration slice, to match
    the reference implementation. For independent split-conformal calibration,
    use ``ConformalizedQuantileCalibrator`` directly with held-out forecasts.
    """

    inner: Forecaster = Field(description="Forecaster whose predictions are calibrated.")
    calibration_length: timedelta = Field(
        default=timedelta(days=14),
        description="Recent training window used to fit conformal corrections.",
    )
    conformalize_median: bool = Field(
        default=False,
        description="Whether to calibrate P50 in addition to the tails.",
    )
    min_calibration_samples: int = Field(
        default=100,
        ge=1,
        description="Minimum valid calibration samples required for corrections.",
    )
    target_column: str = Field(default="load", description="Name of the target column.")

    _calibrator: ConformalizedQuantileCalibrator | None = PrivateAttr(default=None)

    @property
    @override
    def hparams(self) -> HyperParams:
        return self.inner.hparams

    @property
    @override
    def is_fitted(self) -> bool:
        return self.inner.is_fitted and self._calibrator is not None and self._calibrator.is_fitted

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        """Fit the inner forecaster and reference-style calibration state."""
        self.inner.fit(data=data, data_val=data_val)
        calibration_input = self._build_calibration_input(data=data, data_val=data_val)
        calibration_forecast = self.inner.predict(calibration_input)
        target = calibration_input.target_series.reindex(calibration_forecast.data.index)
        calibration_data = calibration_forecast.data.copy()
        calibration_data[self.target_column] = target
        calibration_dataset = ForecastDataset(
            data=calibration_data,
            sample_interval=calibration_forecast.sample_interval,
            target_column=self.target_column,
        )
        self._calibrator = ConformalizedQuantileCalibrator(
            quantiles=self.quantiles,
            conformalize_median=self.conformalize_median,
            min_calibration_samples=self.min_calibration_samples,
        )
        self._calibrator.fit(calibration_dataset)

    def _build_calibration_input(
        self,
        data: ForecastInputDataset,
        data_val: ForecastInputDataset | None,
    ) -> ForecastInputDataset:
        """Build a recent contiguous input window from training and validation data."""
        frames = [data.to_pandas()]
        if data_val is not None:
            frames.append(data_val.to_pandas())
        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        calibration_start = combined.index.max() - self.calibration_length
        calibration_data = combined.loc[combined.index >= calibration_start]
        calibration_data.attrs.pop("forecast_start", None)
        return ForecastInputDataset(
            data=calibration_data,
            sample_interval=data.sample_interval,
            forecast_start=calibration_start.to_pydatetime(),
            target_column=self.target_column,
        )

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        """Predict with the inner forecaster and apply fitted corrections."""
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)
        calibrator = self._calibrator
        if calibrator is None:
            raise NotFittedError(self.__class__.__name__)
        return calibrator.transform(self.inner.predict(data))


__all__ = ["ConformalizedForecaster"]
