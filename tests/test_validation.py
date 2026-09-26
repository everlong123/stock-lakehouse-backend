"""OHLC validation and duplicate handling tests."""

import pandas as pd

from app.pipelines.validation import validate_ohlc_frame
from tests.conftest import make_ohlcv


def test_valid_ohlc() -> None:
    frame = make_ohlcv(80)
    report = validate_ohlc_frame(frame)
    assert report["invalid_ohlc_count"] == 0
    assert report["invalid_volume_count"] == 0
    assert report["duplicate_count"] == 0


def test_invalid_ohlc_detected() -> None:
    frame = make_ohlcv(10)
    frame.loc[0, "high"] = frame.loc[0, "low"] - 1
    report = validate_ohlc_frame(frame)
    assert report["invalid_ohlc_count"] >= 1


def test_negative_volume_detected() -> None:
    frame = make_ohlcv(10)
    frame.loc[2, "volume"] = -5
    report = validate_ohlc_frame(frame)
    assert report["invalid_volume_count"] == 1


def test_duplicate_handling() -> None:
    frame = make_ohlcv(8)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = validate_ohlc_frame(duplicated)
    assert report["duplicate_count"] == 1
