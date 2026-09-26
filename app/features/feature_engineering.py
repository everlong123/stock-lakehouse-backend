"""Gold-layer feature engineering orchestrator."""

from __future__ import annotations

import pandas as pd

from app.core.exceptions import DataValidationError
from app.features.macro_features import MacroFeatureBuilder, add_market_regime_features
from app.features.price_features import add_price_features
from app.features.target_features import add_target_features
from app.features.technical_features import add_technical_features


def build_gold_features(frame: pd.DataFrame, include_macro: bool = True) -> pd.DataFrame:
    """
    Build Gold features per symbol, sorted by timestamp, with no future leakage.

    Args:
        frame: DataFrame with OHLCV columns
        include_macro: Whether to include macro features (slower but more context for ML)

    Returns:
        DataFrame with all features
    """
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"Feature engineering missing columns: {sorted(missing)}")

    parts: list[pd.DataFrame] = []
    macro_builder = MacroFeatureBuilder() if include_macro else None

    for symbol, group in frame.groupby("symbol", sort=True):
        ordered = group.sort_values("timestamp").copy().reset_index(drop=True)

        # Core features
        ordered = add_technical_features(ordered)
        ordered = add_price_features(ordered)
        ordered = add_target_features(ordered)

        # Market regime features
        ordered = add_market_regime_features(ordered)

        # Macro features (optional - adds context but slower)
        if macro_builder is not None and len(ordered) > 50:
            try:
                ordered = macro_builder.build(ordered)
            except Exception:
                pass  # Skip macro if provider fails

        ordered["symbol"] = symbol
        parts.append(ordered)

    result = pd.concat(parts, ignore_index=True)
    return result.loc[:, ~result.columns.duplicated()]


def build_gold_features_simple(frame: pd.DataFrame) -> pd.DataFrame:
    """Build Gold features without macro features (faster for simple use cases)."""
    return build_gold_features(frame, include_macro=False)


def drop_warmup_rows(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Drop rows with NaN features. Target NaNs on the last row are expected."""
    return frame.dropna(subset=feature_columns).reset_index(drop=True)
