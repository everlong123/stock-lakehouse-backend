"""Feature engineering for macro and sector data."""

from __future__ import annotations

import pandas as pd


def add_macro_features(df: pd.DataFrame, macro_data: dict | None = None) -> pd.DataFrame:
    """
    Add macro economic features to OHLCV data.

    This allows ML models to consider macro context when making predictions.
    """
    if df.empty:
        return df

    result = df.copy()

    # If macro data is provided, merge it
    if macro_data is not None:
        # Exchange rate
        if "exchange_rate" in macro_data:
            result["usd_vnd"] = macro_data["exchange_rate"].get("usd_vnd")
            # Compute return vs exchange rate movement
            result["fx_adjusted_close"] = result["close"] / result["usd_vnd"] * 10000  # normalized

        # Gold correlation
        if "gold" in macro_data:
            result["gold_usd_oz"] = macro_data["gold"].get("price_usd_oz")

        # Oil correlation
        if "oil" in macro_data:
            result["wti_usd_barrel"] = macro_data["oil"].get("wti_usd_barrel")

    # Add placeholder macro features if not provided
    # In production, these would come from actual macro data
    if "usd_vnd" not in result.columns:
        result["usd_vnd"] = 25000  # Default USD/VND rate

    if "gold_usd_oz" not in result.columns:
        result["gold_usd_oz"] = 2000  # Default gold price

    if "wti_usd_barrel" not in result.columns:
        result["wti_usd_barrel"] = 80  # Default oil price

    # Computed macro features
    result["macro_normalized"] = (
        result["close"] / result["usd_vnd"] * 1000  # VND normalized price
    )

    return result


def add_sector_features(df: pd.DataFrame, sector_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Add sector performance features.

    This can help ML models understand sector momentum and rotation.
    """
    if df.empty:
        return df

    result = df.copy()

    # If sector data provided, merge by date
    if sector_data is not None and not sector_data.empty:
        # For simplicity, merge on year-month
        result["year_month"] = result["timestamp"].dt.to_period("M").astype(str)

        sector_data = sector_data.copy()
        sector_data["year_month"] = pd.to_datetime(sector_data["date"]).dt.to_period("M").astype(str)

        # Merge sector performance
        sector_cols = ["year_month", "sector_performance", "sector_momentum"]
        if all(col in sector_data.columns for col in ["year_month", "sector"]):
            merged = result.merge(
                sector_data[["year_month", "sector", "sector_performance", "sector_momentum"]],
                on="year_month",
                how="left"
            )
            for col in ["sector_performance", "sector_momentum"]:
                if col in merged.columns:
                    result[col] = merged[col].values

    return result


def add_market_regime_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Add market regime indicators.

    This can help ML models adapt to different market conditions.
    """
    if df.empty or len(df) < lookback:
        return df

    result = df.copy()

    # Volatility regime (rolling std of returns)
    result["returns"] = result["close"].pct_change()
    result["volatility_20d"] = result["returns"].rolling(lookback).std()
    result["volatility_actual"] = result["volatility_20d"] > result["volatility_20d"].quantile(0.75)

    # Trend regime (SMA crossover)
    result["sma_20"] = result["close"].rolling(20).mean()
    result["sma_50"] = result["close"].rolling(50).mean() if len(result) >= 50 else result["close"].rolling(len(result)).mean()
    result["trend_bull"] = result["close"] > result["sma_20"]
    result["trend_strong"] = (result["close"] > result["sma_20"]) & (result["sma_20"] > result["sma_50"])

    # Volume regime
    result["volume_ma20"] = result["volume"].rolling(lookback).mean()
    result["volume_spike"] = result["volume"] > result["volume_ma20"] * 1.5

    # Regime combination
    result["regime"] = "neutral"
    result.loc[result["trend_strong"] & ~result["volatility_actual"], "regime"] = "bull_low_vol"
    result.loc[result["trend_strong"] & result["volatility_actual"], "regime"] = "bull_high_vol"
    result.loc[~result["trend_bull"] & ~result["volatility_actual"], "regime"] = "bear_low_vol"
    result.loc[~result["trend_bull"] & result["volatility_actual"], "regime"] = "bear_high_vol"

    # Encode regime as numeric
    regime_map = {
        "bull_low_vol": 1,
        "bull_high_vol": 2,
        "neutral": 0,
        "bear_low_vol": -1,
        "bear_high_vol": -2,
    }
    result["regime_numeric"] = result["regime"].map(regime_map)

    return result


def add_correlation_features(df: pd.DataFrame, index_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Add correlation features with market index.

    This helps ML models understand beta and correlation dynamics.
    """
    if df.empty:
        return df

    result = df.copy()

    # If index data provided, compute beta and correlation
    if index_df is not None and not index_df.empty:
        # Merge on timestamp
        merged = result.merge(
            index_df[["timestamp", "close", "returns"]].rename(columns={
                "close": "index_close",
                "returns": "index_returns"
            }),
            on="timestamp",
            how="left"
        )

        if "index_returns" in merged.columns:
            result["market_returns"] = merged["index_returns"].values
            result["index_close"] = merged["index_close"].values

            # Rolling correlation
            result["correlation_20d"] = merged["returns"].rolling(20).corr(merged["index_returns"])

            # Beta
            var_market = merged["index_returns"].rolling(20).var()
            cov = merged["returns"].rolling(20).cov(merged["index_returns"])
            result["beta_20d"] = cov / var_market

    return result


class MacroFeatureBuilder:
    """Builder class for macro features with caching."""

    def __init__(self):
        self._macro_cache: dict | None = None
        self._macro_timestamp: pd.Timestamp | None = None

    def build(self, df: pd.DataFrame, macro_provider=None, force_refresh: bool = False) -> pd.DataFrame:
        """Build all macro features."""
        if df.empty:
            return df

        result = df.copy()

        # Fetch macro data if needed
        if macro_provider is not None and (self._macro_cache is None or force_refresh):
            try:
                self._macro_cache = macro_provider.get_macro_summary()
                self._macro_timestamp = pd.Timestamp.now(tz="UTC")
            except Exception:
                pass

        # Add macro features
        result = add_macro_features(result, self._macro_cache)

        # Add market regime
        result = add_market_regime_features(result)

        return result

    def is_cache_valid(self, max_age_minutes: int = 60) -> bool:
        """Check if macro cache is still valid."""
        if self._macro_timestamp is None:
            return False
        age = pd.Timestamp.now(tz="UTC") - self._macro_timestamp
        return age < pd.Timedelta(minutes=max_age_minutes)
