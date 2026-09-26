"""Prediction helper that loads a saved model and forecasts the next close."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.constants import LINEAR_REGRESSION_FEATURES, LSTM_FEATURE_COLUMNS
from app.core.exceptions import PredictionError
from app.forecasting.model_registry import load_model, load_registry_record, model_directory
from app.lakehouse.gold import GoldLayer


FORECAST_DISCLAIMER = """
⚠️ CẢNH BÁO QUAN TRỌNG - CHỈ Mang Tính Tham khảo

1. DỰ BÁO GIÁ CỔ PHIẾU CÓ ĐỘ KHÔNG CHẮC CHẮN CAO
   - Kết quả dự báo chỉ phản ánh xu hướng quá khứ, không phải dự đoán chắc chắn về giá tương lai.
   - Thị trường chứng khoán bị ảnh hưởng bởi nhiều yếu tố không thể dự đoán (tin tức, tâm lý nhà đầu tư, biến động vĩ mô).

2. KHÔNG PHẢI KHUYẾN NGHỊ ĐẦU TƯ
   - Hệ thống này được xây dựng cho mục đích NGHIÊN CỨU HỌC THUẬT.
   - Tuyệt đối KHÔNG sử dụng kết quả dự báo để quyết định mua/bán thực tế.

3. VỀ ĐỘ CHÍNH XÁC CỦA MÔ HÌNH
   - MAE/RMSE/MAPE chỉ đo lỗi trên dữ liệu LỊCH SỬ, không phản ánh hiệu suất tương lai.
   - Các mô hình đơn giản (Linear Regression, ARIMA) có giới hạn trong việc nắm bắt động thái thị trường phức tạp.
   - LSTM cũng chỉ là mô hình thống kê, không có "trí tuệ" về thị trường.

4. BACKTEST KHÔNG ĐẢM BẢO LỢI NHUẬN THỰC
   - Hiệu suất quá khứ trong backtest KHÔNG đảm bảo lợi nhuận trong tương lai.
   - Backtest không tính đến chi phí giao dịch thực tế, slippage, và điều kiện thị trường khác nhau.

5. NGUỒN GỐC DỮ LIỆU
   - Dữ liệu được lấy từ các nguồn công khai, có thể có độ trễ hoặc sai sót.
   - Không có bảo đảm về tính chính xác hoàn toàn của dữ liệu.

Người dùng tự chịu trách nhiệm về mọi quyết định đầu tư của mình.
"""


RISK_ASSESSMENT_DISCLAIMER = """
⚠️ ĐÁNH GIÁ RỦI RO - CHỈ Mang Tính tham khảo

Chỉ báo rủi ro (VaR, drawdown, Sharpe ratio) được tính toán từ dữ liệu lịch sử
với các giả định về phân phối lợi nhuận. Trong thực tế:
- Phân phối lợi nhuận thị trường thường có "đuôi béo" (fat tails), dẫn đến đánh giá rủi ro thấp hơn thực tế.
- Sự kiện "thiên nga đen" có thể gây ra tổn thất lớn hơn nhiều so với dự đoán.
- Correlation giữa các tài sản thay đổi trong thời kỳ khủng hoảng.

Đây là công cụ phân tích, không phải tư vấn tài chính chuyên nghiệp.
"""


GENERAL_DISCLAIMER = """
📚 MỤC ĐÍCH SỬ DỤNG

Hệ thống Stock Lakehouse AI được phát triển cho mục đích:
- Nghiên cứu học thuật về tài chính định lượng
- Học tập về xây dựng hệ thống dữ liệu (Lakehouse architecture)
- Demo các kỹ thuật ML/AI trong lĩnh vực chứng khoán

Mọi nội dung mang tính THAM KHẢO, KHÔNG phải lời khuyên đầu tư.
"""


def predict_symbol(symbol: str, model_name: str, horizon: int = 5) -> dict[str, Any]:
    """Generate a next-close prediction series from the latest trained model."""
    model = load_model(symbol, model_name)
    registry = load_registry_record(symbol, model_name)
    gold = GoldLayer().read(symbol)
    if gold.empty:
        raise PredictionError(f"No Gold data available for {symbol}.")
    gold = gold.sort_values("timestamp")

    if model_name == "arima":
        predicted = model.predict(list(range(horizon)))
        last_ts = pd.to_datetime(gold["timestamp"].iloc[-1])
        freq = pd.infer_freq(gold["timestamp"]) or "B"
        future_index = pd.date_range(last_ts, periods=horizon + 1, freq=freq)[1:]
        points = [
            {"timestamp": str(ts), "actual": None, "predicted": float(value)}
            for ts, value in zip(future_index, predicted)
        ]
    elif model_name == "lstm":
        cols = [col for col in LSTM_FEATURE_COLUMNS if col in gold.columns]
        frame = gold.dropna(subset=cols)
        predicted_all = model.predict(frame[cols])
        actual = frame["close"].to_numpy()[model.sequence_length - 1 :]  # type: ignore[attr-defined]
        timestamps = frame["timestamp"].to_numpy()[model.sequence_length - 1 :]
        n = min(len(predicted_all), len(actual), len(timestamps))
        points = [
            {
                "timestamp": str(timestamps[i]),
                "actual": float(actual[i]),
                "predicted": float(predicted_all[i]),
            }
            for i in range(max(0, n - 120), n)
        ]
    else:
        cols = [col for col in LINEAR_REGRESSION_FEATURES if col in gold.columns]
        frame = gold.dropna(subset=cols)
        predicted_all = model.predict(frame[cols])
        actual = frame["close"].to_numpy()
        timestamps = frame["timestamp"].to_numpy()
        n = min(len(predicted_all), len(actual))
        points = [
            {
                "timestamp": str(timestamps[i]),
                "actual": float(actual[i]),
                "predicted": float(predicted_all[i]),
            }
            for i in range(max(0, n - 120), n)
        ]

    directory = model_directory(symbol, model_name)
    stored = directory / "predictions.parquet"
    if stored.exists() and model_name != "arima":
        stored_frame = pd.read_parquet(stored)
        points = [
            {
                "timestamp": str(row.timestamp),
                "actual": float(row.actual) if pd.notna(row.actual) else None,
                "predicted": float(row.predicted),
            }
            for row in stored_frame.itertuples(index=False)
        ]

    return {
        "symbol": symbol.upper(),
        "model_name": model_name,
        "horizon": horizon,
        "metrics": {
            "mae": registry.get("mae"),
            "rmse": registry.get("rmse"),
            "mape": registry.get("mape"),
            "directional_accuracy": registry.get("directional_accuracy"),
        },
        "parameters": registry.get("parameters"),
        "predictions": points,
        "disclaimer": FORECAST_DISCLAIMER,
    }
