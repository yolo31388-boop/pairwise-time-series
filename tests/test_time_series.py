"""
游戏内时间序列分析与预测系统 - 验收测试
运行方式：python -m pytest tests/test_time_series.py -q
共 15 个测试用例
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.time_series import TimeSeries


class TestTimeSeries:
    def setup_method(self):
        self.system = TimeSeries()

    # ---- 序列分析（analyze_series）-------------------------------------
    def test_case_01(self):
        result = self.system.analyze_series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["mean"] == pytest.approx(3.0)
        assert result["length"] == 5
        assert result["min"] == 1.0
        assert result["max"] == 5.0

    def test_case_02(self):
        result = self.system.analyze_series([1.0, 3.0, 5.0, 7.0, 9.0])
        assert result["trend"] == "increasing"
        assert result["slope"] == pytest.approx(2.0)

    def test_case_03(self):
        result = self.system.analyze_series([9.0, 7.0, 5.0, 3.0, 1.0])
        assert result["trend"] == "decreasing"
        assert result["slope"] == pytest.approx(-2.0)

    def test_case_04(self):
        result = self.system.analyze_series([5.0, 5.0, 5.0, 5.0, 5.0])
        assert result["trend"] == "stable"
        assert result["slope"] == pytest.approx(0.0)

    def test_case_05(self):
        period = 4
        data = [
            10.0 + 5.0 * math.sin(2.0 * math.pi * t / period)
            for t in range(32)
        ]
        result = self.system.analyze_series(data)
        assert result["seasonal"] is True
        assert result["period"] == period

    def test_case_06(self):
        result = self.system.analyze_series([1.0, 2.0, 1.5, 2.5, 1.8, 2.2])
        assert result["seasonal"] is False
        assert result["period"] is None

    # ---- ARIMA（自回归 + 差分）-----------------------------------------
    def test_case_07(self):
        data = [0.5 * (i % 1) for i in range(1)] + [
            0.0,
        ]
        # AR(1): y_t = 0.8 * y_{t-1} + 1.0
        series = [1.0]
        for _ in range(30):
            series.append(0.8 * series[-1] + 1.0)
        forecast = self.system.arima(series, order=(1, 0, 0), steps=1)
        assert len(forecast) == 1
        assert forecast[0] == pytest.approx(5.0, abs=0.5)

    def test_case_08(self):
        linear = [2.0 * t + 1.0 for t in range(20)]
        forecast = self.system.arima(linear, order=(1, 1, 0), steps=3)
        assert len(forecast) == 3
        assert forecast[0] == pytest.approx(41.0, abs=0.5)
        assert forecast[2] == pytest.approx(45.0, abs=1.0)

    def test_case_09(self):
        constant = [7.0] * 10
        forecast = self.system.arima(constant, order=(1, 1, 1), steps=2)
        assert len(forecast) == 2
        assert forecast[0] == pytest.approx(7.0, abs=1e-6)
        assert forecast[1] == pytest.approx(7.0, abs=1e-6)

    # ---- 移动平均（简单 + 加权）----------------------------------------
    def test_case_10(self):
        result = self.system.moving_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        assert result == pytest.approx([2.0, 3.0, 4.0])

    def test_case_11(self):
        result = self.system.moving_average(
            [1.0, 2.0, 3.0], window=3, weighted=True
        )
        assert result == pytest.approx([(1 * 1 + 2 * 2 + 3 * 3) / 6.0])
        custom = self.system.moving_average(
            [1.0, 2.0, 3.0], window=3, weights=[1.0, 2.0, 1.0]
        )
        assert custom == pytest.approx([2.0])

    # ---- 指数平滑（简单 + 双指数）--------------------------------------
    def test_case_12(self):
        result = self.system.exponential_smoothing(
            [3.0, 5.0, 9.0, 20.0, 12.0], alpha=0.5, steps=2
        )
        assert result["method"] == "ses"
        assert len(result["smoothed"]) == 5
        assert result["smoothed"][1] == pytest.approx(4.0)
        assert result["forecast"] == pytest.approx(
            [result["smoothed"][-1], result["smoothed"][-1]]
        )

    def test_case_13(self):
        data = [10.0 + 2.0 * t for t in range(15)]
        result = self.system.exponential_smoothing(data, alpha=0.8, beta=0.3, steps=3)
        assert result["method"] == "double"
        assert len(result["forecast"]) == 3
        assert result["forecast"][0] == pytest.approx(40.0, abs=1.5)
        assert result["forecast"][2] == pytest.approx(44.0, abs=3.0)

    # ---- 预测评估（MAE + RMSE）----------------------------------------
    def test_case_14(self):
        result = self.system.forecast_evaluation(
            [3.0, 5.0, 7.0], [2.0, 5.0, 10.0]
        )
        assert result["mae"] == pytest.approx((1 + 0 + 3) / 3.0)

    def test_case_15(self):
        result = self.system.forecast_evaluation(
            [3.0, 5.0, 7.0], [2.0, 5.0, 10.0]
        )
        assert result["rmse"] == pytest.approx(math.sqrt((1 + 0 + 9) / 3.0))
        assert result["mse"] == pytest.approx(10.0 / 3.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
