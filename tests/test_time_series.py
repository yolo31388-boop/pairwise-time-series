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

    # ---- 序列分析：趋势检测 ----
    def test_case_01(self):
        # 上升趋势检测
        data = [10.0 + 2.0 * i for i in range(30)]
        result = self.system.analyze_series(data)
        assert result["trend"] == "increasing"
        assert result["slope"] == pytest.approx(2.0)

    def test_case_02(self):
        # 下降趋势检测
        data = [100.0 - 3.0 * i for i in range(30)]
        result = self.system.analyze_series(data)
        assert result["trend"] == "decreasing"
        assert result["slope"] == pytest.approx(-3.0)

    def test_case_03(self):
        # 平稳序列检测
        data = [50.0 + (1 if i % 2 == 0 else -1) * 0.01 for i in range(40)]
        result = self.system.analyze_series(data)
        assert result["trend"] == "stationary"

    # ---- 序列分析：季节性检测 ----
    def test_case_04(self):
        # 周期为 7 的季节性序列（模拟每周活跃周期）
        data = [50.0 + 10.0 * math.sin(2 * math.pi * i / 7) for i in range(56)]
        result = self.system.analyze_series(data)
        assert result["seasonal"] is True
        assert result["period"] == 7

    def test_case_05(self):
        # 线性趋势序列不应误报季节性
        data = [5.0 * i for i in range(40)]
        result = self.system.analyze_series(data)
        assert result["seasonal"] is False
        assert result["period"] is None

    # ---- ARIMA：自回归 + 差分 ----
    def test_case_06(self):
        # ARIMA 返回指定步数的预测
        data = [20.0 + 1.5 * i for i in range(30)]
        forecasts = self.system.arima(data, p=2, d=1, steps=5)
        assert len(forecasts) == 5
        assert all(isinstance(f, float) for f in forecasts)

    def test_case_07(self):
        # 差分使 ARIMA 能跟踪线性趋势：一阶差分后近似常数
        data = [10.0 + 2.0 * i for i in range(40)]
        forecasts = self.system.arima(data, p=1, d=1, steps=3)
        expected = [data[-1] + 2.0 * k for k in range(1, 4)]
        for f, e in zip(forecasts, expected):
            assert f == pytest.approx(e, abs=1.0)

    def test_case_08(self):
        # 自回归能拟合 AR(1) 过程并给出合理预测
        coef = 0.8
        data = [0.0]
        value = 0.0
        for i in range(1, 60):
            value = coef * value + math.sin(i)  # 确定性驱动的 AR(1)
            data.append(value)
        forecasts = self.system.arima(data, p=2, d=0, steps=4)
        assert len(forecasts) == 4
        assert all(math.isfinite(f) for f in forecasts)

    # ---- 移动平均：简单 + 加权 ----
    def test_case_09(self):
        # 简单移动平均 = 最近 window 个值的算术平均
        data = [1.0, 2.0, 3.0, 4.0, 10.0]
        forecast = self.system.moving_average(data, window=3, method="simple")
        assert forecast == pytest.approx((3.0 + 4.0 + 10.0) / 3)

    def test_case_10(self):
        # 加权移动平均：默认线性权重，近期权重更大
        data = [1.0, 2.0, 3.0, 4.0, 10.0]
        forecast = self.system.moving_average(data, window=3, method="weighted")
        expected = (1 * 3.0 + 2 * 4.0 + 3 * 10.0) / 6
        assert forecast == pytest.approx(expected)
        # 自定义权重
        custom = self.system.moving_average(data, window=3, method="weighted",
                                            weights=[1.0, 1.0, 2.0])
        assert custom == pytest.approx((3.0 + 4.0 + 20.0) / 4)

    # ---- 指数平滑：简单 + 双指数 ----
    def test_case_11(self):
        # 简单指数平滑：常数序列预测等于常数
        data = [7.5] * 20
        forecast = self.system.exponential_smoothing(data, alpha=0.3)
        assert forecast == pytest.approx(7.5)
        # 介于序列最小最大值之间
        data2 = [float(i % 5) for i in range(30)]
        f2 = self.system.exponential_smoothing(data2, alpha=0.5)
        assert min(data2) <= f2 <= max(data2)

    def test_case_12(self):
        # 双指数平滑能跟踪线性趋势，预测接近下一个真实值
        data = [5.0 + 2.0 * i for i in range(30)]
        forecast = self.system.exponential_smoothing(data, alpha=0.5, beta=0.5)
        expected_next = 5.0 + 2.0 * 30
        assert forecast == pytest.approx(expected_next, abs=2.0)

    # ---- 预测评估：MAE + RMSE ----
    def test_case_13(self):
        # MAE 计算正确
        actual = [10.0, 20.0, 30.0, 40.0]
        predicted = [12.0, 18.0, 33.0, 36.0]
        result = self.system.forecast_evaluation(actual, predicted)
        assert result["mae"] == pytest.approx((2 + 2 + 3 + 4) / 4)

    def test_case_14(self):
        # RMSE 计算正确，且 RMSE >= MAE
        actual = [10.0, 20.0, 30.0, 40.0]
        predicted = [12.0, 18.0, 33.0, 36.0]
        result = self.system.forecast_evaluation(actual, predicted)
        expected_rmse = math.sqrt((4 + 4 + 9 + 16) / 4)
        assert result["rmse"] == pytest.approx(expected_rmse)
        assert result["rmse"] >= result["mae"]

    # ---- 端到端：预测误差必须远低于 50% ----
    def test_case_15(self):
        # 带趋势 + 季节性的游戏在线人数序列，评估预测质量
        data = [100.0 + 1.0 * i + 20.0 * math.sin(2 * math.pi * i / 7)
                for i in range(56)]
        analysis = self.system.analyze_series(data)
        assert analysis["trend"] == "increasing"
        assert analysis["seasonal"] is True

        forecasts = self.system.arima(data, p=7, d=1, steps=7)
        actual = [100.0 + 1.0 * i + 20.0 * math.sin(2 * math.pi * i / 7)
                  for i in range(56, 63)]
        evaluation = self.system.forecast_evaluation(actual, forecasts)
        mean_actual = sum(actual) / len(actual)
        assert evaluation["mae"] / mean_actual < 0.5
        assert evaluation["rmse"] / mean_actual < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
