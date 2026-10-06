"""
游戏内时间序列分析与预测系统 - 核心模块

纯 Python 标准库实现，提供：
- analyze_series: 序列分析（趋势检测 / 季节性检测）
- arima: ARIMA 模型（自回归 + 差分 + 移动平均项）
- moving_average: 简单移动平均 / 加权移动平均
- exponential_smoothing: 简单指数平滑 / 双指数平滑（Holt 线性趋势）
- forecast_evaluation: 预测质量评估（MAE / RMSE / MAPE）
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math


@dataclass
class Config:
    alpha: float = 0.3
    beta: float = 0.1
    window: int = 3
    arima_order: Tuple[int, int, int] = (1, 1, 0)


class TimeSeries:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state: Dict[str, object] = {}
        self._history: List[float] = []

    def update(self, dt: float):
        self._history.append(float(dt))

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 序列分析
    # ------------------------------------------------------------------
    def analyze_series(self, data: Sequence[float]) -> Dict:
        """分析时间序列：基本统计量、趋势检测、季节性检测。"""
        values = [float(x) for x in data]
        n = len(values)
        if n == 0:
            raise ValueError("data 不能为空")

        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        stddev = math.sqrt(variance)

        # 最小二乘线性回归 y = a + b*t，斜率用于趋势检测
        mean_t = (n - 1) / 2.0
        var_t = sum((t - mean_t) ** 2 for t in range(n))
        if var_t > 0:
            slope = sum((t - mean_t) * (values[t] - mean) for t in range(n)) / var_t
            intercept = mean - slope * mean_t
        else:
            slope = 0.0
            intercept = mean

        data_range = max(values) - min(values)
        threshold = 0.01 * max(data_range, 1.0)
        if slope > threshold:
            trend = "increasing"
        elif slope < -threshold:
            trend = "decreasing"
        else:
            trend = "stable"

        period, seasonal_strength = self._detect_seasonality(values, mean, variance)
        seasonal = period is not None and seasonal_strength >= 0.5

        return {
            "length": n,
            "mean": mean,
            "variance": variance,
            "stddev": stddev,
            "min": min(values),
            "max": max(values),
            "slope": slope,
            "intercept": intercept,
            "trend": trend,
            "seasonal": seasonal,
            "seasonality": seasonal,
            "period": period,
            "seasonal_strength": seasonal_strength,
        }

    @staticmethod
    def _detect_seasonality(
        values: List[float], mean: float, variance: float
    ) -> Tuple[Optional[int], float]:
        """通过自相关函数（ACF）寻找显著周期。"""
        n = len(values)
        if variance == 0 or n < 4:
            return None, 0.0

        denom = sum((x - mean) ** 2 for x in values)
        best_lag: Optional[int] = None
        best_acf = 0.0
        for lag in range(2, n // 2 + 1):
            acf = (
                sum(
                    (values[t] - mean) * (values[t + lag] - mean)
                    for t in range(n - lag)
                )
                / denom
            )
            if acf > best_acf:
                best_acf = acf
                best_lag = lag
        if best_acf >= 0.5:
            return best_lag, best_acf
        return None, best_acf

    # ------------------------------------------------------------------
    # ARIMA
    # ------------------------------------------------------------------
    def arima(
        self,
        data: Sequence[float],
        order: Tuple[int, int, int] = (1, 1, 0),
        steps: int = 1,
    ) -> List[float]:
        """ARIMA(p, d, q) 预测，返回未来 steps 个点的预测值。"""
        values = [float(x) for x in data]
        p, d, q = order
        if len(values) <= d:
            raise ValueError("数据长度不足以进行指定阶数的差分")
        if steps < 1:
            return []

        # 记录每一阶差分序列，积分还原时以上一阶序列末值为基底
        levels = [values]
        current = values
        for _ in range(d):
            current = self._difference(current, 1)
            levels.append(current)

        forecast_diff = self._forecast_stationary(current, p, q, steps)

        for level in range(d, 0, -1):
            base = levels[level - 1][-1]
            restored = []
            running = base
            for increment in forecast_diff:
                running += increment
                restored.append(running)
            forecast_diff = restored
        return forecast_diff

    @staticmethod
    def _difference(values: List[float], d: int) -> List[float]:
        result = values
        for _ in range(d):
            result = [result[i] - result[i - 1] for i in range(1, len(result))]
        return result

    def _forecast_stationary(
        self, values: List[float], p: int, q: int, steps: int
    ) -> List[float]:
        """对平稳序列拟合 AR(p) + MA(q) 并向前预测。"""
        n = len(values)

        # ---- AR(p) 最小二乘：z_t = c + phi_1*z_{t-1} + ... + phi_p*z_{t-p}
        ar_coeffs: List[float] = [0.0] * (p + 1)  # 首位为截距 c
        if p > 0 and n > p:
            rows: List[List[float]] = []
            targets: List[float] = []
            for t in range(p, n):
                rows.append([1.0] + values[t - p : t][::-1])
                targets.append(values[t])
            ar_coeffs = self._least_squares(rows, targets, p + 1)
        elif p == 0 and n > 0:
            ar_coeffs = [sum(values) / n]

        c = ar_coeffs[0]
        phis = ar_coeffs[1:]

        def ar_hat(history: List[float], t: int) -> float:
            total = c
            for i in range(p):
                lag = i + 1
                if t - lag >= 0:
                    total += phis[i] * history[t - lag]
            return total

        # ---- 残差与 MA(q)：r_t = theta_1*r_{t-1} + ... + theta_q*r_{t-q}
        residuals: List[float] = []
        if p > 0:
            for t in range(p, n):
                residuals.append(values[t] - ar_hat(values, t))
        thetas: List[float] = [0.0] * q
        if q > 0 and len(residuals) > q:
            ma_rows = []
            ma_targets = []
            for t in range(q, len(residuals)):
                ma_rows.append(residuals[t - q : t][::-1])
                ma_targets.append(residuals[t])
            thetas = self._least_squares(ma_rows, ma_targets, q)

        # ---- 向前递推预测
        future = list(values)
        future_resid = list(residuals)
        forecasts: List[float] = []
        for _ in range(steps):
            t = len(future)
            value = ar_hat(future, t)

            resid_forecast = 0.0
            for j in range(q):
                lag = j + 1
                idx = len(future_resid) - lag
                if idx >= 0:
                    resid_forecast += thetas[j] * future_resid[idx]
            value += resid_forecast

            forecasts.append(value)
            future.append(value)
            future_resid.append(resid_forecast)
        return forecasts

    @staticmethod
    def _least_squares(
        rows: List[List[float]], targets: List[float], k: int
    ) -> List[float]:
        """带截距的最小二乘（正规方程 + 部分主元高斯消元）。"""
        if len(rows) < k:
            return [0.0] * k
        xtx = [[0.0] * k for _ in range(k)]
        xty = [0.0] * k
        for row, y in zip(rows, targets):
            for i in range(k):
                xty[i] += row[i] * y
                for j in range(k):
                    xtx[i][j] += row[i] * row[j]
        # 轻微岭项，避免常数序列导致奇异矩阵
        for i in range(k):
            xtx[i][i] += 1e-10
        return TimeSeries._solve(xtx, xty)

    @staticmethod
    def _solve(a: List[List[float]], b: List[float]) -> List[float]:
        n = len(b)
        m = [row[:] + [b[i]] for i, row in enumerate(a)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
            if abs(m[pivot][col]) < 1e-14:
                continue
            m[col], m[pivot] = m[pivot], m[col]
            pivot_val = m[col][col]
            for r in range(n):
                if r == col:
                    continue
                factor = m[r][col] / pivot_val
                for j in range(col, n + 1):
                    m[r][j] -= factor * m[col][j]
        return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-14 else 0.0 for i in range(n)]

    # ------------------------------------------------------------------
    # 移动平均
    # ------------------------------------------------------------------
    def moving_average(
        self,
        data: Sequence[float],
        window: int = 3,
        weighted: bool = False,
        weights: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """简单移动平均（SMA）或加权移动平均（WMA）。"""
        values = [float(x) for x in data]
        if window < 1:
            raise ValueError("window 必须 >= 1")
        if window > len(values):
            raise ValueError("window 不能大于数据长度")

        if weights is not None:
            w = [float(x) for x in weights]
            if len(w) != window:
                raise ValueError("weights 长度必须等于 window")
        elif weighted:
            w = [float(i + 1) for i in range(window)]  # 越新的点权重越大
        else:
            w = [1.0] * window
        total_weight = sum(w)

        result = []
        for start in range(len(values) - window + 1):
            chunk = values[start : start + window]
            result.append(sum(a * b for a, b in zip(chunk, w)) / total_weight)
        return result

    # ------------------------------------------------------------------
    # 指数平滑
    # ------------------------------------------------------------------
    def exponential_smoothing(
        self,
        data: Sequence[float],
        alpha: float = 0.3,
        beta: Optional[float] = None,
        steps: int = 1,
    ) -> Dict:
        """简单指数平滑（beta=None）或双指数平滑 / Holt 线性趋势（beta 给定）。"""
        values = [float(x) for x in data]
        if not values:
            raise ValueError("data 不能为空")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha 必须在 [0, 1] 区间")

        if beta is None:
            smoothed = [values[0]]
            for t in range(1, len(values)):
                smoothed.append(
                    alpha * values[t] + (1.0 - alpha) * smoothed[t - 1]
                )
            forecast = [smoothed[-1]] * steps
            return {"smoothed": smoothed, "forecast": forecast, "method": "ses"}

        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta 必须在 [0, 1] 区间")
        if len(values) < 2:
            raise ValueError("双指数平滑至少需要 2 个数据点")

        level = values[0]
        trend = values[1] - values[0]
        levels = [level]
        trends: List[float] = []
        for t in range(1, len(values)):
            prev_level = level
            level = alpha * values[t] + (1.0 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1.0 - beta) * trend
            levels.append(level)
            trends.append(trend)
        forecast = [level + h * trend for h in range(1, steps + 1)]
        return {
            "smoothed": levels,
            "forecast": forecast,
            "level": level,
            "trend": trend,
            "method": "double",
        }

    # ------------------------------------------------------------------
    # 预测评估
    # ------------------------------------------------------------------
    def forecast_evaluation(
        self, actual: Sequence[float], predicted: Sequence[float]
    ) -> Dict:
        """计算 MAE、RMSE、MAPE 等预测误差指标。"""
        a = [float(x) for x in actual]
        p = [float(x) for x in predicted]
        if not a or not p:
            raise ValueError("actual 和 predicted 不能为空")
        size = min(len(a), len(p))
        errors = [a[i] - p[i] for i in range(size)]
        mae = sum(abs(e) for e in errors) / size
        mse = sum(e ** 2 for e in errors) / size
        rmse = math.sqrt(mse)
        nonzero = [(x, e) for x, e in zip(a[:size], errors) if x != 0]
        mape = (
            sum(abs(e) / abs(x) for x, e in nonzero) / len(nonzero)
            if nonzero
            else 0.0
        )
        return {
            "mae": mae,
            "rmse": rmse,
            "mse": mse,
            "mape": mape,
            "samples": size,
        }
