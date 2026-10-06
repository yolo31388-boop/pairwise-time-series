"""
游戏内时间序列分析与预测系统 - 核心模块

提供：
- analyze_series():       序列分析（趋势检测 + 季节性检测）
- arima():                ARIMA 预测（自回归 AR + 差分 I）
- moving_average():       移动平均（简单移动平均 + 加权移动平均）
- exponential_smoothing():指数平滑（简单指数平滑 + 双指数平滑）
- forecast_evaluation():  预测评估（MAE + RMSE）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    pass


def _validate_series(data: List[float], min_len: int = 2) -> List[float]:
    values = [float(x) for x in data]
    if len(values) < min_len:
        raise ValueError(f"序列长度至少为 {min_len}，当前为 {len(values)}")
    return values


def _linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """最小二乘拟合 y = slope * x + intercept，返回 (slope, intercept)。"""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, mean_y
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """高斯消元（带部分主元）求解 Ax = b。"""
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] = 1e-12
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        s = aug[row][n] - sum(aug[row][k] * x[k] for k in range(row + 1, n))
        x[row] = s / aug[row][row]
    return x


class TimeSeries:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []

    def update(self, dt: float):
        self._history.append(float(dt))

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 序列分析：趋势检测 + 季节性检测
    # ------------------------------------------------------------------
    def analyze_series(self, data: List[float]) -> Dict:
        """分析时间序列，返回趋势与季节性检测结果。

        返回字典：
            trend:      'increasing' / 'decreasing' / 'stationary'
            slope:      线性回归斜率
            seasonal:   是否存在显著季节性
            period:     检测到的季节周期（无季节性时为 None）
            mean / std: 序列均值与标准差
        """
        values = _validate_series(data)
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)

        # ---- 趋势检测：线性回归斜率 + 残差显著性 ----
        xs = list(range(n))
        slope, intercept = _linear_regression(xs, values)
        residuals = [y - (slope * x + intercept) for x, y in zip(xs, values)]
        residual_std = math.sqrt(sum(r * r for r in residuals) / n)
        total_change = slope * (n - 1)
        threshold = max(2.0 * residual_std, 1e-9)
        if total_change > threshold:
            trend = "increasing"
        elif total_change < -threshold:
            trend = "decreasing"
        else:
            trend = "stationary"

        # ---- 季节性检测：去趋势后的自相关峰值 ----
        detrended = residuals
        dmean = sum(detrended) / n
        centered = [v - dmean for v in detrended]
        denom = sum(c * c for c in centered)
        best_lag, best_acf = None, 0.0
        if denom > 1e-12:
            for lag in range(2, n // 2 + 1):
                num = sum(centered[i] * centered[i - lag] for i in range(lag, n))
                acf = num / denom
                if acf > best_acf:
                    best_acf, best_lag = acf, lag
        seasonal = best_lag is not None and best_acf >= 0.5
        period = best_lag if seasonal else None

        return {
            "trend": trend,
            "slope": slope,
            "seasonal": seasonal,
            "period": period,
            "mean": mean,
            "std": std,
        }

    # ------------------------------------------------------------------
    # ARIMA：自回归(AR) + 差分(I)
    # ------------------------------------------------------------------
    def arima(self, data: List[float], p: int = 2, d: int = 1,
              steps: int = 1) -> List[float]:
        """ARIMA(p, d, 0) 预测未来 steps 个值。

        - 差分(I)：对序列做 d 阶差分使其平稳
        - 自回归(AR)：在差分序列上用最小二乘拟合 AR(p) 模型
        - 预测后逐阶反差分还原到原始尺度
        """
        values = _validate_series(data, min_len=max(3, p + d + 1))
        if p < 1 or d < 0 or steps < 1:
            raise ValueError("要求 p >= 1, d >= 0, steps >= 1")

        # ---- 差分（I 部分），记录每一阶差分序列的末值用于还原 ----
        levels = [values]
        for _ in range(d):
            prev = levels[-1]
            levels.append([prev[i] - prev[i - 1] for i in range(1, len(prev))])
        diffed = levels[-1]

        # ---- 自回归（AR 部分）：最小二乘拟合 AR(p) ----
        p_eff = min(p, len(diffed) - 1)
        train = diffed
        mean_d = sum(train) / len(train)
        centered = [v - mean_d for v in train]
        rows = len(centered) - p_eff
        if rows > 0:
            ata = [[0.0] * p_eff for _ in range(p_eff)]
            atb = [0.0] * p_eff
            for t in range(p_eff, len(centered)):
                feats = [centered[t - j - 1] for j in range(p_eff)]
                target = centered[t]
                for i in range(p_eff):
                    atb[i] += feats[i] * target
                    for j in range(p_eff):
                        ata[i][j] += feats[i] * feats[j]
            for i in range(p_eff):
                ata[i][i] += 1e-8  # 岭正则，保证可解
            coeffs = _solve_linear_system(ata, atb)
        else:
            coeffs = [0.0] * p_eff

        # ---- 在差分尺度上滚动预测 ----
        history = list(centered)
        diff_forecasts = []
        for _ in range(steps):
            pred = sum(coeffs[j] * history[-j - 1] for j in range(p_eff))
            history.append(pred)
            diff_forecasts.append(pred + mean_d)

        # ---- 反差分：从最深一层逐阶累加还原 ----
        forecasts = diff_forecasts
        for level in range(d - 1, -1, -1):
            last = levels[level][-1]
            restored = []
            acc = last
            for f in forecasts:
                acc += f
                restored.append(acc)
            forecasts = restored
        return forecasts

    # ------------------------------------------------------------------
    # 移动平均：简单移动平均(SMA) + 加权移动平均(WMA)
    # ------------------------------------------------------------------
    def moving_average(self, data: List[float], window: int = 3,
                       method: str = "simple",
                       weights: Optional[List[float]] = None) -> float:
        """用移动平均预测下一个值。

        method='simple'   简单移动平均：最近 window 个值的算术平均
        method='weighted' 加权移动平均：默认线性权重 1..window（近期权重更大），
                          也可通过 weights 自定义
        """
        values = _validate_series(data)
        if window < 1:
            raise ValueError("window 必须 >= 1")
        window = min(window, len(values))
        segment = values[-window:]

        if method == "simple":
            return sum(segment) / window
        if method == "weighted":
            if weights is None:
                weights = list(range(1, window + 1))
            if len(weights) != window:
                raise ValueError("weights 长度必须等于 window")
            total = sum(weights)
            if total == 0:
                raise ValueError("weights 之和不能为 0")
            return sum(w * v for w, v in zip(weights, segment)) / total
        raise ValueError(f"未知的移动平均方法: {method}")

    # ------------------------------------------------------------------
    # 指数平滑：简单指数平滑(SES) + 双指数平滑(Holt)
    # ------------------------------------------------------------------
    def exponential_smoothing(self, data: List[float], alpha: float = 0.5,
                              beta: Optional[float] = None) -> float:
        """指数平滑预测下一个值。

        beta=None   简单指数平滑(SES)：适用于无趋势序列
        beta 给定   双指数平滑(Holt)：同时平滑水平项与趋势项
        """
        values = _validate_series(data)
        if not 0 < alpha <= 1:
            raise ValueError("alpha 必须在 (0, 1] 区间")

        if beta is None:
            # 简单指数平滑
            level = values[0]
            for v in values[1:]:
                level = alpha * v + (1 - alpha) * level
            return level

        # 双指数平滑（Holt 线性趋势）
        if not 0 < beta <= 1:
            raise ValueError("beta 必须在 (0, 1] 区间")
        level = values[0]
        trend = values[1] - values[0]
        for v in values[1:]:
            prev_level = level
            level = alpha * v + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        return level + trend

    # ------------------------------------------------------------------
    # 预测评估：MAE + RMSE
    # ------------------------------------------------------------------
    def forecast_evaluation(self, actual: List[float],
                            predicted: List[float]) -> Dict[str, float]:
        """评估预测质量，返回 MAE（平均绝对误差）与 RMSE（均方根误差）。"""
        actual = _validate_series(actual, min_len=1)
        predicted = _validate_series(predicted, min_len=1)
        if len(actual) != len(predicted):
            raise ValueError("actual 与 predicted 长度必须一致")
        n = len(actual)
        mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
        rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n)
        return {"mae": mae, "rmse": rmse}
