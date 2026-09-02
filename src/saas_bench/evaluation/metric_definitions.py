"""单次实验指标的固定计算口径。"""

from __future__ import annotations

from collections.abc import Sequence


TERMINAL_WINDOW_DAYS = 28


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """分母为零时返回空值，不把“无法计算”伪装成零。"""
    if denominator == 0:
        return None
    return numerator / denominator


def calculate_drawdowns(
    cash_values: Sequence[float],
    *,
    initial_cash: float,
) -> tuple[float, float]:
    """计算相对历史峰值的最大绝对回撤和最大回撤率。"""
    peak = initial_cash
    max_absolute = 0.0
    max_rate = 0.0
    for cash in cash_values:
        peak = max(peak, cash)
        absolute = peak - cash
        max_absolute = max(max_absolute, absolute)
        if peak > 0:
            max_rate = max(max_rate, absolute / peak)
    return max_absolute, max_rate


def terminal_window_start(end_day: int) -> int:
    """返回包含终止日在内的末 28 个模拟日的起始日。"""
    return max(0, end_day - TERMINAL_WINDOW_DAYS + 1)

