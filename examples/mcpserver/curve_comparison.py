"""MCPServer Curve Comparison Server

Provides tools for smoothing data curves and comparing them using
Pearson correlation and Dynamic Time Warping (DTW) distance.
"""

import statistics

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Curve Comparison Server")


def _moving_average(values: list[float], window: int) -> list[float]:
    """Compute moving average with the given window size."""
    return [sum(values[i : i + window]) / window for i in range(len(values) - window + 1)]


def _dtw_distance(seq1: list[float], seq2: list[float]) -> float:
    """Compute DTW distance between two sequences using dynamic programming."""
    n, m = len(seq1), len(seq2)
    dtw = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
    return dtw[n][m]


@mcp.tool()
def smooth_curve(values: list[float], window: int = 7) -> list[float]:
    """Apply moving average smoothing to a list of values.

    Args:
        values: The input data series to smooth.
        window: The size of the moving average window (must be >= 1).
            Defaults to 7, suitable for weekly smoothing of daily data.

    Returns:
        A smoothed list of floats with length len(values) - window + 1.
    """
    if window < 1:
        raise ValueError("Window size must be at least 1")
    if len(values) < window:
        raise ValueError(f"Values list must have at least {window} elements")
    return _moving_average(values, window)


@mcp.tool()
def compare_curves(curve1: list[float], curve2: list[float]) -> dict[str, float]:
    """Compare two curves using Pearson correlation and DTW distance.

    Truncates both curves to the shorter length before comparison.

    Args:
        curve1: First data series.
        curve2: Second data series.

    Returns:
        A dict with "pearson_correlation" and "dtw_distance" keys.
        Pearson correlation ranges from -1 (inverse) to 1 (identical trend).
        DTW distance is non-negative; lower values mean more similar shapes.
    """
    min_len = min(len(curve1), len(curve2))
    if min_len < 2:
        raise ValueError("Both curves must have at least 2 elements for comparison")
    c1 = curve1[:min_len]
    c2 = curve2[:min_len]
    corr = statistics.correlation(c1, c2)
    dtw_dist = _dtw_distance(c1, c2)
    return {"pearson_correlation": corr, "dtw_distance": dtw_dist}
