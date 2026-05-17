"""Bullwhip effect analysis via SciPy + NumPy (Blueprint §4.3.3).

Stage 4 implementation.

Analysis pipeline (Chen et al. 1999 + simulation):
    1. AR(1) estimation:
           D_t = ρ · D_{t-1} + ε_t
       Estimated via OLS: regress D[1:] on D[:-1] using scipy.stats.linregress.

    2. Echelon simulation (order-up-to / base-stock policy):
       For each echelon k = 0, 1, …, num_echelons−1:
         • Demand seen by echelon k is the order stream from echelon k−1.
         • Each echelon uses a moving-average forecast with window = forecast_window.
         • Order_t = max(0,  D_t  +  (F_t − F_{t-1}) · lead_time)
           where F_t is the MA(forecast_window) estimate at period t.
         • Amplification ratio at echelon k (k > 0):
               r[k] = Var(orders_k) / Var(orders_{k-1})

    3. Spectral radius:
       For an AR(1) process with lead time L and forecast window p, the
       dominant eigenvalue of the order-update transfer matrix is approximated
       by the maximum of the normalised power spectral density of the simulated
       order series (via numpy.fft.rfft).  When the series is too short for
       spectral analysis, fall back to |ρ|.

    References:
        Chen, F., Drezner, Z., Ryan, J. K., & Simchi-Levi, D. (1999).
        Quantifying the bullwhip effect in a simple supply chain: The impact
        of forecasting, lead times, and information. *Management Science*.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _simulate_echelon(
    demand_in: np.ndarray, lead_time: int, forecast_window: int
) -> np.ndarray:
    """Simulate base-stock orders for one echelon.

    Order_t = max(0, D_t + (F_t - F_{t-1}) * lead_time)
    where F_t is a moving-average forecast of demand_in up to period t.
    """
    T = len(demand_in)
    orders = np.zeros(T)
    forecasts = np.zeros(T)

    for t in range(T):
        window_start = max(0, t - forecast_window + 1)
        forecasts[t] = float(np.mean(demand_in[window_start : t + 1]))

        if t == 0:
            orders[t] = max(0.0, float(demand_in[t]))
        else:
            adjustment = (forecasts[t] - forecasts[t - 1]) * lead_time
            orders[t] = max(0.0, float(demand_in[t]) + adjustment)

    return orders


def analyze_bullwhip(
    demand_series: list[float],
    lead_time: int,
    forecast_window: int,
    num_echelons: int,
) -> dict:
    """Compute amplification ratios, AR(1) rho, and spectral radius.

    Args:
        demand_series: Observed end-customer demand (time-ordered).
        lead_time: Replenishment lead time (periods), same for each echelon.
        forecast_window: Moving-average window size (periods).
        num_echelons: Number of echelons in the supply chain.

    Returns:
        {amplification_ratios, ar1_rho, spectral_radius, simulation_plot_data}
    """
    demand = np.array(demand_series, dtype=float)
    T = len(demand)

    # --- 1. AR(1) estimation ---
    ar1_rho = 0.0
    if T >= 3:
        slope, _intercept, _r, _p, _se = stats.linregress(demand[:-1], demand[1:])
        ar1_rho = float(np.clip(slope, -1.0, 1.0))  # clip to stable range

    # --- 2. Echelon simulation ---
    echelon_orders: list[np.ndarray] = [demand]
    for _ in range(max(1, num_echelons) - 1):
        prev = echelon_orders[-1]
        next_orders = _simulate_echelon(prev, lead_time, max(1, forecast_window))
        echelon_orders.append(next_orders)

    # Amplification ratios: Var(orders_k) / Var(orders_{k-1})
    amplification_ratios: list[float] = []
    for k in range(1, len(echelon_orders)):
        var_prev = float(np.var(echelon_orders[k - 1]))
        var_curr = float(np.var(echelon_orders[k]))
        ratio = var_curr / var_prev if var_prev > 1e-12 else 1.0
        amplification_ratios.append(round(ratio, 6))

    # --- 3. Spectral radius ---
    if T >= 16:
        # Use the power spectrum of the last echelon's order series
        spectrum = np.abs(np.fft.rfft(echelon_orders[-1])) ** 2
        total_power = float(np.sum(spectrum))
        spectral_radius = (
            float(np.max(spectrum)) / total_power if total_power > 1e-12 else abs(ar1_rho)
        )
    else:
        spectral_radius = abs(ar1_rho)

    # --- 4. Simulation plot data (one point per period per echelon) ---
    simulation_plot_data = [
        {
            "echelon": k,
            "period": t,
            "demand": round(float(echelon_orders[k][t]), 4),
        }
        for k in range(len(echelon_orders))
        for t in range(T)
    ]

    return {
        "amplification_ratios": amplification_ratios,
        "ar1_rho": round(ar1_rho, 6),
        "spectral_radius": round(spectral_radius, 6),
        "simulation_plot_data": simulation_plot_data,
    }
