"""Time-series forecasting with TimesFM 2.5."""

import requests
import json

BASE = "http://localhost:8080"


def forecast(series: list[float], horizon: int = 12) -> dict:
    """Forecast future values for a time series."""
    resp = requests.post(f"{BASE}/v1/forecast", json={
        "time_series": [series],
        "horizon": horizon,
    })
    resp.raise_for_status()
    return resp.json()


# --- Example 1: Linear trend ---
data = list(range(1, 51))  # [1, 2, 3, ..., 50]
result = forecast(data, horizon=5)
print("Linear trend forecast:")
print(f"  Input:    ...{data[-5:]}")
print(f"  Forecast: {[round(x, 1) for x in result['point_forecast'][0]]}")
print()

# --- Example 2: Seasonal pattern ---
import math
seasonal = [10 + 5 * math.sin(2 * math.pi * i / 12) for i in range(60)]
result = forecast(seasonal, horizon=12)
print("Seasonal forecast (12 months ahead):")
print(f"  Forecast: {[round(x, 1) for x in result['point_forecast'][0]]}")
print()

# --- Example 3: With confidence intervals ---
sales = [100, 120, 115, 130, 125, 140, 135, 150, 145, 160, 155, 170]
result = forecast(sales, horizon=6)
point = result["point_forecast"][0]
quantiles = result["quantile_forecast"][0]

print("Sales forecast with confidence intervals:")
for i, (p, q) in enumerate(zip(point, quantiles)):
    low = round(q[1], 1)   # 10th percentile
    mid = round(p, 1)      # median
    high = round(q[9], 1)  # 90th percentile
    print(f"  Step {i+1}: {mid:>7.1f}  (90% CI: {low:.1f} – {high:.1f})")
print()

# --- Example 4: Batch forecast ---
result = requests.post(f"{BASE}/v1/forecast", json={
    "time_series": [
        [1, 2, 3, 4, 5, 6, 7, 8],       # linear
        [10, 5, 10, 5, 10, 5, 10, 5],    # oscillating
        [1, 1, 2, 3, 5, 8, 13, 21, 34],  # fibonacci-like
    ],
    "horizon": 4,
}).json()

print("Batch forecast (3 series, 4 steps each):")
for i, pf in enumerate(result["point_forecast"]):
    print(f"  Series {i+1}: {[round(x, 1) for x in pf]}")
