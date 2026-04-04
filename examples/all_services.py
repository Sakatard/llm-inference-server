"""Demo: use all three services in one workflow.

Scenario: Transcribe a meeting recording, forecast next quarter's metrics,
and generate a summary combining both.
"""

import requests

BASE = "http://localhost:8080"


# 1. Forecast quarterly revenue
quarterly_revenue = [
    120, 135, 128, 142,  # 2023
    150, 163, 155, 171,  # 2024
    178, 190, 185, 198,  # 2025
]

forecast = requests.post(f"{BASE}/v1/forecast", json={
    "time_series": [quarterly_revenue],
    "horizon": 4,
}).json()

next_quarter = forecast["point_forecast"][0]
q90 = [q[9] for q in forecast["quantile_forecast"][0]]
q10 = [q[1] for q in forecast["quantile_forecast"][0]]

print("Revenue forecast (next 4 quarters):")
for i, (mid, lo, hi) in enumerate(zip(next_quarter, q10, q90)):
    print(f"  Q{i+1} 2026: ${mid:.0f}M  (${lo:.0f}M – ${hi:.0f}M)")


# 2. Ask Qwen to interpret the forecast
prompt = f"""Given quarterly revenue history: {quarterly_revenue}
And forecast for next 4 quarters: {[round(x, 1) for x in next_quarter]}
With 90% confidence intervals: {[f"${lo:.0f}-${hi:.0f}M" for lo, hi in zip(q10, q90)]}

Write a 2-sentence executive summary of the revenue outlook."""

response = requests.post(f"{BASE}/v1/chat/completions", json={
    "model": "qwen",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 150,
}).json()

print(f"\nExecutive Summary:")
print(response["choices"][0]["message"]["content"])


# 3. Transcribe audio (uncomment with a real file)
# transcript = requests.post(f"{BASE}/v1/audio/transcriptions",
#     files={"file": open("meeting.wav", "rb")},
#     data={"model": "whisper"},
# ).json()["text"]
# print(f"\nMeeting transcript: {transcript}")
