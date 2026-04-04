"""Streaming chat — tokens arrive as they're generated."""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

stream = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    max_tokens=50,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
