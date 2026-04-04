"""Chat with Qwen 3.5 — OpenAI SDK compatible."""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

# Simple chat
response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain recursion in one sentence."},
    ],
    max_tokens=100,
    temperature=0.7,
)
print(response.choices[0].message.content)


# Multi-turn conversation
messages = [{"role": "system", "content": "You are a helpful assistant."}]

for user_input in ["What's the capital of France?", "What's its population?"]:
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="qwen", messages=messages, max_tokens=100
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print(f"User: {user_input}")
    print(f"Qwen: {reply}\n")
