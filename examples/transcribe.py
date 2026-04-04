"""Audio transcription with Whisper."""

import requests
import sys

BASE = "http://localhost:8080"


def transcribe(audio_path: str, response_format: str = "json") -> dict:
    """Transcribe an audio file."""
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{BASE}/v1/audio/transcriptions",
            files={"file": f},
            data={"model": "whisper", "response_format": response_format},
        )
    resp.raise_for_status()
    return resp.json() if response_format != "text" else resp.text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
        print("  Supports: wav, mp3, m4a, ogg, flac")
        sys.exit(1)

    result = transcribe(sys.argv[1])
    print(result["text"])
