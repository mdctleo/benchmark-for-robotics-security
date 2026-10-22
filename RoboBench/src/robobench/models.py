"""Target models: a callable that answers a prompt about one or more images.

A model is anything with ``__call__(images, prompt) -> str``. The classes
here wrap the OpenAI, Anthropic and Gemini APIs with the request settings used
in the RoboBench experiments; ``make_model`` picks one from a model name.

    from robobench.models import make_model
    model = make_model("gpt-5.4-mini-2026-03-17")
    reply = model([frame_bytes], "Describe the scene.")

Clients are created on the first call, so constructing a model needs no API
key. The key is read from ``OPENAI_API_KEY``, from ``CLAUDE_API_KEY`` /
``ANTHROPIC_API_KEY`` for Claude models, or from ``GOOGLE_API_KEY`` for Gemini
models (``pip install robobench[gemini]``).
"""

import base64
import os
import random
import time
from threading import Lock
from typing import Protocol


class Model(Protocol):
    def __call__(self, images: list[bytes], prompt: str) -> str: ...


def detect_media_type(raw: bytes) -> str:
    """Media type from magic bytes.

    The Anthropic API checks the declared media type against the payload, so
    the type comes from the bytes rather than the file name.
    """
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:2] == b"\xff\xd8":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def encode_image(raw: bytes) -> tuple[str, str]:
    """Return (base64 payload, media type) for image bytes."""
    return base64.b64encode(raw).decode(), detect_media_type(raw)


def is_anthropic_model(name: str) -> bool:
    return name.startswith("claude")


def is_gemini_model(name: str) -> bool:
    return name.startswith("gemini")


class _ApiModel:
    """Shared retry loop and lazy client construction."""

    def __init__(self, name: str, retry_count: int = 5, client=None):
        self.name = name
        self.retry_count = retry_count
        self._client = client
        self._client_lock = Lock()

    def client(self):
        with self._client_lock:
            if self._client is None:
                self._client = self._make_client()
            return self._client

    def _make_client(self):
        raise NotImplementedError

    def _query(self, client, images: list[bytes], prompt: str) -> str:
        raise NotImplementedError

    def __call__(self, images: list[bytes], prompt: str) -> str:
        client = self.client()
        last_error = None
        for attempt in range(1, self.retry_count + 1):
            try:
                return self._query(client, images, prompt)
            except Exception as error:  # noqa: BLE001 - provider errors vary
                last_error = error
                if attempt < self.retry_count:
                    time.sleep(min(2 ** attempt, 30) + random.random())
        raise RuntimeError(f"failed model={self.name}") from last_error


class OpenAIModel(_ApiModel):
    """OpenAI chat completion: the prompt text, then each image as a data URL."""

    def _make_client(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI
        return OpenAI()

    def _query(self, client, images: list[bytes], prompt: str) -> str:
        content = [{"type": "text", "text": prompt}]
        for raw in images:
            image_b64, media_type = encode_image(raw)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
            })
        response = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=20000,
            temperature=0.3,
        )
        return response.choices[0].message.content


class AnthropicModel(_ApiModel):
    """Anthropic message: each image as a base64 block, then the prompt text."""

    def _make_client(self):
        # The project's .env stores the Anthropic key as CLAUDE_API_KEY, which is
        # not the variable the SDK reads by default.
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("CLAUDE_API_KEY is not set")
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    def _query(self, client, images: list[bytes], prompt: str) -> str:
        content = []
        for raw in images:
            image_b64, media_type = encode_image(raw)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_b64,
                },
            })
        content.append({"type": "text", "text": prompt})
        message = client.messages.create(
            model=self.name,
            max_tokens=1024,
            temperature=0.3,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(
            block.text for block in message.content
            if getattr(block, "type", None) == "text"
        )


class GeminiModel(_ApiModel):
    """Gemini generate_content: each image as a bytes part, then the prompt."""

    def _make_client(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        from google import genai
        return genai.Client(api_key=api_key)

    def _query(self, client, images: list[bytes], prompt: str) -> str:
        from google.genai import types
        contents = [
            types.Part.from_bytes(data=raw, mime_type=detect_media_type(raw))
            for raw in images
        ]
        contents.append(prompt)
        response = client.models.generate_content(
            model=self.name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text


def make_model(name: str, retry_count: int = 5) -> Model:
    """Model for a name: Claude -> Anthropic, Gemini -> Google, all others OpenAI."""
    if is_anthropic_model(name):
        return AnthropicModel(name, retry_count=retry_count)
    if is_gemini_model(name):
        return GeminiModel(name, retry_count=retry_count)
    return OpenAIModel(name, retry_count=retry_count)
