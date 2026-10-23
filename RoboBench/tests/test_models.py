import base64
from types import SimpleNamespace

import pytest

from robobench.models import (
    AnthropicModel,
    GeminiModel,
    OpenAIModel,
    detect_media_type,
    make_model,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8


class FakeOpenAI:
    def __init__(self, reply="okay"):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.reply = reply

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))]
        )


class FakeAnthropic:
    def __init__(self, reply="deny"):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self.reply = reply

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[
            SimpleNamespace(type="text", text=self.reply[:2]),
            SimpleNamespace(type="tool_use"),
            SimpleNamespace(type="text", text=self.reply[2:]),
        ])


class FakeGemini:
    def __init__(self, reply="okay"):
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self.reply = reply

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.reply)


@pytest.mark.parametrize("raw,expected", [
    (PNG, "image/png"),
    (JPEG, "image/jpeg"),
    (b"RIFF\x00\x00\x00\x00WEBP", "image/webp"),
    (b"GIF89a", "image/gif"),
    (b"not an image", "image/png"),
])
def test_detect_media_type(raw, expected):
    assert detect_media_type(raw) == expected


@pytest.mark.parametrize("n_images", [1, 2])
def test_openai_message_shape(n_images):
    client = FakeOpenAI()
    model = OpenAIModel("gpt-x", client=client)
    images = [PNG, JPEG][:n_images]

    assert model(images, "hello") == "okay"

    call = client.calls[0]
    assert call["model"] == "gpt-x"
    assert call["max_completion_tokens"] == 20000
    assert call["temperature"] == 0.3
    content = call["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "hello"}
    assert len(content) == 1 + n_images
    assert content[1]["image_url"]["url"].startswith(
        "data:image/png;base64," + base64.b64encode(PNG).decode()
    )
    if n_images == 2:
        assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("n_images", [1, 2])
def test_anthropic_message_shape(n_images):
    client = FakeAnthropic()
    model = AnthropicModel("claude-x", client=client)
    images = [PNG, JPEG][:n_images]

    assert model(images, "hello") == "deny"  # text blocks joined, others skipped

    call = client.calls[0]
    assert call["model"] == "claude-x"
    assert call["max_tokens"] == 1024
    assert call["temperature"] == 0.3
    content = call["messages"][0]["content"]
    assert len(content) == n_images + 1
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[-1] == {"type": "text", "text": "hello"}


@pytest.mark.parametrize("n_images", [1, 2])
def test_gemini_message_shape(n_images):
    client = FakeGemini()
    model = GeminiModel("gemini-x", client=client)
    images = [PNG, JPEG][:n_images]

    assert model(images, "hello") == "okay"

    call = client.calls[0]
    assert call["model"] == "gemini-x"
    assert call["config"].temperature == 0.5
    assert call["config"].thinking_config.thinking_budget == 0
    contents = call["contents"]
    assert len(contents) == n_images + 1
    assert contents[0].inline_data.data == PNG
    assert contents[0].inline_data.mime_type == "image/png"
    if n_images == 2:
        assert contents[1].inline_data.mime_type == "image/jpeg"
    assert contents[-1] == "hello"


def test_retry_then_raise(monkeypatch):
    class Failing(FakeOpenAI):
        def _create(self, **kwargs):
            self.calls.append(kwargs)
            raise ConnectionError("down")

    import robobench.models as models

    monkeypatch.setattr(models.time, "sleep", lambda _s: None)  # no backoff in tests
    client = Failing()
    model = OpenAIModel("gpt-x", retry_count=2, client=client)
    with pytest.raises(RuntimeError, match="failed model=gpt-x"):
        model([PNG], "hello")
    assert len(client.calls) == 2


def test_make_model_dispatch():
    assert isinstance(make_model("claude-haiku-4-5"), AnthropicModel)
    assert isinstance(make_model("gpt-5.4-mini-2026-03-17"), OpenAIModel)
    assert isinstance(make_model("gemini-3-flash-preview"), GeminiModel)


def test_missing_key_raises_on_first_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = OpenAIModel("gpt-x", retry_count=1)   # no key needed here
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        model([PNG], "hello")
