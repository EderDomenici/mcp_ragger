from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence


DEFAULT_OPENROUTER_JUDGE_MODELS = (
    "deepseek/deepseek-v4-flash",
    "google/gemini-2.5-flash",
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class JudgeSmokeResult:
    text: str
    model: str


def _csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    return float(value) if value else default


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    return int(value) if value else default


@dataclass(frozen=True)
class OpenRouterJudgeConfig:
    api_key: str
    models: tuple[str, ...] = DEFAULT_OPENROUTER_JUDGE_MODELS
    temperature: float = 0.0
    max_tokens: int = 4096
    base_url: str = OPENROUTER_BASE_URL
    app_referer: str = ""
    app_title: str = "langchain-rag-mcp"
    require_parameters: bool = True
    provider_ignore: tuple[str, ...] = ("deepinfra",)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "OpenRouterJudgeConfig":
        env = os.environ if env is None else env
        api_key = env.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter Ragas judge.")

        models = _csv(env.get("RAGAS_JUDGE_MODELS")) or DEFAULT_OPENROUTER_JUDGE_MODELS
        if len(models) < 2:
            raise RuntimeError("RAGAS_JUDGE_MODELS must include at least free and paid fallback models.")

        return cls(
            api_key=api_key,
            models=models,
            temperature=_env_float(env, "RAGAS_JUDGE_TEMPERATURE", cls.temperature),
            max_tokens=_env_int(env, "RAGAS_JUDGE_MAX_TOKENS", cls.max_tokens),
            base_url=env.get("OPENROUTER_BASE_URL", cls.base_url),
            app_referer=env.get("OPENROUTER_HTTP_REFERER", ""),
            app_title=env.get("OPENROUTER_APP_TITLE", cls.app_title),
            require_parameters=env.get("OPENROUTER_REQUIRE_PARAMETERS", "true").strip().lower()
            not in {"0", "false", "no"},
            provider_ignore=_csv(env.get("OPENROUTER_PROVIDER_IGNORE")) or cls.provider_ignore,
        )

    @property
    def primary_model(self) -> str:
        return self.models[0]

    def extra_headers(self) -> dict[str, str]:
        headers = {}
        if self.app_referer:
            headers["HTTP-Referer"] = self.app_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers

    def chat_request(self, messages: Sequence[Mapping[str, str]]) -> dict:
        return {
            "model": self.primary_model,
            "extra_body": self.openrouter_extra_body(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": list(messages),
        }

    def openrouter_extra_body(self) -> dict:
        body = {"models": list(self.models)}
        provider = {}
        if self.require_parameters:
            provider["require_parameters"] = True
        if self.provider_ignore:
            provider["ignore"] = list(self.provider_ignore)
        if provider:
            body["provider"] = provider
        return body


class _CompletionsProxy:
    def __init__(self, completions, config: OpenRouterJudgeConfig) -> None:
        self._completions = completions
        self._config = config

    def create(self, **kwargs):
        kwargs.setdefault("model", self._config.primary_model)
        kwargs.setdefault("temperature", self._config.temperature)
        kwargs.setdefault("max_tokens", self._config.max_tokens)
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("models", list(self._config.models))
        provider = dict(extra_body.get("provider") or {})
        if self._config.require_parameters:
            provider.setdefault("require_parameters", True)
        if self._config.provider_ignore:
            ignored = list(provider.get("ignore") or [])
            for provider_name in self._config.provider_ignore:
                if provider_name not in ignored:
                    ignored.append(provider_name)
            provider["ignore"] = ignored
        if provider:
            extra_body["provider"] = provider
        kwargs["extra_body"] = extra_body
        return self._completions.create(**kwargs)


class _ChatProxy:
    def __init__(self, chat, config: OpenRouterJudgeConfig) -> None:
        self.completions = _CompletionsProxy(chat.completions, config)


class OpenRouterOpenAIProxy:
    """Duck-typed OpenAI client wrapper that injects OpenRouter model fallback."""

    def __init__(self, client, config: OpenRouterJudgeConfig) -> None:
        self._client = client
        self._config = config
        self.chat = _ChatProxy(client.chat, config)

    def __getattr__(self, name: str):
        return getattr(self._client, name)

    def smoke_judge(self) -> JudgeSmokeResult:
        completion = self.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Return only the word ok if you can act as an evaluation judge.",
                }
            ],
        )
        text = completion.choices[0].message.content or ""
        model = getattr(completion, "model", self._config.primary_model)
        return JudgeSmokeResult(text=text.strip(), model=model)


def create_openai_compatible_client(config: OpenRouterJudgeConfig):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenRouter judge requires the openai package. Install with: pip install openai") from exc

    client = OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        default_headers=config.extra_headers() or None,
    )
    return OpenRouterOpenAIProxy(client, config)


class _FallbackCompletions:
    """Wraps openai.Completions to inject the full OpenRouter extra_body on every call."""

    def __init__(self, completions, config: OpenRouterJudgeConfig) -> None:
        self._completions = completions
        self._config = config

    def create(self, **kwargs):
        base = self._config.openrouter_extra_body()
        incoming = dict(kwargs.pop("extra_body", None) or {})
        # Caller values take precedence; base fills in what's missing.
        merged = {**base, **incoming}
        return self._completions.create(**kwargs, extra_body=merged)

    def __getattr__(self, name: str):
        return getattr(self._completions, name)


class _FallbackChat:
    def __init__(self, chat, config: OpenRouterJudgeConfig) -> None:
        self._chat = chat
        self.completions = _FallbackCompletions(chat.completions, config)

    def __getattr__(self, name: str):
        return getattr(self._chat, name)


def _make_openrouter_client(config: OpenRouterJudgeConfig):
    """Returns a real openai.OpenAI subclass that injects the full OpenRouter extra_body.

    Using a subclass (not a wrapper) so that isinstance(client, OpenAI) passes,
    which is required by instructor.from_openai inside llm_factory.
    """
    from openai import OpenAI

    class _Client(OpenAI):
        def __init__(self) -> None:
            super().__init__(
                base_url=config.base_url,
                api_key=config.api_key,
                default_headers=config.extra_headers() or None,
            )
            # Override after super().__init__ sets self.chat as an instance attribute.
            self.chat = _FallbackChat(self.chat, config)

    return _Client()


def create_ragas_llm(config: OpenRouterJudgeConfig):
    try:
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError("Ragas judge requires ragas and openai. Install with: pip install ragas openai") from exc

    client = _make_openrouter_client(config)
    return llm_factory(
        config.primary_model,
        provider="openai",
        client=client,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        extra_body=config.openrouter_extra_body(),
    )
