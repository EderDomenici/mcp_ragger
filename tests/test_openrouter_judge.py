import os
import unittest
from unittest.mock import patch

from langchain_rag_mcp.evals.openrouter_judge import (
    DEFAULT_OPENROUTER_JUDGE_MODELS,
    OpenRouterJudgeConfig,
    OpenRouterOpenAIProxy,
    create_ragas_llm,
)


class RecordingCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "Completion",
            (),
            {
                "model": "deepseek/deepseek-v4-flash",
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": "ok"})()},
                    )()
                ],
            },
        )()


class RecordingClient:
    def __init__(self):
        self.completions = RecordingCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


class OpenRouterJudgeTests(unittest.TestCase):
    def test_default_models_try_free_then_paid(self):
        self.assertEqual(
            DEFAULT_OPENROUTER_JUDGE_MODELS,
            (
                "deepseek/deepseek-v4-flash",
                "google/gemini-2.5-flash",
            ),
        )

    def test_config_from_env_uses_default_model_order(self):
        config = OpenRouterJudgeConfig.from_env({"OPENROUTER_API_KEY": "sk-test"})

        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(config.models, DEFAULT_OPENROUTER_JUDGE_MODELS)
        self.assertEqual(config.primary_model, "deepseek/deepseek-v4-flash")

    def test_config_from_env_parses_model_override(self):
        config = OpenRouterJudgeConfig.from_env(
            {
                "OPENROUTER_API_KEY": "sk-test",
                "RAGAS_JUDGE_MODELS": "free-model, paid-model ",
            }
        )

        self.assertEqual(config.models, ("free-model", "paid-model"))

    def test_chat_request_uses_primary_model_and_models_fallback_array(self):
        config = OpenRouterJudgeConfig.from_env({"OPENROUTER_API_KEY": "sk-test"})

        request = config.chat_request([{"role": "user", "content": "judge this"}])

        self.assertEqual(request["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(
            request["extra_body"]["models"],
            [
                "deepseek/deepseek-v4-flash",
                "google/gemini-2.5-flash",
            ],
        )
        self.assertEqual(
            request["extra_body"]["provider"],
            {
                "require_parameters": True,
                "ignore": ["deepinfra"],
            },
        )
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["max_tokens"], OpenRouterJudgeConfig.max_tokens)

    def test_openai_proxy_injects_openrouter_fallback_models(self):
        config = OpenRouterJudgeConfig.from_env({"OPENROUTER_API_KEY": "sk-test"})
        client = RecordingClient()
        proxy = OpenRouterOpenAIProxy(client, config)

        result = proxy.chat.completions.create(
            messages=[{"role": "user", "content": "judge this"}],
        )

        self.assertEqual(result.choices[0].message.content, "ok")
        self.assertEqual(client.completions.calls[0]["model"], config.primary_model)
        self.assertEqual(client.completions.calls[0]["extra_body"]["models"], list(config.models))
        self.assertEqual(
            client.completions.calls[0]["extra_body"]["provider"],
            {
                "require_parameters": True,
                "ignore": ["deepinfra"],
            },
        )

    def test_provider_ignore_can_be_overridden_from_env(self):
        config = OpenRouterJudgeConfig.from_env(
            {
                "OPENROUTER_API_KEY": "sk-test",
                "OPENROUTER_PROVIDER_IGNORE": "deepinfra, other",
            }
        )

        self.assertEqual(config.provider_ignore, ("deepinfra", "other"))

    def test_create_ragas_llm_passes_openrouter_extra_body_to_factory(self):
        config = OpenRouterJudgeConfig.from_env({"OPENROUTER_API_KEY": "sk-test"})

        with (
            patch("langchain_rag_mcp.evals.openrouter_judge.create_openai_compatible_client", return_value=RecordingClient()),
            patch("ragas.llms.llm_factory", return_value="llm") as llm_factory,
        ):
            result = create_ragas_llm(config)

        self.assertEqual(result, "llm")
        self.assertEqual(llm_factory.call_args.kwargs["extra_body"]["provider"]["ignore"], ["deepinfra"])
        self.assertTrue(llm_factory.call_args.kwargs["extra_body"]["provider"]["require_parameters"])

    def test_smoke_judge_returns_text_and_used_model(self):
        config = OpenRouterJudgeConfig.from_env({"OPENROUTER_API_KEY": "sk-test"})
        client = RecordingClient()

        result = OpenRouterOpenAIProxy(client, config).smoke_judge()

        self.assertEqual(result.text, "ok")
        self.assertEqual(result.model, "deepseek/deepseek-v4-flash")
        self.assertIn("Return only", client.completions.calls[0]["messages"][0]["content"])

    def test_missing_api_key_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
            OpenRouterJudgeConfig.from_env({})


if __name__ == "__main__":
    unittest.main()
