from typing import Any, Callable

from loguru import logger
from pydantic_ai import Agent, Tool
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import OutputSpec
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from config.config import get_settings

settings = get_settings()


class AgentFactory:
    def __init__(
        self,
        model_name: str,
        tools: list[Callable[..., Any] | Tool] | None = None,
    ) -> None:
        self.model_name = model_name
        self.mounted_tools: list[Callable[..., Any] | Tool] = tools or []

    def create_agent(
        self,
        system_prompt: str,
        output_type: type[Any] | OutputSpec[Any] | None = None,
    ) -> Agent:
        model = self._identify_provider_from_model(self.model_name)
        kwargs = {
            "model": model,
            "system_prompt": system_prompt,
            "tools": self.mounted_tools,
        }

        if output_type is not None:
            kwargs["output_type"] = output_type

        return Agent(**kwargs)

    def mount_tool(self, func: Callable[..., Any] | Tool) -> Callable[..., Any] | Tool:
        """decorator method"""
        logger.info(f"Mounting tool to agent: {getattr(func, '__name__', repr(func))}")
        self.mounted_tools.append(func)
        return func

    def _identify_provider_from_model(
        self, model_name: str
    ) -> AnthropicModel | OpenAIChatModel:
        if model_name.startswith(("claude-", "sonnet", "opus", "haiku")):
            if not settings.ANTHROPIC_API_KEY:
                logger.error("ANTHROPIC_API_KEY not configured")
                raise ValueError("ANTHROPIC_API_KEY not configured")

            return AnthropicModel(
                model_name=model_name,
                provider=AnthropicProvider(api_key=settings.ANTHROPIC_API_KEY),
            )

        if model_name.startswith("gpt-") or "gpt" in model_name.lower():
            if not settings.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY not configured")
                raise ValueError("OPENAI_API_KEY not configured")

            return OpenAIChatModel(
                model_name=model_name,
                provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY),
            )

        raise ValueError(
            f"Model '{model_name}' not supported. Use 'claude-*' or 'gpt-*' models."
        )
