"""
Factory for creating configured PydanticAI agents with multiple LLM providers.

This module provides a flexible factory pattern for instantiating PydanticAI agents
with support for Anthropic (Claude) and OpenAI (GPT) models.
"""

from typing import Any, Callable, TypeVar

from loguru import logger
from pydantic_ai import Agent, Tool
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import OutputSpec
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.config.config import get_settings

settings = get_settings()

# Type variables for generic agent configuration
DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


class AgentFactory:
    """
    Factory for creating PydanticAI agents with automatic provider detection.

    This factory simplifies agent creation by:
    - Auto-detecting LLM provider from model name
    - Managing API keys from environment configuration
    - Supporting tool mounting for agent capabilities
    - Providing consistent agent configuration

    Supports:
    - Anthropic: claude-*, sonnet, opus, haiku models
    - OpenAI: gpt-* models

    Attributes:
        model_name: LLM model identifier (e.g., "claude-haiku-4-5-20251001", "gpt-4")
        mounted_tools: List of tools/functions available to the agent
    """

    def __init__(
        self,
        model_name: str,
        tools: list[Callable[..., Any] | Tool] | None = None,
    ) -> None:
        """
        Initialize the agent factory.

        Args:
            model_name: Model identifier string (e.g., "claude-haiku-4-5-20251001", "gpt-4")
            tools: Optional list of callable tools or Tool objects for the agent
        """
        self.model_name = model_name
        self.mounted_tools: list[Callable[..., Any] | Tool] = tools or []

    def create_agent(
        self,
        instruction_prompt: str,
        output_type: type[OutputT] | None = None,
        deps_type: type[DepsT] | None = None,
        **kwargs,
    ) -> Agent[DepsT, OutputT]:
        """
        Create a configured PydanticAI agent instance.

        Automatically detects the LLM provider based on model_name and configures
        the agent with the specified parameters.

        Args:
            instruction_prompt: System prompt/instructions for the agent
            output_type: Pydantic model for structured output (optional)
            deps_type: Type for dependency injection (optional)
            **kwargs: Additional arguments passed to Agent constructor
                     (e.g., retries, temperature_override)

        Returns:
            Configured PydanticAI Agent instance

        Raises:
            ValueError: If model_name doesn't match supported providers or API keys are missing

        Example:
            >>> factory = AgentFactory("claude-haiku-4-5-20251001", tools=[my_tool])
            >>> agent = factory.create_agent(
            ...     instruction_prompt="You are a helpful assistant",
            ...     output_type=MyOutputSchema,
            ...     deps_type=MyDeps,
            ...     retries=3
            ... )
        """
        model = self._identify_provider_from_model(self.model_name)
        agent_kwargs = {
            "model": model,
            "instructions": instruction_prompt,
            "tools": self.mounted_tools,
        }

        if output_type is not None:
            agent_kwargs["output_type"] = output_type

        if deps_type is not None:
            agent_kwargs["deps_type"] = deps_type

        agent_kwargs.update(kwargs)

        return Agent(**agent_kwargs)  # type: ignore[return-value]

    def mount_tool(self, func: Callable[..., Any] | Tool) -> Callable[..., Any] | Tool:
        """
        Mount a tool to this factory (decorator pattern).

        Can be used as a decorator to register tools that will be available
        to all agents created by this factory.

        Args:
            func: Callable function or Tool object to mount

        Returns:
            The same function/tool (for decorator chaining)

        Example:
            >>> factory = AgentFactory("claude-haiku-4-5-20251001")
            >>> @factory.mount_tool
            ... def my_custom_tool(ctx, query: str) -> str:
            ...     return "result"
        """
        logger.info(f"Mounting tool to agent: {getattr(func, '__name__', repr(func))}")
        self.mounted_tools.append(func)
        return func

    def _identify_provider_from_model(
        self, model_name: str
    ) -> AnthropicModel | OpenAIChatModel:
        """
        Identify and configure the LLM provider based on model name.

        Examines the model name prefix to determine whether to use Anthropic or OpenAI,
        then configures the appropriate model with API keys from settings.

        Args:
            model_name: Model identifier string

        Returns:
            Configured AnthropicModel or OpenAIChatModel instance

        Raises:
            ValueError: If model name doesn't match supported providers or API key is missing

        Note:
            - Anthropic models: claude-*, sonnet, opus, haiku
            - OpenAI models: gpt-*
            - Anthropic models are configured with temperature=0.0 for deterministic output
        """
        if model_name.startswith(("claude-", "sonnet", "opus", "haiku")):
            if not settings.ANTHROPIC_API_KEY:
                logger.error("ANTHROPIC_API_KEY not configured")
                raise ValueError("ANTHROPIC_API_KEY not configured")

            return AnthropicModel(
                model_name=model_name,
                provider=AnthropicProvider(api_key=settings.ANTHROPIC_API_KEY),
                settings={"temperature": 0.0},
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
