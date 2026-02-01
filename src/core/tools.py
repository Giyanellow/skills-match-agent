from typing import Any

from pydantic_ai import RunContext


def nlp_parser(ctx: RunContext): ...


def skill_extractor(ctx: RunContext, text: str) -> Any: ...
