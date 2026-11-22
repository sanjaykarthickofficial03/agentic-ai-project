from typing import Any, TypedDict


class AnalystState(TypedDict, total=False):
    df: Any
    source_df: Any
    processed_df: Any
    model_df: Any
    goal: str
    target_override: str | None
    llm_provider: str
    llm_model: str | None
    profile: dict
    preprocessing: dict
    plan: list
    task: str
    target: str | None
    decision: dict
    eda: dict
    model: dict
    validation: dict
    insights: str
    logs: list
    agent_errors: dict
