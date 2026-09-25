from typing import TypedDict, Any


class AnalystState(TypedDict, total=False):
    df: Any
    goal: str
    target_override: str | None
    profile: dict
    plan: list
    task: str
    target: str | None
    decision: dict
    eda: dict
    model: dict
    validation: dict
    insights: str
    logs: list
