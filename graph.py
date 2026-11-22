try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None
    END = None
    HAS_LANGGRAPH = False

from state import AnalystState
from agents.profile_agent import profile_agent
from agents.planner_agent import planner_agent
from agents.preprocess_agent import preprocess_agent
from agents.eda_agent import eda_agent
from agents.model_agent import model_agent
from agents.validation_agent import validation_agent
from agents.insight_agent import insight_agent

NODES = [
    ("profile", profile_agent),
    ("planner", planner_agent),
    ("preprocess", preprocess_agent),
    ("eda", eda_agent),
    ("model", model_agent),
    ("validation", validation_agent),
    ("insights", insight_agent),
]


def _safe_node(name, fn, progress_callback=None):
    def wrapped(state):
        if progress_callback:
            progress_callback(name, "running")
        try:
            state = fn(state)
        except Exception as exc:
            state.setdefault("logs", []).append(f"{name.title()} Agent failed safely: {type(exc).__name__}: {exc}")
            state.setdefault("agent_errors", {})[name] = f"{type(exc).__name__}: {exc}"
        state.setdefault("logs", []).append(f"Graph completed node '{name}'.")
        if progress_callback:
            progress_callback(name, "completed" if name not in state.get("agent_errors", {}) else "failed")
        return state
    return wrapped


class SequentialGraph:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
    def invoke(self, state):
        for name, fn in NODES:
            state = _safe_node(name, fn, self.progress_callback)(state)
        return state


def build_graph(progress_callback=None):
    if not HAS_LANGGRAPH:
        return SequentialGraph(progress_callback)
    g = StateGraph(AnalystState)
    wrapped = {name: _safe_node(name, fn, progress_callback) for name, fn in NODES}
    for name in wrapped:
        g.add_node(name, wrapped[name])
    g.set_entry_point("profile")
    for a, b in [("profile", "planner"), ("planner", "preprocess"), ("preprocess", "eda"), ("eda", "model"), ("model", "validation"), ("validation", "insights")]:
        g.add_edge(a, b)
    g.add_edge("insights", END)
    return g.compile()
