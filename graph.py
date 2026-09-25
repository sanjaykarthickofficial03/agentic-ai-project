from langgraph.graph import StateGraph, END
from state import AnalystState
from agents.profile_agent import profile_agent
from agents.planner_agent import planner_agent
from agents.eda_agent import eda_agent
from agents.model_agent import model_agent
from agents.validation_agent import validation_agent
from agents.insight_agent import insight_agent


def build_graph():
    g = StateGraph(AnalystState)
    for name, fn in [
        ("profile", profile_agent), ("planner", planner_agent),
        ("eda", eda_agent), ("model", model_agent),
        ("validation", validation_agent), ("insights", insight_agent)
    ]:
        g.add_node(name, fn)
    g.set_entry_point("profile")
    for a, b in [("profile", "planner"), ("planner", "eda"),
                 ("eda", "model"), ("model", "validation"),
                 ("validation", "insights")]:
        g.add_edge(a, b)
    g.add_edge("insights", END)
    return g.compile()
