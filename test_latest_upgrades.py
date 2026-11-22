import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.model_agent import _classification, _regression
from agents.insight_agent import insight_agent, _compact_evidence


def test_classification_tie():
    n = 60
    df = pd.DataFrame({"feature": [1.0] * n, "target": ["A"] * (n // 2) + ["B"] * (n // 2)})
    r = _classification(df, "target", "fast")
    assert r["selected_model"] is not None
    assert r["baseline_comparison"]["tied"] is True


def test_regression_tie():
    n = 60
    df = pd.DataFrame({"feature": [1.0] * n, "target": np.arange(n, dtype=float)})
    r = _regression(df, "target", "fast")
    assert r["selected_model"] is not None
    assert r["baseline_comparison"]["tied"] is True


def test_compact_insight_offline():
    state = {
        "llm_provider": "offline",
        "goal": "predict target",
        "profile": {"rows": 10, "columns": 2, "missing_total": 0, "duplicates": 0},
        "preprocessing": {"remaining_missing": 0, "duplicates_removed": 0},
        "task": "classification", "target": "target",
        "model": {"status": "completed", "selected_model": "LogisticRegression", "best_candidate": "LogisticRegression", "metric": "F1", "score": 0.8, "models_evaluated": 7, "cv": "2-fold", "useful_model": True, "baseline_summary": "real model tied"},
        "eda": {}, "validation": {}, "logs": []
    }
    assert len(_compact_evidence(state)) < 2000
    out = insight_agent(state)
    assert out["insights"]


if __name__ == "__main__":
    test_classification_tie()
    test_regression_tie()
    test_compact_insight_offline()
    print("Latest upgrades validation: PASS")
