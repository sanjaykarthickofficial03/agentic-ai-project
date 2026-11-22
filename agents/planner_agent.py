import os
import re

from .llm_utils import choose_model, ollama_json, server_status


def _goal_has_any(goal, terms):
    text = (goal or "").lower()
    return any(t in text for t in terms)


def _find_target_from_goal(columns, goal):
    text = (goal or "").lower()
    for col in sorted(columns, key=lambda x: len(str(x)), reverse=True):
        name = str(col).lower()
        if re.search(rf"\b{re.escape(name)}\b", text):
            return col
    return None


def _candidate_target(profile):
    columns = profile.get("columns_list", [])
    rows = max(int(profile.get("rows", 0)), 1)
    dtypes = profile.get("dtypes", {})
    nunique_map = profile.get("nunique", {})
    id_words = ("id", "index", "uuid", "code", "serial", "key")
    target_words = (
        "target", "label", "class", "outcome", "response", "result", "churn",
        "price", "salary", "income", "score", "rating", "age", "status",
        "default", "fraud", "demand", "sales", "revenue", "profit", "y"
    )
    ranked = []
    for position, col in enumerate(columns):
        name = str(col).lower().strip().replace(" ", "_")
        dtype = str(dtypes.get(col, ""))
        unique = int(nunique_map.get(col, rows))
        ratio = unique / rows
        if any(name == w or name.endswith("_" + w) or name.startswith(w + "_") for w in id_words):
            continue
        score = 0.0
        reasons = []
        if name in target_words or any(w in name.split("_") for w in target_words):
            score += 6.0
            reasons.append("target-like column name")
        if position == len(columns) - 1:
            score += 2.0
            reasons.append("last-column convention")
        if dtype == "object" or dtype.startswith("category") or dtype.startswith("bool"):
            if 1 < unique <= max(2, min(20, int(rows * 0.2))):
                score += 3.0
                reasons.append("low-cardinality categorical")
            elif ratio <= 0.5:
                score += 1.0
                reasons.append("categorical feature")
        elif dtype.startswith(("int", "float")):
            if 1 < unique <= 10 and ratio <= 0.05:
                score += 3.0
                reasons.append("low-cardinality numeric")
            elif unique >= max(10, min(rows, 20)):
                score += 1.5
                reasons.append("continuous numeric")
        if unique == rows and position == 0:
            score -= 2.0
        ranked.append((score, col, reasons))
    if not ranked:
        return None, "No plausible non-identifier target column was found."
    ranked.sort(key=lambda x: (x[0], -columns.index(x[1])), reverse=True)
    score, target, reasons = ranked[0]
    if score < 2.0:
        return None, "No target column has enough evidence for reliable automatic selection."
    confidence = "high" if score >= 7 else "medium"
    return target, f"Auto-detected '{target}' using {', '.join(reasons)} ({confidence} confidence)."


def _deterministic_decision(state, target_override=None):
    profile = state["profile"]
    goal = state.get("goal", "")
    columns = profile.get("columns_list", [])
    dtypes = profile.get("dtypes", {})
    nunique = profile.get("nunique", {})
    rows = max(int(profile.get("rows", 0)), 1)
    explicit_target = target_override or _find_target_from_goal(columns, goal)
    auto_target, auto_reason = _candidate_target(profile)
    target = explicit_target or auto_target
    clustering_requested = _goal_has_any(goal, ["cluster", "clustering", "segment", "segmentation", "group similar", "customer groups"])
    prediction_requested = _goal_has_any(goal, ["predict", "prediction", "forecast", "classify", "classification", "estimate", "regression", "model"])
    if clustering_requested and not target_override and not explicit_target:
        return {"task": "clustering", "target": None, "reason": "The objective requests grouping or segmentation, so an unsupervised workflow is appropriate.", "confidence": "high"}
    if target:
        dtype = str(dtypes.get(target, ""))
        unique = int(nunique.get(target, rows))
        unique_ratio = unique / rows
        if dtype == "object" or dtype.startswith("category") or dtype.startswith("bool"):
            task = "classification"
            reason = f"Target '{target}' is categorical ({dtype})."
        elif unique <= 10 and unique_ratio <= 0.05:
            task = "classification"
            reason = f"Target '{target}' is numeric but has only {unique} distinct values, so it is treated as discrete classification."
        else:
            task = "regression"
            reason = f"Target '{target}' is numeric with {unique} distinct values, so it is treated as regression."
        if not target_override and not explicit_target:
            reason = f"{auto_reason} {reason}"
        confidence = "high" if target_override or explicit_target else ("high" if "high confidence" in auto_reason else "medium")
        return {"task": task, "target": target, "reason": reason, "confidence": confidence}
    if prediction_requested:
        return {"task": "unknown", "target": None, "reason": "Prediction was requested, but no sufficiently plausible target column was found.", "confidence": "low"}
    return {"task": "clustering", "target": None, "reason": "No target was specified and no supervised objective was detected; using unsupervised clustering.", "confidence": "medium"}


def _llm_refine(state, deterministic):
    status = server_status(os.getenv("OLLAMA_BASE_URL"))
    if not status["server_ok"]:
        return None, f"Ollama server not reachable: {status['version_error']}"
    requested = os.getenv("OLLAMA_MODEL", "")
    model, model_reason = choose_model(status, requested)
    if not model:
        return None, model_reason
    prompt = (
        "You are the Planner Agent for an autonomous tabular-data analyst.\n"
        "Return ONLY a JSON object with task, target, reason, confidence.\n"
        "Choose one task: classification, regression, clustering, or unknown.\n"
        "Respect an explicit target and preserve the deterministic decision unless clearly invalid.\n"
        f"User goal: {state.get('goal','')}\n"
        f"Dataset profile: rows={state['profile'].get('rows')}, columns={state['profile'].get('columns')}, "
        f"columns={state['profile'].get('columns_list')}\n"
        f"Deterministic decision: {deterministic}"
    )
    candidate, error = ollama_json(prompt, model, status["base_url"], timeout=20.0)
    if error:
        return None, f"Ollama reached at {status['base_url']} using model '{model}', but inference failed: {error}"
    if candidate and candidate.get("task") in {"classification", "regression", "clustering", "unknown"}:
        target = candidate.get("target")
        if target in state["profile"].get("columns_list", []) or (target is None and candidate.get("task") == "clustering"):
            candidate["llm_model"] = model
            candidate["llm_server"] = status["base_url"]
            return candidate, f"Ollama planner succeeded with model '{model}' ({model_reason})."
    return None, f"Ollama returned an unusable planner response from model '{model}'. Deterministic decision retained."


def planner_agent(state):
    target_override = state.get("target_override") or None
    deterministic = _deterministic_decision(state, target_override)
    decision = deterministic
    provider = (state.get("llm_provider") or os.getenv("LLM_PROVIDER", "auto")).lower()
    state["llm_provider"] = provider or "auto"

    if provider in {"auto", "ollama"}:
        candidate, message = _llm_refine(state, deterministic)
        if candidate:
            decision = candidate
            state["logs"].append(message)
        else:
            state["logs"].append(message)
            state["logs"].append("Planner Agent safely retained the deterministic decision.")
    elif provider == "offline":
        state["logs"].append("Planner Agent ran in offline deterministic mode.")
    elif provider == "openai":
        state["logs"].append("OpenAI provider selected, but this stable build keeps planning deterministic unless an external OpenAI integration is added.")

    state["task"] = decision["task"]
    state["target"] = decision.get("target")
    state["decision"] = decision
    plan = ["profile_dataset", "plan_workflow", "preprocess_data", "exploratory_analysis"]
    if decision["task"] in {"classification", "regression"}:
        plan += ["candidate_model_search", "cross_validation", "validation", "insights"]
    elif decision["task"] == "clustering":
        plan += ["cluster_search", "validation", "insights"]
    else:
        plan += ["human_target_selection"]
    state["plan"] = plan
    state["logs"].append(f"Planner Agent selected task='{decision['task']}', target='{decision.get('target')}', confidence='{decision.get('confidence')}'.")
    return state
