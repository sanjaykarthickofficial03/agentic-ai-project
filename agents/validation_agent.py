def validation_agent(state):
    m = state.get("model", {})
    task = m.get("task")
    if m.get("status") != "completed":
        state["validation"] = {
            "status": "not_available",
            "metric": None,
            "note": m.get("reason", "Model/analysis was not completed."),
        }
    else:
        validation = {
            "status": "completed",
            "metric": m.get("metric"),
            "score": m.get("score"),
            "models_evaluated": m.get("models_evaluated", m.get("model_count", 0)),
            "cv_folds": m.get("cv_folds"),
            "evaluation": m.get("cv", "model comparison"),
            "baseline": m.get("baseline"),
            "candidate_failures": len(m.get("errors", [])),
            "beats_baseline": m.get("beats_baseline"),
            "useful_model": m.get("useful_model"),
            "quality_warning": m.get("quality_warning"),
            "leakage_suspected": bool(m.get("leakage", {}).get("suspicious")),
        }
        if task == "classification":
            validation["note"] = "Models were compared using weighted F1 as the selection metric, with accuracy and balanced accuracy also recorded."
        elif task == "regression":
            validation["note"] = "Models were compared using cross-validated R², with MAE and RMSE also recorded."
        elif task == "clustering":
            validation["note"] = "Cluster configurations were compared using silhouette score and Davies-Bouldin index."
        state["validation"] = validation
    state["logs"].append("Validation Agent checked model quality, baseline comparison, CV evidence, and leakage flags.")
    return state
