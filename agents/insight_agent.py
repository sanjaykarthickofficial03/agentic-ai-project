from __future__ import annotations

import os

from .llm_utils import choose_model, ollama_text, server_status


def _compact_evidence(state: dict) -> str:
    """Build a small evidence packet for the local LLM. Never send full raw state."""
    p = state.get("profile", {})
    prep = state.get("preprocessing", {})
    eda = state.get("eda", {})
    m = state.get("model", {})
    v = state.get("validation", {})

    lines = [
        f"Rows: {p.get('rows', 0)}",
        f"Columns: {p.get('columns', 0)}",
        f"Initial missing cells: {p.get('missing_total', 0)}",
        f"Duplicates before preprocessing: {p.get('duplicates', 0)}",
        f"Missing cells after preprocessing: {prep.get('remaining_missing', 0)}",
        f"Task: {state.get('task', 'unknown')}",
        f"Target: {state.get('target') or 'None'}",
    ]

    if m:
        lines.extend([
            f"Model status: {m.get('status')}",
            f"Selected model: {m.get('selected_model') or m.get('model') or 'None'}",
            f"Best candidate: {m.get('best_candidate') or 'None'}",
            f"Primary metric: {m.get('metric') or 'None'}",
            f"Primary score: {m.get('score') if m.get('score') is not None else 'None'}",
            f"Models evaluated: {m.get('models_evaluated', m.get('model_count', 0))}",
            f"CV: {m.get('cv', 'None')}",
            f"Useful model: {m.get('useful_model')}",
            f"Baseline comparison: {m.get('baseline_summary', 'not available')}",
        ])
        if m.get("quality_warning"):
            lines.append(f"Quality warning: {m['quality_warning']}")
        if m.get("replan"):
            lines.append(f"Replan recommendation: {m['replan'].get('recommended_task')} | {m['replan'].get('reason')}")

    if v:
        lines.extend([
            f"Validation metric: {v.get('metric', 'None')}",
            f"Validation score: {v.get('score', v.get('baseline', 'None'))}",
        ])

    corr = eda.get("top_correlations") or []
    if corr:
        top = corr[:3]
        lines.append("Top numeric relationships: " + "; ".join(
            f"{x.get('feature_a')} <-> {x.get('feature_b')} (r={x.get('correlation')})" for x in top
        ))
    outliers = eda.get("outliers_iqr") or {}
    if outliers:
        notable = sorted(outliers.items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append("Largest IQR outlier counts: " + ", ".join(f"{k}={v}" for k, v in notable))
    if eda.get("excluded_identifier_columns"):
        lines.append("Identifier-like numeric fields excluded from relationship analysis.")

    return "\n".join(lines)


def _deterministic_insights(state):
    p = state.get("profile", {})
    prep = state.get("preprocessing", {})
    eda = state.get("eda", {})
    m = state.get("model", {})
    lines = [
        "### Automated Summary",
        f"- Dataset contains **{p.get('rows', 0):,} rows** and **{p.get('columns', 0):,} columns**.",
        f"- Initial profile contained **{p.get('missing_total', 0):,} missing cells** and **{p.get('duplicates', 0):,} duplicate rows**.",
        f"- Preprocessing produced a downloadable dataset with **{prep.get('remaining_missing', 0):,} missing cells remaining** and removed **{prep.get('duplicates_removed', 0):,} duplicate rows**.",
        f"- Planner selected **{state.get('task', 'unknown')}**" + (f" with target **{state.get('target')}**." if state.get('target') else "."),
        f"- Model search evaluated **{m.get('models_evaluated', m.get('model_count', 0))} models**." if m else "- Model stage was not completed.",
    ]
    if m.get("status") == "completed":
        shown = m.get("selected_model") or m.get("best_candidate") or m.get("model")
        lines.append(f"- Model selection result: **{shown or 'No useful real model'}** with **{m.get('metric')} = {m.get('score')}**.")
        if m.get("useful_model") is False:
            lines.append("- No real candidate clearly outperformed the dummy baseline; the system treated the result conservatively.")
        if m.get("baseline"):
            metric = "f1_weighted_mean" if m.get("task") == "classification" else "R2_mean"
            base = m["baseline"].get(metric)
            if base is not None:
                lines.append(f"- Dummy baseline primary score: **{float(base):.4f}**.")
        if m.get("baseline_summary"):
            lines.append(f"- Baseline comparison: **{m['baseline_summary']}**")
        if m.get("quality_warning"):
            lines.append(f"- **Quality warning:** {m.get('quality_warning')}")
        if m.get("replan"):
            lines.append(f"- **Replanning suggestion:** {m['replan'].get('recommended_task')}: {m['replan'].get('reason')}")
    if eda.get("top_correlations"):
        strongest = eda["top_correlations"][0]
        lines.append(f"- Strongest observed linear relationship: **{strongest['feature_a']} ↔ {strongest['feature_b']} (r={strongest['correlation']})**.")
    if eda.get("excluded_identifier_columns"):
        lines.append("- Identifier-like fields were excluded from correlation analysis to avoid meaningless numeric relationships.")
    lines.append("- These findings describe the analysed dataset and model evaluation; they are not causal claims.")
    return "\n".join(lines)


def insight_agent(state):
    provider = (state.get("llm_provider") or os.getenv("LLM_PROVIDER", "auto")).lower()
    text = None
    if provider in {"auto", "ollama"}:
        status = server_status(os.getenv("OLLAMA_BASE_URL"))
        if status["server_ok"]:
            model, reason = choose_model(status, os.getenv("OLLAMA_MODEL", ""))
            if model:
                evidence = _compact_evidence(state)
                prompt = (
                    "You are the Insight Agent in an autonomous tabular-data analysis system. "
                    "Write a concise Markdown report (maximum 180 words) with: key findings, "
                    "model interpretation, up to three useful insights, limitations, and next steps. "
                    "Use only the supplied evidence. Do not invent metrics or facts. "
                    "Do not repeat the evidence line-by-line.\n\n"
                    f"User goal: {state.get('goal', '')}\n"
                    f"Evidence:\n{evidence}"
                )
                # Insight is optional. Keep a strict latency budget so the analytical workflow
                # remains useful even on CPU-only machines running a local model.
                text, error = ollama_text(
                    prompt, model, status["base_url"], timeout=18.0, num_predict=220
                )
                if text:
                    state["logs"].append(f"Insight Agent used Ollama model '{model}' ({reason}) with a compact evidence prompt.")
                else:
                    state["logs"].append(f"Insight Agent used deterministic fallback; Ollama inference failed: {error}")
            else:
                state["logs"].append(f"Insight Agent used deterministic fallback: {reason}")
        else:
            state["logs"].append(f"Insight Agent used deterministic fallback; Ollama server status: {status['version_error']}")
    elif provider == "openai" and os.getenv("OPENAI_API_KEY"):
        state["logs"].append("Insight Agent used deterministic mode in this stable build; cloud inference is intentionally not required for completion.")
    else:
        state["logs"].append("Insight Agent ran in deterministic offline mode.")
    state["insights"] = text or _deterministic_insights(state)
    state["logs"].append("Insight Agent generated the analytical summary.")
    return state
