import os
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from graph import build_graph
from agents.profile_agent import profile_agent
from agents.planner_agent import planner_agent
from agents.preprocess_agent import preprocess_agent
from agents.eda_agent import eda_agent
from agents.model_agent import model_agent
from agents.validation_agent import validation_agent
from agents.insight_agent import insight_agent
from agents.llm_utils import server_status, choose_model

load_dotenv()

ROOT = Path(__file__).resolve().parent

def dataset_signature(df):
    head_hash = int(pd.util.hash_pandas_object(df.head(25), index=True).sum()) if not df.empty else 0
    return f"{df.shape}|{tuple(map(str, df.columns))}|{head_hash}"

AGENT_ORDER = ["profile", "planner", "preprocess", "eda", "model", "validation", "insights"]
AGENT_META = {
    "profile": ("Profile Agent", "🔍", "Understand rows, columns, types, missingness, duplicates, and feature structure."),
    "planner": ("Planner Agent", "🧭", "Choose the analytical task and target from the objective and dataset profile."),
    "preprocess": ("Preprocessing Agent", "🧹", "Clean structure, normalize columns, remove duplicates, and prepare an analysis-ready view."),
    "eda": ("EDA Agent", "📊", "Generate statistics, distributions, correlations, and outlier signals."),
    "model": ("Model Agent", "🧠", "Search a broad model pool and select the strongest candidate with cross-validation."),
    "validation": ("Validation Agent", "✅", "Summarize evaluation quality, baselines, candidate failures, and validation evidence."),
    "insights": ("Insight Agent", "💡", "Turn the executed evidence into a concise, traceable analytical narrative."),
}

st.set_page_config(page_title="Agentic Data Analyst", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{--ink:#172033;--muted:#667085;--line:#e6eaf0;--panel:#fff;--soft:#f6f8fb;--accent:#5b5ce2;--accent2:#7c3aed;--good:#147d4a;--warn:#a15c00;--bad:#b42318}
.stApp{background:linear-gradient(180deg,#f8f9fc 0%,#f3f5f9 100%)}
[data-testid="stSidebar"]{background:#0f172a;border-right:1px solid #1f2937}
[data-testid="stSidebar"] *{color:#f8fafc!important}
.brand{display:flex;gap:12px;align-items:center;margin-bottom:14px}.brand-icon{width:42px;height:42px;border-radius:13px;display:grid;place-items:center;background:linear-gradient(135deg,#6366f1,#8b5cf6);font-size:22px;box-shadow:0 10px 24px rgba(99,102,241,.25)}.brand-title{font-weight:800;font-size:18px}.brand-sub{font-size:12px;color:#a7b0c2!important;margin-top:2px}
.side-section{margin:22px 0 8px;font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#94a3b8!important;font-weight:800}
.agent-btn{padding:8px 10px;border-radius:12px;border:1px solid #273247;background:rgba(255,255,255,.03);margin-bottom:7px}.agent-btn-active{background:linear-gradient(135deg,rgba(99,102,241,.24),rgba(139,92,246,.18));border-color:#5b5ce2}.agent-status{float:right;font-size:11px;color:#9fb0c7!important}
.hero{background:linear-gradient(135deg,#fff 0%,#f2f3ff 65%,#f7f1ff 100%);border:1px solid #e4e7ec;border-radius:24px;padding:30px 32px;margin-bottom:18px;box-shadow:0 12px 34px rgba(15,23,42,.06)}.eyebrow{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#5b5ce2}.hero h1{font-size:38px;line-height:1.06;margin:8px 0 10px;color:var(--ink);letter-spacing:-.035em}.hero p{max-width:850px;color:var(--muted);line-height:1.6;font-size:15px;margin:0}
.panel{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 6px 18px rgba(15,23,42,.035);margin-bottom:16px}.panel-title{color:var(--ink);font-size:17px;font-weight:800;margin-bottom:4px}.panel-sub{color:var(--muted);font-size:12px;margin-bottom:14px}.metric-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:15px 16px;min-height:92px}.metric-label{color:#7b8497;font-size:10px;text-transform:uppercase;letter-spacing:.08em;font-weight:800}.metric-value{font-size:28px;line-height:1.1;font-weight:800;color:var(--ink);margin-top:8px}.metric-note{color:#8b95a7;font-size:11px;margin-top:4px}
.decision-card{border:1px solid #dfe2ff;background:#f7f7ff;border-radius:16px;padding:16px;height:100%}.decision-kicker{font-size:10px;text-transform:uppercase;letter-spacing:.09em;font-weight:800;color:#6c6fd4}.decision-main{font-size:22px;font-weight:800;color:#252a59;margin-top:7px}.decision-note{color:#5f6780;font-size:12px;line-height:1.45;margin-top:5px}
.agent-page{border:1px solid #dfe2ff;background:linear-gradient(135deg,#fafaff,#fff);border-radius:18px;padding:18px;margin-bottom:14px}.agent-head{display:flex;align-items:center;gap:12px}.agent-symbol{font-size:30px}.agent-name{font-size:22px;font-weight:800;color:var(--ink)}.agent-desc{font-size:13px;color:var(--muted);margin-top:3px}.status-pill{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:6px 10px;font-size:11px;font-weight:800;background:#edf8f1;color:#147d4a}.status-pill-warn{background:#fff6e8;color:#a15c00}.status-pill-off{background:#eef2f6;color:#667085}
.stButton>button{border-radius:11px;font-weight:750}
[data-testid="stSidebar"] .stButton>button:not([kind="primary"]){
    background:#1b2538 !important;color:#f8fafc !important;border:1px solid #334155 !important;
    box-shadow:none !important;text-align:left !important;
}
[data-testid="stSidebar"] .stButton>button:not([kind="primary"]):hover{
    background:#25324a !important;border-color:#64748b !important;color:#ffffff !important;
}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#6366f1,#7c3aed) !important;color:#ffffff !important;
    border:1px solid #818cf8 !important;box-shadow:0 8px 20px rgba(99,102,241,.22) !important;
    text-align:left !important;
}
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{filter:brightness(1.06)}
.run-card{border:1px dashed #b9c1d0;background:#fbfcfe;border-radius:16px;padding:14px}.small-note{font-size:11px;color:#7b8497;line-height:1.5}.tag{display:inline-block;background:#eef0ff;color:#4c46c7;border-radius:999px;padding:5px 9px;margin:3px 5px 3px 0;font-size:11px;font-weight:700}.leader-best{background:#eefaf3;color:#147d4a;border-radius:7px;padding:2px 7px;font-size:10px;font-weight:800}
</style>
""", unsafe_allow_html=True)


def fresh_state(df, goal, target_override, llm_provider):
    return {
        "df": df.copy(), "source_df": df.copy(), "goal": goal,
        "target_override": target_override, "llm_provider": llm_provider, "llm_model": None, "model_search_mode": "balanced",
        "profile": {}, "preprocessing": {}, "plan": [], "task": "", "target": None,
        "decision": {}, "eda": {}, "model": {}, "validation": {}, "insights": "", "logs": [],
    }


def current_state_from_ui():
    if st.session_state.get("agent_state") is None:
        return None
    state = st.session_state.agent_state
    state["goal"] = st.session_state.get("objective", state.get("goal", ""))
    target = st.session_state.get("target_choice")
    state["target_override"] = None if not target or target == "Auto-detect (recommended)" else target
    state["llm_provider"] = st.session_state.get("llm_provider", "auto")
    state["llm_model"] = st.session_state.get("llm_model") or None
    state["model_search_mode"] = st.session_state.get("model_search_mode", "balanced")
    return state


def run_named_agent(agent_key):
    if st.session_state.get("df") is None:
        st.error("Upload a dataset first.")
        return
    state = current_state_from_ui()
    if state is None:
        state = fresh_state(
            st.session_state.df,
            st.session_state.get("objective", "Analyze the dataset and find useful patterns."),
            None,
            st.session_state.get("llm_provider", "auto"),
        )
    fns = {"profile": profile_agent, "planner": planner_agent, "preprocess": preprocess_agent, "eda": eda_agent, "model": model_agent, "validation": validation_agent, "insights": insight_agent}
    prerequisites = {
        "profile": [], "planner": ["profile"], "preprocess": ["profile", "planner"], "eda": ["profile", "planner", "preprocess"],
        "model": ["profile", "planner", "preprocess", "eda"], "validation": ["profile", "planner", "preprocess", "eda", "model"],
        "insights": ["profile", "planner", "preprocess", "eda", "model", "validation"],
    }
    for req in prerequisites[agent_key]:
        if not _agent_done(state, req):
            state = fns[req](state)
    state = fns[agent_key](state)
    state["agent_done"] = {k: _agent_done(state, k) for k in AGENT_ORDER}
    st.session_state.agent_state = state
    st.session_state.result = state
    st.session_state.run_time = datetime.now()


def _agent_done(state, key):
    mapping = {"profile": "profile", "planner": "decision", "preprocess": "preprocessing", "eda": "eda", "model": "model", "validation": "validation", "insights": "insights"}
    value = state.get(mapping[key])
    if key == "insights":
        return bool(value)
    return bool(value)


def run_full_workflow():
    state = fresh_state(
        st.session_state.df,
        st.session_state.get("objective", "Analyze the dataset and build a suitable analysis workflow."),
        None if st.session_state.get("target_choice") in (None, "Auto-detect (recommended)") else st.session_state.target_choice,
        st.session_state.get("llm_provider", "auto"),
    )
    state["model_search_mode"] = st.session_state.get("model_search_mode", "balanced")
    state["llm_model"] = st.session_state.get("llm_model") or None
    progress = st.progress(0, text="Starting agent workflow...")
    stage_box = st.empty()
    stages = ["profile", "planner", "preprocess", "eda", "model", "validation", "insights"]
    stage_labels = {
        "profile":"Profile Agent", "planner":"Planner Agent", "preprocess":"Preprocessing Agent",
        "eda":"EDA Agent", "model":"Model Agent", "validation":"Validation Agent", "insights":"Insight Agent"
    }
    completed = set()
    def on_progress(name, status):
        if status == "running":
            stage_box.info(f"Running **{stage_labels.get(name, name)}**...")
        elif status == "completed":
            completed.add(name)
            progress.progress(len(completed)/len(stages), text=f"Completed {stage_labels.get(name, name)}")
        elif status == "failed":
            completed.add(name)
            progress.progress(len(completed)/len(stages), text=f"{stage_labels.get(name, name)} failed safely")
    graph = build_graph(progress_callback=on_progress)
    state = graph.invoke(state)
    state["agent_done"] = {k: _agent_done(state, k) for k in AGENT_ORDER}
    st.session_state.agent_state = state
    st.session_state.result = state
    st.session_state.run_time = datetime.now()
    stage_box.success("Workflow finished. Check the Agent Trace for stage-level details.")

# session
for key, default in {
    "result": None, "agent_state": None, "df": None, "run_time": None,
    "objective": "Analyze the dataset, find important patterns, and build a suitable prediction model.",
    "active_agent": "profile", "target_choice": "Auto-detect (recommended)", "model_search_mode": "balanced", "llm_provider": os.getenv("LLM_PROVIDER", "auto").lower() or "auto", "llm_model": os.getenv("OLLAMA_MODEL", "") or "", "dataset_signature": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# sidebar
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">🤖</div><div><div class="brand-title">Agentic Data Analyst</div><div class="brand-sub">Interactive multi-agent workspace</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-section">Agent control panel</div>', unsafe_allow_html=True)
    done_map = st.session_state.get("agent_state", {}).get("agent_done", {}) if st.session_state.get("agent_state") else {}
    for key in AGENT_ORDER:
        title, icon, desc = AGENT_META[key]
        active = st.session_state.active_agent == key
        label = f"{icon}  {title}" + ("  ✓" if done_map.get(key) else "")
        if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.active_agent = key
            st.rerun()
    st.markdown('<div class="side-section">Runtime</div>', unsafe_allow_html=True)
    st.session_state.llm_provider = st.selectbox("LLM provider", ["auto", "ollama", "offline", "openai"], index=["auto","ollama","offline","openai"].index(st.session_state.llm_provider) if st.session_state.llm_provider in ["auto","ollama","offline","openai"] else 0, label_visibility="collapsed")
    if st.session_state.llm_provider in {"auto", "ollama"}:
        st.session_state.llm_model = st.text_input("Ollama model (optional)", value=st.session_state.get("llm_model", ""), placeholder="e.g. qwen2.5:1.5b", label_visibility="collapsed")
        with st.expander("Ollama connection", expanded=False):
            status = server_status(os.getenv("OLLAMA_BASE_URL"))
            if status.get("server_ok"):
                chosen, why = choose_model(status, st.session_state.llm_model)
                st.success(f"Ollama connected · {status.get('version') or 'version unknown'}")
                st.caption(f"Models: {', '.join(status.get('models', [])) or 'none installed'}")
                if chosen:
                    st.caption(f"Model selected: {chosen} · {why}")
                else:
                    st.warning(why)
            else:
                st.warning(f"Ollama server check failed: {status.get('version_error')}")
    if st.button("↻ Reset workspace", use_container_width=True):
        for k in ["result", "agent_state", "df", "run_time", "dataset_signature"]:
            st.session_state[k] = None
        st.session_state.active_agent = "profile"
        st.rerun()
    if st.session_state.run_time:
        st.caption(f"Last run: {st.session_state.run_time.strftime('%d %b %Y • %H:%M')}")

# hero
st.markdown('<div class="hero"><div class="eyebrow">✦ Autonomous Data Science</div><h1>Let the agents decide what the data needs.</h1><p>Run the complete workflow or open any agent individually. The Model Agent searches a broad model pool, validates candidates with cross-validation, and explains why the winner was selected.</p></div>', unsafe_allow_html=True)

# input area
left, right = st.columns([1.2, 1], gap="large")
with left:
    st.markdown('<div class="panel"><div class="panel-title">1. Load dataset</div><div class="panel-sub">CSV input for the current project. A demo dataset is available for a quick test.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="visible")
    demo = st.selectbox("Demo dataset", ["None", "Classification demo", "Regression demo"])
    if demo == "Classification demo":
        new_df = pd.read_csv(ROOT / "demo_classification.csv")
        sig = dataset_signature(new_df)
        if sig != st.session_state.dataset_signature:
            st.session_state.agent_state = None
            st.session_state.result = None
            st.session_state.dataset_signature = sig
        st.session_state.df = new_df
        st.caption("Loaded built-in classification demo.")
    elif demo == "Regression demo":
        new_df = pd.read_csv(ROOT / "demo_regression.csv")
        sig = dataset_signature(new_df)
        if sig != st.session_state.dataset_signature:
            st.session_state.agent_state = None
            st.session_state.result = None
            st.session_state.dataset_signature = sig
        st.session_state.df = new_df
        st.caption("Loaded built-in regression demo.")
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            sig = dataset_signature(new_df)
            if sig != st.session_state.dataset_signature:
                st.session_state.agent_state = None
                st.session_state.result = None
                st.session_state.dataset_signature = sig
            st.session_state.df = new_df
            st.success(f"Loaded {uploaded.name} · {len(st.session_state.df):,} rows × {len(st.session_state.df.columns):,} columns")
        except Exception as exc:
            st.session_state.df = None
            st.error(f"Could not read CSV: {exc}")
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(8), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="panel-title">2. Set the objective</div><div class="panel-sub">The Planner Agent combines this objective with the dataset profile.</div>', unsafe_allow_html=True)
    quick = st.selectbox("Quick objective", [
        "Custom objective", "Predict a target variable", "Find important patterns and compare models", "Segment the records into groups", "Explore the dataset and explain major insights"
    ], label_visibility="collapsed")
    defaults = {
        "Predict a target variable": "Predict the target variable using the most suitable supervised learning approach.",
        "Find important patterns and compare models": "Analyze the dataset, find important patterns, and compare suitable models to identify the strongest candidate.",
        "Segment the records into groups": "Group similar records and identify meaningful segments in the dataset.",
        "Explore the dataset and explain major insights": "Explore the dataset, explain important patterns, and identify notable relationships.",
    }
    if quick != "Custom objective":
        st.session_state.objective = defaults[quick]
    st.session_state.objective = st.text_area("Objective", value=st.session_state.objective, height=110, label_visibility="collapsed")
    target_options = ["Auto-detect (recommended)"] + (list(st.session_state.df.columns) if st.session_state.df is not None else [])
    current = st.session_state.get("target_choice", "Auto-detect (recommended)")
    if current not in target_options:
        current = "Auto-detect (recommended)"
    st.session_state.target_choice = st.selectbox("Target column", target_options, index=target_options.index(current), help="Explicit target selection overrides auto-detection.")
    mode_options = ["fast", "balanced", "deep"]
    mode_help = {
        "fast": "6-7 diverse models with 2-fold CV. Best for quick demos.",
        "balanced": "A broad model pool with 3-fold CV. Recommended for most datasets.",
        "deep": "All supported candidates with up to 5-fold CV. Slowest, most exhaustive search.",
    }
    current_mode = st.session_state.get("model_search_mode", "balanced")
    st.session_state.model_search_mode = st.select_slider("Model search depth", options=mode_options, value=current_mode, format_func=lambda x: x.title(), help=mode_help[current_mode])
    st.caption(mode_help[st.session_state.model_search_mode])
    run_full = st.button("🚀 Run Full Autonomous Workflow", type="primary", use_container_width=True, disabled=st.session_state.df is None)
    if run_full:
        with st.status("Running all agents...", expanded=True) as status:
            try:
                run_full_workflow()
                status.update(label="All agents completed", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Workflow failed", state="error", expanded=True)
                st.exception(exc)
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.df is None and st.session_state.result is None:
    st.markdown('<div class="panel" style="text-align:center;padding:48px"><div style="font-size:42px">🧭</div><h3>Choose a dataset and start an agent</h3><p class="small-note">You can run the full pipeline or click a single agent from the left control panel.</p></div>', unsafe_allow_html=True)
    st.stop()

# active agent workspace
state = st.session_state.get("agent_state") or {}
active = st.session_state.active_agent
active_title, active_icon, active_desc = AGENT_META[active]
done = _agent_done(state, active)

st.markdown(f'<div class="agent-page"><div class="agent-head"><div class="agent-symbol">{active_icon}</div><div><div class="agent-name">{active_title}</div><div class="agent-desc">{active_desc}</div></div></div><div style="margin-top:12px"><span class="status-pill {"" if done else "status-pill-off"}">{"✓ Ready / completed" if done else "○ Not run yet"}</span></div></div>', unsafe_allow_html=True)

# Agent-specific execution
if st.button(f"▶ Run {active_title}", type="primary", use_container_width=True, disabled=st.session_state.df is None):
    with st.status(f"Running {active_title}...", expanded=True) as status:
        try:
            run_named_agent(active)
            status.update(label=f"{active_title} completed", state="complete", expanded=False)
        except Exception as exc:
            status.update(label=f"{active_title} failed", state="error", expanded=True)
            st.exception(exc)
    st.rerun()

state = st.session_state.get("agent_state") or {}
profile = state.get("profile", {})
prep = state.get("preprocessing", {})
decision = state.get("decision", {})
eda = state.get("eda", {})
model = state.get("model", {})
validation = state.get("validation", {})

# selected agent body
if active == "profile":
    if profile:
        m1, m2, m3, m4 = st.columns(4)
        cards = [(m1,"Rows",profile.get("rows",0),"observations"),(m2,"Columns",profile.get("columns",0),"features + target"),(m3,"Missing",profile.get("missing_total",0),"blank cells"),(m4,"Duplicates",profile.get("duplicates",0),"duplicate rows")]
        for c,l,v,n in cards:
            with c: st.markdown(f'<div class="metric-card"><div class="metric-label">{l}</div><div class="metric-value">{v:,}</div><div class="metric-note">{n}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="panel-title">Column profile</div>', unsafe_allow_html=True)
        rows=[]
        for c in profile.get("columns_list",[]):
            rows.append({"Column":c,"Type":profile.get("dtypes",{}).get(c,""),"Unique":profile.get("nunique",{}).get(c,0),"Missing":profile.get("missing_values",{}).get(c,0)})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.markdown('</div>',unsafe_allow_html=True)
    else: st.info("Run the Profile Agent to inspect the dataset.")

elif active == "planner":
    if decision:
        a,b,c=st.columns(3)
        a.metric("Task",str(state.get("task","unknown")).title())
        b.metric("Target",state.get("target") or "None")
        c.metric("Confidence",str(decision.get("confidence","unknown")).title())
        st.markdown('<div class="panel"><div class="panel-title">Decision reasoning</div>',unsafe_allow_html=True)
        st.write(decision.get("reason","No reason available."))
        st.markdown('</div>',unsafe_allow_html=True)
        if state.get("plan"):
            st.markdown('<div class="panel"><div class="panel-title">Adaptive execution plan</div>',unsafe_allow_html=True)
            for i,step in enumerate(state["plan"],1): st.markdown(f"**{i:02d}.** {step.replace('_',' ').title()}")
            st.markdown('</div>',unsafe_allow_html=True)
    else: st.info("Run the Planner Agent. It will profile first if necessary.")

elif active == "preprocess":
    if prep:
        a,b,c,d=st.columns(4)
        a.metric("Duplicates removed",prep.get("duplicates_removed",0))
        b.metric("Numeric missing filled",len(prep.get("numeric_missing_filled",[])))
        c.metric("Categorical missing filled",len(prep.get("categorical_missing_filled",[])))
        d.metric("Missing cells remaining",prep.get("remaining_missing",0))
        if prep.get("fully_processed"):
            st.success("✓ Downloadable dataset is fully processed and contains zero missing cells.")
        else:
            st.warning("Some missing cells remain and require review.")
        st.markdown('<div class="panel"><div class="panel-title">Preprocessing actions</div><div class="panel-sub">The downloadable dataset is fully processed. Model evaluation keeps feature imputation inside each CV fold to avoid data leakage.</div>',unsafe_allow_html=True)
        for x in prep.get("actions",[]): st.markdown(f"- {x}")
        if prep.get("target_rows_removed", 0):
            st.info(f"{prep['target_rows_removed']:,} rows with a missing target were removed instead of fabricating target values.")
        processed = st.session_state.get("agent_state",{}).get("processed_df")
        if processed is not None:
            csv=processed.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download fully preprocessed dataset",csv,"fully_preprocessed_dataset.csv","text/csv",use_container_width=True)
            st.caption(f"{len(processed):,} rows × {len(processed.columns):,} columns · {int(processed.isna().sum().sum())} missing cells")
        st.markdown('</div>',unsafe_allow_html=True)
    else: st.info("Run the Preprocessing Agent.")

elif active == "eda":
    if eda:
        e1,e2=st.columns(2)
        with e1:
            st.markdown('<div class="panel"><div class="panel-title">Numeric summary</div>',unsafe_allow_html=True)
            if eda.get("summary"): st.dataframe(pd.DataFrame(eda["summary"]).T,use_container_width=True)
            else: st.info("No numeric columns.")
            st.markdown('</div>',unsafe_allow_html=True)
        with e2:
            st.markdown('<div class="panel"><div class="panel-title">Top correlations</div>',unsafe_allow_html=True)
            if eda.get("top_correlations"): st.dataframe(pd.DataFrame(eda["top_correlations"]),use_container_width=True,hide_index=True)
            else: st.info("No pairwise numeric correlations available after excluding identifier-like columns.")
            if eda.get("excluded_identifier_columns"):
                st.caption("Excluded identifiers: " + ", ".join(eda["excluded_identifier_columns"]))
            st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="panel-title">Outlier scan (IQR)</div>',unsafe_allow_html=True)
        if eda.get("outliers_iqr"): st.dataframe(pd.DataFrame([{"Feature":k,"Outliers":v} for k,v in eda["outliers_iqr"].items()]).sort_values("Outliers",ascending=False),use_container_width=True,hide_index=True)
        else: st.info("No usable numeric outlier scan.")
        st.markdown('</div>',unsafe_allow_html=True)
    else: st.info("Run the EDA Agent.")

elif active == "model":
    if model:
        if model.get("status")=="completed":
            a,b,c,d,e=st.columns(5)
            selected = model.get("model", "n/a")
            a.metric("Task",str(model.get("task","" )).title())
            b.metric("Selected model",selected)
            c.metric(model.get("metric","Score"),model.get("score","n/a"))
            d.metric("Models evaluated",model.get("models_evaluated", model.get("model_count",0)))
            e.metric("CV folds", model.get("cv_folds", "—"))
            sample_note = ""
            if model.get("sampled"):
                sample_note = f" · Selection rows: {model.get('selection_rows', 0):,} / {model.get('original_rows', 0):,}"
            st.caption('Search mode: **' + str(model.get('search_mode', 'balanced')).title() + '** · ' + str(model.get('cv', 'evaluation not available')) + sample_note)
            if model.get("useful_model") is False and model.get("best_candidate"):
                st.warning(f"No real candidate beat the dummy baseline. Best real candidate: **{model.get('best_candidate')}** with {model.get('score')}. The dummy baseline is not promoted as the best model.")
            if model.get("quality_warning"):
                st.warning("⚠️ " + str(model["quality_warning"]))
            st.markdown('<div class="panel"><div class="panel-title">Model leaderboard</div><div class="panel-sub">Sorted by the primary selection metric. Cross-validation prevents choosing a model from a single lucky split.</div>',unsafe_allow_html=True)
            candidates=pd.DataFrame(model.get("candidates") or [])
            if not candidates.empty:
                candidates.insert(0,"Rank",range(1,len(candidates)+1))
                st.dataframe(candidates,use_container_width=True,hide_index=True)
            if model.get("errors"):
                with st.expander(f"{len(model['errors'])} candidate models skipped"):
                    for e in model["errors"]: st.write(e)
            leak = model.get("leakage", {})
            with st.expander("Leakage & feature-quality checks", expanded=bool(leak.get("suspicious"))):
                if model.get("excluded_id_columns"):
                    st.write("**Identifier-like columns excluded from modeling:**", ", ".join(model["excluded_id_columns"]))
                if leak.get("suspicious"):
                    st.error("Potential target leakage detected. Review these features before trusting a near-perfect score.")
                    st.dataframe(pd.DataFrame(leak.get("findings", [])), use_container_width=True, hide_index=True)
                else:
                    st.success(leak.get("note", "No strong leakage signal detected."))
            if model.get("replan"):
                rp=model["replan"]
                st.warning(f"🧭 Agentic replanning suggestion: **{rp.get('recommended_task','unknown').title()}**. {rp.get('reason','')}")
            st.markdown('</div>',unsafe_allow_html=True)
            if model.get("feature_importance"):
                st.markdown('<div class="panel"><div class="panel-title">Why the winner?</div><div class="panel-sub">Top features from the fitted winning model. This is descriptive importance, not causality.</div>',unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(model["feature_importance"]),use_container_width=True,hide_index=True)
                st.markdown('</div>',unsafe_allow_html=True)
        else: st.warning(model.get("reason","Modeling did not complete."))
    else: st.info("Run the Model Agent. It will prepare prerequisites automatically.")

elif active == "validation":
    if validation:
        st.markdown('<div class="panel"><div class="panel-title">Validation summary</div>',unsafe_allow_html=True)
        st.json(validation)
        st.markdown('</div>',unsafe_allow_html=True)
    else: st.info("Run the Validation Agent.")

elif active == "insights":
    if state.get("insights"):
        st.markdown('<div class="panel"><div class="panel-title">Analyst report</div><div class="panel-sub">Generated from the actual agent outputs. LLM text is optional and never supplies metrics.</div>',unsafe_allow_html=True)
        st.markdown(state["insights"])
        st.markdown('</div>',unsafe_allow_html=True)
    else: st.info("Run the Insight Agent.")

# common sections
if state:
    st.markdown('<div class="panel"><div class="panel-title">Workflow trace</div><div class="panel-sub">Every completed action is recorded below.</div>',unsafe_allow_html=True)
    for entry in state.get("logs", [])[-20:]: st.markdown(f"- {entry}")
    st.markdown('</div>',unsafe_allow_html=True)

    with st.expander("Raw agent state"):
        safe = {k:v for k,v in state.items() if k not in {"df","source_df","processed_df","model_df"}}
        st.json(safe)
