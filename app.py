import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from graph import build_graph

load_dotenv()
st.set_page_config(page_title="Agentic Data Analyst", layout="wide")
st.title("🤖 Agentic AI Data Analyst")
st.caption("Automatically detects classification, regression, or clustering from the dataset and user objective.")

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
goal = st.text_input(
    "Analysis objective",
    "Analyze the dataset, find important patterns, and build a suitable prediction model."
)

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_options = ["Auto-detect (recommended)"] + list(df.columns)
    target_choice = st.selectbox(
        "Target column (optional)",
        target_options,
        help="Choose a target for supervised learning, or leave Auto-detect for the agent to decide."
    )

    if st.button("Run Autonomous Analysis", type="primary"):
        target_override = None if target_choice == "Auto-detect (recommended)" else target_choice
        result = build_graph().invoke({
            "df": df,
            "goal": goal,
            "target_override": target_override,
            "profile": {},
            "plan": [],
            "task": "",
            "target": None,
            "decision": {},
            "eda": {},
            "model": {},
            "validation": {},
            "insights": "",
            "logs": []
        })

        st.subheader("Agent Decision")
        decision = result.get("decision", {})
        st.write(f"**Task:** {result.get('task', 'unknown').title()}")
        st.write(f"**Target:** {result.get('target') or 'None (unsupervised)'}")
        st.write(f"**Confidence:** {decision.get('confidence', 'unknown').title()}")
        st.info(decision.get("reason", "No decision explanation available."))

        st.subheader("Agent Plan")
        st.write(result["plan"])
        st.subheader("Profile")
        st.json(result["profile"])
        st.subheader("Model / Analysis")
        st.json(result["model"])
        st.subheader("Validation")
        st.json(result["validation"])
        st.subheader("Insights")
        st.markdown(result["insights"])
        st.subheader("Execution Log")
        for x in result["logs"]:
            st.write("•", x)
