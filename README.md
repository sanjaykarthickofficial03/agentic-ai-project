# Agentic Data Analyst - Stable Build

This build provides a reliable multi-agent tabular-data workflow with optional local Ollama reasoning.

## Workflow
Profile -> Planner -> Preprocessing -> EDA -> Model -> Validation -> Insights

## Fully preprocessed dataset
The Preprocessing Agent now creates `processed_df`, a user-facing dataset with zero missing cells. Feature missing values are filled using median/mode. Rows with a missing target are dropped rather than inventing labels or values. This processed dataframe is downloadable from the Preprocessing Agent page.

For machine-learning evaluation, the Model Agent uses a separate `model_df`: target-missing rows are removed, while feature missing values remain and are imputed inside the model pipeline during each CV fold. This prevents leakage.

## Ollama
The planner and insight stages use native Ollama endpoints when available. The app checks `/api/version` and `/api/tags`, verifies the requested model is installed, and falls back to deterministic reasoning when Ollama or the selected model is unavailable.

Suggested setup:

```powershell
ollama serve
ollama pull qwen2.5:1.5b
python diagnose_ollama.py
streamlit run app.py
```

`OLLAMA_MODEL=` can be left blank. Auto mode selects an installed small model when possible.

## Large datasets
Model search uses adaptive representative sampling for large datasets, removes identifier-like columns from modeling, skips poorly scaling algorithms where appropriate, and keeps cross-validation at a controlled resource level.
## Latest reliability upgrades

- Insight Agent sends a compact evidence packet instead of the full analytical state, uses a bounded local-LLM response budget, and falls back immediately to deterministic insights when inference is slow/unavailable.
- Classification/regression model selection is tie-aware: if a real model ties the DummyClassifier/DummyRegressor on the primary metric and does not materially lose on the secondary metric, the real model is surfaced as the recommendation.
- The model result exposes the baseline comparison and selection reason so the UI does not misleadingly report a dummy model as the winner.

