def profile_agent(state):
    df = state["df"]
    missing = df.isna().sum().to_dict()
    state["profile"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "columns_list": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_values": {str(k): int(v) for k, v in missing.items()},
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "nunique": {str(k): int(v) for k, v in df.nunique(dropna=True).items()},
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(exclude="number").columns),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 3),
    }
    state["logs"].append("Profile Agent completed dataset profiling.")
    return state
