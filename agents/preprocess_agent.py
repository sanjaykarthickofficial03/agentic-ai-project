import re
import numpy as np
import pandas as pd


def _clean_columns(columns):
    seen = {}
    out = []
    mapping = {}
    for col in columns:
        base = re.sub(r"\s+", "_", str(col).strip()) or "unnamed"
        base = re.sub(r"[^0-9A-Za-z_]+", "", base) or "unnamed"
        idx = seen.get(base, 0)
        name = base if idx == 0 else f"{base}_{idx + 1}"
        seen[base] = idx + 1
        out.append(name)
        mapping[col] = name
    return out, mapping


def _coerce_numeric_like(df):
    converted = []
    for col in list(df.select_dtypes(exclude="number").columns):
        s = df[col]
        numeric = pd.to_numeric(s, errors="coerce")
        non_null = int(s.notna().sum())
        if non_null >= 5 and numeric.notna().sum() / non_null >= 0.95:
            df[col] = numeric
            converted.append(col)
    return converted


def _fill_missing(df, feature_columns):
    numeric = []
    categorical = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(0 if pd.isna(median) else median)
                numeric.append(col)
        else:
            if df[col].isna().any():
                mode = df[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else "__MISSING__"
                df[col] = df[col].fillna(fill)
                categorical.append(col)
    return numeric, categorical


def preprocess_agent(state):
    original = state.get("source_df", state["df"]).copy()
    df = original.copy()
    original_shape = tuple(df.shape)
    cleaned_columns, column_mapping = _clean_columns(df.columns)
    df.columns = cleaned_columns

    # Replace infinities before missing-value handling.
    inf_cells = 0
    for col in df.select_dtypes(include="number").columns:
        mask = ~np.isfinite(df[col].to_numpy(dtype=float, copy=False))
        count = int(mask.sum())
        if count:
            df.loc[mask, col] = np.nan
            inf_cells += count

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        df = df.drop_duplicates().reset_index(drop=True)

    whitespace_cols = []
    for col in df.select_dtypes(include="object").columns:
        before = df[col].copy()
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        if not before.equals(df[col]):
            whitespace_cols.append(col)

    converted = _coerce_numeric_like(df)
    target_original = state.get("target")
    target = column_mapping.get(target_original, target_original)
    if target is not None:
        state["target"] = target
        if state.get("decision"):
            state["decision"]["target"] = target

    # Missing target rows are removed rather than inventing labels/values.
    target_rows_removed = 0
    modeling_df = df.copy()
    if target in modeling_df.columns:
        target_missing = modeling_df[target].isna()
        target_rows_removed = int(target_missing.sum())
        if target_rows_removed:
            modeling_df = modeling_df.loc[~target_missing].reset_index(drop=True)

    # Build the user-facing fully processed dataset.
    processed_df = df.copy()
    feature_columns = [c for c in processed_df.columns if c != target]
    num_filled, cat_filled = _fill_missing(processed_df, feature_columns)

    # A fully processed downloadable dataset should contain no missing cells.
    # For a missing target, fill with the most frequent/median value ONLY in the
    # downloadable artifact? No. We avoid fabricated target labels by dropping
    # those rows from the downloadable dataset as well.
    if target in processed_df.columns and target_rows_removed:
        processed_df = processed_df.loc[~processed_df[target].isna()].reset_index(drop=True)

    # The modeling dataframe keeps feature NaNs so the Model Agent can impute
    # inside each CV fold. This prevents data leakage.
    modeling_feature_columns = [c for c in modeling_df.columns if c != target]
    remaining_before_model = int(modeling_df[modeling_feature_columns].isna().sum().sum()) if modeling_feature_columns else 0

    state["source_df"] = original
    state["processed_df"] = processed_df
    state["model_df"] = modeling_df
    state["df"] = processed_df.copy()

    state["preprocessing"] = {
        "original_shape": [int(original_shape[0]), int(original_shape[1])],
        "final_shape": [int(processed_df.shape[0]), int(processed_df.shape[1])],
        "model_shape": [int(modeling_df.shape[0]), int(modeling_df.shape[1])],
        "duplicates_removed": duplicate_rows,
        "infinite_values_replaced": inf_cells,
        "numeric_like_columns_converted": converted,
        "text_columns_trimmed": whitespace_cols,
        "numeric_missing_filled": num_filled,
        "categorical_missing_filled": cat_filled,
        "target_rows_removed": target_rows_removed,
        "remaining_missing": int(processed_df.isna().sum().sum()),
        "model_feature_missing_before_fold_imputation": remaining_before_model,
        "fully_processed": bool(processed_df.isna().sum().sum() == 0),
        "modeling_note": "The downloadable dataset is fully processed. For model validation, feature imputation is repeated inside each CV fold to avoid leakage; missing target rows are removed rather than fabricated.",
        "actions": [
            "Column names normalized",
            f"{duplicate_rows} exact duplicate rows removed" if duplicate_rows else "No exact duplicate rows found",
            f"{inf_cells} infinite numeric values replaced with missing values" if inf_cells else "No infinite numeric values found",
            "Text values trimmed",
            f"{len(converted)} numeric-like text columns converted" if converted else "No numeric-like text conversions needed",
            f"Filled missing numeric feature values: {', '.join(num_filled)}" if num_filled else "No missing numeric feature values",
            f"Filled missing categorical feature values: {', '.join(cat_filled)}" if cat_filled else "No missing categorical feature values",
            f"Dropped {target_rows_removed} rows with missing target '{target}'" if target_rows_removed else "No rows dropped for missing target",
            "Downloadable dataset now contains zero missing cells" if processed_df.isna().sum().sum() == 0 else "Some missing cells remain and require review",
        ],
    }
    state["logs"].append(
        f"Preprocessing Agent produced a fully processed dataset: {processed_df.shape[0]:,} rows, {processed_df.shape[1]:,} columns, {int(processed_df.isna().sum().sum())} missing cells remaining."
    )
    return state
