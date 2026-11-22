import re
import numpy as np
import pandas as pd

ID_WORDS = ("id", "uuid", "index", "identifier", "serial", "key")


def _is_id_like(col, series):
    name = str(col).strip().lower().replace(" ", "_")
    explicit = ("customerid", "productid", "userid", "user_id", "orderid", "order_id",
                "transactionid", "transaction_id", "accountid", "account_id", "recordid",
                "record_id", "itemid", "item_id", "customer_id", "product_id")
    if name in explicit:
        return True
    return name in ID_WORDS or any(name.endswith("_" + w) for w in ID_WORDS) or name.endswith("uuid") or name.endswith("identifier") or name.endswith("_code") or name.endswith("_serial")


def _iqr_outlier_counts(numeric):
    out = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        out[col] = 0 if iqr == 0 else int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    return out


def eda_agent(state):
    df = state.get("processed_df", state["df"]).copy()
    numeric_all = df.select_dtypes(include="number")
    id_like = [c for c in numeric_all.columns if _is_id_like(c, df[c])]
    numeric = numeric_all.drop(columns=id_like, errors="ignore")
    categorical = df.select_dtypes(exclude="number")

    correlations = numeric.corr().round(3) if len(numeric.columns) > 1 else pd.DataFrame()
    pairs = []
    if not correlations.empty:
        cols = list(correlations.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                value = correlations.loc[a, b]
                if pd.notna(value):
                    pairs.append((abs(float(value)), a, b, float(value)))
    pairs.sort(reverse=True)
    top_corr = [
        {"feature_a": a, "feature_b": b, "correlation": round(v, 3)}
        for _, a, b, v in pairs[:10]
    ]

    numeric_summary = numeric.describe().round(3).to_dict() if not numeric.empty else {}
    categorical_summary = {}
    for col in categorical.columns:
        vc = categorical[col].value_counts(dropna=False).head(10)
        categorical_summary[col] = {str(k): int(v) for k, v in vc.items()}

    state["eda"] = {
        "summary": numeric_summary,
        "correlations": correlations.to_dict() if not correlations.empty else {},
        "top_correlations": top_corr,
        "unique_counts": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        "categorical_top_values": categorical_summary,
        "outliers_iqr": _iqr_outlier_counts(numeric),
        "excluded_identifier_columns": [str(x) for x in id_like],
        "correlation_note": "Identifier-like numeric columns were excluded because Pearson correlation on IDs is generally not analytically meaningful.",
        "strongest_relationship_label": "Strongest observed linear relationship",
    }
    state["logs"].append(
        f"EDA Agent analysed {len(numeric.columns)} numeric feature columns for correlations and excluded {len(id_like)} identifier-like columns."
    )
    return state
