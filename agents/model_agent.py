import time
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score, davies_bouldin_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

RANDOM_STATE = 42
SELECTION_ROWS = {"fast": 5000, "balanced": 10000, "deep": 15000}

SEARCH_MODES = {
    "fast": {"class_folds": 2, "reg_folds": 2, "max_class": 7, "max_reg": 7},
    "balanced": {"class_folds": 3, "reg_folds": 3, "max_class": 9, "max_reg": 10},
    "deep": {"class_folds": 5, "reg_folds": 5, "max_class": 99, "max_reg": 99},
}

# Models that require a dense design matrix. They are skipped automatically when
# the estimated encoded width would make dense conversion risky.
DENSE_ONLY = {
    "GradientBoostingClassifier", "HistGradientBoostingClassifier", "GaussianNB",
    "GradientBoostingRegressor", "HistGradientBoostingRegressor",
}

ID_NAME_WORDS = ("id", "uuid", "index", "identifier", "serial", "key")

# A tie with the dummy baseline should still surface a real model when the
# real model matches the baseline on the primary and secondary metrics.
PRIMARY_TOL = 1e-6
SECONDARY_TOL = 1e-6


def _select_real_vs_baseline_classification(results, baseline):
    real = [r for r in results if r.get("model") != "DummyClassifier"]
    if not real:
        return None, {"status": "no_real_model", "primary_delta": None, "secondary_delta": None, "tied": False}
    real.sort(key=lambda r: (r["f1_weighted_mean"], r.get("accuracy_mean", -1), r.get("balanced_accuracy_mean", -1)), reverse=True)
    best_real = real[0]
    if not baseline:
        return best_real, {"status": "no_baseline", "primary_delta": None, "secondary_delta": None, "tied": False}
    primary_delta = float(best_real["f1_weighted_mean"] - baseline["f1_weighted_mean"])
    secondary_delta = float(best_real.get("accuracy_mean", 0.0) - baseline.get("accuracy_mean", 0.0))
    tied = abs(primary_delta) <= PRIMARY_TOL and abs(secondary_delta) <= SECONDARY_TOL
    beats = primary_delta > PRIMARY_TOL or tied
    return (best_real if beats else None), {"status": "selected_real" if beats else "baseline_wins", "primary_delta": primary_delta, "secondary_delta": secondary_delta, "tied": tied}


def _select_real_vs_baseline_regression(results, baseline):
    real = [r for r in results if r.get("model") != "DummyRegressor"]
    if not real:
        return None, {"status": "no_real_model", "primary_delta": None, "secondary_delta": None, "tied": False}
    real.sort(key=lambda r: ((r["R2_mean"] if np.isfinite(r["R2_mean"]) else -1e9), -r["RMSE_mean"]), reverse=True)
    best_real = real[0]
    if not baseline:
        return best_real, {"status": "no_baseline", "primary_delta": None, "secondary_delta": None, "tied": False}
    primary_delta = float(best_real["R2_mean"] - baseline["R2_mean"])
    scale = max(abs(float(baseline.get("RMSE_mean", 0.0))), 1e-9)
    secondary_delta = float((best_real["RMSE_mean"] - baseline["RMSE_mean"]) / scale)
    tied = abs(primary_delta) <= 0.005 and secondary_delta <= 0.01
    beats = primary_delta > 0.005 or tied
    return (best_real if beats else None), {"status": "selected_real" if beats else "baseline_wins", "primary_delta": primary_delta, "secondary_delta": secondary_delta, "tied": tied}


def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _preprocessor(X):
    nums = list(X.select_dtypes(include="number").columns)
    cats = list(X.select_dtypes(exclude="number").columns)
    transformers = []
    if nums:
        transformers.append(("num", Pipeline([
            ("imputer", __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), nums))
    if cats:
        transformers.append(("cat", Pipeline([
            ("imputer", __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="most_frequent")),
            ("onehot", _onehot()),
        ]), cats))
    if not transformers:
        raise ValueError("No usable feature columns remain after removing the target.")
    from sklearn.compose import ColumnTransformer
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


def _is_id_like_name(col):
    name = str(col).strip().lower().replace(" ", "_")
    explicit = ("customerid", "productid", "userid", "user_id", "orderid", "order_id",
                "transactionid", "transaction_id", "accountid", "account_id", "recordid",
                "record_id", "itemid", "item_id", "customer_id", "product_id")
    if name in explicit:
        return True
    if name in ID_NAME_WORDS or any(name.endswith("_" + word) for word in ID_NAME_WORDS):
        return True
    return name.endswith("uuid") or name.endswith("identifier") or name.endswith("_code") or name.endswith("_serial")


def _id_like_columns(X):
    # Prefer semantic/name evidence. High uniqueness alone is not enough because
    # legitimate business measures (e.g., revenue) can also be nearly unique.
    return [col for col in X.columns if _is_id_like_name(col)]


def _estimated_encoded_width(X):
    width = len(X.select_dtypes(include="number").columns)
    for col in X.select_dtypes(exclude="number").columns:
        width += min(int(X[col].nunique(dropna=True)), 500)
    return width


def _prepare_features(X, notes):
    ids = _id_like_columns(X)
    if ids:
        X = X.drop(columns=ids).copy()
        notes.append("Excluded identifier-like columns from model features: " + ", ".join(map(str, ids[:8])) + ("..." if len(ids) > 8 else ""))
    if X.shape[1] == 0:
        raise ValueError("No model features remain after removing identifier-like columns.")
    return X, ids


def _pipeline(X, estimator):
    return Pipeline([("prep", _preprocessor(X)), ("model", estimator)])


def _candidate_classifiers(n_rows):
    k = max(3, min(7, int(np.sqrt(max(n_rows, 9)))))
    return {
        "DummyClassifier": DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=1500, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=120, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=80, random_state=RANDOM_STATE),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_STATE),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "AdaBoostClassifier": AdaBoostClassifier(n_estimators=80, random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB(),
        "KNNClassifier": KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=1),
        "SVC": SVC(kernel="rbf", C=1.0, gamma="scale"),
    }


def _candidate_regressors(n_rows):
    k = max(3, min(7, int(np.sqrt(max(n_rows, 9)))))
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "SGDRegressor": SGDRegressor(loss="squared_error", max_iter=1500, random_state=RANDOM_STATE),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=120, random_state=RANDOM_STATE, n_jobs=1),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=120, random_state=RANDOM_STATE, n_jobs=1),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=80, random_state=RANDOM_STATE),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=100, random_state=RANDOM_STATE),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "AdaBoostRegressor": AdaBoostRegressor(n_estimators=80, random_state=RANDOM_STATE),
        "KNNRegressor": KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=1),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1),
    }


def _ordered_subset(models, mode, kind):
    cfg = SEARCH_MODES.get(mode, SEARCH_MODES["balanced"])
    if kind == "classification":
        preferred = [
            "DummyClassifier", "LogisticRegression", "SGDClassifier", "RandomForestClassifier",
            "ExtraTreesClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier",
            "DecisionTreeClassifier", "AdaBoostClassifier", "GaussianNB", "KNNClassifier", "SVC",
        ]
        limit = cfg["max_class"]
    else:
        preferred = [
            "DummyRegressor", "LinearRegression", "Ridge", "Lasso", "ElasticNet", "SGDRegressor",
            "RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor",
            "HistGradientBoostingRegressor", "DecisionTreeRegressor", "AdaBoostRegressor", "KNNRegressor", "SVR",
        ]
        limit = cfg["max_reg"]
    names = [n for n in preferred if n in models]
    return names if limit >= len(names) else names[:limit]


def _sample_for_selection(X, y, task, notes, mode):
    n = len(y)
    size = SELECTION_ROWS.get(mode, SELECTION_ROWS["balanced"])
    if n <= size:
        return X, y, False
    if task == "classification":
        Xs, _, ys, _ = train_test_split(
            X, y, train_size=size, random_state=RANDOM_STATE, stratify=y
        )
    else:
        Xs, _, ys, _ = train_test_split(X, y, train_size=size, random_state=RANDOM_STATE)
    notes.append(f"Large-dataset optimization: model selection used a representative sample of {size:,} rows from {n:,}.")
    return Xs, ys, True


def _leakage_report(X, y, task):
    findings = []
    checks = []
    for col in X.columns:
        s = X[col]
        # Exact duplicate / near duplicate target values.
        if task == "regression":
            ys = pd.to_numeric(y, errors="coerce")
            xs = pd.to_numeric(s, errors="coerce")
            mask = ys.notna() & xs.notna()
            if int(mask.sum()) >= 10:
                corr = float(xs[mask].corr(ys[mask])) if mask.sum() > 1 else 0.0
                if np.isfinite(corr):
                    checks.append((abs(corr), col, f"numeric correlation with target r={corr:.4f}"))
                    if abs(corr) >= 0.999:
                        findings.append({"feature": str(col), "type": "near-perfect numeric correlation", "evidence": f"|r|={abs(corr):.4f}"})
        else:
            same = (s.astype(str).fillna("<NA>") == y.astype(str).fillna("<NA>"))
            if bool(same.mean() >= 0.999):
                findings.append({"feature": str(col), "type": "exact/near-exact target copy", "evidence": f"matching rate={same.mean():.4f}"})
            # Deterministic mapping for reasonably low-cardinality features.
            if s.nunique(dropna=True) <= 50:
                mapping = pd.DataFrame({"x": s, "y": y}).dropna().groupby("x")["y"].nunique()
                if not mapping.empty and int(mapping.max()) == 1 and len(mapping) >= 2:
                    purity = len(mapping) / max(s.notna().sum(), 1)
                    if purity >= 0.98:
                        findings.append({"feature": str(col), "type": "target-deterministic mapping", "evidence": f"unique target per feature value; coverage={purity:.4f}"})
    return {
        "suspicious": bool(findings),
        "findings": findings,
        "note": "These are screening signals, not proof of leakage. Review the flagged features before trusting a near-perfect score."
        if findings else "No strong target-leakage signal was detected by the automatic checks.",
    }


def _fit_feature_importance(X, y, model_name, candidates):
    if not model_name or model_name.startswith("Dummy") or model_name not in candidates:
        return []
    try:
        pipe = _pipeline(X, candidates[model_name])
        pipe.fit(X, y)
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        names = prep.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            vals = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            vals = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return []
        pairs = sorted(zip(names, vals), key=lambda x: float(x[1]), reverse=True)[:12]
        return [{"feature": str(name), "importance": round(float(value), 6)} for name, value in pairs]
    except Exception:
        return []


def _classification(df, target, mode):
    work = df.dropna(subset=[target]).copy()
    X0, y0 = work.drop(columns=[target]), work[target].astype(str)
    if y0.nunique() < 2:
        return {"status": "skipped", "reason": "Classification requires at least two target classes."}
    if len(y0) < 6:
        return {"status": "skipped", "reason": "Classification requires at least six usable rows."}

    notes = []
    X, ids = _prepare_features(X0, notes)
    X, y, sampled = _sample_for_selection(X, y0, "classification", notes, mode)
    models = _candidate_classifiers(len(y))
    names = _ordered_subset(models, mode, "classification")
    width = _estimated_encoded_width(X)
    if width > 5000:
        for name in ["GaussianNB", "GradientBoostingClassifier", "HistGradientBoostingClassifier"]:
            if name in names:
                names.remove(name)
                notes.append(f"{name} skipped because estimated one-hot width ({width:,}) is too large for a dense model.")
    if len(y) > 15000:
        for name in ["SVC", "KNNClassifier"]:
            if name in names:
                names.remove(name)
                notes.append(f"{name} skipped for scalability on the model-selection sample.")

    min_class = int(y.value_counts().min())
    desired_folds = SEARCH_MODES.get(mode, SEARCH_MODES["balanced"])["class_folds"]
    folds = min(desired_folds, min_class)
    results, errors = [], []
    if folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"f1_weighted": "f1_weighted", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"}
        eval_mode = f"{folds}-fold stratified cross-validation"
        for name in names:
            started = time.perf_counter()
            try:
                cv = cross_validate(_pipeline(X, models[name]), X, y, cv=splitter, scoring=scoring, n_jobs=1, error_score="raise")
                results.append({
                    "model": name,
                    "f1_weighted_mean": float(np.mean(cv["test_f1_weighted"])),
                    "f1_weighted_std": float(np.std(cv["test_f1_weighted"])),
                    "accuracy_mean": float(np.mean(cv["test_accuracy"])),
                    "balanced_accuracy_mean": float(np.mean(cv["test_balanced_accuracy"])),
                    "time_s": round(float(time.perf_counter() - started), 3),
                })
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
    else:
        return {"status": "skipped", "reason": "Classification cannot be cross-validated because at least one class has fewer than two examples."}

    if not results:
        return {"status": "skipped", "reason": "All classification candidates failed. " + " | ".join(errors[-5:]), "errors": errors}

    results.sort(key=lambda x: (x["f1_weighted_mean"], x.get("accuracy_mean", -1), x.get("balanced_accuracy_mean", -1)), reverse=True)
    baseline = next((r for r in results if r["model"] == "DummyClassifier"), None)
    selected, baseline_cmp = _select_real_vs_baseline_classification(results, baseline)
    best_candidate = selected or next((r for r in results if r["model"] != "DummyClassifier"), None) or results[0]
    beats = bool(selected)
    leakage = _leakage_report(X, y, "classification")
    perfect = float(best_candidate["f1_weighted_mean"]) >= 0.999

    return {
        "status": "completed", "task": "classification", "target": target,
        "model": selected["model"] if selected else "No useful model found",
        "selected_model": selected["model"] if selected else None,
        "best_candidate": best_candidate["model"],
        "metric": "F1-weighted (CV mean)",
        "score": round(float(best_candidate["f1_weighted_mean"]), 4),
        "cv": eval_mode, "cv_folds": folds, "candidates": results, "errors": errors,
        "baseline": baseline, "beats_baseline": beats, "useful_model": bool(selected),
        "baseline_comparison": baseline_cmp,
        "baseline_summary": (
            "real model tied with the dummy baseline on primary and secondary metrics; surfaced the real model" if baseline_cmp.get("tied") else
            (f"real model primary metric delta vs dummy = {baseline_cmp.get('primary_delta'):.4f}" if baseline_cmp.get("primary_delta") is not None else "not available")
        ),
        "model_count": len(results), "models_evaluated": len(results), "search_mode": mode,
        "selection_rows": len(y), "original_rows": len(y0), "sampled": sampled,
        "excluded_id_columns": [str(x) for x in ids],
        "search_notes": notes, "leakage": leakage, "perfect_score_check": perfect,
        "quality_warning": "Perfect-score check triggered. Review feature leakage and target duplication before trusting this result." if perfect else None,
        "reason": (
            f"Compared {len(results)} classification models using {eval_mode}. "
            + (f"Selected {selected['model']} because it beat or tied the dummy baseline; ties favor the real model when F1 and accuracy are equal." if selected else "No real model beat or tied the dummy baseline on the primary/secondary criteria; the system reports weak predictive performance." )
        ),
        "feature_importance": _fit_feature_importance(X, y, selected["model"] if selected else None, models),
    }


def _regression(df, target, mode):
    work = df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])
    X0, y0 = work.drop(columns=[target]), work[target]
    if y0.nunique() < 2:
        return {"status": "skipped", "reason": "Regression requires at least two distinct target values."}
    if len(y0) < 6:
        return {"status": "skipped", "reason": "Regression requires at least six usable rows."}

    notes = []
    X, ids = _prepare_features(X0, notes)
    X, y, sampled = _sample_for_selection(X, y0, "regression", notes, mode)
    models = _candidate_regressors(len(y))
    names = _ordered_subset(models, mode, "regression")
    width = _estimated_encoded_width(X)
    if width > 5000:
        for name in ["SVR", "GradientBoostingRegressor", "HistGradientBoostingRegressor"]:
            if name in names:
                names.remove(name)
                notes.append(f"{name} skipped because estimated one-hot width ({width:,}) is too large for a dense model.")
    if len(y) > 15000:
        for name in ["SVR", "KNNRegressor"]:
            if name in names:
                names.remove(name)
                notes.append(f"{name} skipped for scalability on the model-selection sample.")

    desired_folds = SEARCH_MODES.get(mode, SEARCH_MODES["balanced"])["reg_folds"]
    folds = min(desired_folds, len(y))
    if folds < 2:
        return {"status": "skipped", "reason": "Regression requires at least two usable folds."}

    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"mae": "neg_mean_absolute_error", "r2": "r2", "rmse": "neg_root_mean_squared_error"}
    eval_mode = f"{folds}-fold cross-validation"
    results, errors = [], []
    for name in names:
        started = time.perf_counter()
        try:
            cv = cross_validate(_pipeline(X, models[name]), X, y, cv=splitter, scoring=scoring, n_jobs=1, error_score="raise")
            results.append({
                "model": name,
                "MAE_mean": float(-np.mean(cv["test_mae"])),
                "RMSE_mean": float(-np.mean(cv["test_rmse"])),
                "R2_mean": float(np.mean(cv["test_r2"])),
                "time_s": round(float(time.perf_counter() - started), 3),
            })
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if not results:
        return {"status": "skipped", "reason": "All regression candidates failed. " + " | ".join(errors[-5:]), "errors": errors}

    results.sort(key=lambda x: ((x["R2_mean"] if np.isfinite(x["R2_mean"]) else -1e9), -x["RMSE_mean"]))
    baseline = next((r for r in results if r["model"] == "DummyRegressor"), None)
    selected, baseline_cmp = _select_real_vs_baseline_regression(results, baseline)
    best_candidate = selected or next((r for r in results if r.get("model") != "DummyRegressor"), None) or results[0]
    beats = bool(selected)
    leakage = _leakage_report(X, y, "regression")
    perfect = False
    discrete = int(y0.nunique()) <= 10 and len(y0) >= 20
    replan = None
    if not selected and discrete:
        replan = {
            "recommended_task": "classification",
            "reason": f"Regression did not beat the dummy baseline, and the numeric target has only {int(y0.nunique())} distinct values. Classification may be more appropriate.",
            "auto_executed": False,
        }

    return {
        "status": "completed", "task": "regression", "target": target,
        "model": selected["model"] if selected else "No useful model found",
        "selected_model": selected["model"] if selected else None,
        "best_candidate": best_candidate["model"],
        "metric": "R² (CV mean)", "score": round(float(best_candidate["R2_mean"]), 4),
        "cv": eval_mode, "cv_folds": folds, "candidates": results, "errors": errors,
        "baseline": baseline, "beats_baseline": beats, "useful_model": bool(selected),
        "baseline_comparison": baseline_cmp,
        "baseline_summary": (
            "real model tied with the dummy baseline on primary and secondary metrics; surfaced the real model" if baseline_cmp.get("tied") else
            (f"real model primary metric delta vs dummy = {baseline_cmp.get('primary_delta'):.4f}" if baseline_cmp.get("primary_delta") is not None else "not available")
        ),
        "model_count": len(results), "models_evaluated": len(results), "search_mode": mode,
        "selection_rows": len(y), "original_rows": len(y0), "sampled": sampled,
        "excluded_id_columns": [str(x) for x in ids],
        "search_notes": notes, "leakage": leakage, "perfect_score_check": perfect,
        "quality_warning": "No useful real regression model beat or tied the dummy baseline under the tie-aware selection rule." if not selected else None,
        "replan": replan,
        "reason": (
            f"Compared {len(results)} regression models using {eval_mode}. "
            + (f"Selected {selected['model']} because it beat or tied the dummy baseline; ties favor the real model when R² is tied and RMSE is not materially worse." if selected else "No real model beat or tied the dummy baseline on the primary/secondary criteria; the system reports weak predictive performance." )
        ),
        "feature_importance": _fit_feature_importance(X, y, selected["model"] if selected else None, models),
    }


def _clustering(df, exclude_target=None):
    numeric = df.select_dtypes(include="number").copy()
    if exclude_target in numeric.columns:
        numeric = numeric.drop(columns=[exclude_target])
    ids = _id_like_columns(numeric)
    numeric = numeric.drop(columns=ids, errors="ignore")
    if numeric.shape[1] < 2 or len(numeric) < 6:
        return {"status": "skipped", "reason": "Clustering requires at least two numeric features and six rows."}
    sampled = False
    if len(numeric) > MAX_SELECTION_ROWS:
        numeric = numeric.sample(MAX_SELECTION_ROWS, random_state=RANDOM_STATE)
        sampled = True
    X = StandardScaler().fit_transform(
        __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median").fit_transform(numeric)
    )
    results = []
    max_k = min(6, len(X) - 1)
    for k in range(2, max_k + 1):
        candidates = {
            "KMeans": KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10),
            "AgglomerativeClustering": AgglomerativeClustering(n_clusters=k),
            "GaussianMixture": GaussianMixture(n_components=k, random_state=RANDOM_STATE),
        }
        for name, estimator in candidates.items():
            try:
                labels = estimator.fit_predict(X) if hasattr(estimator, "fit_predict") else estimator.fit(X).predict(X)
                if len(set(labels)) >= 2:
                    results.append({"model": name, "k": k, "silhouette": float(silhouette_score(X, labels)), "davies_bouldin": float(davies_bouldin_score(X, labels))})
            except Exception:
                continue
    try:
        labels = DBSCAN(eps=0.5, min_samples=max(3, min(10, len(X)//20 or 3))).fit_predict(X)
        mask = labels != -1
        if mask.sum() >= 4 and len(set(labels[mask])) >= 2:
            results.append({"model": "DBSCAN", "k": int(len(set(labels[mask]))), "silhouette": float(silhouette_score(X[mask], labels[mask])), "davies_bouldin": float(davies_bouldin_score(X[mask], labels[mask]))})
    except Exception:
        pass
    if not results:
        return {"status": "skipped", "reason": "No clustering candidate produced at least two usable clusters."}
    results.sort(key=lambda x: (x["silhouette"], -x["davies_bouldin"]), reverse=True)
    best = results[0]
    return {
        "status": "completed", "task": "clustering", "target": None, "model": best["model"],
        "metric": "Silhouette score", "score": round(best["silhouette"], 4), "selected_k": best["k"],
        "candidates": results, "model_count": len(results), "models_evaluated": len(results),
        "sampled": sampled, "excluded_id_columns": [str(x) for x in ids], "search_mode": "deep",
        "reason": f"Compared {len(results)} clustering configurations using silhouette score with Davies-Bouldin as a secondary signal.",
    }


def model_agent(state):
    try:
        df = state.get("model_df", state["df"]).copy()
        task = state.get("task", "unknown")
        target = state.get("target")
        mode = str(state.get("model_search_mode", "balanced")).lower()
        if mode not in SEARCH_MODES:
            mode = "balanced"
        if task == "classification" and target in df.columns:
            result = _classification(df, target, mode)
        elif task == "regression" and target in df.columns:
            result = _regression(df, target, mode)
        elif task == "clustering":
            result = _clustering(df, target)
        else:
            result = {"status": "needs_input", "task": "unknown", "target": target, "reason": "The system could not confidently identify a supervised target. Select a target column or request clustering/segmentation."}
    except Exception as exc:
        result = {"status": "failed", "task": state.get("task", "unknown"), "target": state.get("target"), "reason": f"Model Agent failed safely: {type(exc).__name__}: {exc}"}
    state["model"] = result
    if result.get("replan"):
        state["logs"].append(f"Model Agent recommends replanning toward {result['replan']['recommended_task']} because predictive quality was weak.")
    state["logs"].append(f"Model Agent finished with status='{result.get('status')}', task='{result.get('task', 'unknown')}', model='{result.get('model', 'none')}'.")
    return state
