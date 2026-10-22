from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "src_stroke_prediction.csv"
OUT_DIR = ROOT / "outputs" / "dashboard_v3"
HTML_PATH = ROOT / "care_risk_outreach_dashboard.html"
INDEX_PATH = ROOT / "index.html"
PAGES_DIR = ROOT / "docs"
PAGES_PATH = PAGES_DIR / "index.html"
JSON_PATH = OUT_DIR / "dashboard_data_v3.json"
AI_OUT_DIR = ROOT / "outputs" / "care_risk_ai"
D1_SEED_PATH = AI_OUT_DIR / "d1_seed.sql"
RANDOM_STATE = 42


def to_records(df: pd.DataFrame):
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def pct_label(fraction: float) -> str:
    return f"Top {fraction:.1%}"


def sql_literal(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NULL"
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return repr(float(value)) if isinstance(value, (float, np.floating)) else str(int(value))
    return "'" + str(value).replace("'", "''") + "'"


def insert_batches(table: str, columns: list[str], rows: list[list], batch_size: int = 100) -> list[str]:
    statements = []
    quoted_columns = ", ".join(columns)
    for start in range(0, len(rows), batch_size):
        values = []
        for row in rows[start : start + batch_size]:
            values.append("(" + ", ".join(sql_literal(value) for value in row) + ")")
        statements.append(
            f"INSERT INTO {table} ({quoted_columns}) VALUES\n  " + ",\n  ".join(values) + ";"
        )
    return statements


def build_d1_seed(data: dict) -> str:
    lines = [
        "DELETE FROM scored_patients;",
        "DELETE FROM model_metrics;",
        "DELETE FROM subgroup_metrics;",
        "DELETE FROM capacity_metrics;",
        "DELETE FROM model_coefficients;",
    ]

    patient_columns = [
        "patient_rank",
        "predicted_risk",
        "observed_event",
        "gender",
        "age",
        "age_group",
        "hypertension",
        "heart_disease",
        "residence_type",
        "avg_glucose_level",
        "glucose_band",
        "bmi",
        "bmi_band",
        "smoking_status",
        "error_group",
    ]
    patient_rows = [
        [
            row["rank"],
            row["predicted_risk"],
            row["event"],
            row["gender"],
            row["age"],
            row["age_group"],
            row["hypertension"],
            row["heart_disease"],
            row["Residence_type"],
            row["avg_glucose_level"],
            row["glucose_band"],
            row["bmi"],
            row["bmi_band"],
            row["smoking_status"],
            row["error_group"],
        ]
        for row in data["records"]
    ]
    lines.extend(insert_batches("scored_patients", patient_columns, patient_rows))

    model_columns = [
        "model",
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "precision",
        "top10_precision",
        "top10_recall",
        "top10_events_reached",
    ]
    model_rows = [
        [
            row["model"],
            row["roc_auc"],
            row["pr_auc"],
            row["sensitivity"],
            row["specificity"],
            row["precision"],
            row["top10_precision"],
            row["top10_recall"],
            row["top10_true_events_reached"],
        ]
        for row in data["modelMetrics"]
    ]
    lines.extend(insert_batches("model_metrics", model_columns, model_rows))

    subgroup_columns = [
        "population",
        "patient_count",
        "event_rate",
        "roc_auc",
        "pr_auc",
        "recall_top_10",
        "global_top_10_share",
    ]
    subgroup_rows = [
        [
            row["population"],
            row["n"],
            row["event_rate"],
            row["auc"],
            row["pr_auc"],
            row["recall_top_10"],
            row["global_top_10_share"],
        ]
        for row in data["subgroupPerformance"]
    ]
    lines.extend(insert_batches("subgroup_metrics", subgroup_columns, subgroup_rows))

    capacity_columns = [
        "capacity_fraction",
        "patients_contacted",
        "events_found",
        "precision_at_k",
        "recall_at_k",
        "lift",
        "minimum_score",
    ]
    capacity_rows = [
        [
            row["capacity_fraction"],
            row["patients_contacted"],
            row["true_events_found"],
            row["precision"],
            row["recall"],
            row["lift"],
            row["min_score"],
        ]
        for row in data["capacityTable"]
    ]
    lines.extend(insert_batches("capacity_metrics", capacity_columns, capacity_rows))

    coefficient_columns = ["feature", "coefficient", "odds_ratio"]
    coefficient_rows = [
        [row["feature"], row["coefficient"], float(np.exp(row["coefficient"]))]
        for row in data["coefficients"]
    ]
    lines.extend(insert_batches("model_coefficients", coefficient_columns, coefficient_rows))
    return "\n\n".join(lines) + "\n"


def build_data() -> dict:
    df = pd.read_csv(DATA_PATH, na_values=["N/A", "NA", ""])
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    numeric_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
    categorical_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    feature_cols = numeric_features + categorical_features

    model_df = df[df["gender"].isin(["Male", "Female"])].copy()
    model_df["work_type"] = model_df["work_type"].replace(
        {"children": "Not_working", "Never_worked": "Not_working"}
    )

    X = model_df[feature_cols].copy()
    y = model_df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["bmi"] = X_train["bmi"].fillna(X_train["bmi"].mean())
    X_test["bmi"] = X_test["bmi"].fillna(X_test["bmi"].mean())

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_features),
            (
                "cat",
                Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]),
                categorical_features,
            ),
        ]
    )

    def make_logistic(class_weight=None):
        return Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight=class_weight,
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    unweighted_model = make_logistic(class_weight=None)
    weighted_model = make_logistic(class_weight="balanced")
    unweighted_model.fit(X_train, y_train)
    weighted_model.fit(X_train, y_train)
    unweighted_scores = unweighted_model.predict_proba(X_test)[:, 1]
    weighted_scores = weighted_model.predict_proba(X_test)[:, 1]

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count
    xgb_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=5,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    xgb_model.fit(X_train, y_train)
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

    model_scores = {
        "Unweighted logistic": (unweighted_model, unweighted_scores),
        "Weighted logistic": (weighted_model, weighted_scores),
        "XGBoost": (xgb_model, xgb_scores),
    }

    def threshold_for_top_fraction(scores, fraction=0.10):
        k = max(1, int(np.ceil(len(scores) * fraction)))
        return np.sort(scores)[::-1][k - 1]

    def outreach_at_fraction(y_true, scores, fraction):
        y_arr = np.asarray(y_true)
        k = max(1, int(np.ceil(len(y_arr) * fraction)))
        selected = np.argsort(scores)[::-1][:k]
        found = int(y_arr[selected].sum())
        total = int(y_arr.sum())
        precision = found / k
        recall = found / total if total else np.nan
        lift = precision / y_arr.mean() if y_arr.mean() else np.nan
        return {
            "capacity_fraction": float(fraction),
            "outreach_capacity": pct_label(fraction),
            "patients_contacted": int(k),
            "true_events_found": found,
            "false_positives": int(k - found),
            "precision": float(precision),
            "recall": float(recall),
            "lift": float(lift),
            "min_score": float(scores[selected[-1]]),
        }

    def evaluate_model(name, scores):
        threshold = threshold_for_top_fraction(scores, 0.10)
        preds = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        top10 = outreach_at_fraction(y_test, scores, 0.10)
        return {
            "model": name,
            "chosen_threshold": float(threshold),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "pr_auc": float(average_precision_score(y_test, scores)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
            "precision": float(tp / (tp + fp)) if (tp + fp) else None,
            "top10_precision": top10["precision"],
            "top10_recall": top10["recall"],
            "top10_true_events_reached": top10["true_events_found"],
        }

    metrics = pd.DataFrame(
        [evaluate_model(name, scores) for name, (_, scores) in model_scores.items()]
    ).sort_values(["pr_auc", "top10_recall"], ascending=False)

    recommended_model_name = "Unweighted logistic"
    recommended_model = unweighted_model
    recommended_scores = unweighted_scores

    def age_group(age):
        return "Age <65" if age < 65 else "Age >=65"

    def glucose_band(value):
        if value < 100:
            return "<100"
        if value < 126:
            return "100-125"
        if value < 180:
            return "126-179"
        return "180+"

    def bmi_band(value):
        if pd.isna(value):
            return "Missing"
        if value < 18.5:
            return "<18.5"
        if value < 25:
            return "18.5-24.9"
        if value < 30:
            return "25-29.9"
        if value < 35:
            return "30-34.9"
        return "35+"

    X_population = model_df[feature_cols].copy()
    y_population = model_df["stroke"].copy()
    X_population["bmi"] = X_population["bmi"].fillna(X_train["bmi"].mean())
    population_scores = recommended_model.predict_proba(X_population)[:, 1]

    scored = X_population.copy()
    scored["event"] = y_population.to_numpy().astype(int)
    scored["predicted_risk"] = population_scores
    scored["age_group"] = scored["age"].map(age_group)
    scored["glucose_band"] = scored["avg_glucose_level"].map(glucose_band)
    scored["bmi_band"] = scored["bmi"].map(bmi_band)
    scored = scored.sort_values("predicted_risk", ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    top10_threshold = threshold_for_top_fraction(population_scores, 0.10)
    scored["predicted_positive_top10"] = scored["predicted_risk"] >= top10_threshold
    scored["error_group"] = np.select(
        [
            (scored["event"] == 1) & (scored["predicted_positive_top10"]),
            (scored["event"] == 1) & (~scored["predicted_positive_top10"]),
            (scored["event"] == 0) & (scored["predicted_positive_top10"]),
            (scored["event"] == 0) & (~scored["predicted_positive_top10"]),
        ],
        ["True positive", "False negative", "False positive", "True negative"],
        default="Unclassified",
    )

    capacity_table = pd.DataFrame(
        [
            outreach_at_fraction(y_population, population_scores, f)
            for f in [0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
        ]
    )

    gains = scored[["event", "predicted_risk"]].copy()
    gains["patients_contacted"] = np.arange(1, len(gains) + 1)
    gains["pct_patients_contacted"] = gains["patients_contacted"] / len(gains)
    gains["cumulative_events"] = gains["event"].cumsum()
    gains["pct_events_captured"] = gains["cumulative_events"] / gains["event"].sum()
    gains["random_baseline"] = gains["pct_patients_contacted"]
    gains["precision_at_k"] = gains["cumulative_events"] / gains["patients_contacted"]
    gains["recall_at_k"] = gains["pct_events_captured"]
    gains["lift_at_k"] = gains["precision_at_k"] / y_test.mean()

    def recall_at_top_fraction(y_true, score_values, fraction=0.10):
        y_arr = np.asarray(y_true)
        s_arr = np.asarray(score_values)
        positives = y_arr.sum()
        if len(y_arr) == 0 or positives == 0:
            return np.nan
        k = max(1, int(np.ceil(len(y_arr) * fraction)))
        top_idx = np.argsort(s_arr)[::-1][:k]
        return y_arr[top_idx].sum() / positives

    def safe_auc(y_true, score_values):
        y_arr = np.asarray(y_true)
        if len(np.unique(y_arr)) < 2:
            return np.nan
        return roc_auc_score(y_arr, score_values)

    def safe_pr_auc(y_true, score_values):
        y_arr = np.asarray(y_true)
        if len(np.unique(y_arr)) < 2:
            return np.nan
        return average_precision_score(y_arr, score_values)

    subgroup_rows = []
    for population, mask in [
        ("Male", scored["gender"] == "Male"),
        ("Female", scored["gender"] == "Female"),
        ("Age <65", scored["age"] < 65),
        ("Age >=65", scored["age"] >= 65),
        ("Urban", scored["Residence_type"] == "Urban"),
        ("Rural", scored["Residence_type"] == "Rural"),
    ]:
        sub = scored.loc[mask]
        subgroup_rows.append(
            {
                "population": population,
                "n": int(len(sub)),
                "event_rate": float(sub["event"].mean()),
                "auc": None if pd.isna(safe_auc(sub["event"], sub["predicted_risk"])) else float(safe_auc(sub["event"], sub["predicted_risk"])),
                "pr_auc": None if pd.isna(safe_pr_auc(sub["event"], sub["predicted_risk"])) else float(safe_pr_auc(sub["event"], sub["predicted_risk"])),
                "recall_top_10": None if pd.isna(recall_at_top_fraction(sub["event"], sub["predicted_risk"], 0.10)) else float(recall_at_top_fraction(sub["event"], sub["predicted_risk"], 0.10)),
                "global_top_10_share": float(sub["predicted_positive_top10"].mean()),
            }
        )

    pre = recommended_model.named_steps["preprocess"]
    cat_names = pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_names])
    coefs = recommended_model.named_steps["model"].coef_[0]
    coefficients = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .assign(abs_coefficient=lambda x: x["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
    )

    curves = []
    for name, scores in [(recommended_model_name, recommended_scores)]:
        precision, recall, _ = precision_recall_curve(y_test, scores)
        curves.append(
            {
                "model": name,
                "roc_auc": float(roc_auc_score(y_test, scores)),
                "pr_auc": float(average_precision_score(y_test, scores)),
                "pr": [{"x": float(x), "y": float(y)} for x, y in zip(recall, precision)],
            }
        )

    records = scored[
        [
            "rank",
            "predicted_risk",
            "event",
            "gender",
            "age",
            "age_group",
            "hypertension",
            "heart_disease",
            "Residence_type",
            "avg_glucose_level",
            "glucose_band",
            "bmi",
            "bmi_band",
            "smoking_status",
            "error_group",
        ]
    ].copy()

    top10 = capacity_table.loc[capacity_table["capacity_fraction"] == 0.10].iloc[0]
    summary = {
        "rows": int(len(df)),
        "model_rows": int(len(model_df)),
        "event_positive": int(df["stroke"].sum()),
        "event_rate": float(df["stroke"].mean()),
        "test_patients": int(len(y_test)),
        "test_events": int(y_test.sum()),
        "test_event_rate": float(y_test.mean()),
        "scored_population_patients": int(len(scored)),
        "scored_population_events": int(scored["event"].sum()),
        "scored_population_event_rate": float(scored["event"].mean()),
        "recommended_model": recommended_model_name,
        "roc_auc": float(metrics.iloc[0]["roc_auc"]),
        "pr_auc": float(metrics.iloc[0]["pr_auc"]),
        "default_capacity": int(top10["patients_contacted"]),
        "default_capacity_fraction": 10,
        "default_cases": int(top10["true_events_found"]),
        "default_precision": float(top10["precision"]),
        "default_recall": float(top10["recall"]),
    }

    return {
        "summary": summary,
        "records": to_records(records),
        "modelMetrics": to_records(metrics),
        "capacityTable": to_records(capacity_table),
        "gains": to_records(gains),
        "subgroupPerformance": to_records(pd.DataFrame(subgroup_rows)),
        "coefficients": to_records(coefficients),
        "modelCurves": curves,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Care Risk Outreach Dashboard</title>
<style>
:root{--ink:#142426;--muted:#617174;--line:#dbe5e3;--soft:#f4f8f7;--surface:#fff;--teal:#0f6b72;--teal2:#4eb5a8;--coral:#c76044;--gold:#d6a63d;--red:#c84436;--navy:#0f2f35;--ai:#145d63;--ai-soft:#eaf6f3;--shadow:0 18px 45px rgba(13,44,48,.11);--radius:8px}
*{box-sizing:border-box}html,body{overflow-x:hidden}body{margin:0;color:var(--ink);font-family:Aptos,Inter,"Segoe UI",Arial,sans-serif;background:linear-gradient(180deg,#eef6f4 0%,#f8fbfa 34%,#fff 100%);line-height:1.45}button,input,textarea{font:inherit}.page{width:100%;max-width:1440px;margin:0 auto;padding:28px 28px 48px}.topbar{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:22px}.eyebrow{color:var(--teal);font-weight:800;letter-spacing:.08em;text-transform:uppercase;font-size:12px}h1{margin:6px 0 8px;font-size:52px;line-height:1;letter-spacing:0}.dashboard-subtitle{margin:0;color:#506468;font-size:16px;font-weight:700}.status-card{min-width:300px;background:var(--navy);color:#fff;border-radius:var(--radius);padding:18px;box-shadow:var(--shadow)}.status-card .small{color:#b9cfcb;font-size:13px}.status-card .large{font-size:32px;font-weight:800;margin-top:6px}.status-ai-button{width:100%;margin-top:14px;border:1px solid #4eb5a8;border-radius:var(--radius);padding:9px 12px;background:#174b50;color:#fff;font-weight:800;cursor:pointer}.status-ai-button:hover{background:#1d5a60}.grid{display:grid;gap:18px}.grid>*,.topbar>*,.card,.metric,.status-card{min-width:0}.summary-grid{grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:18px}.main-grid{grid-template-columns:minmax(360px,.86fr) minmax(520px,1.14fr);align-items:stretch}.two-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.patient-grid{grid-template-columns:minmax(0,1.7fr) minmax(360px,.9fr);align-items:start}.card{background:rgba(255,255,255,.88);border:1px solid rgba(219,229,227,.9);border-radius:var(--radius);padding:20px;box-shadow:var(--shadow);backdrop-filter:blur(8px);overflow-x:auto}.metric{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:16px 17px}.metric.capacity-state .metric-label,.metric.capacity-state .metric-note{color:rgba(255,255,255,.82)}.metric.capacity-state .metric-value{color:#fff}.metric.capacity-state.enough{background:linear-gradient(135deg,#0f6b72,#4eb5a8);border-color:#4eb5a8}.metric.capacity-state.close{background:linear-gradient(135deg,#d6a63d,#efc95a);border-color:#efc95a}.metric.capacity-state.close .metric-label,.metric.capacity-state.close .metric-note,.metric.capacity-state.close .metric-value{color:#142426}.metric.capacity-state.over{background:linear-gradient(135deg,#c84436,#e46d5b);border-color:#e46d5b}.metric-label{color:var(--muted);font-size:13px;margin-bottom:8px}.metric-value{font-size:30px;font-weight:800;line-height:1}.metric-note{margin-top:7px;color:var(--muted);font-size:12px}h2{margin:0 0 8px;font-size:24px;letter-spacing:0}h3{margin:0 0 10px;font-size:17px}.subtle{color:var(--muted);font-size:14px;margin:0 0 16px}.control-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.pill,.tab-button{border:1px solid var(--line);background:#fff;color:var(--ink);padding:8px 12px;border-radius:999px;cursor:pointer;transition:.18s ease;font-size:14px}.tab-button{border-radius:var(--radius);padding:11px 15px;font-weight:800;box-shadow:0 8px 20px rgba(13,44,48,.07)}.tab-button.ai-tab{border-color:#8fc8be;background:#f1faf7}.tab-button.ai-tab.active{background:var(--ai);color:#fff;border-color:var(--ai)}.tab-live{display:inline-flex;align-items:center;justify-content:center;min-width:24px;height:20px;margin-right:6px;border-radius:4px;background:#d8f0eb;color:#0f6b72;font-size:10px;font-weight:900}.tab-button.active .tab-live{background:#fff;color:var(--ai)}.pill.active,.tab-button.active{background:var(--navy);color:#fff;border-color:var(--navy)}.tab-row{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 18px}.tab-panel{display:none}.tab-panel.active{display:block}.capacity-input-row{display:grid;grid-template-columns:minmax(170px,1fr) minmax(150px,.75fr);gap:12px;align-items:end;margin-top:16px}.input-label{color:var(--muted);font-size:13px;display:block;margin-bottom:6px}.number-input{width:100%;border:1px solid var(--line);border-radius:var(--radius);padding:10px 11px;color:var(--ink);background:#fff}.slider-wrap{margin:20px 0 16px}.slider-meta{display:flex;justify-content:space-between;color:var(--muted);font-size:13px;margin-bottom:8px}input[type=range]{width:100%;accent-color:var(--teal)}.capacity-visual{margin:14px 0 4px;padding:12px;border:1px solid var(--line);border-radius:var(--radius);background:#fbfdfc}.capacity-bar{height:15px;border-radius:999px;background:#e3eeec;overflow:hidden}.capacity-fill{height:100%;width:10%;border-radius:999px}.capacity-fill.enough{background:linear-gradient(90deg,#239b73,var(--teal2))}.capacity-fill.close{background:linear-gradient(90deg,#d6a63d,#efc95a)}.capacity-fill.over{background:linear-gradient(90deg,#c84436,#e46d5b)}.capacity-visual-text{display:flex;justify-content:space-between;gap:12px;margin-top:8px;color:var(--muted);font-size:12px}.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));margin-top:16px}.chart-box{min-height:380px}svg{max-width:100%;display:block}.chart-label{fill:var(--muted);font-size:12px}.chart-title{fill:var(--ink);font-weight:800;font-size:14px}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;background:#eaf3f1;color:#24464a;font-size:12px;text-transform:uppercase;letter-spacing:.04em;padding:10px}td{border-bottom:1px solid var(--line);padding:10px;vertical-align:top}tr:last-child td{border-bottom:0}.patient-table tbody tr{cursor:pointer}.patient-table tbody tr:hover,.patient-table tbody tr.selected{background:#f1f7f5}.priority-badge{display:inline-flex;align-items:center;justify-content:center;min-width:68px;border-radius:999px;padding:5px 10px;color:#fff;font-weight:800;font-size:12px}.priority-high{background:#c84436}.priority-medium{background:#d8832f}.priority-low{background:#d6a63d;color:#142426}.insight-card{position:sticky;top:18px}.insight-score{font-size:42px;line-height:1;font-weight:900;margin:10px 0;color:var(--navy)}.insight-box{border:1px solid var(--line);border-radius:var(--radius);padding:12px;margin-top:12px;background:#fbfdfc}.insight-box strong{display:block;margin-bottom:6px}.insight-box ul{margin:0;padding-left:18px;color:#44595c}.section{margin-top:18px}.callout{border-left:4px solid var(--coral);padding:12px 14px;background:#fff7f3;color:#4f332c;border-radius:0 var(--radius) var(--radius) 0;font-size:14px}.error-profile{grid-template-columns:repeat(3,minmax(0,1fr))}.mini{padding:12px;border:1px solid var(--line);border-radius:var(--radius);background:#fbfdfc}.mini strong{display:block;font-size:20px}.mini span{color:var(--muted);font-size:12px}.bar-row{display:grid;grid-template-columns:190px 1fr 70px;gap:10px;align-items:center;margin:9px 0;font-size:13px}.bar-track{height:12px;background:#e9f1ef;border-radius:999px;overflow:hidden}.bar-fill{height:100%;border-radius:999px;background:var(--teal)}.bar-fill.negative{background:var(--coral)}@media(max-width:1050px){.topbar,.main-grid,.two-grid,.patient-grid{grid-template-columns:1fr;display:grid}.summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.status-card{min-width:0}.insight-card{position:static}}@media(max-width:640px){.page{width:100vw;max-width:100vw;padding:18px}.topbar,.summary-grid,.main-grid,.two-grid,.patient-grid{width:100%;max-width:calc(100vw - 36px)}.status-card,.card,.metric{width:100%;max-width:calc(100vw - 36px)}.summary-grid,.kpi-grid,.error-profile,.capacity-input-row{grid-template-columns:1fr}.bar-row{grid-template-columns:1fr}h1{max-width:calc(100vw - 36px);font-size:29px;overflow-wrap:break-word}table{min-width:620px}.slider-meta{display:grid;gap:4px}}
.metric.capacity-state,.metric.capacity-state.enough,.metric.capacity-state.close,.metric.capacity-state.over{background:var(--surface);border-color:var(--line)}
.metric.capacity-state .metric-label,.metric.capacity-state .metric-note,.metric.capacity-state.close .metric-label,.metric.capacity-state.close .metric-note{color:var(--muted)}
.metric.capacity-state.enough .metric-value{color:#0f6b72}.metric.capacity-state.close .metric-value{color:#b47b13}.metric.capacity-state.over .metric-value{color:#c84436}
.priority-low{background:#f7e58a;color:#142426}.sortable{cursor:pointer;user-select:none}.sortable:hover{background:#dcebe8}.sort-mark{color:#617174;font-size:10px;margin-left:4px}
.caregiver-profile{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0}.match-score{font-size:42px;line-height:1;font-weight:900;color:#0f6b72;margin:8px 0}.match-badge{display:inline-flex;border-radius:999px;padding:5px 10px;font-weight:800;font-size:12px;background:#e5f3ef;color:#0f6b72}.caregiver-table tbody tr{cursor:pointer}.caregiver-table tbody tr:hover,.caregiver-table tbody tr.selected{background:#f1f7f5}.fit-good{color:#0f6b72;font-weight:800}.fit-watch{color:#b47b13;font-weight:800}
.outcome-bars{display:grid;gap:12px}.outcome-bar-row{display:grid;grid-template-columns:190px 1fr 80px;gap:12px;align-items:center}.outcome-track{height:14px;background:#e3eeec;border-radius:999px;overflow:hidden}.outcome-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#0f6b72,#4eb5a8)}.outcome-target{height:100%;border-left:3px solid #c76044}
.trend{display:inline-flex;align-items:center;gap:7px}.tri{width:0;height:0;display:inline-block}.tri.up{border-left:7px solid transparent;border-right:7px solid transparent;border-bottom:12px solid #239b73}.tri.down{border-left:7px solid transparent;border-right:7px solid transparent;border-top:12px solid #c84436}.feedback-svg{width:100%;max-width:440px;margin:8px 0 14px}.feedback-label{fill:#617174;font-size:12px}.feedback-title{fill:#142426;font-size:14px;font-weight:800}
.ai-panel{background:#fff;border:1px solid #a9cec8;border-radius:var(--radius);box-shadow:0 20px 48px rgba(13,44,48,.16);overflow:hidden}.ai-panel-head{display:flex;align-items:flex-start;gap:12px;padding:18px;background:var(--navy);color:#fff}.ai-panel-head h2{margin:0 0 4px;font-size:21px}.ai-panel-head p{margin:0;color:#b9cfcb;font-size:12px}.ai-panel-body{padding:18px}.ai-mark{display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;width:34px;height:34px;border-radius:6px;background:#d9f1ec;color:#0f6b72;font-size:12px;font-weight:900}.ai-workbench{margin-top:20px;padding-top:18px;border-top:2px solid #cfe5e0}.ai-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}.ai-head h3{margin:0}.ai-kicker{color:var(--teal);font-size:11px;font-weight:900;text-transform:uppercase}.ai-status{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase}.ai-dot{width:8px;height:8px;border-radius:50%;background:#9aa7a5}.ai-status.ready .ai-dot{background:#239b73}.ai-status.checking .ai-dot{background:#d6a63d}.ai-status.offline .ai-dot{background:#c84436}.ai-button{width:100%;border:0;border-radius:var(--radius);padding:11px 13px;background:var(--teal);color:#fff;font-weight:800;cursor:pointer}.ai-button.secondary{width:auto;background:var(--navy)}.ai-button:disabled{opacity:.55;cursor:wait}.ai-context-strip{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:10px 0;padding:9px 10px;border-left:3px solid var(--teal2);background:var(--ai-soft);color:#355256;font-size:12px}.ai-context-chip{display:inline-flex;align-items:center;border:1px solid #b8d8d2;border-radius:6px;padding:4px 7px;background:#fff;color:#245056;font-size:11px;font-weight:800}.ai-scope{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:12px 0}.ai-scope button{border:1px solid var(--line);border-radius:var(--radius);background:#fff;color:var(--ink);padding:9px;font-size:12px;font-weight:800;cursor:pointer}.ai-scope button.active{background:var(--navy);border-color:var(--navy);color:#fff}.ai-question{width:100%;min-height:88px;resize:vertical;border:1px solid var(--line);border-radius:var(--radius);padding:11px;background:#fff;color:var(--ink)}.ai-actions{display:flex;justify-content:flex-end;margin-top:8px}.ai-suggestions{display:grid;grid-template-columns:1fr;gap:6px;margin:9px 0}.ai-suggestion{border:1px solid #c9dcda;border-radius:var(--radius);background:#fff;color:#284c50;padding:9px 10px;font-size:12px;text-align:left;cursor:pointer}.ai-suggestion:hover{border-color:var(--teal);background:#f1f8f6}.ai-result{margin-top:12px;padding:14px;border:1px solid #bcded8;border-left:4px solid var(--teal);border-radius:var(--radius);background:#eef8f5}.ai-result.error{border-color:#efc5ba;border-left-color:var(--red);background:#fff5f2}.ai-result h3{font-size:16px;margin-bottom:6px}.ai-result p{font-size:13px;margin:0 0 8px}.ai-result ul{font-size:12px;margin:8px 0;padding-left:18px}.ai-caveat{padding-top:8px;border-top:1px solid rgba(15,107,114,.15);color:var(--muted);font-size:11px}.ai-sql{margin-top:10px}.ai-sql summary{cursor:pointer;color:var(--teal);font-size:12px;font-weight:800}.ai-sql pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#102f35;color:#dff4ef;border-radius:var(--radius);padding:10px;font-size:11px}.ai-divider{margin:16px 0;border-top:1px solid var(--line)}.ai-command{overflow:hidden;border:1px solid #9dc9c1;border-radius:var(--radius);background:#fff;box-shadow:0 22px 55px rgba(13,44,48,.15)}.ai-command-header{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:22px 24px;background:var(--navy);color:#fff}.ai-command-title{display:flex;align-items:center;gap:13px}.ai-command-title .ai-mark{width:42px;height:42px}.ai-command-title h2{margin:0 0 4px;font-size:25px}.ai-command-title p{margin:0;color:#bdd2ce;font-size:13px}.ai-command-header .ai-status{color:#d3e7e3}.ai-command-grid{display:grid;grid-template-columns:minmax(270px,.72fr) minmax(480px,1.55fr);min-height:560px}.ai-context-pane{padding:22px;background:#f3f8f7;border-right:1px solid var(--line)}.ai-section-label{margin-bottom:8px;color:#4f6669;font-size:11px;font-weight:900;text-transform:uppercase}.ai-scope.large{grid-template-columns:1fr;margin:0 0 20px}.ai-scope.large button{text-align:left;padding:11px 12px}.ai-context-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.ai-context-stat{min-height:78px;padding:11px;border:1px solid #cfdfdc;border-radius:var(--radius);background:#fff}.ai-context-stat strong{display:block;font-size:19px;color:var(--navy)}.ai-context-stat span{display:block;margin-top:4px;color:var(--muted);font-size:11px}.ai-trust{margin-top:18px;padding-top:16px;border-top:1px solid var(--line)}.ai-trust-row{display:flex;gap:8px;margin:9px 0;color:#486064;font-size:12px}.ai-trust-check{color:var(--teal);font-weight:900}.ai-chat-pane{display:flex;flex-direction:column;min-width:0;padding:24px}.ai-thread{flex:1;min-height:270px}.ai-thread-empty{display:flex;min-height:250px;align-items:center;justify-content:center;text-align:center;border:1px dashed #b8d1cd;border-radius:var(--radius);background:#f8fbfa;padding:28px}.ai-thread-empty strong{display:block;margin-bottom:7px;font-size:20px;color:var(--navy)}.ai-thread-empty p{max-width:520px;margin:0;color:var(--muted);font-size:13px}.ai-chat-pane .ai-suggestions{grid-template-columns:repeat(3,minmax(0,1fr));margin:12px 0}.ai-composer{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:end;margin-top:10px;padding:10px;border:2px solid #87bdb4;border-radius:var(--radius);background:#fff;box-shadow:0 10px 26px rgba(13,44,48,.08)}.ai-composer .ai-question{min-height:70px;border:0;padding:4px;outline:0}.ai-composer .ai-button{width:auto;min-width:150px}.ai-generate{margin-top:14px}.ai-model-note{margin-top:10px;color:var(--muted);font-size:11px}.ai-workspace-answer .ai-result{margin-top:0}.ai-busy{display:inline-flex;align-items:center;gap:8px;color:var(--teal);font-weight:800}.ai-busy:before{content:"";width:12px;height:12px;border:2px solid #b7d9d3;border-top-color:var(--teal);border-radius:50%;animation:ai-spin .8s linear infinite}@keyframes ai-spin{to{transform:rotate(360deg)}}
.ai-primary-panel{position:relative}.ai-primary-panel .ai-panel-head{padding:20px 22px}.ai-primary-panel .ai-panel-body{padding:20px 22px}.ai-primary-panel .ai-workbench{margin:0;padding:0;border:0}.ai-live-grid{display:grid;grid-template-columns:minmax(250px,.62fr) minmax(520px,1.38fr);gap:22px}.ai-live-summary{padding-right:22px;border-right:1px solid var(--line)}.ai-live-query{min-width:0}.ai-live-query .ai-suggestions{grid-template-columns:repeat(3,minmax(0,1fr))}.ai-live-query .ai-scope{grid-template-columns:repeat(2,minmax(0,1fr));margin-top:0}.ai-live-output:empty{display:none}.narrative-panel{background:#fbfdfc}.narrative-panel h2{font-size:21px}.table-pager{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}.table-pager span{color:var(--muted);font-size:12px}.pager-actions{display:flex;gap:8px}.pager-button{border:1px solid var(--line);border-radius:var(--radius);background:#fff;color:var(--ink);padding:7px 11px;font-weight:800;cursor:pointer}.pager-button:disabled{opacity:.4;cursor:not-allowed}
.outreach-ai-panel{margin-top:0}.optimizer-head{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}.optimizer-status{display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:6px 10px;background:#e8f5f1;color:#0f6b72;font-size:12px;font-weight:900}.optimizer-status.nearest{background:#fff2d6;color:#8b5d07}.optimizer-layout{display:grid;grid-template-columns:minmax(360px,.9fr) minmax(440px,1.1fr);gap:22px;margin-top:18px}.optimizer-controls{padding-right:22px;border-right:1px solid var(--line)}.optimizer-presets{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 16px}.optimizer-preset{border:1px solid #b8d5d0;border-radius:var(--radius);padding:8px 10px;background:#fff;color:#254c50;font-size:12px;font-weight:800;cursor:pointer}.optimizer-preset:hover{border-color:var(--teal);background:#f1f8f6}.optimizer-field{display:grid;gap:6px;margin-top:12px}.optimizer-field select,.optimizer-field input[type=number]{width:100%;border:1px solid var(--line);border-radius:var(--radius);padding:10px 11px;background:#fff;color:var(--ink)}.constraint-row{display:grid;grid-template-columns:auto minmax(130px,1fr) 120px;gap:10px;align-items:center;margin-top:10px;padding:10px 0;border-top:1px solid #e8efee;font-size:13px}.constraint-row input[type=checkbox]{width:17px;height:17px;accent-color:var(--teal)}.constraint-row input:disabled{background:#eef3f2;color:#83908e}.optimizer-result{min-width:0}.optimizer-compare{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:12px 0}.optimizer-stat{padding:11px;border:1px solid #cfdfdc;border-radius:var(--radius);background:#f8fbfa}.optimizer-stat strong{display:block;color:var(--navy);font-size:20px}.optimizer-stat span{display:block;margin-top:4px;color:var(--muted);font-size:11px}.optimizer-delta{margin:12px 0;padding:11px 12px;border-left:3px solid var(--teal);background:#eef8f5;color:#355256;font-size:13px}.optimizer-delta.nearest{border-left-color:#d6a63d;background:#fff8e8}.optimizer-actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:14px}.optimizer-actions .ai-button{width:auto}.constraint-summary{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.constraint-chip{border:1px solid #ccdcd9;border-radius:999px;padding:5px 8px;background:#fff;color:#53686b;font-size:11px}.constraint-chip.fail{border-color:#e7b1a5;background:#fff4f1;color:#9c372b}
@media(max-width:1050px){.ai-command-grid,.ai-live-grid,.optimizer-layout{grid-template-columns:1fr}.ai-context-pane{border-right:0;border-bottom:1px solid var(--line)}.ai-context-grid{grid-template-columns:repeat(4,minmax(0,1fr))}.ai-live-summary,.optimizer-controls{padding-right:0;padding-bottom:18px;border-right:0;border-bottom:1px solid var(--line)}}
@media(max-width:640px){.page{width:100%;max-width:100%;padding:18px}.topbar,.summary-grid,.main-grid,.two-grid,.patient-grid,.tab-row{width:100%;max-width:100%}.status-card,.card,.metric{width:100%;max-width:100%}.tab-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}.tab-button{width:100%;min-height:48px;white-space:normal}h1{max-width:100%}.ai-command-header,.optimizer-head{align-items:flex-start;padding:18px}.ai-command-grid{display:block}.ai-context-pane,.ai-chat-pane{padding:18px}.ai-context-grid,.optimizer-compare{grid-template-columns:repeat(2,minmax(0,1fr))}.ai-chat-pane .ai-suggestions,.ai-live-query .ai-suggestions{grid-template-columns:1fr}.ai-composer{grid-template-columns:1fr}.ai-composer .ai-button{width:100%}.ai-panel-head,.ai-primary-panel .ai-panel-head{padding:16px}.ai-primary-panel .ai-panel-body{padding:16px}.table-pager{align-items:flex-start;flex-direction:column}.constraint-row{grid-template-columns:auto 1fr}.constraint-row input[type=number]{grid-column:1/-1}.optimizer-head{display:grid;padding:0}}
</style>
</head>
<body>
<main class="page">
<section class="topbar"><div><div class="eyebrow">High-risk care coordination</div><h1>Care risk outreach</h1><p class="dashboard-subtitle">MVP demo for capacity-based high-risk outreach planning</p></div><aside class="status-card"><div class="small">Current outreach list</div><div class="large" id="statusPercent">Top 10%</div><div class="small" id="statusSummary">103 calls, 26 high-risk events reached</div><button class="status-ai-button" id="openAiWorkspace">Open Care Intelligence</button></aside></section>
<section class="grid summary-grid" id="summaryCards"></section>
<nav class="tab-row" aria-label="Dashboard sections"><button class="tab-button ai-tab active" data-tab="ai"><span class="tab-live">AI</span>Care Intelligence</button><button class="tab-button" data-tab="outreach">Outreach planner</button><button class="tab-button" data-tab="performance">Model &amp; subgroup review</button><button class="tab-button" data-tab="patients">Priority patient list</button><button class="tab-button" data-tab="caregiver">Caregiver match</button><button class="tab-button" data-tab="outcomes">Program outcomes</button></nav>
<section id="tab-ai" class="tab-panel active"><div id="aiWorkspace" class="section"></div></section>
<section id="tab-outreach" class="tab-panel"><section class="ai-panel ai-primary-panel outreach-ai-panel"><header class="ai-panel-head"><span class="ai-mark">AI</span><div><h2>AI outreach copilot</h2><p>Threshold strategy, capacity tradeoffs, and constraint-aware planning</p></div></header><div class="ai-panel-body" id="outreachAiPanel"></div></section><div class="grid main-grid section"><div class="card"><h2>Outreach capacity planner</h2><p class="subtle">Use the population filter, team capacity, and outreach slider to build an achievable prioritized call plan.</p><div class="control-row" id="populationFilters"></div><div class="capacity-input-row"><div><label class="input-label" for="manualCapacityInput">Manual patient capacity</label><input class="number-input" id="manualCapacityInput" type="number" min="1" step="1" value="__DEFAULT_CAPACITY__"></div><div><span class="input-label">Current outreach share</span><strong id="capacityLabel">10%</strong></div></div><div class="slider-wrap"><div class="slider-meta"><span>Adjust by percent</span><span>1% to 30% of selected patients</span></div><input id="capacitySlider" type="range" min="1" max="30" value="10" step="1"></div><div class="capacity-visual"><div class="capacity-bar"><div class="capacity-fill enough" id="capacityFill"></div></div><div class="capacity-visual-text"><span id="capacityUsedText">__DEFAULT_CAPACITY__ selected</span><span id="capacityRemainingText">0 capacity remaining</span></div></div><div class="grid kpi-grid" id="capacityKpis"></div><div class="callout" id="capacityMessage"></div></div><div class="card chart-box"><h2>Outreach value curve</h2><p class="subtle">Cumulative gains estimates operational event capture; the lower chart provides held-out precision-recall context.</p><div id="gainsChart"></div></div></div><section class="card section optimizer-card"><div class="optimizer-head"><div><h2>Constraint optimizer</h2><p class="subtle">Search the available top-risk thresholds for the best feasible plan under the selected operating objective.</p></div><div id="optimizerStatus"></div></div><div class="optimizer-layout"><div class="optimizer-controls"><div class="ai-section-label">Scenario presets</div><div class="optimizer-presets"><button class="optimizer-preset" data-optimizer-preset="balanced">Balanced plan</button><button class="optimizer-preset" data-optimizer-preset="capture">Maximize capture</button><button class="optimizer-preset" data-optimizer-preset="efficiency">High efficiency</button></div><label class="optimizer-field"><span class="input-label">Optimization objective</span><select id="optimizerObjective"><option value="max_events">Maximize events reached</option><option value="max_precision">Maximize outreach precision</option><option value="min_contacts">Minimize contacts</option></select></label><label class="optimizer-field"><span class="input-label">Maximum patients contacted</span><input id="optimizerMaxContacts" type="number" min="1" step="1"></label><div class="ai-section-label section">Optional requirements</div><label class="constraint-row"><input id="optimizerUseRecall" type="checkbox"><span>Minimum recall</span><input id="optimizerMinRecall" type="number" min="1" max="100" step="1" aria-label="Minimum recall percent"></label><label class="constraint-row"><input id="optimizerUsePrecision" type="checkbox"><span>Minimum precision</span><input id="optimizerMinPrecision" type="number" min="1" max="100" step="1" aria-label="Minimum precision percent"></label><label class="constraint-row"><input id="optimizerUseLift" type="checkbox"><span>Minimum lift</span><input id="optimizerMinLift" type="number" min="1" max="20" step="0.1" aria-label="Minimum lift"></label></div><div class="optimizer-result" id="optimizerResult"></div></div></section><div class="card section"><h2>Capacity reference points</h2><p class="subtle">Compare practical call-list sizes and expected yield across the full scored population.</p><div id="capacityTable"></div></div></section>
<section id="tab-performance" class="tab-panel"><div class="callout section">This MVP presents the source dataset's binary outcome as a high-risk event. A production program would require a clinically governed event definition, prediction window, and validation plan.</div><div class="grid two-grid section"><div class="card"><h2>Model performance</h2><p class="subtle">Unweighted logistic regression is recommended by held-out PR AUC and supports transparent review.</p><div id="modelTable"></div></div><div class="card"><h2>Initial subgroup assessment</h2><p class="subtle">This is not a fairness conclusion. It is an initial performance check requiring deeper validation before deployment.</p><div id="subgroupTable"></div></div></div><div class="grid two-grid section"><div class="card"><h2>Classification review</h2><p class="subtle">The top 10% is treated as outreach-positive. Compare correct and incorrect classifications to identify model limitations.</p><div class="control-row" id="errorButtons"><button class="pill active" data-error="False negative">False negatives</button><button class="pill" data-error="False positive">False positives</button><button class="pill" data-error="True positive">True positives</button><button class="pill" data-error="True negative">True negatives</button></div><div class="grid error-profile section" id="errorProfile"></div><h3 class="section" id="patternTitle">Patterns</h3><div id="patternTable"></div></div><div class="card"><h2>Risk drivers</h2><p class="subtle">Predictive associations used for ranking, not evidence of causal effects.</p><div class="control-row"><button class="pill active" data-driver="positive">Higher predicted risk</button><button class="pill" data-driver="negative">Lower predicted risk</button></div><div id="driverChart" class="section"></div></div></div></section>
<section id="tab-patients" class="tab-panel"><section class="ai-panel ai-primary-panel section"><header class="ai-panel-head"><span class="ai-mark">AI</span><div><h2>AI patient brief</h2><p>Priority drivers, queue context, and outreach questions</p></div></header><div class="ai-panel-body" id="patientAiPanel"></div></section><div class="grid patient-grid section"><div class="card"><h2>Priority patient list</h2><p class="subtle" id="patientListNote">Patients currently selected for outreach.</p><div class="control-row section" id="priorityFilters"></div><div id="patientTable"></div></div><aside class="card insight-card narrative-panel"><h2>Patient review</h2><p class="subtle">Deterministic ranking context for the selected patient.</p><div id="patientNarrative"></div></aside></div></section>
<section id="tab-caregiver" class="tab-panel"><section class="ai-panel ai-primary-panel section"><header class="ai-panel-head"><span class="ai-mark">AI</span><div><h2>AI match brief</h2><p>Fit, availability, capacity, and routing tradeoffs</p></div></header><div class="ai-panel-body" id="caregiverAiPanel"></div></section><div class="grid patient-grid section"><div class="card"><h2>Caregiver match</h2><p class="subtle">Demo routing view for assigning a high-risk outreach patient to an available care-team member.</p><div id="caregiverPatient"></div><h3 class="section">Available caregivers</h3><div id="caregiverTable"></div></div><aside class="card insight-card narrative-panel"><h2>Routing rationale</h2><p class="subtle">Deterministic fit summary for the highlighted caregiver.</p><div id="caregiverNarrative"></div></aside></div></section>
<section id="tab-outcomes" class="tab-panel"><div class="grid two-grid section"><div class="card"><h2>Program outcomes</h2><p class="subtle">MVP tracking view for patient engagement and outreach operations after intervention launch.</p><div class="grid kpi-grid" id="outcomeKpis"></div><div class="section" id="outcomeFunnel"></div></div><div class="card"><h2>Caregiver feedback</h2><p class="subtle">Demo readout from care-team feedback after the outreach workflow is running.</p><div id="outcomeInsight"></div></div></div><div class="card section"><h2>Patient and operations scorecard</h2><p class="subtle">Measures to monitor weekly once the program is running.</p><div id="outcomeTable"></div></div></section>
</main>
<script>
const DATA = __DATA__;
const CONFIG = {aiApiBase:"__AI_API_BASE__"};
const state = {population:"All patients", capacity:10, manualCapacity:DATA.summary.default_capacity, dashboard:"ai", driver:"positive", errorGroup:"False negative", patientIndex:0, patientPage:0, priorityFilter:"All", patientSortKey:"contact_rank", patientSortDir:"asc", caregiverId:"cg-003",optimizer:{objective:"max_events",maxContacts:DATA.summary.default_capacity,useRecall:true,minRecall:40,usePrecision:true,minPrecision:15,useLift:false,minLift:3,recommendation:null}};
const aiSession={health:"checking",workspace:{scope:"population",contextKey:"",context:null,insight:null,answer:null,question:"",busy:false,error:""},outreach:{scope:"outreach",contextKey:"",context:null,insight:null,answer:null,question:"",busy:false,error:""},patient:{scope:"patient",contextKey:"",context:null,insight:null,answer:null,question:"",busy:false,error:""},caregiver:{scope:"caregiver",contextKey:"",context:null,insight:null,answer:null,question:"",busy:false,error:""}};
const filters=["All patients","Male","Female","Age <65","Age >=65","Urban","Rural"];
const colors=["#0f6b72","#c76044","#d6a63d","#4eb5a8"];
const CAREGIVERS=[
{id:"cg-001",name:"Maya Chen, RN",role:"Care Coordinator",languages:"English, Mandarin",specialty:"Diabetes education",availability:"Today",distance:7,capacity:6,score:88,strengths:["Strong fit for elevated glucose coaching","Same-day availability","Bilingual support if needed"],watchouts:["Farther from patient than top match"]},
{id:"cg-002",name:"Jordan Patel, LPN",role:"Outreach Nurse",languages:"English, Spanish",specialty:"Medication adherence",availability:"Tomorrow",distance:4,capacity:3,score:76,strengths:["Good general outreach fit","Close distance"],watchouts:["Less clinical focus on cardiovascular risk","Limited remaining capacity"]},
{id:"cg-003",name:"Elena Rivera, RN",role:"Senior Care Manager",languages:"English, Spanish",specialty:"Cardio-metabolic risk",availability:"Today",distance:3,capacity:8,score:94,strengths:["Best specialty fit for the patient's risk profile","Today availability with open capacity","Closest available high-acuity caregiver"],watchouts:["Reserve for high-priority cases when volume spikes"]},
{id:"cg-004",name:"Sam Brooks, SW",role:"Social Worker",languages:"English",specialty:"Transportation and access",availability:"This week",distance:2,capacity:10,score:63,strengths:["Excellent for access barriers","High remaining capacity"],watchouts:["Lower clinical match for risk-factor counseling"]},
{id:"cg-005",name:"Priya Nair, NP",role:"Nurse Practitioner",languages:"English, Hindi",specialty:"Hypertension management",availability:"Tomorrow",distance:9,capacity:2,score:82,strengths:["Strong clinical fit for hypertension and risk review","Can escalate medication questions"],watchouts:["Limited capacity and later availability"]}
];
const PATIENT_PAGE_SIZE=20;
const fmtPct=(v,d=1)=>Number.isFinite(v)?`${(v*100).toFixed(d)}%`:"N/A";
const fmtNum=v=>Number.isFinite(v)?Math.round(v).toLocaleString():"N/A";
const fmtFloat=(v,d=1)=>Number.isFinite(v)?v.toFixed(d):"N/A";
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const friendly=s=>String(s).replace("avg_glucose_level","Glucose").replace("heart_disease","Heart disease").replace("hypertension","Hypertension").replace("smoking_status_","Smoking: ").replace("work_type_","Work: ").replace("Residence_type_","Residence: ").replace("ever_married_","Ever married: ").replace("gender_","Gender: ").replace("bmi","BMI").replace("age","Age");
function aiApiBase(){if(CONFIG.aiApiBase)return CONFIG.aiApiBase;if(["localhost","127.0.0.1"].includes(location.hostname)||location.protocol==="file:")return"http://127.0.0.1:8787";return""}
function patientAiContext(r,m){if(!r)return{};return{priority:r.priority?.label||"Selected",predicted_risk:r.predicted_risk,queue_rank:r.contact_rank||r.rank,contacted_count:m?.k||null,age:r.age,gender:r.gender,residence:r.Residence_type,hypertension:Boolean(r.hypertension),heart_disease:Boolean(r.heart_disease),avg_glucose_level:r.avg_glucose_level,glucose_band:r.glucose_band,bmi:r.bmi,bmi_band:r.bmi_band,smoking_status:r.smoking_status,model_signals:[{feature:"Age",odds_ratio:oddsRatio("age")},{feature:"Hypertension",odds_ratio:oddsRatio("hypertension")},{feature:"Heart disease",odds_ratio:oddsRatio("heart_disease")},{feature:"Glucose",odds_ratio:oddsRatio("avg_glucose_level")}]}}
function caregiverAiContext(c,p){return{patient:patientAiContext({...p,priority:{label:"High"},contact_rank:p.rank},{k:Math.max(1,Math.ceil(DATA.summary.model_rows*.1))}),caregiver:{name:c.name,role:c.role,languages:c.languages,specialty:c.specialty,availability:c.availability,distance_miles:c.distance,open_capacity:c.capacity,match_score:c.score,strengths:c.strengths,watch_outs:c.watchouts}}}
function aiStatusMarkup(){const labels={checking:"Checking AI",ready:"Workers AI live",offline:"Fallback active"};return`<span class="ai-status ${aiSession.health}"><span class="ai-dot"></span>${labels[aiSession.health]}</span>`}
function aiList(items){return items?.length?`<ul>${items.map(item=>`<li>${esc(item)}</li>`).join("")}</ul>`:""}
function aiInsightMarkup(value){if(!value)return"";return`<div class="ai-result"><h3>${esc(value.headline)}</h3><p>${esc(value.summary)}</p>${aiList(value.supporting_factors)}${value.watch_outs?.length?`<strong>Watch-outs</strong>${aiList(value.watch_outs)}`:""}<div class="ai-caveat">Generated from the displayed model and routing facts.</div></div>`}
function aiAnswerMarkup(value){if(!value)return"";return`<div class="ai-result"><h3>${esc(value.title||"AI answer")}</h3><p>${esc(value.answer)}</p>${aiList(value.key_points)}<div class="ai-caveat">${esc(value.caveat)}</div>${value.sql?`<details class="ai-sql"><summary>View generated SQL</summary><pre>${esc(value.sql)}</pre></details>`:""}</div>`}
function aiErrorMarkup(message){return message?`<div class="ai-result error"><h3>AI request unavailable</h3><p>${esc(message)}</p><div class="ai-caveat">The deterministic model explanation above remains available.</div></div>`:""}
function suggestionsFor(kind,scope){if(scope==="population")return["How many scored patients are age 65 or older?","Compare average predicted risk by age group.","Which subgroup has the highest PR AUC?"];return kind==="patient"?["Why is this patient prioritized?","Which model signals matter most?","What should the reviewer verify?"]:["Why is this caregiver recommended?","What are the main tradeoffs?","Is this caregiver available today?"]}
async function apiPost(path,payload){const base=aiApiBase();if(!base)throw new Error("The live AI endpoint has not been published. Use the local preview after starting the Worker.");const response=await fetch(`${base}${path}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const data=await response.json().catch(()=>({error:"The AI service returned an unreadable response."}));if(!response.ok)throw new Error(data.error||`AI request failed (${response.status}).`);return data}
function mountAiWorkbench(kind,context,contextKey){const target=document.getElementById(`${kind}AiPanel`);if(!target)return;const session=aiSession[kind];if(session.contextKey!==contextKey){session.contextKey=contextKey;session.context=context;session.insight=null;session.answer=null;session.question="";session.error=""}else{session.context=context}target.querySelector(".ai-workbench")?.remove();const patientMode=kind==="patient",scopeLabel=patientMode?"Selected patient":"Selected match",copilotTitle=patientMode?"Patient triage copilot":"Caregiver routing copilot",contextName=patientMode?"Ranking evidence and outreach prioritization":"Match fit and operational availability",generateLabel=patientMode?"Generate patient triage brief":"Generate routing brief",askLabel=patientMode?"Ask about patient":"Ask about match";const suggestions=suggestionsFor(kind,session.scope);const workbench=document.createElement("section");workbench.className="ai-workbench";workbench.innerHTML=`<div class="ai-live-grid"><section class="ai-live-summary"><div class="ai-head"><div><div class="ai-kicker">${copilotTitle}</div><h3>${patientMode?"Explain priority":"Explain match quality"}</h3></div>${aiStatusMarkup()}</div><div class="ai-context-strip"><span class="ai-context-chip">Context locked</span>${contextName}</div><button class="ai-button ai-generate" data-ai-generate="${kind}" ${session.busy?"disabled":""}>${session.busy?"Generating...":generateLabel}</button><div class="ai-live-output">${aiInsightMarkup(session.insight)}</div></section><section class="ai-live-query"><div class="ai-section-label">Ask this decision</div><div class="ai-scope"><button data-ai-scope="${kind}" data-scope-value="${kind}" class="${session.scope===kind?"active":""}">${scopeLabel}</button><button data-ai-scope="${kind}" data-scope-value="population" class="${session.scope==="population"?"active":""}">Outreach population</button></div><div class="ai-suggestions">${suggestions.map(q=>`<button class="ai-suggestion" data-ai-suggestion="${kind}" data-question="${esc(q)}">${esc(q)}</button>`).join("")}</div><div class="ai-composer"><textarea class="ai-question" data-ai-question="${kind}" maxlength="500" placeholder="${patientMode?"Ask about ranking signals or outreach priority":"Ask about fit, availability, or routing tradeoffs"}">${esc(session.question)}</textarea><button class="ai-button secondary" data-ai-ask="${kind}" ${session.busy?"disabled":""}>${session.busy?"Working...":askLabel}</button></div>${aiAnswerMarkup(session.answer)}${aiErrorMarkup(session.error)}<div class="ai-model-note">No direct identifiers are sent. Population questions use read-only aggregate SQL.</div></section></div>`;target.appendChild(workbench);workbench.querySelector(`[data-ai-generate="${kind}"]`).onclick=async()=>{session.busy=true;session.error="";mountAiWorkbench(kind,context,contextKey);try{session.insight=await apiPost("/api/insight",{kind,context})}catch(error){session.error=error.message}finally{session.busy=false;mountAiWorkbench(kind,context,contextKey)}};workbench.querySelectorAll(`[data-ai-scope="${kind}"]`).forEach(button=>button.onclick=()=>{session.scope=button.dataset.scopeValue;session.answer=null;session.error="";mountAiWorkbench(kind,context,contextKey)});workbench.querySelectorAll(`[data-ai-suggestion="${kind}"]`).forEach(button=>button.onclick=()=>{session.question=button.dataset.question;mountAiWorkbench(kind,context,contextKey)});const input=workbench.querySelector(`[data-ai-question="${kind}"]`);input.oninput=()=>{session.question=input.value};workbench.querySelector(`[data-ai-ask="${kind}"]`).onclick=async()=>{session.question=input.value.trim();if(!session.question){session.error="Enter a question before asking the AI assistant.";mountAiWorkbench(kind,context,contextKey);return}session.busy=true;session.error="";session.answer=null;mountAiWorkbench(kind,context,contextKey);try{session.answer=await apiPost("/api/ask",{question:session.question,scope:session.scope,context})}catch(error){session.error=error.message}finally{session.busy=false;mountAiWorkbench(kind,context,contextKey)}}}
function outreachAiContext(records){const current=capacityMetrics(records),recommended=state.optimizer.recommendation,o=state.optimizer;return{population:state.population,patient_count:current.n,current_outreach_percent:current.percent,patients_contacted:current.k,observed_events_total:current.totalEvents,events_reached:current.found,precision:current.precision,recall:current.recall,lift:current.lift,minimum_score:current.minRisk,manual_capacity:state.manualCapacity,capacity_used:state.manualCapacity?current.k/state.manualCapacity:null,objective:o.objective,active_constraints:optimizerConstraintLabels(o),recommendation:recommended?{feasible:recommended.feasible,outreach_percent:recommended.percent,patients_contacted:recommended.k,events_reached:recommended.found,precision:recommended.precision,recall:recommended.recall,lift:recommended.lift,minimum_score:recommended.minRisk,unmet_constraints:recommended.checks.filter(c=>!c.pass).map(c=>c.label)}:null}}
function outreachSuggestions(scope){return scope==="population"?["How does outreach yield vary by capacity?","Which subgroup has the highest event rate?","Compare model PR AUC and top-10% recall."]:["What is the best contact threshold under these constraints?","How many patients should we contact to reach the recall target?","What changes if team capacity is reduced by 20%?"]}
function mountOutreachAi(records){const target=document.getElementById("outreachAiPanel");if(!target)return;const session=aiSession.outreach,context=outreachAiContext(records),rec=context.recommendation,key=`${state.population}-${context.patients_contacted}-${state.manualCapacity}-${state.optimizer.objective}-${state.optimizer.maxContacts}-${state.optimizer.useRecall}-${state.optimizer.minRecall}-${state.optimizer.usePrecision}-${state.optimizer.minPrecision}-${state.optimizer.useLift}-${state.optimizer.minLift}-${rec?.patients_contacted||0}`;if(session.contextKey!==key){session.contextKey=key;session.context=context;session.insight=null;session.answer=null;session.question="";session.error=""}else{session.context=context}const suggestions=outreachSuggestions(session.scope);target.innerHTML=`<section class="ai-workbench"><div class="ai-live-grid"><section class="ai-live-summary"><div class="ai-head"><div><div class="ai-kicker">Outreach strategy copilot</div><h3>Explain the current plan</h3></div>${aiStatusMarkup()}</div><div class="ai-context-strip"><span class="ai-context-chip">Context locked</span>${esc(state.population)} &middot; ${fmtNum(context.patients_contacted)} planned contacts &middot; ${rec?.feasible?"feasible recommendation":"constraint review"}</div><button class="ai-button ai-generate" data-outreach-generate ${session.busy?"disabled":""}>${session.busy?"Generating...":"Generate contextual insight"}</button><div class="ai-live-output">${aiInsightMarkup(session.insight)}</div></section><section class="ai-live-query"><div class="ai-section-label">Ask the outreach decision</div><div class="ai-scope"><button data-outreach-scope="outreach" class="${session.scope==="outreach"?"active":""}">Current plan</button><button data-outreach-scope="population" class="${session.scope==="population"?"active":""}">Population data</button></div><div class="ai-suggestions">${suggestions.map(q=>`<button class="ai-suggestion" data-outreach-suggestion data-question="${esc(q)}">${esc(q)}</button>`).join("")}</div><div class="ai-composer"><textarea class="ai-question" data-outreach-question maxlength="500" placeholder="Ask about thresholds, capacity, recall, precision, or constraints">${esc(session.question)}</textarea><button class="ai-button secondary" data-outreach-ask ${session.busy?"disabled":""}>${session.busy?"Working...":"Ask outreach copilot"}</button></div>${aiAnswerMarkup(session.answer)}${aiErrorMarkup(session.error)}<div class="ai-model-note">The optimizer calculates the plan deterministically. AI explains the supplied results and cannot change model scores.</div></section></div></section>`;target.querySelector("[data-outreach-generate]").onclick=async()=>{session.busy=true;session.error="";mountOutreachAi(records);try{session.insight=await apiPost("/api/insight",{kind:"outreach",context})}catch(error){session.error=error.message}finally{session.busy=false;mountOutreachAi(records)}};target.querySelectorAll("[data-outreach-scope]").forEach(button=>button.onclick=()=>{session.scope=button.dataset.outreachScope;session.answer=null;session.error="";mountOutreachAi(records)});target.querySelectorAll("[data-outreach-suggestion]").forEach(button=>button.onclick=()=>{session.question=button.dataset.question;mountOutreachAi(records)});const input=target.querySelector("[data-outreach-question]");input.oninput=()=>{session.question=input.value};target.querySelector("[data-outreach-ask]").onclick=async()=>{session.question=input.value.trim();if(!session.question){session.error="Enter a planning question before asking the outreach copilot.";mountOutreachAi(records);return}session.busy=true;session.error="";session.answer=null;mountOutreachAi(records);try{session.answer=await apiPost("/api/ask",{question:session.question,scope:session.scope,context})}catch(error){session.error=error.message}finally{session.busy=false;mountOutreachAi(records)}}}
function mountAiPanels(records){mountOutreachAi(records);const m=capacityMetrics(records),flagged=priorityRows(records.slice(0,m.k)),filtered=state.priorityFilter==="All"?flagged:flagged.filter(r=>r.priority.label===state.priorityFilter),rows=sortRows(filtered),patient=rows[state.patientIndex];mountAiWorkbench("patient",patientAiContext(patient,m),patient?`patient-${patient.rank}-${m.k}`:"patient-empty");const p=demoCaregiverPatient(),caregiver=CAREGIVERS.find(c=>c.id===state.caregiverId)||CAREGIVERS[0];mountAiWorkbench("caregiver",caregiverAiContext(caregiver,p),`caregiver-${caregiver.id}`)}
function workspaceContext(records){const session=aiSession.workspace,m=capacityMetrics(records),flagged=priorityRows(records.slice(0,m.k)),filtered=state.priorityFilter==="All"?flagged:flagged.filter(r=>r.priority.label===state.priorityFilter),rows=sortRows(filtered),patient=rows[state.patientIndex]||flagged[0]||records[0],demoPatient=demoCaregiverPatient(),caregiver=CAREGIVERS.find(c=>c.id===state.caregiverId)||CAREGIVERS[0];if(session.scope==="patient"){return{kind:"patient",key:patient?`workspace-patient-${patient.rank}-${m.k}`:"workspace-patient-empty",context:patientAiContext(patient,m),facts:[["Patient risk",patient?fmtPct(patient.predicted_risk):"N/A"],["Queue position",patient?`#${fmtNum(patient.contact_rank)}`:"N/A"],["Priority",patient?.priority?.label||"N/A"],["Age",patient?fmtFloat(patient.age,0):"N/A"]],emptyTitle:"Review the selected patient",emptyText:"Generate a concise patient insight or ask a question grounded in the score, queue position, and model signals."}}if(session.scope==="caregiver"){return{kind:"caregiver",key:`workspace-caregiver-${caregiver.id}`,context:caregiverAiContext(caregiver,demoPatient),facts:[["Caregiver",caregiver.name],["Match score",fmtNum(caregiver.score)],["Availability",caregiver.availability],["Open capacity",fmtNum(caregiver.capacity)]],emptyTitle:"Review the selected match",emptyText:"Generate a routing explanation or ask about fit, availability, capacity, and tradeoffs."}}return{kind:"population",key:`workspace-population-${state.population}-${m.k}`,context:{},facts:[["Population",state.population],["Patients scored",fmtNum(m.n)],["Outreach list",fmtNum(m.k)],["Events reached",fmtNum(m.found)]],emptyTitle:"Ask the outreach data",emptyText:"Questions are translated into guarded, read-only aggregate SQL and answered from the model-results database."}}
function renderAiWorkspace(records){const target=document.getElementById("aiWorkspace");if(!target)return;const session=aiSession.workspace,view=workspaceContext(records);if(session.contextKey!==view.key){session.contextKey=view.key;session.context=view.context;session.insight=null;session.answer=null;session.question="";session.error=""}else{session.context=view.context}const suggestions=suggestionsFor(view.kind,session.scope),scopeButtons=[["population","Outreach population"],["patient","Selected patient"],["caregiver","Selected caregiver"]],output=session.busy?`<div class="ai-thread-empty"><div><span class="ai-busy">Care Intelligence is working</span><p>Grounding the response in the selected context.</p></div></div>`:[aiInsightMarkup(session.insight),aiAnswerMarkup(session.answer),aiErrorMarkup(session.error)].filter(Boolean).join("")||`<div class="ai-thread-empty"><div><strong>${view.emptyTitle}</strong><p>${view.emptyText}</p></div></div>`;target.innerHTML=`<section class="ai-command"><header class="ai-command-header"><div class="ai-command-title"><span class="ai-mark">AI</span><div><h2>Care Intelligence</h2><p>Grounded decision support for outreach operations</p></div></div>${aiStatusMarkup()}</header><div class="ai-command-grid"><aside class="ai-context-pane"><div class="ai-section-label">Analysis context</div><div class="ai-scope large">${scopeButtons.map(([value,label])=>`<button data-workspace-scope="${value}" class="${session.scope===value?"active":""}">${label}</button>`).join("")}</div><div class="ai-section-label">Context snapshot</div><div class="ai-context-grid">${view.facts.map(([label,value])=>`<div class="ai-context-stat"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join("")}</div>${view.kind!=="population"?`<button class="ai-button ai-generate" data-workspace-generate ${session.busy?"disabled":""}>${session.busy?"Generating...":"Generate contextual insight"}</button>`:""}<div class="ai-trust"><div class="ai-section-label">Grounding &amp; controls</div><div class="ai-trust-row"><span class="ai-trust-check">OK</span><span>Uses only displayed model and routing facts</span></div><div class="ai-trust-row"><span class="ai-trust-check">OK</span><span>No direct patient identifiers are sent</span></div><div class="ai-trust-row"><span class="ai-trust-check">OK</span><span>Population SQL is read-only and aggregate-only</span></div><div class="ai-trust-row"><span class="ai-trust-check">OK</span><span>Predictive associations are not clinical conclusions</span></div></div></aside><section class="ai-chat-pane"><div class="ai-thread ai-workspace-answer">${output}</div><div class="ai-section-label">Suggested questions</div><div class="ai-suggestions">${suggestions.map(q=>`<button class="ai-suggestion" data-workspace-suggestion data-question="${esc(q)}">${esc(q)}</button>`).join("")}</div><div class="ai-composer"><textarea class="ai-question" data-workspace-question maxlength="500" placeholder="Ask about the selected context or model results">${esc(session.question)}</textarea><button class="ai-button secondary" data-workspace-ask ${session.busy?"disabled":""}>${session.busy?"Working...":"Ask Care Intelligence"}</button></div><div class="ai-model-note">Answers support human review. Generated SQL can be inspected when population data is queried.</div></section></div></section>`;target.querySelectorAll("[data-workspace-scope]").forEach(button=>button.onclick=()=>{session.scope=button.dataset.workspaceScope;session.contextKey="";renderAiWorkspace(records)});target.querySelectorAll("[data-workspace-suggestion]").forEach(button=>button.onclick=()=>{session.question=button.dataset.question;renderAiWorkspace(records)});const input=target.querySelector("[data-workspace-question]");input.oninput=()=>{session.question=input.value};const generate=target.querySelector("[data-workspace-generate]");if(generate)generate.onclick=async()=>{session.busy=true;session.error="";renderAiWorkspace(records);try{session.insight=await apiPost("/api/insight",{kind:view.kind,context:view.context})}catch(error){session.error=error.message}finally{session.busy=false;renderAiWorkspace(records)}};target.querySelector("[data-workspace-ask]").onclick=async()=>{session.question=input.value.trim();if(!session.question){session.error="Enter a question before asking Care Intelligence.";renderAiWorkspace(records);return}session.busy=true;session.error="";session.answer=null;renderAiWorkspace(records);try{session.answer=await apiPost("/api/ask",{question:session.question,scope:session.scope,context:view.context})}catch(error){session.error=error.message}finally{session.busy=false;renderAiWorkspace(records)}}}
function refreshAiStatus(){document.querySelectorAll(".ai-status").forEach(el=>{el.outerHTML=aiStatusMarkup()})}
async function checkAiHealth(){const base=aiApiBase();if(!base){aiSession.health="offline";refreshAiStatus();return}try{const response=await fetch(`${base}/api/health`);const data=await response.json();aiSession.health=response.ok&&data.ai_binding?"ready":"offline"}catch{aiSession.health="offline"}refreshAiStatus()}
function populationFilter(r){if(state.population==="All patients")return true;if(state.population==="Male")return r.gender==="Male";if(state.population==="Female")return r.gender==="Female";if(state.population==="Age <65")return r.age<65;if(state.population==="Age >=65")return r.age>=65;if(state.population==="Urban")return r.Residence_type==="Urban";if(state.population==="Rural")return r.Residence_type==="Rural";return true}
function selectedRecords(){return DATA.records.filter(populationFilter).slice().sort((a,b)=>b.predicted_risk-a.predicted_risk)}
function capacityMetrics(records){const n=records.length,total=records.reduce((s,r)=>s+r.event,0),k=Math.max(1,Math.min(n,Math.ceil(n*state.capacity/100))),sel=records.slice(0,k),found=sel.reduce((s,r)=>s+r.event,0),precision=found/k,recall=total?found/total:NaN,base=n?total/n:NaN,lift=base?precision/base:NaN,minRisk=sel.length?sel[sel.length-1].predicted_risk:NaN;return{n,totalEvents:total,k,found,precision,recall,lift,minRisk,percent:n?k/n*100:0,baseRate:base}}
function candidateMetrics(records,inputPercent){const n=records.length,total=records.reduce((s,r)=>s+r.event,0),k=Math.max(1,Math.min(n,Math.ceil(n*inputPercent/100))),sel=records.slice(0,k),found=sel.reduce((s,r)=>s+r.event,0),precision=k?found/k:NaN,recall=total?found/total:NaN,base=n?total/n:NaN,lift=base?precision/base:NaN,minRisk=sel.length?sel[sel.length-1].predicted_risk:NaN;return{inputPercent,n,totalEvents:total,k,found,precision,recall,lift,minRisk,percent:n?k/n*100:0,baseRate:base}}
function optimizerConstraintLabels(o=state.optimizer){const labels=[`No more than ${fmtNum(o.maxContacts)} contacts`];if(o.useRecall)labels.push(`Recall at least ${fmtNum(o.minRecall)}%`);if(o.usePrecision)labels.push(`Precision at least ${fmtNum(o.minPrecision)}%`);if(o.useLift)labels.push(`Lift at least ${fmtFloat(o.minLift,1)}x`);return labels}
function optimizerChecks(candidate,o=state.optimizer){const checks=[{label:`Contacts <= ${fmtNum(o.maxContacts)}`,pass:candidate.k<=o.maxContacts,violation:Math.max(0,(candidate.k-o.maxContacts)/Math.max(1,o.maxContacts))}];if(o.useRecall)checks.push({label:`Recall >= ${fmtNum(o.minRecall)}%`,pass:candidate.recall>=o.minRecall/100,violation:Math.max(0,o.minRecall/100-candidate.recall)/Math.max(.01,o.minRecall/100)});if(o.usePrecision)checks.push({label:`Precision >= ${fmtNum(o.minPrecision)}%`,pass:candidate.precision>=o.minPrecision/100,violation:Math.max(0,o.minPrecision/100-candidate.precision)/Math.max(.01,o.minPrecision/100)});if(o.useLift)checks.push({label:`Lift >= ${fmtFloat(o.minLift,1)}x`,pass:candidate.lift>=o.minLift,violation:Math.max(0,o.minLift-candidate.lift)/Math.max(.1,o.minLift)});return checks}
function optimizeOutreach(records){const o=state.optimizer,candidates=Array.from({length:30},(_,i)=>candidateMetrics(records,i+1)).map(c=>({...c,checks:optimizerChecks(c,o)})),feasible=candidates.filter(c=>c.checks.every(x=>x.pass));let pool=feasible,selected;if(feasible.length){if(o.objective==="max_precision")pool.sort((a,b)=>b.precision-a.precision||b.found-a.found||a.k-b.k);else if(o.objective==="min_contacts")pool.sort((a,b)=>a.k-b.k||b.recall-a.recall||b.precision-a.precision);else pool.sort((a,b)=>b.found-a.found||b.recall-a.recall||a.k-b.k);selected=pool[0]}else{selected=candidates.slice().sort((a,b)=>a.checks.reduce((s,c)=>s+c.violation,0)-b.checks.reduce((s,c)=>s+c.violation,0)||b.found-a.found||a.k-b.k)[0]}return{...selected,feasible:feasible.length>0}}
function objectiveLabel(value){return({max_events:"maximize events reached",max_precision:"maximize outreach precision",min_contacts:"minimize contacts"})[value]||value}
function applyOptimizerPreset(name){const o=state.optimizer;if(name==="capture")Object.assign(o,{objective:"max_events",maxContacts:state.manualCapacity,useRecall:true,minRecall:50,usePrecision:false,minPrecision:12,useLift:false,minLift:3});else if(name==="efficiency")Object.assign(o,{objective:"max_precision",maxContacts:Math.max(1,Math.round(state.manualCapacity*.75)),useRecall:false,minRecall:30,usePrecision:true,minPrecision:25,useLift:true,minLift:4});else Object.assign(o,{objective:"max_events",maxContacts:state.manualCapacity,useRecall:true,minRecall:40,usePrecision:true,minPrecision:15,useLift:false,minLift:3});renderAll()}
function renderOptimizationStudio(records){const o=state.optimizer,current=capacityMetrics(records),r=optimizeOutreach(records);o.recommendation=r;document.getElementById("optimizerObjective").value=o.objective;document.getElementById("optimizerMaxContacts").value=o.maxContacts;document.getElementById("optimizerUseRecall").checked=o.useRecall;document.getElementById("optimizerMinRecall").value=o.minRecall;document.getElementById("optimizerMinRecall").disabled=!o.useRecall;document.getElementById("optimizerUsePrecision").checked=o.usePrecision;document.getElementById("optimizerMinPrecision").value=o.minPrecision;document.getElementById("optimizerMinPrecision").disabled=!o.usePrecision;document.getElementById("optimizerUseLift").checked=o.useLift;document.getElementById("optimizerMinLift").value=o.minLift;document.getElementById("optimizerMinLift").disabled=!o.useLift;document.getElementById("optimizerStatus").innerHTML=`<span class="optimizer-status ${r.feasible?"":"nearest"}">${r.feasible?"Feasible plan found":"No exact match"}</span>`;const contactDelta=r.k-current.k,eventDelta=r.found-current.found,recallDelta=(r.recall-current.recall)*100;document.getElementById("optimizerResult").innerHTML=`<div class="ai-section-label">${r.feasible?"Recommended operating point":"Closest available operating point"}</div><h3>Top ${r.percent.toFixed(1)}% for ${objectiveLabel(o.objective)}</h3><p class="subtle">Evaluated all integer thresholds from 1% to 30% for ${esc(state.population.toLowerCase())}.</p><div class="optimizer-compare"><div class="optimizer-stat"><strong>${fmtNum(r.k)}</strong><span>patients contacted</span></div><div class="optimizer-stat"><strong>${fmtNum(r.found)}</strong><span>events reached</span></div><div class="optimizer-stat"><strong>${fmtPct(r.recall)}</strong><span>recall</span></div><div class="optimizer-stat"><strong>${fmtPct(r.precision)}</strong><span>precision</span></div></div><div class="optimizer-delta ${r.feasible?"":"nearest"}">${r.feasible?"Compared with the current plan":"This option has the smallest normalized constraint gap"}: ${contactDelta>=0?"+":""}${fmtNum(contactDelta)} contacts, ${eventDelta>=0?"+":""}${fmtNum(eventDelta)} events, and ${recallDelta>=0?"+":""}${fmtFloat(recallDelta,1)} recall points. Minimum score ${fmtPct(r.minRisk)}; lift ${fmtFloat(r.lift,1)}x.</div><div class="constraint-summary">${r.checks.map(c=>`<span class="constraint-chip ${c.pass?"":"fail"}">${c.pass?"Met":"Unmet"}: ${esc(c.label)}</span>`).join("")}</div><div class="optimizer-actions"><button class="ai-button" id="applyOptimizerPlan">Apply ${r.feasible?"recommended":"closest"} plan</button><span class="subtle">Human review remains required before operational use.</span></div>`;document.querySelectorAll("[data-optimizer-preset]").forEach(button=>button.onclick=()=>applyOptimizerPreset(button.dataset.optimizerPreset));document.getElementById("optimizerObjective").onchange=e=>{o.objective=e.target.value;renderAll()};document.getElementById("optimizerMaxContacts").onchange=e=>{o.maxContacts=Math.max(1,Math.round(Number(e.target.value)||1));renderAll()};[["optimizerUseRecall","useRecall"],["optimizerUsePrecision","usePrecision"],["optimizerUseLift","useLift"]].forEach(([id,key])=>document.getElementById(id).onchange=e=>{o[key]=e.target.checked;renderAll()});[["optimizerMinRecall","minRecall",1,100],["optimizerMinPrecision","minPrecision",1,100],["optimizerMinLift","minLift",.1,20]].forEach(([id,key,min,max])=>document.getElementById(id).onchange=e=>{o[key]=Math.min(max,Math.max(min,Number(e.target.value)||min));renderAll()});document.getElementById("applyOptimizerPlan").onclick=()=>{state.capacity=r.inputPercent;document.getElementById("capacitySlider").value=r.inputPercent;state.patientIndex=0;state.patientPage=0;renderAll()}}
function capacityStatus(util){if(!Number.isFinite(util))return"over";if(util<.70||util>1)return"over";if(util<.90)return"close";return"enough"}
function gainsPoints(records){const total=records.reduce((s,r)=>s+r.event,0);let cum=0;return records.map((r,i)=>{cum+=r.event;return{pctPatients:(i+1)/records.length,pctEvents:total?cum/total:0,patients:i+1,events:cum}})}
function table(headers,rows){return`<table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`}
function renderSummaryCards(m){const allK=Math.max(1,Math.ceil(DATA.summary.model_rows*m.percent/100));const capacityUsed=state.manualCapacity?allK/state.manualCapacity:NaN;const statusClass=capacityStatus(capacityUsed);const cards=[["Recommended Outreach",`${fmtNum(m.k)} patients`,`Top ${m.percent.toFixed(1)}% prioritized call list`,`capacity-state ${statusClass}`],["Expected Events Reached",`${fmtNum(m.found)} events`,`${fmtPct(m.recall)} of observed high-risk events captured`,""],["Outreach Efficiency",`${fmtFloat(m.lift,1)}x lift`,"Versus random selection",""],["Capacity Used",fmtPct(capacityUsed),"Based on manual capacity setting",`capacity-state ${statusClass}`]];document.getElementById("summaryCards").innerHTML=cards.map(([l,v,n,c])=>`<div class="metric ${c}"><div class="metric-label">${l}</div><div class="metric-value">${v}</div><div class="metric-note">${n}</div></div>`).join("")}
function renderFilters(){document.getElementById("populationFilters").innerHTML=filters.map(f=>`<button class="pill ${state.population===f?"active":""}" data-pop="${f}">${f}</button>`).join("");document.querySelectorAll("[data-pop]").forEach(btn=>btn.onclick=()=>{state.population=btn.dataset.pop;state.patientIndex=0;state.patientPage=0;renderAll()})}
function renderCapacity(records,m){document.getElementById("capacityLabel").textContent=`${m.percent.toFixed(1)}%`;document.getElementById("statusPercent").textContent=`Top ${m.percent.toFixed(1)}%`;document.getElementById("statusSummary").textContent=`${fmtNum(m.k)} selected, ${fmtNum(m.found)} high-risk events reached`;const allK=Math.max(1,Math.ceil(DATA.summary.model_rows*m.percent/100));const util=state.manualCapacity?allK/state.manualCapacity:NaN;const fill=document.getElementById("capacityFill");fill.style.width=`${Math.min(100,Math.max(0,allK/DATA.summary.model_rows*100))}%`;fill.className=`capacity-fill ${capacityStatus(util)}`;document.getElementById("capacityUsedText").textContent=`${fmtNum(allK)} selected of ${fmtNum(DATA.summary.model_rows)} total scored patients`;const rem=state.manualCapacity-allK;document.getElementById("capacityRemainingText").textContent=rem>=0?`${fmtNum(rem)} manual capacity remaining`:`${fmtNum(Math.abs(rem))} over manual capacity`;document.getElementById("capacityKpis").innerHTML=[["Patients contacted",fmtNum(m.k),`${m.percent.toFixed(1)}% of ${fmtNum(m.n)} selected patients`],["High-risk events found",fmtNum(m.found),`of ${fmtNum(m.totalEvents)} observed events`],["Precision@K",fmtPct(m.precision),"share of contacted patients with an observed event"],["Recall@K",fmtPct(m.recall),"share of observed high-risk events captured"],["Lift",`${fmtFloat(m.lift,1)}x`,"compared with random selection"],["Minimum risk score",fmtPct(m.minRisk),"lowest score in contacted group"]].map(([l,v,n])=>`<div class="metric"><div class="metric-label">${l}</div><div class="metric-value">${v}</div><div class="metric-note">${n}</div></div>`).join("");document.getElementById("capacityMessage").textContent=`If the team contacts ${fmtNum(m.k)} ${state.population==="All patients"?"patients":state.population+" patients"}, the prioritized list would reach ${fmtNum(m.found)} of ${fmtNum(m.totalEvents)} observed high-risk events in the scored population. Manual capacity is set to ${fmtNum(state.manualCapacity)} patients.`}
function drawLineChart(id,series,opts){const w=760,h=340,p={l:54,r:24,t:18,b:45};const xMax=opts.xMax??1,yMax=opts.yMax??1;const xy=pt=>[p.l+(pt.x/xMax)*(w-p.l-p.r),h-p.b-(pt.y/yMax)*(h-p.t-p.b)];const grid=[0,.25,.5,.75,1].map(t=>{const x=p.l+t*(w-p.l-p.r),y=h-p.b-t*(h-p.t-p.b);return`<line x1="${x}" x2="${x}" y1="${p.t}" y2="${h-p.b}" stroke="#e2ecea"/><text x="${x}" y="${h-16}" text-anchor="middle" class="chart-label">${Math.round(t*xMax*100)}%</text><line x1="${p.l}" x2="${w-p.r}" y1="${y}" y2="${y}" stroke="#e2ecea"/><text x="10" y="${y+4}" class="chart-label">${Math.round(t*yMax*100)}%</text>`}).join("");const lines=series.map((s,i)=>{const d=s.points.map((pt,j)=>{const [x,y]=xy(pt);return`${j?"L":"M"} ${x.toFixed(1)} ${y.toFixed(1)}`}).join(" ");return`<path d="${d}" fill="none" stroke="${s.color||colors[i%colors.length]}" stroke-width="3" ${s.dash?'stroke-dasharray="8 7"':""}/>`}).join("");const legend=series.map((s,i)=>`<span style="display:inline-flex;align-items:center;gap:6px;margin-right:12px;font-size:12px;color:#617174"><span style="width:22px;height:3px;background:${s.color||colors[i%colors.length]};display:inline-block"></span>${s.name}</span>`).join("");document.getElementById(id).innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img">${grid}${lines}<text x="${p.l}" y="14" class="chart-title">${opts.title}</text><text x="${w/2}" y="${h-2}" text-anchor="middle" class="chart-label">${opts.xLabel}</text></svg><div>${legend}</div>`}
function renderGains(records,m){const w=760,h=700;const top={l:54,r:24,t:28,b:392};const bot={l:54,r:24,t:374,b:46};const map=(pt,p,xMax=1,yMax=1)=>[p.l+(pt.x/xMax)*(w-p.l-p.r),h-p.b-(pt.y/yMax)*(h-p.t-p.b)];const path=(pts,p)=>pts.map((pt,i)=>{const[x,y]=map(pt,p);return`${i?"L":"M"} ${x.toFixed(1)} ${y.toFixed(1)}`}).join(" ");const grid=(p)=>[0,.25,.5,.75,1].map(t=>{const x=p.l+t*(w-p.l-p.r),y=h-p.b-t*(h-p.t-p.b);return`<line x1="${x}" x2="${x}" y1="${p.t}" y2="${h-p.b}" stroke="#e2ecea"/><text x="${x}" y="${h-p.b+18}" text-anchor="middle" class="chart-label">${Math.round(t*100)}%</text><line x1="${p.l}" x2="${w-p.r}" y1="${y}" y2="${y}" stroke="#e2ecea"/><text x="10" y="${y+4}" class="chart-label">${Math.round(t*100)}%</text>`}).join("");const gainPts=gainsPoints(records).map(p=>({x:p.pctPatients,y:p.pctEvents}));const random=[{x:0,y:0},{x:1,y:1}];const markerX=top.l+(m.percent/100)*(w-top.l-top.r);const current=gainPts[Math.max(0,Math.min(gainPts.length-1,m.k-1))]||{x:0,y:0};const[cx,cy]=map(current,top);const labelAnchor=cx>w-210?"end":"start";const labelX=cx>w-210?cx-12:cx+12;const prLines=DATA.modelCurves.map((c,i)=>`<path d="${path(c.pr,bot)}" fill="none" stroke="${colors[i%colors.length]}" stroke-width="3"/>`).join("");const prLegend=DATA.modelCurves.map((c,i)=>`<span style="display:inline-flex;align-items:center;gap:6px;margin-right:12px;font-size:12px;color:#617174"><span style="width:22px;height:3px;background:${colors[i%colors.length]};display:inline-block"></span>${c.model} AP=${c.pr_auc.toFixed(3)}</span>`).join("");document.getElementById("gainsChart").innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Cumulative gains and precision-recall chart">${grid(top)}<path d="${path(random,top)}" fill="none" stroke="#9aa7a5" stroke-width="3" stroke-dasharray="8 7"/><path d="${path(gainPts,top)}" fill="none" stroke="#0f6b72" stroke-width="4"/><line x1="${markerX}" x2="${markerX}" y1="${top.t}" y2="${h-top.b}" stroke="#d6a63d" stroke-width="3"/><circle cx="${cx}" cy="${cy}" r="7" fill="#d6a63d" stroke="white" stroke-width="3"/><text x="${labelX}" y="${cy-8}" text-anchor="${labelAnchor}" class="chart-title">${fmtNum(m.k)} patients reached</text><text x="${labelX}" y="${cy+10}" text-anchor="${labelAnchor}" class="chart-label">${fmtNum(m.found)} events found</text><text x="${top.l}" y="18" class="chart-title">${m.percent.toFixed(1)}% captures ${fmtPct(m.recall,0)} of observed high-risk events</text><text x="${w/2}" y="${h-top.b+38}" text-anchor="middle" class="chart-label">% of patients contacted</text>${grid(bot)}${prLines}<text x="${bot.l}" y="${bot.t-10}" class="chart-title">Precision-recall curve (held-out test set)</text><text x="${w/2}" y="${h-4}" text-anchor="middle" class="chart-label">Recall</text></svg><div><span style="display:inline-flex;align-items:center;gap:6px;margin-right:12px;font-size:12px;color:#617174"><span style="width:22px;height:3px;background:#0f6b72;display:inline-block"></span>Prioritized outreach</span><span style="display:inline-flex;align-items:center;gap:6px;margin-right:12px;font-size:12px;color:#617174"><span style="width:22px;height:3px;background:#d6a63d;display:inline-block"></span>Current capacity marker</span>${prLegend}</div>`}
function renderCurves(){}
function renderTables(){document.getElementById("modelTable").innerHTML=table(["Model","PR AUC","Top 10% precision","Top 10% recall"],DATA.modelMetrics.map(r=>[r.model,r.pr_auc.toFixed(3),fmtPct(r.top10_precision),fmtPct(r.top10_recall)]));document.getElementById("capacityTable").innerHTML=table(["Capacity","Contacted","Events found","Precision","Recall","Lift"],DATA.capacityTable.filter(r=>[.01,.05,.10,.15,.20,.25].includes(r.capacity_fraction)).map(r=>[r.outreach_capacity,fmtNum(r.patients_contacted),fmtNum(r.true_events_found),fmtPct(r.precision),fmtPct(r.recall),`${fmtFloat(r.lift,1)}x`]));document.getElementById("subgroupTable").innerHTML=table(["Population","N","Event rate","PR AUC","Recall@Top 10%","Global top 10% share"],DATA.subgroupPerformance.map(r=>[r.population,fmtNum(r.n),fmtPct(r.event_rate),r.pr_auc?r.pr_auc.toFixed(3):"N/A",fmtPct(r.recall_top_10),fmtPct(r.global_top_10_share)]))}
function priorityFor(rank,total){const third=Math.ceil(total/3);if(rank<=third)return{label:"High",cls:"high"};if(rank<=third*2)return{label:"Medium",cls:"medium"};return{label:"Low",cls:"low"}}
function oddsRatio(feature){const row=DATA.coefficients.find(r=>r.feature===feature);return row?Math.exp(row.coefficient):NaN}
function priorityRows(flagged){return flagged.map((r,i)=>({...r,contact_rank:i+1,priority:priorityFor(i+1,flagged.length)}))}
function renderPriorityFilters(rows){const counts={All:rows.length,High:0,Medium:0,Low:0};rows.forEach(r=>counts[r.priority.label]++);document.getElementById("priorityFilters").innerHTML=["All","High","Medium","Low"].map(label=>`<button class="pill ${state.priorityFilter===label?"active":""}" data-priority="${label}">${label} ${fmtNum(counts[label])}</button>`).join("");document.querySelectorAll("[data-priority]").forEach(btn=>btn.onclick=()=>{state.priorityFilter=btn.dataset.priority;state.patientIndex=0;state.patientPage=0;renderPatientDetails(selectedRecords())})}
function sortVal(r,key){if(key==="priority")return r.contact_rank;if(key==="risk")return r.predicted_risk;if(key==="age")return r.age;if(key==="gender")return r.gender;if(key==="residence")return r.Residence_type;if(key==="hypertension")return r.hypertension;if(key==="heart_disease")return r.heart_disease;if(key==="glucose")return r.avg_glucose_level;if(key==="bmi")return Number(r.bmi)||0;if(key==="smoking")return r.smoking_status;return r.contact_rank}
function sortRows(rows){const dir=state.patientSortDir==="asc"?1:-1;return rows.slice().sort((a,b)=>{const av=sortVal(a,state.patientSortKey),bv=sortVal(b,state.patientSortKey);if(typeof av==="string"||typeof bv==="string")return String(av).localeCompare(String(bv))*dir;return (av-bv)*dir})}
function sortHead(label,key){const mark=state.patientSortKey===key?`<span class="sort-mark">${state.patientSortDir==="asc"?"▲":"▼"}</span>`:"";return`<th class="sortable" data-sort="${key}">${label}${mark}</th>`}
function renderPatientDetails(records){const m=capacityMetrics(records),flagged=priorityRows(records.slice(0,m.k)),filtered=state.priorityFilter==="All"?flagged:flagged.filter(r=>r.priority.label===state.priorityFilter),rows=sortRows(filtered),pageCount=Math.max(1,Math.ceil(rows.length/PATIENT_PAGE_SIZE));state.patientPage=Math.max(0,Math.min(state.patientPage,pageCount-1));const start=state.patientPage*PATIENT_PAGE_SIZE,visibleRows=rows.slice(start,start+PATIENT_PAGE_SIZE),end=start+visibleRows.length;if(!rows.length){state.patientIndex=0}else if(state.patientIndex<start||state.patientIndex>=end){state.patientIndex=start}renderPriorityFilters(flagged);document.getElementById("patientListNote").textContent=rows.length?`Showing ${fmtNum(start+1)}-${fmtNum(end)} of ${fmtNum(rows.length)} filtered patients from ${fmtNum(flagged.length)} flagged for contact.`:`No patients match the current outreach and priority filters.`;document.getElementById("patientTable").innerHTML=`<table class="patient-table"><thead><tr>${sortHead("Priority","priority")}${sortHead("Rank","contact_rank")}${sortHead("Risk score","risk")}${sortHead("Age","age")}${sortHead("Gender","gender")}${sortHead("Residence","residence")}${sortHead("Hypertension","hypertension")}${sortHead("Heart disease","heart_disease")}${sortHead("Glucose","glucose")}${sortHead("BMI","bmi")}${sortHead("Smoking","smoking")}</tr></thead><tbody>${visibleRows.map((r,i)=>{const p=r.priority,rowIndex=start+i;return`<tr class="${rowIndex===state.patientIndex?"selected":""}" data-patient-row="${rowIndex}"><td><span class="priority-badge priority-${p.cls}">${p.label}</span></td><td>${fmtNum(r.contact_rank)}</td><td><strong>${fmtPct(r.predicted_risk)}</strong></td><td>${fmtFloat(r.age,0)}</td><td>${esc(r.gender)}</td><td>${esc(r.Residence_type)}</td><td>${r.hypertension?"Yes":"No"}</td><td>${r.heart_disease?"Yes":"No"}</td><td>${esc(r.glucose_band)}</td><td>${esc(r.bmi_band)}</td><td>${esc(r.smoking_status)}</td></tr>`}).join("")}</tbody></table><div class="table-pager"><span>Page ${fmtNum(state.patientPage+1)} of ${fmtNum(pageCount)} &middot; ${fmtNum(visibleRows.length)} patients shown</span><div class="pager-actions"><button class="pager-button" data-patient-page="previous" ${state.patientPage===0?"disabled":""}>Previous</button><button class="pager-button" data-patient-page="next" ${state.patientPage>=pageCount-1?"disabled":""}>Next</button></div></div>`;document.querySelectorAll("[data-sort]").forEach(th=>th.onclick=()=>{const key=th.dataset.sort;if(state.patientSortKey===key){state.patientSortDir=state.patientSortDir==="asc"?"desc":"asc"}else{state.patientSortKey=key;state.patientSortDir="asc"}state.patientIndex=0;state.patientPage=0;renderPatientDetails(selectedRecords())});document.querySelectorAll("[data-patient-row]").forEach(row=>row.onclick=()=>{state.patientIndex=Number(row.dataset.patientRow);renderPatientDetails(selectedRecords())});document.querySelectorAll("[data-patient-page]").forEach(button=>button.onclick=()=>{state.patientPage+=button.dataset.patientPage==="next"?1:-1;state.patientIndex=state.patientPage*PATIENT_PAGE_SIZE;renderPatientDetails(selectedRecords())});renderPatientInsight(rows[state.patientIndex],m)}
function renderPatientInsight(r,m){const el=document.getElementById("patientNarrative");if(!r){el.innerHTML=`<div class="insight-box"><strong>No patients flagged</strong><p class="subtle">Adjust the outreach percentage, population filter, or priority filter to generate a contact list.</p></div>`;return}const p=r.priority;const drivers=[];if(r.age>=65)drivers.push("older age");if(r.hypertension)drivers.push("hypertension");if(r.heart_disease)drivers.push("heart disease");if(r.avg_glucose_level>=126)drivers.push("elevated glucose");if(Number(r.bmi)>=30)drivers.push("higher BMI");if(!drivers.length)drivers.push("relative ranking within the selected population");const ageOr=oddsRatio("age"),htnOr=oddsRatio("hypertension"),heartOr=oddsRatio("heart_disease"),glucoseOr=oddsRatio("avg_glucose_level");const tierText=p.label==="High"?"This patient is in the top third of the current contacted list and should be near the front of the call queue.":p.label==="Medium"?"This patient is in the middle third of the current contacted list and is appropriate for outreach after the highest-priority group.":"This patient is in the final third of the current contacted list, closer to the operating cutoff but still selected by the outreach plan.";el.innerHTML=`<div class="priority-badge priority-${p.cls}">${p.label} priority</div><div class="insight-score">${fmtPct(r.predicted_risk)}</div><p>${tierText}</p><div class="insight-box"><strong>Queue position</strong><p class="subtle">Rank ${fmtNum(r.contact_rank)} of ${fmtNum(m.k)} contacted patients at the current slicer setting.</p></div><div class="insight-box"><strong>Ranking signals</strong><ul>${drivers.map(d=>`<li>${esc(d)}</li>`).join("")}</ul></div><div class="insight-box"><strong>Model signal context</strong><p class="subtle">In the logistic model, higher scaled age has an odds ratio of ${fmtFloat(ageOr,2)}; hypertension ${fmtFloat(htnOr,2)}; heart disease ${fmtFloat(heartOr,2)}; and glucose ${fmtFloat(glucoseOr,2)}. These are predictive associations used for ranking, not causal claims.</p></div>`}
function demoCaregiverPatient(){return DATA.records.slice().sort((a,b)=>b.predicted_risk-a.predicted_risk)[0]}
function caregiverFit(c){if(c.score>=90)return{label:"Recommended",cls:"match-badge"};if(c.score>=80)return{label:"Strong backup",cls:"match-badge"};if(c.score>=70)return{label:"Usable",cls:"match-badge"};return{label:"Lower fit",cls:"match-badge"}}
function renderCaregiverMatch(){const p=demoCaregiverPatient();const selected=CAREGIVERS.find(c=>c.id===state.caregiverId)||CAREGIVERS.slice().sort((a,b)=>b.score-a.score)[0];const profile=[["Patient risk",fmtPct(p.predicted_risk)],["Age",fmtFloat(p.age,0)],["Glucose",esc(p.glucose_band)],["BMI",esc(p.bmi_band)],["Hypertension",p.hypertension?"Yes":"No"],["Heart disease",p.heart_disease?"Yes":"No"],["Residence",esc(p.Residence_type)],["Smoking",esc(p.smoking_status)]];document.getElementById("caregiverPatient").innerHTML=`<div class="caregiver-profile">${profile.map(([l,v])=>`<div class="mini"><strong>${v}</strong><span>${l}</span></div>`).join("")}</div><div class="callout">Fixed demo patient: high-scoring outreach candidate used to show how caregiver routing could work after the model creates a prioritized call list.</div>`;const rows=CAREGIVERS.slice().sort((a,b)=>b.score-a.score);document.getElementById("caregiverTable").innerHTML=`<table class="caregiver-table"><thead><tr><th>Caregiver</th><th>Role</th><th>Specialty</th><th>Availability</th><th>Distance</th><th>Open capacity</th><th>Match</th></tr></thead><tbody>${rows.map(c=>{const fit=caregiverFit(c);return`<tr class="${c.id===selected.id?"selected":""}" data-caregiver="${c.id}"><td><strong>${esc(c.name)}</strong><br><span class="subtle">${esc(c.languages)}</span></td><td>${esc(c.role)}</td><td>${esc(c.specialty)}</td><td>${esc(c.availability)}</td><td>${fmtFloat(c.distance,0)} mi</td><td>${fmtNum(c.capacity)}</td><td><strong>${fmtNum(c.score)}</strong><br><span class="${fit.cls}">${fit.label}</span></td></tr>`}).join("")}</tbody></table>`;document.querySelectorAll("[data-caregiver]").forEach(row=>row.onclick=()=>{state.caregiverId=row.dataset.caregiver;renderCaregiverMatch()});renderCaregiverInsight(selected,p)}
function renderCaregiverInsight(c,p){const isBest=c.score===Math.max(...CAREGIVERS.map(x=>x.score));const fitText=isBest?"This is the recommended caregiver for the demo patient. The match is strong because clinical specialty, availability, distance, and open capacity all line up with the patient's risk profile.":c.score>=80?"This caregiver is a reasonable backup, but the match is not as strong as the recommended option after balancing specialty, timing, and capacity.":c.score>=70?"This caregiver could support outreach if the recommended team members are unavailable, but there are clear tradeoffs.":"This is a weaker match for the demo patient and should usually be reserved for a different type of need.";document.getElementById("caregiverNarrative").innerHTML=`<span class="match-badge">${isBest?"Recommended caregiver":"Highlighted caregiver"}</span><div class="match-score">${fmtNum(c.score)}</div><h3>${esc(c.name)}</h3><p>${fitText}</p><div class="insight-box"><strong class="fit-good">Fit strengths</strong><ul>${c.strengths.map(s=>`<li>${esc(s)}</li>`).join("")}</ul></div><div class="insight-box"><strong class="fit-watch">Routing tradeoffs</strong><ul>${c.watchouts.map(s=>`<li>${esc(s)}</li>`).join("")}</ul></div><div class="insight-box"><strong>Patient context</strong><p class="subtle">The selected demo patient has ${fmtPct(p.predicted_risk)} predicted risk with ${p.hypertension?"hypertension":"no hypertension"}, ${p.heart_disease?"heart disease":"no heart disease"}, ${esc(p.glucose_band).toLowerCase()} glucose, and ${esc(p.bmi_band).toLowerCase()} BMI. The routing recommendation is demo logic for interview storytelling.</p></div>`}
function programOutcomeMetrics(records){const m=capacityMetrics(records);const contacted=m.k;const reached=Math.round(contacted*.74);const enrolled=Math.round(reached*.68);const active90=Math.round(enrolled*.72);const carePlan=Math.round(enrolled*.61);const riskReview=Math.round(enrolled*.58);const caregiverAssigned=Math.round(contacted*.92);const escalations=Math.round(enrolled*.18);const closedEscalations=Math.round(escalations*.83);const allK=Math.max(1,Math.ceil(DATA.summary.model_rows*m.percent/100));const util=state.manualCapacity?allK/state.manualCapacity:1;const firstContactDays=util>1?5.1:util<.7?2.4:3.2;return{contacted,reached,enrolled,active90,carePlan,riskReview,caregiverAssigned,escalations,closedEscalations,firstContactDays,retention:enrolled?active90/enrolled:NaN,contactRate:contacted?reached/contacted:NaN,enrollmentRate:reached?enrolled/reached:NaN,carePlanRate:enrolled?carePlan/enrolled:NaN,riskReviewRate:enrolled?riskReview/enrolled:NaN,assignmentRate:contacted?caregiverAssigned/contacted:NaN,escalationClosure:escalations?closedEscalations/escalations:NaN}}
function outcomeBar(label,value,target){return`<div class="outcome-bar-row"><div>${label}</div><div class="outcome-track"><div class="outcome-fill" style="width:${Math.min(100,Math.max(0,value*100))}%"><div class="outcome-target" style="margin-left:${Math.min(100,Math.max(0,target*100))}%"></div></div></div><div><strong>${fmtPct(value)}</strong></div></div>`}
function trendValue(value,dir="up"){return`<span class="trend"><span class="tri ${dir}"></span><span>${value}</span></span>`}
function feedbackBar(label,value,y,color){const width=260*value;return`<text x="12" y="${y+14}" class="feedback-label">${label}</text><rect x="132" y="${y}" width="260" height="16" rx="8" fill="#e3eeec"/><rect x="132" y="${y}" width="${width.toFixed(1)}" height="16" rx="8" fill="${color}"/><text x="404" y="${y+13}" class="feedback-label">${Math.round(value*100)}%</text>`}
function caregiverFeedbackSvg(o){const confidence=Math.min(.96,.74+o.assignmentRate*.18),workload=Math.min(.95,.52+o.contacted/DATA.summary.model_rows*2.4),handoff=Math.min(.96,o.assignmentRate);return`<svg class="feedback-svg" viewBox="0 0 450 150" role="img" aria-label="Caregiver feedback summary"><text x="12" y="18" class="feedback-title">Care-team feedback pulse</text>${feedbackBar("Workflow clarity",confidence,42,"#0f6b72")}${feedbackBar("Handoff confidence",handoff,76,"#4eb5a8")}${feedbackBar("Workload pressure",workload,110,workload>.82?"#c84436":"#d6a63d")}</svg>`}
function renderProgramOutcomes(records){const o=programOutcomeMetrics(records);document.getElementById("outcomeKpis").innerHTML=[["90-day active patients",trendValue(fmtNum(o.active90),"up"),`${fmtPct(o.retention)} of enrolled patients stayed engaged`],["Patients enrolled",trendValue(fmtNum(o.enrolled),"up"),`${fmtPct(o.enrollmentRate)} of reached patients accepted intervention`],["Care plans completed",trendValue(fmtNum(o.carePlan),"up"),`${fmtPct(o.carePlanRate)} of enrolled patients`],["Avg first contact",trendValue(`${fmtFloat(o.firstContactDays,1)} days`,"down"),"From prioritized list to first successful outreach"]].map(([l,v,n])=>`<div class="metric"><div class="metric-label">${l}</div><div class="metric-value">${v}</div><div class="metric-note">${n}</div></div>`).join("");document.getElementById("outcomeFunnel").innerHTML=`<h3>Intervention funnel</h3><div class="outcome-bars">${outcomeBar("First contact success",o.contactRate,.75)}${outcomeBar("Intervention enrollment",o.enrollmentRate,.65)}${outcomeBar("90-day retention",o.retention,.70)}${outcomeBar("Care plan completion",o.carePlanRate,.60)}</div>`;document.getElementById("outcomeTable").innerHTML=table(["Area","Metric","Current","Target","Why it matters"],[["Patient","90-day active participants",`${fmtNum(o.active90)} patients`,">=70% retained","Shows whether patients stay engaged after intervention."],["Patient","Risk-factor review complete",fmtPct(o.riskReviewRate),">=55%","Tracks clinical follow-up on glucose, BMI, hypertension, and heart disease."],["Patient","Care plan completed",fmtPct(o.carePlanRate),">=60%","Measures whether outreach turns into documented next steps."],["Operations","Successful first contact",fmtPct(o.contactRate),">=75%","Tests whether the team can actually reach prioritized patients."],["Operations","Caregiver assigned",fmtPct(o.assignmentRate),">=90%","Shows handoff coverage from risk list to care team."],["Operations","Escalations closed",fmtPct(o.escalationClosure),">=80%","Tracks resolution of cases needing clinical or access follow-up."],["Operations","Average first-contact time",`${fmtFloat(o.firstContactDays,1)} days`,"<=3.5 days","Measures outreach speed after the model generates the queue."]]);document.getElementById("outcomeInsight").innerHTML=`${caregiverFeedbackSvg(o)}<div class="insight-box"><strong>Caregiver feedback summary</strong><p class="subtle">Caregivers report strong handoff confidence when assignments stay above 90% and the prioritized list is clearly staged by capacity.</p></div><div class="insight-box"><strong class="fit-good">Positive signal</strong><ul><li>${fmtPct(o.assignmentRate)} of contacted patients are projected to have a caregiver assigned.</li><li>90-day retention of ${fmtPct(o.retention)} suggests outreach is converting into continued engagement.</li></ul></div><div class="insight-box"><strong class="fit-watch">Watch-outs</strong><ul><li>Workload pressure rises as the selected outreach list approaches or exceeds team capacity.</li><li>Caregiver comments should be segmented by role before using feedback as a staffing decision.</li></ul></div>`}
function rateForGroup(records,label,mask){const sub=records.filter(mask);let denom,hit,metric;if(state.errorGroup==="False negative"){const cases=sub.filter(r=>r.event===1);denom=cases.length;hit=cases.filter(r=>r.error_group==="False negative").length;metric="FN rate"}else if(state.errorGroup==="False positive"){const non=sub.filter(r=>r.event===0);denom=non.length;hit=non.filter(r=>r.error_group==="False positive").length;metric="FP rate"}else if(state.errorGroup==="True positive"){const cases=sub.filter(r=>r.event===1);denom=cases.length;hit=cases.filter(r=>r.error_group==="True positive").length;metric="TP rate"}else{const non=sub.filter(r=>r.event===0);denom=non.length;hit=non.filter(r=>r.error_group==="True negative").length;metric="TN rate"}const rate=denom?hit/denom:NaN;return[label,fmtPct(rate),`${fmtNum(hit)} of ${fmtNum(denom)} relevant patients`]}
function renderPatterns(records){const rows=[rateForGroup(records,"Age <65",r=>r.age<65),rateForGroup(records,"Age >=65",r=>r.age>=65),rateForGroup(records,"No hypertension",r=>r.hypertension===0),rateForGroup(records,"Hypertension",r=>r.hypertension===1),rateForGroup(records,"No heart disease",r=>r.heart_disease===0),rateForGroup(records,"Heart disease",r=>r.heart_disease===1),rateForGroup(records,"Glucose <126",r=>r.avg_glucose_level<126),rateForGroup(records,"Glucose >=126",r=>r.avg_glucose_level>=126)];document.getElementById("patternTitle").textContent=`Patterns for ${state.errorGroup.toLowerCase()}s`;document.getElementById("patternTable").innerHTML=table(["Group","Rate","Observation"],rows)}
function renderErrorProfile(records){document.querySelectorAll("[data-error]").forEach(b=>b.classList.toggle("active",b.dataset.error===state.errorGroup));const group=records.filter(r=>r.error_group===state.errorGroup);const avg=(arr,f)=>arr.length?arr.reduce((s,r)=>s+(Number(r[f])||0),0)/arr.length:NaN;const rate=(arr,f)=>arr.length?arr.reduce((s,r)=>s+Number(r[f]||0),0)/arr.length:NaN;const bmi=group.filter(r=>Number.isFinite(Number(r.bmi)));const metrics=[["Patients",fmtNum(group.length)],["Mean age",fmtFloat(avg(group,"age"),1)],["Hypertension",fmtPct(rate(group,"hypertension"))],["Heart disease",fmtPct(rate(group,"heart_disease"))],["Mean glucose",fmtFloat(avg(group,"avg_glucose_level"),1)],["Mean BMI",fmtFloat(avg(bmi,"bmi"),1)],["Avg predicted risk",fmtPct(avg(group,"predicted_risk"))]];document.getElementById("errorProfile").innerHTML=metrics.map(([l,v])=>`<div class="mini"><strong>${v}</strong><span>${l}</span></div>`).join("");renderPatterns(records)}
function renderDrivers(){document.querySelectorAll("[data-driver]").forEach(b=>b.classList.toggle("active",b.dataset.driver===state.driver));const rows=DATA.coefficients.slice().sort((a,b)=>state.driver==="positive"?b.coefficient-a.coefficient:a.coefficient-b.coefficient).slice(0,8);const maxAbs=Math.max(...rows.map(r=>Math.abs(r.coefficient)))||1;const html=rows.map(r=>`<div class="bar-row"><div>${friendly(r.feature)}</div><div class="bar-track"><div class="bar-fill ${r.coefficient<0?"negative":""}" style="width:${Math.max(3,Math.abs(r.coefficient)/maxAbs*100)}%"></div></div><div>${r.coefficient.toFixed(3)}</div></div>`).join("");document.getElementById("driverChart").innerHTML=html}
function renderTabs(){document.querySelectorAll("[data-tab]").forEach(b=>b.classList.toggle("active",b.dataset.tab===state.dashboard));document.querySelectorAll(".tab-panel").forEach(p=>p.classList.toggle("active",p.id===`tab-${state.dashboard}`))}
function renderAll(){renderFilters();const records=selectedRecords();const m=capacityMetrics(records);renderSummaryCards(m);renderCapacity(records,m);renderGains(records,m);renderCurves();renderTables();renderOptimizationStudio(records);renderPatientDetails(records);renderCaregiverMatch();mountAiPanels(records);renderAiWorkspace(records);renderProgramOutcomes(records);renderErrorProfile(records);renderDrivers();renderTabs()}
document.getElementById("capacitySlider").addEventListener("input",e=>{state.capacity=Number(e.target.value);state.patientIndex=0;state.patientPage=0;renderAll()});
document.getElementById("manualCapacityInput").addEventListener("input",e=>{const previous=state.manualCapacity,v=Number(e.target.value);state.manualCapacity=Number.isFinite(v)?Math.max(1,Math.round(v)):1;if(state.optimizer.maxContacts===previous)state.optimizer.maxContacts=state.manualCapacity;renderAll()});
document.querySelectorAll("[data-tab]").forEach(b=>b.onclick=()=>{state.dashboard=b.dataset.tab;renderTabs()});
document.getElementById("openAiWorkspace").onclick=()=>{state.dashboard="ai";renderTabs();document.getElementById("tab-ai").scrollIntoView({behavior:"smooth",block:"start"})};
document.querySelectorAll("[data-error]").forEach(b=>b.onclick=()=>{state.errorGroup=b.dataset.error;renderErrorProfile(selectedRecords())});
document.querySelectorAll("[data-driver]").forEach(b=>b.onclick=()=>{state.driver=b.dataset.driver;renderDrivers()});
document.addEventListener("click",event=>{if(event.target.closest("[data-patient-row],[data-patient-page],[data-sort],[data-priority],[data-caregiver]")){setTimeout(()=>{const records=selectedRecords();mountAiPanels(records);renderAiWorkspace(records)},0)}});
document.getElementById("manualCapacityInput").value=String(state.manualCapacity);
renderAll();
checkAiHealth();
</script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the care risk outreach dashboard preview.")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Also copy the generated dashboard to index.html and docs/index.html.",
    )
    parser.add_argument(
        "--ai-api-base",
        default="",
        help="Production Worker URL to embed. Local previews automatically use port 8787.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()
    JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    D1_SEED_PATH.write_text(build_d1_seed(data), encoding="utf-8")
    html = (
        HTML_TEMPLATE
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__DEFAULT_CAPACITY__", str(data["summary"]["default_capacity"]))
        .replace("__AI_API_BASE__", args.ai_api_base.rstrip("/"))
    )
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {D1_SEED_PATH}")
    print(f"Wrote {HTML_PATH}")
    if args.publish:
        PAGES_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(html, encoding="utf-8")
        PAGES_PATH.write_text(html, encoding="utf-8")
        print(f"Published {INDEX_PATH}")
        print(f"Published {PAGES_PATH}")
    else:
        print("Preview build only; release copies were not changed.")


if __name__ == "__main__":
    main()
