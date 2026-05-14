"""Metricas de desempeno y reportes de evaluacion con validacion cruzada temporal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


def evaluate_models_cv(
    models: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv: Any,
    df_meta: pd.DataFrame | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evalua modelos con CV estratificada por grupos; devuelve metricas, resumen, OOF y reporte."""
    _ = random_state

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    class_names = list(le.classes_)

    metrics_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []

    for model_name, model_tpl in models.items():
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            m = clone(model_tpl)
            X_tr = X.iloc[train_idx]
            X_te = X.iloc[test_idx]
            X_tr_np = X_tr.to_numpy(dtype=np.float64)
            X_te_np = X_te.to_numpy(dtype=np.float64)
            y_tr_enc = y_enc[train_idx]
            y_te_enc = y_enc[test_idx]
            y_te_lab = y.iloc[test_idx].astype(str).to_numpy()
            sw_tr = compute_sample_weight("balanced", y_tr_enc)
            fit_warning = ""

            try:
                if model_name == "xgboost":
                    m.fit(X_tr_np, y_tr_enc, sample_weight=sw_tr)
                elif model_name == "soft_voting_ensemble":
                    try:
                        m.fit(X_tr_np, y_tr_enc, sample_weight=sw_tr)
                    except TypeError:
                        m.fit(X_tr_np, y_tr_enc)
                        fit_warning = "sample_weight_no_soportado_voting"
                else:
                    m.fit(X_tr_np, y_tr_enc)
            except Exception as exc:  # noqa: BLE001
                fit_warning = f"fit_error:{exc}"
                raise

            y_pred_enc = m.predict(X_te_np)
            y_pred_lab = le.inverse_transform(y_pred_enc.astype(int))

            acc = float(accuracy_score(y_te_enc, y_pred_enc))
            bal = float(balanced_accuracy_score(y_te_enc, y_pred_enc))
            f1m = float(f1_score(y_te_enc, y_pred_enc, average="macro", zero_division=0))
            precm = float(
                precision_score(y_te_enc, y_pred_enc, average="macro", zero_division=0)
            )
            recm = float(
                recall_score(y_te_enc, y_pred_enc, average="macro", zero_division=0)
            )
            mcc = float(matthews_corrcoef(y_te_enc, y_pred_enc))
            kappa = float(cohen_kappa_score(y_te_enc, y_pred_enc))

            metrics_rows.append(
                {
                    "model_name": model_name,
                    "fold": int(fold),
                    "accuracy": acc,
                    "balanced_accuracy": bal,
                    "f1_macro": f1m,
                    "precision_macro": precm,
                    "recall_macro": recm,
                    "mcc": mcc,
                    "cohen_kappa": kappa,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "fit_warning": fit_warning,
                }
            )

            P = m.predict_proba(X_te_np) if hasattr(m, "predict_proba") else None
            cls_arr = np.asarray(m.classes_) if P is not None else None

            for pos in range(len(test_idx)):
                row_pos = int(test_idx[pos])
                row: dict[str, Any] = {
                    "model_name": model_name,
                    "fold": int(fold),
                    "row_index": row_pos,
                    "window_id": np.nan,
                    "y_true": int(y_te_enc[pos]),
                    "y_pred": int(y_pred_enc[pos]),
                    "y_true_label": str(y_te_lab[pos]),
                    "y_pred_label": str(y_pred_lab[pos]),
                }
                if df_meta is not None and "window_id" in df_meta.columns:
                    row["window_id"] = float(df_meta.iloc[row_pos]["window_id"])
                if P is not None and cls_arr is not None:
                    for cname in class_names:
                        enc = int(le.transform([cname])[0])
                        col_j = int(np.where(cls_arr == enc)[0][0])
                        row[f"prob_{cname}"] = float(P[pos, col_j])
                else:
                    for cname in class_names:
                        row[f"prob_{cname}"] = np.nan
                oof_rows.append(row)

    metrics_by_fold_df = pd.DataFrame(metrics_rows)
    oof_predictions_df = pd.DataFrame(oof_rows)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "mcc",
        "cohen_kappa",
    ]
    melted = metrics_by_fold_df.melt(
        id_vars=["model_name", "fold"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    metrics_summary_df = (
        melted.groupby(["model_name", "metric"], as_index=False)["value"]
        .agg(mean="mean", std="std", min="min", max="max")
    )

    report_rows: list[dict[str, Any]] = []
    for model_name in models:
        sdf = oof_predictions_df[oof_predictions_df["model_name"] == model_name]
        if sdf.empty:
            continue
        rep = classification_report(
            sdf["y_true_label"],
            sdf["y_pred_label"],
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
        for cls in class_names:
            if cls not in rep:
                continue
            report_rows.append(
                {
                    "model_name": model_name,
                    "class_label": cls,
                    "precision": float(rep[cls]["precision"]),
                    "recall": float(rep[cls]["recall"]),
                    "f1_score": float(rep[cls]["f1-score"]),
                    "support": float(rep[cls]["support"]),
                }
            )
    classification_report_df = pd.DataFrame(report_rows)

    return (
        metrics_by_fold_df,
        metrics_summary_df,
        oof_predictions_df,
        classification_report_df,
    )


def build_confusion_matrices(
    oof_predictions_df: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    """Matriz de confusion en formato largo con porcentaje por fila (verdadero)."""
    _ = class_names
    g = (
        oof_predictions_df.groupby(
            ["model_name", "y_true_label", "y_pred_label"], as_index=False
        )
        .size()
        .rename(columns={"size": "count"})
    )
    totals = (
        oof_predictions_df.groupby(["model_name", "y_true_label"], as_index=False)
        .size()
        .rename(columns={"size": "total_true"})
    )
    merged = g.merge(totals, on=["model_name", "y_true_label"])
    merged["percent_by_true"] = 100.0 * merged["count"].astype(float) / merged[
        "total_true"
    ].astype(float)
    out = merged.rename(
        columns={"y_true_label": "true_label", "y_pred_label": "pred_label"}
    )
    return out[
        ["model_name", "true_label", "pred_label", "count", "percent_by_true"]
    ]


def select_best_model_from_summary(
    metrics_summary_df: pd.DataFrame,
    primary_metric: str = "f1_macro",
    secondary_metric: str = "balanced_accuracy",
) -> str:
    """Selecciona el modelo con mejor media de f1_macro y desempate por balanced_accuracy."""
    pvt = metrics_summary_df.pivot_table(
        index="model_name", columns="metric", values="mean", aggfunc="first"
    )
    if primary_metric not in pvt.columns or secondary_metric not in pvt.columns:
        raise ValueError("Faltan metricas requeridas en el resumen.")
    pvt = pvt.sort_values(
        [primary_metric, secondary_metric], ascending=[False, False]
    )
    return str(pvt.index[0])


def plot_model_metric_comparison(
    metrics_summary_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> None:
    """Barras de media +/- std por modelo para una metrica."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = metrics_summary_df[metrics_summary_df["metric"] == metric].copy()
    if sub.empty:
        return
    sub = sub.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    ax.bar(x, sub["mean"], yerr=sub["std"], capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model_name"], rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix_for_model(
    confusion_long_df: pd.DataFrame,
    model_name: str,
    output_path: str | Path,
) -> None:
    """Heatmap de confusion normalizada por fila (percent_by_true)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = confusion_long_df[confusion_long_df["model_name"] == model_name].copy()
    labels = sorted(sub["true_label"].unique().tolist())
    mat = np.zeros((len(labels), len(labels)))
    for _, r in sub.iterrows():
        i = labels.index(str(r["true_label"]))
        j = labels.index(str(r["pred_label"]))
        mat[i, j] = float(r["percent_by_true"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("pred_label")
    ax.set_ylabel("true_label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
