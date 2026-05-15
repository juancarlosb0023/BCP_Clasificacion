"""Generacion de archivos y tablas listos para consumo externo, en particular para el dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DATA_DASHBOARD_DIR,
    DATA_PROCESSED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TABLES_DIR,
)


def _to_jsonable(obj: Any) -> Any:
    """Convierte tipos de numpy/pandas a tipos nativos serializables en JSON."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def export_table(df: pd.DataFrame, path: str | Path) -> None:
    """Guarda un DataFrame en CSV UTF-8 sin indice, creando carpetas si faltan."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")


def export_json(data: Any, path: str | Path) -> None:
    """Guarda datos en JSON indentado, normalizando tipos numpy/pandas."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(data)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Serializa un objeto con joblib, creando la carpeta destino si no existe."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)


def save_json(data: Any, path: str | Path) -> None:
    """Alias de export_json para API explicita de guardado JSON."""
    export_json(data, path)


def read_required_csv(path: str | Path, *, required: bool = True) -> pd.DataFrame:
    """Lee CSV UTF-8; si required y no existe, lanza FileNotFoundError claro."""
    p = Path(path)
    if not p.is_file():
        if required:
            raise FileNotFoundError(
                f"No se encontro archivo requerido para dashboard: {p.as_posix()}"
            )
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8")


def export_dashboard_table(df: pd.DataFrame, filename: str) -> None:
    """Guarda tabla en data/dashboard/<filename>, UTF-8, sin indice."""
    DATA_DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DASHBOARD_DIR / filename
    df.to_csv(out, index=False, encoding="utf-8")


def _kv(df: pd.DataFrame, key: str) -> Any:
    if df.empty or "metric" not in df.columns:
        return None
    m = df.loc[df["metric"].astype(str) == str(key), "value"]
    if len(m) == 0:
        return None
    return m.iloc[0]


def _scalar_row(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or len(df.columns) == 0:
        return {}
    return df.iloc[0].to_dict()


def build_dashboard_kpis(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    KPIs consolidados para dashboard: columnas metric, value, category, description.
    Solo lectura de tablas ya exportadas por etapas anteriores.
    """
    dq = sources.get("data_quality_summary", pd.DataFrame())
    miss = sources.get("missing_values", pd.DataFrame())
    gaps = sources.get("timestamp_gaps", pd.DataFrame())
    trans = sources.get("batch_transitions", pd.DataFrame())
    win = sources.get("windowed_dataset_summary", pd.DataFrame())
    sval = sources.get("statistical_validation_summary", pd.DataFrame())
    perm = sources.get("permanova_results", pd.DataFrame())
    disp = sources.get("permdisp_results", pd.DataFrame())
    best = sources.get("best_model_selection", pd.DataFrame())
    cms = sources.get("classifier_metrics_summary", pd.DataFrame())
    ftrain = sources.get("final_model_training_summary", pd.DataFrame())
    intr = sources.get("interpretability_summary", pd.DataFrame())
    asum = sources.get("assessment_summary", pd.DataFrame())
    csum = sources.get("condition_index_summary", pd.DataFrame())
    ath = sources.get("assessment_thresholds_summary", pd.DataFrame())
    cs_cur = sources.get("condition_current_state", pd.DataFrame())
    cs_alt = sources.get("condition_alerts_active", pd.DataFrame())
    cs_tr = sources.get("condition_trend_summary", pd.DataFrame())

    def add(rows: list[dict[str, Any]], metric: str, value: Any, category: str, desc: str) -> None:
        rows.append(
            {
                "metric": metric,
                "value": value,
                "category": category,
                "description": desc,
            }
        )

    rows: list[dict[str, Any]] = []
    n_miss = int(miss["missing_count"].sum()) if not miss.empty and "missing_count" in miss.columns else 0
    n_gaps = int(len(gaps)) if not gaps.empty else 0
    n_trans = int(len(trans)) if not trans.empty else 0

    def _sval_col(col: str) -> Any:
        if sval.empty or col not in sval.columns:
            return None
        return sval.iloc[0][col]

    add(rows, "n_rows_raw", _kv(dq, "n_rows"), "calidad", "Filas en serie cruda")
    add(rows, "n_features_raw", _kv(dq, "n_features"), "calidad", "Columnas de sensores numericos en crudo")
    add(rows, "n_missing_values_total", n_miss, "calidad", "Suma de missing por columna (quality)")
    add(rows, "n_timestamp_gaps", n_gaps, "calidad", "Filas en timestamp_gaps")
    add(rows, "n_batch_transitions", n_trans, "calidad", "Transiciones de Batch detectadas")

    w0 = _scalar_row(win)
    add(
        rows,
        "n_windows_total",
        w0.get("n_windows_total_after_drop")
        if w0.get("n_windows_total_after_drop") is not None
        else _sval_col("n_windows_original"),
        "ventaneo",
        "Ventanas despues de filtros de ventaneo",
    )
    add(
        rows,
        "n_windows_modeling",
        _sval_col("n_windows_used"),
        "ventaneo",
        "Ventanas usadas en modelado (sin transicion)",
    )
    add(
        rows,
        "n_transition_windows_excluded",
        _sval_col("n_transition_windows_excluded"),
        "ventaneo",
        "Ventanas excluidas por proximidad a transicion",
    )
    n_mf = w0.get("n_model_features")
    if (n_mf is None or (isinstance(n_mf, float) and pd.isna(n_mf))) and _sval_col("n_model_features") is not None:
        n_mf = _sval_col("n_model_features")
    add(
        rows,
        "n_model_features",
        n_mf,
        "ventaneo",
        "Features numericas por ventana para ML",
    )

    if not perm.empty:
        add(
            rows,
            "permanova_r2",
            float(perm.iloc[0]["r2"]),
            "estadistica",
            "PERMANOVA R\u00b2 sobre ventanas",
        )
        add(
            rows,
            "permanova_p_value",
            float(perm.iloc[0]["p_value"]),
            "estadistica",
            "PERMANOVA p-value",
        )
    else:
        add(rows, "permanova_r2", None, "estadistica", "PERMANOVA R\u00b2")
        add(rows, "permanova_p_value", None, "estadistica", "PERMANOVA p-value")

    if not disp.empty:
        add(rows, "permdisp_F", float(disp.iloc[0]["F"]), "estadistica", "PERMDISP F")
        add(rows, "permdisp_p_value", float(disp.iloc[0]["p_value"]), "estadistica", "PERMDISP p-value")
    else:
        add(rows, "permdisp_F", None, "estadistica", "PERMDISP F")
        add(rows, "permdisp_p_value", None, "estadistica", "PERMDISP p-value")

    if not sval.empty:
        add(
            rows,
            "pca_explained_variance_PC1",
            float(sval.iloc[0]["pca_explained_variance_PC1"]),
            "estadistica",
            "Varianza explicada PC1",
        )
        add(
            rows,
            "pca_explained_variance_PC2",
            float(sval.iloc[0]["pca_explained_variance_PC2"]),
            "estadistica",
            "Varianza explicada PC2",
        )
    else:
        add(rows, "pca_explained_variance_PC1", None, "estadistica", "Varianza explicada PC1")
        add(rows, "pca_explained_variance_PC2", None, "estadistica", "Varianza explicada PC2")

    b0 = _scalar_row(best)
    best_name = b0.get("best_model_name")
    add(rows, "best_model_name", best_name, "modelo", "Modelo seleccionado por metrica primaria")
    add(
        rows,
        "best_model_f1_macro_cv",
        b0.get("f1_macro_mean"),
        "modelo",
        "F1 macro promedio en CV (comparativa)",
    )
    add(
        rows,
        "best_model_balanced_accuracy_cv",
        b0.get("balanced_accuracy_mean"),
        "modelo",
        "Exactitud balanceada promedio en CV",
    )

    if best_name and not cms.empty:
        sub = cms[(cms["model_name"] == best_name) & (cms["metric"] == "accuracy")]
        add(
            rows,
            "best_model_accuracy_cv",
            float(sub.iloc[0]["mean"]) if len(sub) else None,
            "modelo",
            "Accuracy CV mejor modelo",
        )
        sub_m = cms[(cms["model_name"] == best_name) & (cms["metric"] == "mcc")]
        add(
            rows,
            "best_model_mcc_cv",
            float(sub_m.iloc[0]["mean"]) if len(sub_m) else None,
            "modelo",
            "MCC CV mejor modelo",
        )
        sub_k = cms[(cms["model_name"] == best_name) & (cms["metric"] == "cohen_kappa")]
        add(
            rows,
            "best_model_kappa_cv",
            float(sub_k.iloc[0]["mean"]) if len(sub_k) else None,
            "modelo",
            "Kappa CV mejor modelo",
        )
    else:
        add(rows, "best_model_accuracy_cv", None, "modelo", "Accuracy CV mejor modelo")
        add(rows, "best_model_mcc_cv", None, "modelo", "MCC CV mejor modelo")
        add(rows, "best_model_kappa_cv", None, "modelo", "Kappa CV mejor modelo")

    ft = _scalar_row(ftrain)
    add(rows, "training_f1_macro", ft.get("training_f1_macro"), "modelo_final", "F1 macro en entrenamiento final")
    add(
        rows,
        "training_balanced_accuracy",
        ft.get("training_balanced_accuracy"),
        "modelo_final",
        "Exactitud balanceada en entrenamiento final",
    )
    add(rows, "mean_confidence", ft.get("mean_confidence"), "modelo_final", "Confianza media prediccion final")
    add(rows, "median_confidence", ft.get("median_confidence"), "modelo_final", "Confianza mediana prediccion final")

    i0 = _scalar_row(intr)
    add(rows, "best_feature_consolidated", i0.get("best_feature_consolidated"), "interpretabilidad", "Mejor feature consolidada")
    add(rows, "best_feature_raw_variable", i0.get("best_feature_raw_variable"), "interpretabilidad", "Variable raw de mejor feature")
    add(rows, "top_raw_variable", i0.get("top_raw_variable"), "interpretabilidad", "Top variable raw agregada")
    add(rows, "top_component", i0.get("top_component"), "interpretabilidad", "Top componente (interpretabilidad)")
    add(rows, "top_family", i0.get("top_family"), "interpretabilidad", "Top familia (interpretabilidad)")
    add(rows, "top_position", i0.get("top_position"), "interpretabilidad", "Top posicion")
    add(rows, "top_statistic", i0.get("top_statistic"), "interpretabilidad", "Top estadistico")
    add(rows, "shap_status", i0.get("shap_status"), "interpretabilidad", "Estado SHAP")

    a0 = _scalar_row(asum)
    add(rows, "n_weights_mapped", a0.get("n_weights_mapped"), "assessment", "Pesos mapeados a columnas")
    add(rows, "n_weights_unmatched", a0.get("n_weights_unmatched"), "assessment", "Pesos sin match")
    add(rows, "median_available_weight", a0.get("median_available_weight"), "assessment", "Peso disponible mediano en indice")

    if not csum.empty:
        def cv(metric_name: str) -> Any:
            m = csum.loc[csum["metric"].astype(str) == metric_name, "value"]
            return float(m.iloc[0]) if len(m) else None

        add(rows, "condition_index_mean", cv("condition_index_mean"), "assessment", "Indice de condicion medio")
        add(rows, "condition_index_median", cv("condition_index_median"), "assessment", "Indice de condicion mediano")
        add(rows, "condition_index_p95", cv("condition_index_p95"), "assessment", "Indice de condicion p95")
    else:
        add(rows, "condition_index_mean", None, "assessment", "Indice de condicion medio")
        add(rows, "condition_index_median", None, "assessment", "Indice de condicion mediano")
        add(rows, "condition_index_p95", None, "assessment", "Indice de condicion p95")

    add(rows, "assessment_method", a0.get("assessment_method"), "assessment", "Metodo dominante en assessment")

    if not ath.empty and "metric" in ath.columns and "value" in ath.columns:
        thr_desc: dict[str, str] = {
            "n_variables_thresholded": "Variables con umbrales V0/H/HH estimados",
            "n_batches_thresholded": "Batches con umbrales por batch",
            "n_thresholds_global_rows": "Filas en umbrales globales (24 variables)",
            "n_thresholds_by_batch_rows": "Filas en umbrales por batch (24x3)",
            "n_threshold_warnings_global": "Advertencias de ajuste epsilon (global)",
            "n_threshold_warnings_by_batch": "Advertencias de ajuste epsilon (por batch)",
            "threshold_method_global": "Metodo estimacion umbrales globales (percentiles)",
            "threshold_method_by_batch": "Metodo estimacion umbrales por batch (percentiles)",
            "condition_index_mean_thresholded_global": "Media indice con umbrales globales (exploratorio)",
            "condition_index_median_thresholded_global": "Mediana indice umbrales globales",
            "condition_index_p95_thresholded_global": "P95 indice umbrales globales",
            "condition_index_min_thresholded_global": "Minimo indice umbrales globales",
            "condition_index_max_thresholded_global": "Maximo indice umbrales globales",
            "condition_index_mean_thresholded_by_batch": "Media indice con baseline por batch",
            "condition_index_median_thresholded_by_batch": "Mediana indice baseline por batch",
            "condition_index_p95_thresholded_by_batch": "P95 indice baseline por batch",
            "condition_index_min_thresholded_by_batch": "Minimo indice baseline por batch",
            "condition_index_max_thresholded_by_batch": "Maximo indice baseline por batch",
        }
        for _, r in ath.iterrows():
            m = str(r["metric"])
            if m == "note":
                add(
                    rows,
                    "threshold_note",
                    r["value"],
                    "assessment_thresholds",
                    "Nota metodologica: umbrales exploratorios, no normativos",
                )
            else:
                add(
                    rows,
                    m,
                    r["value"],
                    "assessment_thresholds",
                    thr_desc.get(m, "KPI assessment_thresholds"),
                )

    if not cs_cur.empty:
        r = cs_cur.iloc[0]
        n_att = (
            int((cs_alt["alert_level"].astype(str) == "attention").sum())
            if not cs_alt.empty and "alert_level" in cs_alt.columns
            else 0
        )
        n_hi = (
            int((cs_alt["alert_level"].astype(str) == "high").sum())
            if not cs_alt.empty and "alert_level" in cs_alt.columns
            else 0
        )
        tr_dir = str(cs_tr.iloc[0]["trend_direction"]) if not cs_tr.empty and "trend_direction" in cs_tr.columns else ""
        add(
            rows,
            "current_window_id",
            int(r["window_id"]),
            "condition_state",
            "Ultima ventana historica en condition_state (Etapa 8C)",
        )
        add(rows, "current_batch_real", r.get("Batch"), "condition_state", "Batch real de la ultima ventana")
        add(
            rows,
            "current_batch_predicted",
            r.get("y_pred_label"),
            "condition_state",
            "Clase predicha para la ultima ventana (modelo final)",
        )
        add(
            rows,
            "current_classification_confidence",
            r.get("confidence"),
            "condition_state",
            "Confianza del clasificador en la ultima ventana",
        )
        add(
            rows,
            "current_condition_index",
            float(r["condition_index"]) if pd.notna(r.get("condition_index")) else None,
            "condition_state",
            "Indice de condicion (baseline por batch, exploratorio)",
        )
        add(
            rows,
            "current_health_index",
            float(r["health_index"]) if pd.notna(r.get("health_index")) else None,
            "condition_state",
            "health_index = 100 - condition_index (exploratorio)",
        )
        add(
            rows,
            "current_condition_state",
            r.get("condition_state"),
            "condition_state",
            "Banda visual sobre condition_index: normal <20, attention 20-40, high >=40 (no normativa)",
        )
        add(
            rows,
            "current_trend_direction",
            tr_dir,
            "condition_state",
            "Tendencia reciente del indice (ultimas 60 ventanas, exploratorio)",
        )
        add(
            rows,
            "n_active_attention_alerts",
            n_att,
            "condition_state",
            "Alertas exploratorias nivel attention en ultima ventana exportada",
        )
        add(
            rows,
            "n_active_high_alerts",
            n_hi,
            "condition_state",
            "Alertas exploratorias nivel high en ultima ventana exportada",
        )
        add(
            rows,
            "top_condition_component",
            r.get("top_component_by_contribution"),
            "condition_state",
            "Componente destacado en el resumen de condicion (KPI agregado; puede diferir del driver principal)",
        )
        add(
            rows,
            "top_condition_family",
            r.get("top_family_by_contribution"),
            "condition_state",
            "Familia destacada en el resumen de condicion (KPI agregado; puede diferir del driver principal)",
        )
        add(
            rows,
            "top_condition_variable",
            r.get("top_variable_by_contribution"),
            "condition_state",
            "Variable principal de condicion (mayor weighted_score en la ventana)",
        )

    return pd.DataFrame(rows)


def build_dashboard_model_metrics(classifier_metrics_summary: pd.DataFrame) -> pd.DataFrame:
    """Una fila por model_name con medias y desviaciones de metricas CV."""
    if classifier_metrics_summary.empty:
        return pd.DataFrame()
    models = sorted(classifier_metrics_summary["model_name"].astype(str).unique())
    metric_keys = [
        ("accuracy", "accuracy_mean", "accuracy_std"),
        ("balanced_accuracy", "balanced_accuracy_mean", "balanced_accuracy_std"),
        ("f1_macro", "f1_macro_mean", "f1_macro_std"),
        ("precision_macro", "precision_macro_mean", None),
        ("recall_macro", "recall_macro_mean", None),
        ("mcc", "mcc_mean", None),
        ("cohen_kappa", "cohen_kappa_mean", None),
    ]
    out_rows: list[dict[str, Any]] = []
    for m in models:
        sub = classifier_metrics_summary[classifier_metrics_summary["model_name"] == m]
        row: dict[str, Any] = {"model_name": m}
        for metric, mean_col, std_col in metric_keys:
            sm = sub[sub["metric"] == metric]
            if len(sm):
                row[mean_col] = float(sm.iloc[0]["mean"])
                if std_col:
                    row[std_col] = float(sm.iloc[0]["std"])
            else:
                row[mean_col] = np.nan
                if std_col:
                    row[std_col] = np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def build_dashboard_confusion_best_model(
    confusion_matrix_by_model: pd.DataFrame,
    best_model_name: str,
) -> pd.DataFrame:
    """Matriz de confusion solo del mejor modelo."""
    if confusion_matrix_by_model.empty:
        return pd.DataFrame(
            columns=["true_label", "pred_label", "count", "percent_by_true", "model_name"]
        )
    df = confusion_matrix_by_model[
        confusion_matrix_by_model["model_name"].astype(str) == str(best_model_name)
    ].copy()
    return df[["true_label", "pred_label", "count", "percent_by_true"]].reset_index(drop=True)


def build_dashboard_predictions(
    final_model_predictions: pd.DataFrame,
    condition_index_by_window: pd.DataFrame | None = None,
    condition_index_thresholded_global: pd.DataFrame | None = None,
    condition_index_thresholded_by_batch: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Predicciones finales con indice de condicion original (fallback percentiles)
    y columnas thresholded (solo lectura de CSV previos, left join por window_id).
    """
    pred = final_model_predictions.copy()
    cols_out = [
        "window_id",
        "window_start",
        "window_end",
        "Batch",
        "y_true_label",
        "y_pred_label",
        "prob_CASTILLA",
        "prob_MEZCLA",
        "prob_RUBIALES",
        "confidence",
        "margin_top2",
        "is_correct",
        "condition_index",
        "assessment_method",
        "condition_index_thresholded_global",
        "assessment_method_thresholded_global",
        "baseline_batch_used",
        "condition_index_thresholded_by_batch",
        "assessment_method_thresholded_by_batch",
        "health_index_thresholded_by_batch",
        "has_near_transition",
    ]
    if condition_index_by_window is None or condition_index_by_window.empty:
        pred["condition_index"] = np.nan
        pred["assessment_method"] = np.nan
    else:
        cond = condition_index_by_window[
            ["window_id", "condition_index", "assessment_method"]
        ].drop_duplicates(subset=["window_id"])
        pred = pred.merge(cond, on="window_id", how="left")

    if condition_index_thresholded_global is not None and not condition_index_thresholded_global.empty:
        gcols = [c for c in ("window_id", "condition_index", "assessment_method") if c in condition_index_thresholded_global.columns]
        if len(gcols) == 3:
            g = condition_index_thresholded_global[gcols].drop_duplicates(subset=["window_id"]).copy()
            g = g.rename(
                columns={
                    "condition_index": "condition_index_thresholded_global",
                    "assessment_method": "assessment_method_thresholded_global",
                }
            )
            pred = pred.merge(g, on="window_id", how="left")
        else:
            pred["condition_index_thresholded_global"] = np.nan
            pred["assessment_method_thresholded_global"] = np.nan
    else:
        pred["condition_index_thresholded_global"] = np.nan
        pred["assessment_method_thresholded_global"] = np.nan

    if condition_index_thresholded_by_batch is not None and not condition_index_thresholded_by_batch.empty:
        need = ["window_id", "baseline_batch_used", "condition_index", "assessment_method"]
        if all(c in condition_index_thresholded_by_batch.columns for c in need):
            b = condition_index_thresholded_by_batch[need].drop_duplicates(subset=["window_id"]).copy()
            b = b.rename(
                columns={
                    "condition_index": "condition_index_thresholded_by_batch",
                    "assessment_method": "assessment_method_thresholded_by_batch",
                }
            )
            pred = pred.merge(b, on="window_id", how="left")
        else:
            pred["baseline_batch_used"] = pd.NA
            pred["condition_index_thresholded_by_batch"] = np.nan
            pred["assessment_method_thresholded_by_batch"] = np.nan
    else:
        pred["baseline_batch_used"] = pd.NA
        pred["condition_index_thresholded_by_batch"] = np.nan
        pred["assessment_method_thresholded_by_batch"] = np.nan

    ci_b = pd.to_numeric(pred["condition_index_thresholded_by_batch"], errors="coerce")
    pred["health_index_thresholded_by_batch"] = np.where(
        ci_b.notna(), 100.0 - ci_b, np.nan
    )

    return pred[cols_out].reset_index(drop=True)


def build_dashboard_feature_importance(ranking_features: pd.DataFrame) -> pd.DataFrame:
    """Columnas utiles del ranking de features para dashboard."""
    cols = [
        "consolidated_rank",
        "feature",
        "raw_variable",
        "statistic",
        "family",
        "component",
        "position",
        "unit",
        "perm_importance_mean",
        "shap_importance_mean_abs",
        "consolidated_score",
    ]
    present = [c for c in cols if c in ranking_features.columns]
    return ranking_features[present].copy().reset_index(drop=True)


def build_dashboard_sensor_rankings(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Tablas de ranking y top ponderadas sin recalcular."""
    keys = [
        "rank_raw_variable",
        "rank_component",
        "rank_family",
        "rank_position",
        "rank_statistic",
        "top_weighted_variables",
    ]
    out: dict[str, pd.DataFrame] = {}
    for k in keys:
        df = sources.get(k, pd.DataFrame())
        out[k] = df.copy() if not df.empty else pd.DataFrame()
    return out


DASHBOARD_PREREQUISITES: list[tuple[Path, str]] = [
    (DATA_PROCESSED_DIR / "bpc_windowed_features.csv", "features"),
    (TABLES_DIR / "classifier_metrics_summary.csv", "modeling"),
    (TABLES_DIR / "best_model_selection.csv", "modeling"),
    (TABLES_DIR / "classification_report_by_model.csv", "modeling"),
    (TABLES_DIR / "confusion_matrix_by_model.csv", "modeling"),
    (TABLES_DIR / "oof_predictions.csv", "modeling"),
    (TABLES_DIR / "final_model_predictions.csv", "final_model"),
    (TABLES_DIR / "final_model_training_summary.csv", "final_model"),
    (TABLES_DIR / "statistical_validation_summary.csv", "stats"),
    (TABLES_DIR / "permanova_results.csv", "stats"),
    (TABLES_DIR / "permdisp_results.csv", "stats"),
    (TABLES_DIR / "pairwise_permanova_fdr_results.csv", "stats"),
    (TABLES_DIR / "pairwise_permdisp_fdr_results.csv", "stats"),
    (TABLES_DIR / "pca_centroids.csv", "stats"),
    (TABLES_DIR / "pairwise_centroid_distances.csv", "stats"),
    (TABLES_DIR / "interpretability_summary.csv", "interpretability"),
    (TABLES_DIR / "ranking_features.csv", "interpretability"),
    (TABLES_DIR / "rank_raw_variable.csv", "interpretability"),
    (TABLES_DIR / "rank_component.csv", "interpretability"),
    (TABLES_DIR / "rank_family.csv", "interpretability"),
    (TABLES_DIR / "rank_position.csv", "interpretability"),
    (TABLES_DIR / "rank_statistic.csv", "interpretability"),
    (TABLES_DIR / "assessment_summary.csv", "assessment"),
    (TABLES_DIR / "sensor_weights_clean.csv", "assessment"),
    (TABLES_DIR / "sensor_weights_mapping.csv", "assessment"),
    (TABLES_DIR / "condition_index_by_window.csv", "assessment"),
    (TABLES_DIR / "condition_index_by_batch.csv", "assessment"),
    (TABLES_DIR / "condition_index_summary.csv", "assessment"),
    (TABLES_DIR / "top_weighted_variables.csv", "assessment"),
    (TABLES_DIR / "data_quality_summary.csv", "quality"),
    (TABLES_DIR / "class_distribution.csv", "quality"),
    (TABLES_DIR / "batch_transitions.csv", "quality"),
    (TABLES_DIR / "missing_values.csv", "quality"),
    (TABLES_DIR / "windowed_dataset_summary.csv", "features"),
    (TABLES_DIR / "assessment_thresholds_summary.csv", "assessment_thresholds"),
    (TABLES_DIR / "assessment_thresholds_global.csv", "assessment_thresholds"),
    (TABLES_DIR / "assessment_thresholds_by_batch.csv", "assessment_thresholds"),
    (TABLES_DIR / "sensor_weights_with_thresholds.csv", "assessment_thresholds"),
    (TABLES_DIR / "condition_index_by_window_thresholded_global.csv", "assessment_thresholds"),
    (TABLES_DIR / "condition_index_by_batch_thresholded_global.csv", "assessment_thresholds"),
    (TABLES_DIR / "condition_index_by_window_thresholded_by_batch.csv", "assessment_thresholds"),
    (TABLES_DIR / "condition_index_by_batch_thresholded_by_batch.csv", "assessment_thresholds"),
    (TABLES_DIR / "condition_contributions_long.csv", "condition_state"),
    (TABLES_DIR / "condition_contributions_top_by_window.csv", "condition_state"),
    (TABLES_DIR / "condition_current_state.csv", "condition_state"),
    (TABLES_DIR / "condition_alerts_active.csv", "condition_state"),
    (TABLES_DIR / "condition_trend_summary.csv", "condition_state"),
    (REPORTS_DIR / "current_asset_state.json", "condition_state"),
]


def verify_dashboard_prerequisites() -> None:
    """Lanza FileNotFoundError si falta algun archivo critico."""
    missing: list[str] = []
    for path, stage in DASHBOARD_PREREQUISITES:
        if not path.is_file():
            missing.append(f"  - {path.relative_to(PROJECT_ROOT)} (ejecute: python run_pipeline.py --stage {stage})")
    if missing:
        msg = "Faltan archivos requeridos para dashboard_exports:\n" + "\n".join(missing)
        if any(
            "assessment_thresholds" in line
            or "sensor_weights_with_thresholds" in line
            or "thresholded" in line
            for line in missing
        ):
            msg += (
                "\n\nEjecutar primero:\n"
                "python run_pipeline.py --stage assessment\n"
                "python run_pipeline.py --stage assessment_thresholds\n"
            )
        if any(
            "condition_contributions" in line
            or "condition_current_state.csv" in line
            or "condition_alerts_active.csv" in line
            or "condition_trend_summary.csv" in line
            or "current_asset_state.json" in line
            for line in missing
        ):
            msg += "\n\nEjecutar primero:\npython run_pipeline.py --stage condition_state\n"
        raise FileNotFoundError(msg)


def run_dashboard_exports() -> dict[str, Any]:
    """
    Lee outputs existentes, construye tablas para dashboard en data/dashboard/.
    No recalcula modelos, SHAP, estadistica, assessment, umbrales ni Etapa 8C (condition_state);
    solo copia/deriva lecturas desde outputs ya generados.
    """
    verify_dashboard_prerequisites()

    sources: dict[str, pd.DataFrame] = {
        "data_quality_summary": read_required_csv(TABLES_DIR / "data_quality_summary.csv"),
        "missing_values": read_required_csv(TABLES_DIR / "missing_values.csv", required=False),
        "timestamp_gaps": read_required_csv(TABLES_DIR / "timestamp_gaps.csv", required=False),
        "batch_transitions": read_required_csv(TABLES_DIR / "batch_transitions.csv"),
        "windowed_dataset_summary": read_required_csv(
            TABLES_DIR / "windowed_dataset_summary.csv", required=False
        ),
        "statistical_validation_summary": read_required_csv(TABLES_DIR / "statistical_validation_summary.csv"),
        "permanova_results": read_required_csv(TABLES_DIR / "permanova_results.csv"),
        "permdisp_results": read_required_csv(TABLES_DIR / "permdisp_results.csv"),
        "classifier_metrics_summary": read_required_csv(TABLES_DIR / "classifier_metrics_summary.csv"),
        "best_model_selection": read_required_csv(TABLES_DIR / "best_model_selection.csv"),
        "final_model_training_summary": read_required_csv(TABLES_DIR / "final_model_training_summary.csv"),
        "interpretability_summary": read_required_csv(TABLES_DIR / "interpretability_summary.csv"),
        "ranking_features": read_required_csv(TABLES_DIR / "ranking_features.csv"),
        "rank_raw_variable": read_required_csv(TABLES_DIR / "rank_raw_variable.csv"),
        "rank_component": read_required_csv(TABLES_DIR / "rank_component.csv"),
        "rank_family": read_required_csv(TABLES_DIR / "rank_family.csv"),
        "rank_position": read_required_csv(TABLES_DIR / "rank_position.csv"),
        "rank_statistic": read_required_csv(TABLES_DIR / "rank_statistic.csv"),
        "assessment_summary": read_required_csv(TABLES_DIR / "assessment_summary.csv"),
        "condition_index_summary": read_required_csv(TABLES_DIR / "condition_index_summary.csv"),
        "condition_index_by_window": read_required_csv(TABLES_DIR / "condition_index_by_window.csv"),
        "condition_index_by_batch": read_required_csv(TABLES_DIR / "condition_index_by_batch.csv"),
        "top_weighted_variables": read_required_csv(TABLES_DIR / "top_weighted_variables.csv"),
        "final_model_predictions": read_required_csv(TABLES_DIR / "final_model_predictions.csv"),
        "pairwise_permanova": read_required_csv(TABLES_DIR / "pairwise_permanova_fdr_results.csv"),
        "pairwise_permdisp": read_required_csv(TABLES_DIR / "pairwise_permdisp_fdr_results.csv"),
        "pca_centroids": read_required_csv(TABLES_DIR / "pca_centroids.csv"),
        "pca_projection": read_required_csv(TABLES_DIR / "pca_projection.csv"),
        "assessment_thresholds_summary": read_required_csv(
            TABLES_DIR / "assessment_thresholds_summary.csv"
        ),
        "assessment_thresholds_global": read_required_csv(TABLES_DIR / "assessment_thresholds_global.csv"),
        "assessment_thresholds_by_batch": read_required_csv(TABLES_DIR / "assessment_thresholds_by_batch.csv"),
        "sensor_weights_with_thresholds": read_required_csv(
            TABLES_DIR / "sensor_weights_with_thresholds.csv"
        ),
        "condition_index_by_window_thresholded_global": read_required_csv(
            TABLES_DIR / "condition_index_by_window_thresholded_global.csv"
        ),
        "condition_index_by_batch_thresholded_global": read_required_csv(
            TABLES_DIR / "condition_index_by_batch_thresholded_global.csv"
        ),
        "condition_index_by_window_thresholded_by_batch": read_required_csv(
            TABLES_DIR / "condition_index_by_window_thresholded_by_batch.csv"
        ),
        "condition_index_by_batch_thresholded_by_batch": read_required_csv(
            TABLES_DIR / "condition_index_by_batch_thresholded_by_batch.csv"
        ),
        "condition_contributions_long": read_required_csv(TABLES_DIR / "condition_contributions_long.csv"),
        "condition_contributions_top_by_window": read_required_csv(
            TABLES_DIR / "condition_contributions_top_by_window.csv"
        ),
        "condition_current_state": read_required_csv(TABLES_DIR / "condition_current_state.csv"),
        "condition_alerts_active": read_required_csv(TABLES_DIR / "condition_alerts_active.csv"),
        "condition_trend_summary": read_required_csv(TABLES_DIR / "condition_trend_summary.csv"),
    }

    kpis = build_dashboard_kpis(sources)
    export_dashboard_table(kpis, "dashboard_kpis.csv")

    dmm = build_dashboard_model_metrics(sources["classifier_metrics_summary"])
    export_dashboard_table(dmm, "dashboard_model_metrics.csv")

    best_name = str(sources["best_model_selection"].iloc[0]["best_model_name"])
    dconf = build_dashboard_confusion_best_model(
        read_required_csv(TABLES_DIR / "confusion_matrix_by_model.csv"),
        best_name,
    )
    export_dashboard_table(dconf, "dashboard_confusion_best_model.csv")

    dpred = build_dashboard_predictions(
        sources["final_model_predictions"],
        sources["condition_index_by_window"],
        sources["condition_index_by_window_thresholded_global"],
        sources["condition_index_by_window_thresholded_by_batch"],
    )
    export_dashboard_table(dpred, "dashboard_predictions.csv")

    export_dashboard_table(sources["assessment_thresholds_summary"], "dashboard_assessment_thresholds_summary.csv")
    export_dashboard_table(sources["assessment_thresholds_global"], "dashboard_assessment_thresholds_global.csv")
    export_dashboard_table(sources["assessment_thresholds_by_batch"], "dashboard_assessment_thresholds_by_batch.csv")
    export_dashboard_table(
        sources["sensor_weights_with_thresholds"], "dashboard_sensor_weights_with_thresholds.csv"
    )
    export_dashboard_table(
        sources["condition_index_by_window_thresholded_global"],
        "dashboard_condition_index_thresholded_global_by_window.csv",
    )
    export_dashboard_table(
        sources["condition_index_by_batch_thresholded_global"],
        "dashboard_condition_index_thresholded_global_by_batch.csv",
    )
    export_dashboard_table(
        sources["condition_index_by_window_thresholded_by_batch"],
        "dashboard_condition_index_thresholded_by_batch_by_window.csv",
    )
    export_dashboard_table(
        sources["condition_index_by_batch_thresholded_by_batch"],
        "dashboard_condition_index_thresholded_by_batch_by_batch.csv",
    )

    dfeat = build_dashboard_feature_importance(sources["ranking_features"])
    export_dashboard_table(dfeat, "dashboard_feature_importance.csv")

    rank_tables = build_dashboard_sensor_rankings(sources)
    export_dashboard_table(rank_tables["rank_raw_variable"], "dashboard_rank_raw_variable.csv")
    export_dashboard_table(rank_tables["rank_component"], "dashboard_rank_component.csv")
    export_dashboard_table(rank_tables["rank_family"], "dashboard_rank_family.csv")
    export_dashboard_table(rank_tables["rank_position"], "dashboard_rank_position.csv")
    export_dashboard_table(rank_tables["rank_statistic"], "dashboard_rank_statistic.csv")
    export_dashboard_table(rank_tables["top_weighted_variables"], "dashboard_top_weighted_variables.csv")

    export_dashboard_table(sources["condition_index_by_batch"], "dashboard_condition_index_by_batch.csv")
    export_dashboard_table(sources["pairwise_permanova"], "dashboard_pairwise_permanova.csv")
    export_dashboard_table(sources["pairwise_permdisp"], "dashboard_pairwise_permdisp.csv")
    export_dashboard_table(sources["pca_centroids"], "dashboard_pca_centroids.csv")
    export_dashboard_table(sources["pca_projection"], "dashboard_pca_projection.csv")
    export_dashboard_table(sources["batch_transitions"], "dashboard_batch_transitions.csv")

    export_dashboard_table(
        sources["condition_contributions_long"], "dashboard_condition_contributions_long.csv"
    )
    export_dashboard_table(
        sources["condition_contributions_top_by_window"],
        "dashboard_condition_contributions_top_by_window.csv",
    )
    export_dashboard_table(sources["condition_current_state"], "dashboard_condition_current_state.csv")
    export_dashboard_table(sources["condition_alerts_active"], "dashboard_condition_alerts_active.csv")
    export_dashboard_table(sources["condition_trend_summary"], "dashboard_condition_trend_summary.csv")

    with (REPORTS_DIR / "current_asset_state.json").open(encoding="utf-8") as f:
        current_asset_state_obj = json.load(f)
    export_json(current_asset_state_obj, DATA_DASHBOARD_DIR / "current_asset_state.json")

    _write_dashboard_readme()

    # Validaciones obligatorias (solo lectura / shape)
    kpi_path = DATA_DASHBOARD_DIR / "dashboard_kpis.csv"
    kdf = pd.read_csv(kpi_path, encoding="utf-8")
    assert set(["metric", "value", "category", "description"]).issubset(set(kdf.columns)), "dashboard_kpis columnas"
    assert len(dmm) == 5, f"dashboard_model_metrics esperaba 5 modelos, hay {len(dmm)}"
    assert len(dconf) > 0, "dashboard_confusion_best_model vacio"
    assert len(dpred) == 1451, f"dashboard_predictions esperaba 1451 filas, hay {len(dpred)}"
    assert len(dfeat) == 120, f"dashboard_feature_importance esperaba 120 filas, hay {len(dfeat)}"
    rc = rank_tables["rank_component"]
    if not rc.empty and "share_percent" in rc.columns:
        s = float(rc["share_percent"].sum())
        assert abs(s - 100.0) < 0.02, f"share_percent componentes suma {s}, se esperaba ~100"
    assert len(rank_tables["top_weighted_variables"]) == 24, "top_weighted_variables debe tener 24 filas"
    assert len(sources["condition_index_by_batch"]) == 3, "condition_index_by_batch debe tener 3 batches"

    aths = sources["assessment_thresholds_summary"]
    assert len(sources["assessment_thresholds_global"]) == 24, "dashboard export thresholds global"
    assert len(sources["assessment_thresholds_by_batch"]) == 72, "dashboard export thresholds by batch"
    assert len(sources["sensor_weights_with_thresholds"]) == 24, "dashboard sensor_weights_with_thresholds"
    assert len(sources["condition_index_by_window_thresholded_global"]) == 1466, "thresholded global by_window"
    assert len(sources["condition_index_by_window_thresholded_by_batch"]) == 1466, "thresholded by_batch by_window"
    assert len(sources["condition_index_by_batch_thresholded_global"]) == 3, "thresholded global by_batch"
    assert len(sources["condition_index_by_batch_thresholded_by_batch"]) == 3, "thresholded by_batch by_batch"
    sw = sources["sensor_weights_with_thresholds"]
    assert sw[["v0", "h", "hh"]].notna().all().all(), "V0/H/HH no deben ser nulos en pesos con umbrales"
    assert not aths.empty, "assessment_thresholds_summary no vacio"
    assert len(aths.loc[aths["metric"].astype(str) == "threshold_method_global", "value"]) > 0
    assert len(aths.loc[aths["metric"].astype(str) == "threshold_method_by_batch", "value"]) > 0
    for col in (
        "condition_index_thresholded_global",
        "condition_index_thresholded_by_batch",
        "baseline_batch_used",
        "health_index_thresholded_by_batch",
    ):
        assert col in dpred.columns, f"falta columna {col} en dashboard_predictions"
    assert dpred["condition_index_thresholded_global"].notna().all(), "thresholded global sin NaN en predicciones"
    assert dpred["condition_index_thresholded_by_batch"].notna().all(), "thresholded by_batch sin NaN en predicciones"
    assert dpred["baseline_batch_used"].notna().all(), "baseline_batch_used sin NaN en predicciones"
    cg = dpred["condition_index_thresholded_global"].dropna()
    cb = dpred["condition_index_thresholded_by_batch"].dropna()
    assert (cg >= -1e-9).all() and (cg <= 100.0 + 1e-9).all(), "condition_index_thresholded_global fuera de [0,100]"
    assert (cb >= -1e-9).all() and (cb <= 100.0 + 1e-9).all(), "condition_index_thresholded_by_batch fuera de [0,100]"
    htb = pd.to_numeric(dpred["health_index_thresholded_by_batch"], errors="coerce")
    m_htb = dpred["condition_index_thresholded_by_batch"].notna()
    assert (htb[m_htb] >= -1e-9).all() and (htb[m_htb] <= 100.0 + 1e-9).all(), "health_index_thresholded_by_batch fuera de [0,100]"
    kdf_thr = kdf[kdf["category"].astype(str) == "assessment_thresholds"]
    assert len(kdf_thr) >= 18, "KPIs assessment_thresholds en dashboard_kpis"
    assert (kdf_thr["metric"].astype(str) == "threshold_note").any(), "threshold_note en KPIs"

    d_long = pd.read_csv(DATA_DASHBOARD_DIR / "dashboard_condition_contributions_long.csv", encoding="utf-8")
    d_top = pd.read_csv(DATA_DASHBOARD_DIR / "dashboard_condition_contributions_top_by_window.csv", encoding="utf-8")
    d_cur = pd.read_csv(DATA_DASHBOARD_DIR / "dashboard_condition_current_state.csv", encoding="utf-8")
    pd.read_csv(DATA_DASHBOARD_DIR / "dashboard_condition_alerts_active.csv", encoding="utf-8")
    d_tr = pd.read_csv(DATA_DASHBOARD_DIR / "dashboard_condition_trend_summary.csv", encoding="utf-8")
    assert len(d_long) == 35184, "dashboard_condition_contributions_long debe tener 35184 filas"
    assert len(d_top) == 7330, "dashboard_condition_contributions_top_by_window debe tener 7330 filas"
    assert len(d_cur) == 1, "dashboard_condition_current_state debe tener 1 fila"
    assert (DATA_DASHBOARD_DIR / "dashboard_condition_alerts_active.csv").is_file(), "dashboard_condition_alerts_active"
    assert len(d_tr) >= 1, "dashboard_condition_trend_summary al menos 1 fila"
    js_dash = DATA_DASHBOARD_DIR / "current_asset_state.json"
    assert js_dash.is_file(), "current_asset_state.json en data/dashboard"
    with js_dash.open(encoding="utf-8") as jf:
        json.load(jf)
    kdf_cs = kdf[kdf["category"].astype(str) == "condition_state"]
    assert not kdf_cs.empty, "dashboard_kpis debe incluir categoria condition_state"
    cs_metrics = {
        "current_window_id",
        "current_batch_real",
        "current_batch_predicted",
        "current_classification_confidence",
        "current_condition_index",
        "current_health_index",
        "current_condition_state",
        "current_trend_direction",
        "n_active_attention_alerts",
        "n_active_high_alerts",
        "top_condition_component",
        "top_condition_family",
        "top_condition_variable",
    }
    assert cs_metrics.issubset(set(kdf_cs["metric"].astype(str))), "KPIs condition_state incompletos"
    ci_cur = float(kdf_cs.loc[kdf_cs["metric"] == "current_condition_index", "value"].iloc[0])
    hi_cur = float(kdf_cs.loc[kdf_cs["metric"] == "current_health_index", "value"].iloc[0])
    assert -1e-9 <= ci_cur <= 100.0 + 1e-9, "current_condition_index fuera de [0,100]"
    assert -1e-9 <= hi_cur <= 100.0 + 1e-9, "current_health_index fuera de [0,100]"

    b0 = sources["best_model_selection"].iloc[0].to_dict()
    intr0 = sources["interpretability_summary"].iloc[0].to_dict()
    as0 = sources["assessment_summary"].iloc[0].to_dict()
    cmean = None
    if not sources["condition_index_summary"].empty:
        cs = sources["condition_index_summary"]
        m = cs.loc[cs["metric"].astype(str) == "condition_index_mean", "value"]
        if len(m):
            cmean = float(m.iloc[0])

    def _ath_val(key: str) -> Any:
        return _kv(aths, key)

    def _ath_float(key: str) -> float | None:
        v = _ath_val(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    cur_state_df = sources["condition_current_state"]
    cur0d = cur_state_df.iloc[0].to_dict() if not cur_state_df.empty else {}
    alt_src = sources["condition_alerts_active"]
    n_att_c = (
        int((alt_src["alert_level"].astype(str) == "attention").sum())
        if not alt_src.empty and "alert_level" in alt_src.columns
        else 0
    )
    n_hi_c = (
        int((alt_src["alert_level"].astype(str) == "high").sum())
        if not alt_src.empty and "alert_level" in alt_src.columns
        else 0
    )
    tr_src = sources["condition_trend_summary"]
    tr_dir_c = (
        str(tr_src.iloc[0]["trend_direction"])
        if not tr_src.empty and "trend_direction" in tr_src.columns
        else ""
    )

    return {
        "dashboard_dir": DATA_DASHBOARD_DIR,
        "best_model_name": str(b0.get("best_model_name", "")),
        "f1_macro_cv": float(b0.get("f1_macro_mean", np.nan)),
        "balanced_accuracy_cv": float(b0.get("balanced_accuracy_mean", np.nan)),
        "permanova_r2": float(sources["permanova_results"].iloc[0]["r2"])
        if not sources["permanova_results"].empty
        else np.nan,
        "top_component": str(intr0.get("top_component", "")),
        "top_family": str(intr0.get("top_family", "")),
        "condition_index_mean": cmean,
        "assessment_method": str(as0.get("assessment_method", "")),
        "threshold_method_global": str(_ath_val("threshold_method_global") or ""),
        "threshold_method_by_batch": str(_ath_val("threshold_method_by_batch") or ""),
        "condition_index_mean_thresholded_global": _ath_float("condition_index_mean_thresholded_global"),
        "condition_index_mean_thresholded_by_batch": _ath_float("condition_index_mean_thresholded_by_batch"),
        "current_window_id": int(cur0d["window_id"])
        if cur0d and pd.notna(cur0d.get("window_id"))
        else None,
        "current_batch_real": cur0d.get("Batch") if cur0d else None,
        "current_batch_predicted": cur0d.get("y_pred_label") if cur0d else None,
        "current_condition_index": float(cur0d["condition_index"])
        if cur0d and pd.notna(cur0d.get("condition_index"))
        else None,
        "current_health_index": float(cur0d["health_index"])
        if cur0d and pd.notna(cur0d.get("health_index"))
        else None,
        "current_condition_state": str(cur0d.get("condition_state", "")) if cur0d else None,
        "trend_direction": tr_dir_c,
        "n_attention_alerts": n_att_c,
        "n_high_alerts": n_hi_c,
    }


def _write_dashboard_readme() -> None:
    DATA_DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DASHBOARD_DIR / "README_dashboard_data.md"
    body = """# Datos derivados para dashboard (Proyecto 4 BPC)

## Origen

Los archivos CSV en esta carpeta son **derivados**: se generan leyendo `outputs/tables/` y
`outputs/reports/` (JSON de estado) y `data/processed/` **sin** recalcular features, modelos, SHAP, estadistica inferencial ni assessment;
los artefactos **thresholded** (Etapa 8B) y **condition_state** (Etapa 8C) solo se **copian** desde `outputs/` si ya existen.

## Uso

No editar manualmente estos archivos en entornos productivos.

Regenerar tras actualizar etapas previas:

```text
python run_pipeline.py --stage dashboard_exports
```

## Archivos

| Archivo | Contenido breve |
|---------|-----------------|
| dashboard_kpis.csv | KPIs consolidados (metric, value, category, description) |
| dashboard_model_metrics.csv | Metricas CV por modelo (filas = modelos) |
| dashboard_confusion_best_model.csv | Matriz de confusion del mejor modelo |
| dashboard_predictions.csv | Predicciones finales + indice de condicion por ventana + umbrales globales/por batch + health por batch derivado (100 - indice) |
| dashboard_feature_importance.csv | Ranking consolidado de features (interpretabilidad) |
| dashboard_rank_*.csv | Rankings agregados por variable, componente, familia, posicion, estadistico |
| dashboard_top_weighted_variables.csv | Variables con mayor peso en assessment |
| dashboard_condition_index_by_batch.csv | Indice de condicion resumido por Batch |
| dashboard_pairwise_permanova.csv | Comparaciones pairwise PERMANOVA (FDR) |
| dashboard_pairwise_permdisp.csv | Comparaciones pairwise PERMDISP (FDR) |
| dashboard_pca_centroids.csv | Centroides PCA por clase |
| dashboard_pca_projection.csv | Proyeccion PCA (PC1, PC2) por ventana y batch |
| dashboard_batch_transitions.csv | Transiciones de operacion entre batches |
| dashboard_assessment_thresholds_summary.csv | KPIs y notas de umbrales V0/H/HH exploratorios (Etapa 8B) |
| dashboard_assessment_thresholds_global.csv | Umbrales globales por variable (P40/P75/P99) |
| dashboard_assessment_thresholds_by_batch.csv | Umbrales por Batch y variable |
| dashboard_sensor_weights_with_thresholds.csv | Pesos normalizados con V0/H/HH globales estimados |
| dashboard_condition_index_thresholded_global_by_window.csv | Indice por ventana con baseline global |
| dashboard_condition_index_thresholded_global_by_batch.csv | Resumen por Batch (baseline global) |
| dashboard_condition_index_thresholded_by_batch_by_window.csv | Indice por ventana con baseline por Batch |
| dashboard_condition_index_thresholded_by_batch_by_batch.csv | Resumen por Batch (baseline por Batch) |
| dashboard_condition_contributions_long.csv | Contribuciones por variable y ventana (Etapa 8C, exploratorio) |
| dashboard_condition_contributions_top_by_window.csv | Top contribuciones agregadas por ventana |
| dashboard_condition_current_state.csv | Estado de condicion de la ultima ventana historica exportada |
| dashboard_condition_alerts_active.csv | Alertas exploratorias activas asociadas a esa ventana (0 o mas filas) |
| dashboard_condition_trend_summary.csv | Resumen de tendencia reciente del indice de condicion |
| current_asset_state.json | Ultimo estado del activo serializado (lectura de `outputs/reports/`) |

## Advertencias

- Las metricas de **generalizacion** relevantes para comparar modelos son las de **validacion cruzada**
  (`dashboard_model_metrics.csv`), no las de entrenamiento del modelo final (pueden ser optimistas).
- El **assessment** de condicion usa en este proyecto principalmente `robust_percentile_fallback`
  cuando los umbrales H, HH y V0 del Excel de pesos estan vacios.
- Los **pesos ponderados** del activo **no** representan importancia ML ni causalidad.
- **SHAP** y permutation importance **no** implican causalidad.
- **MEZCLA** se trata como clase operacional independiente (no es solo transicion).

### Etapa 8C — Estado de condicion y alertas (exploratorio)

- Las **alertas** en `dashboard_condition_alerts_active.csv` son **exploratorias**; **no** son alarmas normativas ni sustituyen procedimientos de operacion.
- **condition_state** usa **bandas visuales** sobre **condition_index**: **normal** si < 20, **attention** si 20 <= indice < 40, **high** si >= 40 (umbrales no normativos). Sobre **EPI_BPC** (= health_index = 100 - condition_index): **normal** si EPI > 80, **attention** si 60 <= EPI <= 80, **high** si EPI < 60.
- **health_index** = 100 − **condition_index** (misma convencion en series y KPIs cuando aplica).
- **No** se calcula **RUL** ni vida util remanente en estos artefactos.
- **current_asset_state.json** representa el **ultimo estado historico disponible** en los datos exportados, **no** una lectura en vivo del activo.

### Umbrales V0 / H / HH (Etapa 8B, exploratorios)

- Los umbrales V0/H/HH globales y por batch son **exploratorios**; no son limites normativos de alarma ni detectan falla real por si solos.
- Se estimaron con percentiles del historico disponible en ventanas: **V0 = P40, H = P75, HH = P99** (por variable; por batch dentro de cada Batch).
- El modo **by_batch** en analisis historico usa el **Batch real** de cada ventana.
- En operacion futura, si se usara batch **predicho**, solo seria aceptable con **confianza y margen** del modelo suficientes; si no, conviene baseline **global** o marcar el assessment como **incierto**.
"""
    path.write_text(body, encoding="utf-8")
