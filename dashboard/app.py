"""
Dashboard visual Proyecto 4 BPC (Dash).
Solo lee CSV en data/dashboard/. No entrena modelos ni recalcula metricas.
Version de la app: constante DASHBOARD_APP_VERSION en este archivo (incrementar al publicar cambios).
Ejecucion: python dashboard/app.py  ->  http://127.0.0.1:8050
Puerto alternativo (si 8050 quedo colgado por un proceso viejo): DASH_PORT=8051 python dashboard/app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dash_table, dcc, html

# Raiz del subproyecto (padre de dashboard/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DATA = PROJECT_ROOT / "data" / "dashboard"
try:
    _app_mt = Path(__file__).resolve().stat().st_mtime
    DASH_APP_BUILD = datetime.fromtimestamp(_app_mt, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
except OSError:
    DASH_APP_BUILD = "?"

# Version semantica del dashboard Dash (incrementar manualmente al publicar cambios de UI o logica).
DASHBOARD_APP_VERSION = "1.4.21"

CRITICAL_FILES = (
    "dashboard_kpis.csv",
    "dashboard_predictions.csv",
    "dashboard_model_metrics.csv",
)

OPTIONAL_FILES = (
    "dashboard_confusion_best_model.csv",
    "dashboard_feature_importance.csv",
    "dashboard_rank_component.csv",
    "dashboard_rank_family.csv",
    "dashboard_rank_position.csv",
    "dashboard_rank_statistic.csv",
    "dashboard_rank_raw_variable.csv",
    "dashboard_top_weighted_variables.csv",
    "dashboard_condition_index_by_batch.csv",
    "dashboard_pairwise_permanova.csv",
    "dashboard_pairwise_permdisp.csv",
    "dashboard_pca_centroids.csv",
    "dashboard_pca_projection.csv",
    "dashboard_batch_transitions.csv",
    "README_dashboard_data.md",
    "dashboard_assessment_thresholds_summary.csv",
    "dashboard_assessment_thresholds_global.csv",
    "dashboard_assessment_thresholds_by_batch.csv",
    "dashboard_sensor_weights_with_thresholds.csv",
    "dashboard_condition_index_thresholded_global_by_window.csv",
    "dashboard_condition_index_thresholded_global_by_batch.csv",
    "dashboard_condition_index_thresholded_by_batch_by_window.csv",
    "dashboard_condition_index_thresholded_by_batch_by_batch.csv",
)

CLASS_ORDER = ("CASTILLA", "MEZCLA", "RUBIALES")
# Serie temporal indices (tab Predicciones): solo estas claves en checklist y por defecto.
PRED_INDEX_SERIES_ALLOWED = frozenset({"by_batch", "health_batch"})
PRED_INDEX_SERIES_DEFAULT: tuple[str, ...] = ("by_batch", "health_batch")
# Paleta por clase de crudo: sin rojo/verde/amarillo (se reservan para semaforos en KPIs y bandas).
CLASS_COLORS = {
    "CASTILLA": "#1f77b4",
    "MEZCLA": "#9467bd",
    "RUBIALES": "#17becf",
}

# Etiquetas visibles para columnas tecnicas (id de columna = clave interna sin cambiar).
MODEL_METRICS_COLUMN_LABELS: dict[str, str] = {
    "model_name": "Modelo",
    "accuracy_mean": "Exactitud (media CV)",
    "accuracy_std": "Exactitud (desv. CV)",
    "balanced_accuracy_mean": "Exactitud balanceada (media CV)",
    "balanced_accuracy_std": "Exactitud balanceada (desv. CV)",
    "f1_macro_mean": "F1 macro (media CV)",
    "f1_macro_std": "F1 macro (desv. CV)",
    "precision_macro_mean": "Precision macro (media CV)",
    "recall_macro_mean": "Exhaustividad macro (media CV)",
    "mcc_mean": "MCC (media CV)",
    "cohen_kappa_mean": "Kappa Cohen (media CV)",
}

PREDICTION_TABLE_COLUMN_LABELS: dict[str, str] = {
    "window_id": "ID ventana",
    "window_start": "Inicio ventana",
    "Batch": "Batch real",
    "y_true_label": "Clase real",
    "y_pred_label": "Clase predicha",
    "prob_CASTILLA": "Prob. CASTILLA",
    "prob_MEZCLA": "Prob. MEZCLA",
    "prob_RUBIALES": "Prob. RUBIALES",
    "confidence": "Confianza del clasificador",
    "margin_top2": "Margen entre las dos clases mas probables",
    "is_correct": "Prediccion correcta",
    "condition_index": "Indice de condicion",
    "condition_index_thresholded_global": "Indice condicion (umbral global)",
    "condition_index_thresholded_by_batch": "Indice condicion (umbral por batch)",
    "baseline_batch_used": "Linea base usada",
    "health_index_thresholded_by_batch": "Salud relativa por batch (CSV)",
    "baseline_batch_operational": "Linea base operacional (predicha)",
    "baseline_status": "Estado linea base operacional",
    "health_index_operational": "EPI / salud relativa (operacional)",
    "condition_index_operational": "Severidad relativa (operacional)",
}

# Tooltips de cabecera (tabla de predicciones): id de columna -> texto corto.
PREDICTION_TABLE_HEADER_TOOLTIPS: dict[str, str] = {
    "confidence": "Probabilidad asignada por el clasificador a la clase predicha.",
    "margin_top2": "Diferencia entre las dos probabilidades mas altas.",
    "baseline_batch_operational": "Linea base seleccionada segun clase predicha y confianza.",
    "baseline_batch_used": "Linea base historica asociada al batch real en el export.",
    "baseline_status": "Indica si la linea base sigue al crudo predicho o a un fallback global.",
    "health_index_operational": "100 - condition_index (EPI_BPC operacional con la linea base vigente).",
    "condition_index_operational": "Severidad relativa 0-100 (operacional con la linea base vigente).",
    "condition_index": "Severidad relativa 0-100.",
    "health_index_thresholded_by_batch": "100 - condition_index segun exportacion CSV (umbral por batch).",
    "weighted_score": "Aporte ponderado de una variable al indice de condicion.",
}


def _prediction_table_tooltip_header_for_datatable(cols: list[str]) -> dict[str, dict[str, str]]:
    return {
        c: {"value": PREDICTION_TABLE_HEADER_TOOLTIPS[c], "type": "text"}
        for c in cols
        if c in PREDICTION_TABLE_HEADER_TOOLTIPS
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(136,136,136,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _activo_sem_style(lev: str) -> dict[str, str]:
    """Semaforo exploratorio: verde / amarillo / rojo (solo UI)."""
    if lev == "neutral":
        return {"border": "1px solid #e2e4e9", "backgroundColor": "#fff"}
    if lev == "green":
        return {
            "borderLeft": "4px solid #2ca02c",
            "backgroundColor": "rgba(44, 160, 44, 0.11)",
            "border": "1px solid rgba(44, 160, 44, 0.38)",
        }
    if lev == "yellow":
        return {
            "borderLeft": "4px solid #e6b800",
            "backgroundColor": "rgba(230, 184, 0, 0.16)",
            "border": "1px solid rgba(200, 160, 0, 0.42)",
        }
    if lev == "red":
        return {
            "borderLeft": "4px solid #d62728",
            "backgroundColor": "rgba(214, 39, 40, 0.10)",
            "border": "1px solid rgba(214, 39, 40, 0.38)",
        }
    return {"border": "1px solid #e2e4e9", "backgroundColor": "#fff"}


def _activo_kpi_card_style(label: str, display_val: str, raw_val: Any = None) -> dict[str, Any]:
    """Colores tarjetas estado del activo: batch = paleta graficos; metricas = semaforo."""
    base: dict[str, Any] = {
        "borderRadius": "8px",
        "padding": "0.65rem 0.85rem",
        "minWidth": "120px",
        "flex": "2 1 200px" if "Alertas exploratorias" in label else "1 1 140px",
    }

    def _to_float(x: Any) -> float | None:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            try:
                return float(str(x).strip().replace(",", "."))
            except ValueError:
                return None

    lbl = str(label)
    vs = str(display_val).strip()

    if lbl in ("Batch real", "Crudo predicho", "Batch predicho"):
        bn = vs.upper()
        c = CLASS_COLORS.get(bn, "#888888")
        return {
            **base,
            "borderLeft": f"4px solid {c}",
            "backgroundColor": _hex_to_rgba(c, 0.14),
            "border": f"1px solid {_hex_to_rgba(c, 0.42)}",
        }

    src = raw_val if raw_val is not None else vs
    if lbl == "Confianza del clasificador":
        v = _to_float(src)
        if v is None:
            return {**base, **_activo_sem_style("neutral")}
        # Escala 0-1 (dashboard); si viniera 0-100, tratar como fraccion >1
        if v > 1.0 + 1e-6:
            v = v / 100.0
        if v >= 0.80:
            lev = "green"
        elif v >= 0.50:
            lev = "yellow"
        else:
            lev = "red"
        return {**base, **_activo_sem_style(lev)}

    if lbl in ("Indice de condicion", "Indice condicion"):
        v = _to_float(src)
        if v is None:
            return {**base, **_activo_sem_style("neutral")}
        # Complemento de bandas EPI 60/80 en misma escala 0-100 (equiv. 100 - zona EPI).
        if v < 20:
            lev = "green"
        elif v < 40:
            lev = "yellow"
        else:
            lev = "red"
        return {**base, **_activo_sem_style(lev)}

    if lbl in ("EPI_BPC / salud relativa", "Health index"):
        v = _to_float(src)
        if v is None:
            return {**base, **_activo_sem_style("neutral")}
        if v >= 80:
            lev = "green"
        elif v >= 60:
            lev = "yellow"
        else:
            lev = "red"
        return {**base, **_activo_sem_style(lev)}

    if lbl in ("Banda condition_state", "Estado exploratorio"):
        s = vs.lower()
        if "normal" in s:
            lev = "green"
        elif "attention" in s or "atencion" in s:
            lev = "yellow"
        elif "high" in s or "alto" in s:
            lev = "red"
        else:
            return {**base, **_activo_sem_style("neutral")}
        return {**base, **_activo_sem_style(lev)}

    if "Alertas exploratorias" in lbl and "attention" in lbl.lower():
        n = int(_to_float(src) or 0)
        lev = "green" if n == 0 else "yellow"
        return {**base, **_activo_sem_style(lev)}

    if "Alertas exploratorias" in lbl and "high" in lbl.lower():
        n = int(_to_float(src) or 0)
        lev = "green" if n == 0 else "red"
        return {**base, **_activo_sem_style(lev)}

    return {**base, "border": "1px solid #e2e4e9", "backgroundColor": "#fafafa"}


def _read_csv(name: str) -> pd.DataFrame:
    p = DASHBOARD_DATA / name
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8")


def _read_text(name: str) -> str:
    p = DASHBOARD_DATA / name
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8")


def _read_json(name: str) -> Any | None:
    """Lee JSON desde data/dashboard; None si no existe o no es JSON valido."""
    p = DASHBOARD_DATA / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_all() -> dict[str, Any]:
    missing_critical: list[str] = []
    missing_optional: list[str] = []
    for f in CRITICAL_FILES:
        if not (DASHBOARD_DATA / f).is_file():
            missing_critical.append(f)
    for f in OPTIONAL_FILES:
        if not (DASHBOARD_DATA / f).is_file():
            missing_optional.append(f)

    return {
        "kpis": _read_csv("dashboard_kpis.csv"),
        "model_metrics": _read_csv("dashboard_model_metrics.csv"),
        "confusion": _read_csv("dashboard_confusion_best_model.csv"),
        "predictions": _read_csv("dashboard_predictions.csv"),
        "feature_importance": _read_csv("dashboard_feature_importance.csv"),
        "rank_component": _read_csv("dashboard_rank_component.csv"),
        "rank_family": _read_csv("dashboard_rank_family.csv"),
        "rank_position": _read_csv("dashboard_rank_position.csv"),
        "rank_statistic": _read_csv("dashboard_rank_statistic.csv"),
        "rank_raw": _read_csv("dashboard_rank_raw_variable.csv"),
        "top_weighted": _read_csv("dashboard_top_weighted_variables.csv"),
        "cond_batch": _read_csv("dashboard_condition_index_by_batch.csv"),
        "pairwise_perm": _read_csv("dashboard_pairwise_permanova.csv"),
        "pairwise_disp": _read_csv("dashboard_pairwise_permdisp.csv"),
        "pca_centroids": _read_csv("dashboard_pca_centroids.csv"),
        "pca_projection": _read_csv("dashboard_pca_projection.csv"),
        "batch_transitions": _read_csv("dashboard_batch_transitions.csv"),
        "readme": _read_text("README_dashboard_data.md"),
        "ath_summary": _read_csv("dashboard_assessment_thresholds_summary.csv"),
        "ath_global": _read_csv("dashboard_assessment_thresholds_global.csv"),
        "ath_by_batch": _read_csv("dashboard_assessment_thresholds_by_batch.csv"),
        "sw_thr": _read_csv("dashboard_sensor_weights_with_thresholds.csv"),
        "ci_g_win": _read_csv("dashboard_condition_index_thresholded_global_by_window.csv"),
        "ci_g_batch": _read_csv("dashboard_condition_index_thresholded_global_by_batch.csv"),
        "ci_b_win": _read_csv("dashboard_condition_index_thresholded_by_batch_by_window.csv"),
        "ci_b_batch": _read_csv("dashboard_condition_index_thresholded_by_batch_by_batch.csv"),
        "cond_contrib_long": _read_csv("dashboard_condition_contributions_long.csv"),
        "cond_contrib_top": _read_csv("dashboard_condition_contributions_top_by_window.csv"),
        "cond_current": _read_csv("dashboard_condition_current_state.csv"),
        "cond_alerts": _read_csv("dashboard_condition_alerts_active.csv"),
        "cond_trend": _read_csv("dashboard_condition_trend_summary.csv"),
        "asset_state_json": _read_json("current_asset_state.json"),
        "_missing_critical": missing_critical,
        "_missing_optional": missing_optional,
    }


def kpi_lookup(kpis: pd.DataFrame, metric: str) -> Any:
    if kpis.empty or "metric" not in kpis.columns:
        return None
    m = kpis.loc[kpis["metric"].astype(str) == metric, "value"]
    return m.iloc[0] if len(m) else None


def abbrev_label(text: str, max_len: int = 42) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    if "\\" in s:
        tail = s.split("\\")[-1]
        if len(tail) <= max_len:
            return "..." + tail[-max_len:]
    return s[: max_len - 3] + "..."


def _bar_sorted_categoryarray(
    df: pd.DataFrame,
    cat: str,
    val: str,
    *,
    largest_first: bool,
    tiebreak_cols: tuple[str, ...] = (),
) -> list[str]:
    """
    Orden estable de categorias para ejes de barras en Plotly.
    largest_first=True: barras verticales (mayor valor a la izquierda).
    largest_first=False: barras horizontales (mayor valor arriba: menor valor primero en la lista).
    """
    if df.empty or cat not in df.columns or val not in df.columns:
        return []
    extras = [c for c in tiebreak_cols if c in df.columns and c not in (cat, val)]
    cols = [cat, val, *extras]
    sub = df.loc[:, cols].copy()
    sub["_v"] = pd.to_numeric(sub[val], errors="coerce")
    asc = not largest_first
    sort_keys = ["_v", cat, *extras]
    asc_flags = [asc] + [True] * (len(sort_keys) - 1)
    sub = sub.sort_values(sort_keys, ascending=asc_flags, na_position="last")
    return sub[cat].astype(str).tolist()


def _unique_bar_category_labels(df: pd.DataFrame, label_col: str, id_col: str) -> pd.Series:
    """
    Garantiza categorias unicas en el eje de barras. Si varias filas comparten la misma
    etiqueta truncada (p. ej. abbrev_label), Plotly las fusiona y el orden por categoryarray falla.
    """
    if label_col not in df.columns:
        return pd.Series(dtype=str)
    if id_col not in df.columns:
        return df[label_col].astype(str)
    labels = df[label_col].astype(str).to_numpy()
    ids = df[id_col].astype(str).to_numpy()
    counts: dict[str, int] = {}
    out: list[str] = []
    for lab, fid in zip(labels, ids):
        k = str(lab)
        n = counts.get(k, 0)
        counts[k] = n + 1
        if n == 0:
            out.append(k)
        else:
            tail = fid[-36:] if len(fid) > 36 else fid
            out.append(f"{k} [{tail}]")
    return pd.Series(out, index=df.index, dtype=str)


# Decimales mostrados en tablas, KPIs y hovers de Plotly (maximo dos).
DASH_DECIMALS = 2


def dash_format_number(val: Any) -> str:
    """Texto para KPIs y tarjetas: como mucho dos decimales; enteros sin .00."""
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    if isinstance(val, (bool, np.bool_)):
        return str(bool(val))
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return ""
        try:
            fv = float(s.replace(",", "."))
        except ValueError:
            return s
    else:
        try:
            fv = float(val)
        except (TypeError, ValueError):
            return str(val)
    if abs(fv - round(fv)) < 1e-9 and abs(fv) < 1e15:
        return str(int(round(fv)))
    return f"{fv:.{DASH_DECIMALS}f}"


def round_numeric_df(df: pd.DataFrame, decimals: int = DASH_DECIMALS) -> pd.DataFrame:
    """Redondea columnas numericas para tablas Dash y hovers derivados de datos."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(decimals)
    return out


def json_round_floats(obj: Any, decimals: int = DASH_DECIMALS) -> Any:
    """Recorre JSON-like para limitar decimales en vista Pre."""
    if isinstance(obj, float):
        return round(obj, decimals) if obj == obj else obj
    if isinstance(obj, dict):
        return {str(k): json_round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_round_floats(v, decimals) for v in obj]
    return obj


def confusion_matrix_normalized(conf: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    labels = list(CLASS_ORDER)
    n = len(labels)
    mat = np.zeros((n, n))
    if conf.empty:
        return mat, labels
    for i, tl in enumerate(labels):
        sub = conf[conf["true_label"].astype(str) == tl]
        tot = sub["count"].sum()
        for j, pl in enumerate(labels):
            row = sub[sub["pred_label"].astype(str) == pl]
            c = float(row["count"].sum()) if len(row) else 0.0
            mat[i, j] = (c / tot) if tot > 0 else 0.0
    return mat, labels


def collect_dashboard_health_warnings(data: dict[str, Any]) -> list[str]:
    """Validaciones internas Etapa 10B/10C: no bloquean el arranque."""
    w: list[str] = []
    pred = data.get("predictions", pd.DataFrame())
    for col in (
        "condition_index_thresholded_global",
        "condition_index_thresholded_by_batch",
        "baseline_batch_used",
        "health_index_thresholded_by_batch",
    ):
        if pred.empty or col not in pred.columns:
            w.append(f"dashboard_predictions.csv: falta columna '{col}'")
    checks: list[tuple[str, str, int | None]] = [
        ("dashboard_assessment_thresholds_global.csv", "ath_global", 24),
        ("dashboard_assessment_thresholds_by_batch.csv", "ath_by_batch", 72),
        ("dashboard_condition_index_thresholded_global_by_batch.csv", "ci_g_batch", 3),
        ("dashboard_condition_index_thresholded_by_batch_by_batch.csv", "ci_b_batch", 3),
    ]
    for fname, key, expected_rows in checks:
        df = data.get(key, pd.DataFrame())
        if df.empty:
            w.append(f"{fname}: vacio o no cargado")
        elif expected_rows is not None and len(df) != expected_rows:
            w.append(f"{fname}: se esperaban {expected_rows} filas, hay {len(df)}")
    if not pred.empty and "health_index_thresholded_by_batch" in pred.columns:
        s = pd.to_numeric(pred["health_index_thresholded_by_batch"], errors="coerce")
        if s.notna().any() and (float(s.min()) < -1e-6 or float(s.max()) > 100.0 + 1e-6):
            w.append(
                "EPI_BPC / health: health_index_thresholded_by_batch tiene valores fuera de [0,100] "
                "(revisar exportaciones; las graficas clipan en 0-100 solo a nivel visual)."
            )
    return w


def collect_epi_bpc_operational_range_warnings(pred_op: pd.DataFrame) -> list[str]:
    """Tras construir predictions_operational: validar health_index_operational en [0,100]."""
    w: list[str] = []
    if pred_op.empty or "health_index_operational" not in pred_op.columns:
        return w
    so = pd.to_numeric(pred_op["health_index_operational"], errors="coerce")
    if so.notna().any() and (float(so.min()) < -1e-6 or float(so.max()) > 100.0 + 1e-6):
        w.append("EPI_BPC / health: health_index_operational fuera de rango [0,100].")
    return w


def collect_condition_state_warnings(data: dict[str, Any]) -> list[str]:
    """Etapa 10C: archivos condition_state / JSON / health index (no bloqueante)."""
    w: list[str] = []
    need = [
        "dashboard_condition_contributions_long.csv",
        "dashboard_condition_contributions_top_by_window.csv",
        "dashboard_condition_current_state.csv",
        "dashboard_condition_alerts_active.csv",
        "dashboard_condition_trend_summary.csv",
        "current_asset_state.json",
    ]
    missing_cs = [fn for fn in need if not (DASHBOARD_DATA / fn).is_file()]
    if missing_cs:
        w.append(
            "Faltan artefactos exportados de estado de condicion. Ejecute: "
            "python run_pipeline.py --stage condition_state  y luego  "
            "python run_pipeline.py --stage dashboard_exports"
        )
        w.extend(f"  - falta: {fn}" for fn in missing_cs)

    pred = data.get("predictions", pd.DataFrame())
    if pred.empty or "health_index_thresholded_by_batch" not in pred.columns:
        w.append("dashboard_predictions.csv debe incluir health_index_thresholded_by_batch (regenerar dashboard_exports).")

    cur_path = DASHBOARD_DATA / "dashboard_condition_current_state.csv"
    cur = data.get("cond_current", pd.DataFrame())
    if cur_path.is_file() and len(cur) != 1:
        w.append(f"dashboard_condition_current_state.csv: se esperaba 1 fila, hay {len(cur)}")

    tr_path = DASHBOARD_DATA / "dashboard_condition_trend_summary.csv"
    tr = data.get("cond_trend", pd.DataFrame())
    if tr_path.is_file() and tr.empty:
        w.append("dashboard_condition_trend_summary.csv esta vacio (se esperaba al menos 1 fila).")

    js_path = DASHBOARD_DATA / "current_asset_state.json"
    if js_path.is_file() and data.get("asset_state_json") is None:
        w.append("current_asset_state.json existe pero no es JSON valido.")

    return w


_PREDICTIONS_OPERATIONAL_COLS = (
    "y_pred_label",
    "confidence",
    "margin_top2",
    "condition_index_thresholded_by_batch",
    "condition_index_thresholded_global",
    "health_index_thresholded_by_batch",
)


def assign_operational_baseline(
    pred_df: pd.DataFrame,
    *,
    confidence_min: float = 0.80,
    margin_min: float = 0.15,
) -> pd.DataFrame:
    """Etapa 10D: baseline operacional segun y_pred_label si la clasificacion es suficientemente fiable."""
    out = pred_df.copy()
    if out.empty:
        return out
    if not all(c in out.columns for c in _PREDICTIONS_OPERATIONAL_COLS):
        return out
    conf = pd.to_numeric(out["confidence"], errors="coerce")
    marg = pd.to_numeric(out["margin_top2"], errors="coerce")
    yp = out["y_pred_label"].astype(str)
    ok = (conf >= confidence_min) & (marg >= margin_min)
    out["baseline_batch_operational"] = np.where(ok, yp, "GLOBAL")
    out["baseline_status"] = np.where(
        ok,
        "predicted_batch_baseline",
        "global_fallback_due_to_low_confidence",
    )
    ci_b = pd.to_numeric(out["condition_index_thresholded_by_batch"], errors="coerce")
    ci_g = pd.to_numeric(out["condition_index_thresholded_global"], errors="coerce")
    hi_b = pd.to_numeric(out["health_index_thresholded_by_batch"], errors="coerce")
    use_p = out["baseline_batch_operational"].astype(str) != "GLOBAL"
    out["condition_index_operational"] = np.where(use_p, ci_b, ci_g)
    out["health_index_operational"] = np.where(use_p, hi_b, 100.0 - ci_g)
    out["condition_index_operational"] = pd.to_numeric(out["condition_index_operational"], errors="coerce").clip(0, 100)
    out["health_index_operational"] = pd.to_numeric(out["health_index_operational"], errors="coerce").clip(0, 100)
    return out


def build_predictions_operational(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pred
    if not all(c in pred.columns for c in _PREDICTIONS_OPERATIONAL_COLS):
        return pred.copy()
    return assign_operational_baseline(pred.copy())


def collect_operational_10d_warnings(data: dict[str, Any], pred_op: pd.DataFrame) -> list[str]:
    """Validaciones Etapa 10D (no bloqueante)."""
    w: list[str] = []
    pred = data.get("predictions", pd.DataFrame())
    for col in (
        "y_pred_label",
        "confidence",
        "margin_top2",
        "condition_index_thresholded_global",
        "condition_index_thresholded_by_batch",
        "health_index_thresholded_by_batch",
    ):
        if pred.empty or col not in pred.columns:
            w.append(f"Predicciones operacionales: falta la columna '{col}' en dashboard_predictions.csv.")
    gl = data.get("ath_global", pd.DataFrame())
    bb = data.get("ath_by_batch", pd.DataFrame())
    if not gl.empty and len(gl) != 24:
        w.append(f"dashboard_assessment_thresholds_global: se esperaban 24 filas, hay {len(gl)}")
    if not bb.empty and len(bb) != 72:
        w.append(f"dashboard_assessment_thresholds_by_batch: se esperaban 72 filas, hay {len(bb)}")
    if pred_op.empty:
        return w
    if "baseline_batch_operational" in pred_op.columns and pred_op["baseline_batch_operational"].isna().any():
        w.append("Linea base operacional: baseline_batch_operational contiene valores no definidos (NaN).")
    for c in ("health_index_operational", "condition_index_operational"):
        if c in pred_op.columns:
            s = pd.to_numeric(pred_op[c], errors="coerce")
            if s.notna().any() and (s.min() < -1e-6 or s.max() > 100.0 + 1e-6):
                w.append(f"Linea base operacional: la columna {c} tiene valores fuera de rango [0,100].")
    return w


DATA = load_all()
DATA["_health_warnings"] = collect_dashboard_health_warnings(DATA)
DATA["_condition_state_warnings"] = collect_condition_state_warnings(DATA)
DATA_LOADED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
DATA["predictions_operational"] = build_predictions_operational(DATA.get("predictions", pd.DataFrame()).copy())
DATA["_operational_10d_warnings"] = collect_operational_10d_warnings(DATA, DATA["predictions_operational"])
DATA["_health_warnings"] = (DATA.get("_health_warnings") or []) + collect_epi_bpc_operational_range_warnings(
    DATA["predictions_operational"]
)


def fig_empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _ensure_plot_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Anade _plot_time desde window_start para eje X temporal (solo lectura CSV)."""
    out = df.copy()
    if "window_start" not in out.columns:
        out["_plot_time"] = pd.NaT
        return out
    ws = out["window_start"]
    if pd.api.types.is_numeric_dtype(ws):
        num = pd.to_numeric(ws, errors="coerce")
        med = float(num.median()) if num.notna().any() else float("nan")
        if med > 1e12:
            ts = pd.to_datetime(num, unit="ms", errors="coerce")
        elif med > 1e9:
            ts = pd.to_datetime(num, unit="s", errors="coerce")
        else:
            ts = pd.to_datetime(ws, errors="coerce")
    else:
        ts = pd.to_datetime(ws, errors="coerce")
    lo, hi = pd.Timestamp("2010-01-01"), pd.Timestamp("2035-12-31")
    ts = ts.where((ts >= lo) & (ts <= hi), pd.NaT)
    out["_plot_time"] = ts
    return out


def _plot_time_axis_usable(dfr: pd.DataFrame, *, min_valid_frac: float = 0.85) -> bool:
    """True solo si la mayoria de filas tiene tiempo parseable y hay mas de un instante distinto."""
    if dfr.empty or "_plot_time" not in dfr.columns:
        return False
    s = dfr["_plot_time"]
    n_ok = int(s.notna().sum())
    if n_ok == 0 or len(s) == 0:
        return False
    if n_ok / len(s) < min_valid_frac:
        return False
    if s.isna().any():
        return False
    nu = s.dropna().nunique()
    if len(s) > 1 and nu <= 1:
        return False
    return True


def _sort_for_time_series_plots(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_plot_time_column(df)
    if _plot_time_axis_usable(d):
        if "window_id" in d.columns:
            return d.sort_values(["_plot_time", "window_id"], na_position="last")
        return d.sort_values("_plot_time", na_position="last")
    if "window_id" in d.columns:
        return d.sort_values("window_id")
    return d


def _plot_time_x(dfr: pd.DataFrame) -> pd.Series:
    """Serie alineada al indice de dfr: tiempo si hay window_start util; si no, window_id."""
    if _plot_time_axis_usable(dfr):
        return dfr["_plot_time"]
    if "window_id" in dfr.columns:
        return dfr["window_id"]
    return pd.Series(np.arange(len(dfr)), index=dfr.index, dtype="float64")


def _plot_xaxis_layout(dfr: pd.DataFrame) -> dict[str, Any]:
    if _plot_time_axis_usable(dfr):
        return {
            "title": "Tiempo (inicio de ventana)",
            "type": "date",
            "tickformat": "%d %b %H:%M",
        }
    return {"title": "window_id"}


def kpi_cards(kpis: pd.DataFrame) -> html.Div:
    keys = [
        ("n_rows_raw", "Filas crudas"),
        ("n_windows_total", "Ventanas totales"),
        ("n_windows_modeling", "Ventanas modelado"),
        ("best_model_name", "Modelo seleccionado"),
        ("best_model_f1_macro_cv", "F1 macro (CV)"),
        ("best_model_balanced_accuracy_cv", "Exactitud balanceada (CV)"),
        ("permanova_r2", "PERMANOVA R\u00b2"),
        ("top_component", "Top componente (interpretabilidad)"),
        ("top_family", "Top familia (interpretabilidad)"),
        ("condition_index_mean", "Indice condicion (media)"),
        ("assessment_method", "Metodo assessment"),
    ]
    cards = []
    for key, label in keys:
        val = kpi_lookup(kpis, key)
        disp = "" if val is None or (isinstance(val, float) and np.isnan(val)) else dash_format_number(val)
        cards.append(
            html.Div(
                [
                    html.Div(label, className="label"),
                    html.Div(disp, className="value"),
                ],
                className="kpi-card",
            )
        )
    return html.Div(cards, className="kpi-row")


def kpi_cards_threshold(kpis: pd.DataFrame) -> html.Div:
    keys = [
        ("threshold_method_global", "Metodo umbral global"),
        ("threshold_method_by_batch", "Metodo umbral por batch"),
        ("condition_index_mean_thresholded_global", "Indice medio (umbral global)"),
        ("condition_index_mean_thresholded_by_batch", "Indice medio (umbral por batch)"),
        ("n_thresholds_by_batch_rows", "Filas umbrales por batch"),
        ("n_threshold_warnings_by_batch", "Advertencias umbrales por batch"),
    ]
    cards = []
    for key, label in keys:
        val = kpi_lookup(kpis, key)
        disp = "" if val is None or (isinstance(val, float) and np.isnan(val)) else dash_format_number(val)
        cards.append(
            html.Div(
                [
                    html.Div(label, className="label"),
                    html.Div(disp, className="value"),
                ],
                className="kpi-card",
            )
        )
    return html.Div(cards, className="kpi-row")


def kpi_cards_condition_state(kpis: pd.DataFrame) -> html.Div:
    """KPIs categoria condition_state (Etapa 9C); mismos colores que estado del activo en Assessment."""
    keys = [
        ("current_window_id", "Ultima ventana historica exportada"),
        ("current_batch_real", "Batch real"),
        ("current_batch_predicted", "Crudo predicho"),
        ("current_classification_confidence", "Confianza del clasificador"),
        ("current_condition_index", "Indice de condicion"),
        ("current_health_index", "EPI_BPC / salud relativa"),
        ("current_condition_state", "Estado exploratorio"),
        ("current_trend_direction", "Tendencia reciente"),
        ("n_active_attention_alerts", "Alertas exploratorias attention"),
        ("n_active_high_alerts", "Alertas exploratorias high"),
        ("top_condition_variable", "Variable principal de condicion"),
        ("top_condition_family", "Familia destacada (resumen KPI)"),
        ("top_condition_component", "Componente destacado (resumen KPI)"),
    ]
    cards = []
    for key, label in keys:
        val = kpi_lookup(kpis, key)
        disp = "" if val is None or (isinstance(val, float) and np.isnan(val)) else dash_format_number(val)
        cards.append(
            html.Div(
                [
                    html.Div(label, className="label"),
                    html.Div(disp, className="value"),
                ],
                className="kpi-card",
                style=_activo_kpi_card_style(label, disp, val),
            )
        )
    return html.Div(cards, className="kpi-row")


def build_alerts() -> list:
    out = []
    if DATA["_missing_critical"]:
        out.append(
            html.Div(
                [
                    html.Strong("Datos de dashboard incompletos."),
                    html.P(
                        "Ejecute primero: python run_pipeline.py --stage dashboard_exports",
                        style={"margin": "0.5rem 0 0 0"},
                    ),
                    html.Ul([html.Li(x) for x in DATA["_missing_critical"]]),
                ],
                className="alert-critical",
            )
        )
    if DATA["_missing_optional"]:
        out.append(
            html.Div(
                [
                    html.Strong("Advertencia: faltan archivos opcionales."),
                    html.Ul([html.Li(x) for x in DATA["_missing_optional"]]),
                ],
                className="alert-warn",
            )
        )
    hw = DATA.get("_health_warnings") or []
    if hw:
        out.append(
            html.Div(
                [
                    html.Strong("Validacion datos threshold / lineas base (no bloqueante)."),
                    html.Ul([html.Li(x) for x in hw]),
                ],
                className="alert-warn",
            )
        )
    csw = DATA.get("_condition_state_warnings") or []
    if csw:
        out.append(
            html.Div(
                [
                    html.Strong("Estado de condicion exportado (avisos de datos, no bloqueante)."),
                    html.Ul([html.Li(x) for x in csw]),
                ],
                className="alert-warn",
            )
        )
    ow = DATA.get("_operational_10d_warnings") or []
    if ow:
        out.append(
            html.Div(
                [
                    html.Strong("Indices operacionales y linea base predicha (avisos de datos, no bloqueante)."),
                    html.Ul([html.Li(x) for x in ow]),
                ],
                className="alert-warn",
            )
        )
    return out


def tab_resumen() -> html.Div:
    kpis = DATA["kpis"]
    pred = DATA["predictions"]
    cb = DATA["cond_batch"]
    best = str(kpi_lookup(kpis, "best_model_name") or "xgboost")
    children: list = [
        html.H3("Resumen ejecutivo"),
        html.P(
            "Este resumen integra tres lecturas: clasificacion del crudo trasegado, desempeno del modelo y condicion relativa del activo. "
            "El estado mostrado corresponde a la ultima ventana historica exportada, no a una lectura en vivo.",
            className="note",
            style={"marginBottom": "0.75rem"},
        ),
        html.Div(
            [
                html.Img(
                    src=_dash_asset("equipo_bpc.png"),
                    alt="Representación del equipo: estación de bombeo BPC",
                    className="tab-hero-equipo",
                ),
                html.P("Activo analizado: estación de bombeo BPC (vista esquemática).", className="tab-hero-caption"),
            ],
            className="tab-hero-wrap",
        ),
        kpi_cards(kpis),
        kpi_cards_threshold(kpis),
        html.H4("Estado exploratorio (ultima ventana historica exportada)"),
        kpi_cards_condition_state(kpis),
        html.P(
            "La variable principal de condicion y la familia destacada en el KPI son agregados del resumen exportado; "
            "pueden diferir del primer driver en JSON por reglas de agregacion distintas.",
            className="note",
            style={"fontSize": "0.85rem", "marginTop": "0.35rem"},
        ),
    ]
    pred_op_rs = DATA.get("predictions_operational", pd.DataFrame())
    if not pred_op_rs.empty and _epi_bpc_y_column(pred_op_rs) is not None:
        children.extend(
            [
                html.H4("EPI_BPC — salud relativa (resumen)"),
                html.P(
                    "EPI_BPC = 100 - condition_index. Valores altos indican mejor condicion relativa frente a la linea base seleccionada; "
                    "valores bajos de EPI corresponden a mayor severidad relativa (mayor condition_index).",
                    className="note",
                ),
                html.P(
                    "El EPI_BPC usa la linea base del crudo predicho cuando la confianza del modelo es suficiente "
                    "(compuerta exploratoria en la app); si no, se usa baseline global o se marca el assessment como incierto.",
                    className="note",
                ),
                html.P(
                    "Misma serie que en Assessment (salud operacional si aplica; bandas visuales exploratorias, no normativas).",
                    className="note",
                ),
                dcc.Graph(figure=fig_epi_bpc_picadora_style(pred_op_rs)),
            ]
        )
    children.extend(
        [
        html.Div(
            [
                html.P(
                    "El estado actual corresponde a la ultima ventana historica exportada. "
                    "Las alertas son exploratorias, calculadas con lineas base por batch derivadas de datos; "
                    "no son alarmas normativas ni diagnostico de falla."
                ),
            ],
            className="note",
            style={"marginBottom": "0.75rem"},
        ),
        html.Div(
            [
                html.P(
                    f"Modelo seleccionado (segun KPI exportados): {best}. "
                    "Las metricas de generalizacion mostradas como F1 macro y exactitud balanceada "
                    "provienen de validacion cruzada temporal, no del ajuste final sobre todo el conjunto."
                ),
                html.P(
                    "Los batches CASTILLA, MEZCLA y RUBIALES son etiquetas operacionales; cada uno se asocia a una firma vibratoria distinta en ventanas."
                ),
                html.P(
                    "En la pestaña Predicciones, la serie temporal de indices superpuestos muestra por defecto "
                    "severidad con umbral por batch y EPI / salud relativa por batch. "
                    "Los umbrales V0/H/HH son exploratorios, estimados como P40/P75/P99, "
                    "y no constituyen limites normativos de alarma."
                ),
                html.P(
                    "El assessment de condicion incluye robust_percentile_fallback y variantes con "
                    "umbrales data-driven cuando los CSV correspondientes estan disponibles."
                ),
                html.P(
                    "Los pesos ponderados describen contribucion al indice de condicion; no representan importancia ML del clasificador."
                ),
            ],
            className="note",
        ),
        ]
    )
    if not pred.empty and "Batch" in pred.columns:
        vc = pred["Batch"].astype(str).value_counts().reindex(CLASS_ORDER, fill_value=0).reset_index()
        vc.columns = ["Batch", "n_windows"]
        vc_plot = vc.assign(_n=pd.to_numeric(vc["n_windows"], errors="coerce")).sort_values(
            ["_n", "Batch"], ascending=[False, True], na_position="last"
        )
        fig1 = px.bar(
            vc_plot,
            x="Batch",
            y="n_windows",
            title="Ventanas por batch (predicciones exportadas)",
            color="Batch",
            color_discrete_map={k: CLASS_COLORS[k] for k in CLASS_ORDER},
        )
        fig1.update_layout(
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                categoryorder="array",
                categoryarray=_bar_sorted_categoryarray(vc_plot, "Batch", "n_windows", largest_first=True),
            ),
        )
        fig1.update_traces(hovertemplate="Batch=%{x}<br>n ventanas=%{y:.0f}<extra></extra>")
        children.append(dcc.Graph(figure=fig1))
    else:
        children.append(dcc.Graph(figure=fig_empty("Sin datos de predicciones")))
    if not cb.empty and "condition_index_mean" in cb.columns:
        cb_plot = cb.assign(_m=pd.to_numeric(cb["condition_index_mean"], errors="coerce")).sort_values(
            ["_m", "Batch"], ascending=[False, True], na_position="last"
        )
        fig2 = px.bar(
            cb_plot,
            x="Batch",
            y="condition_index_mean",
            title="Indice de condicion medio por batch",
            color="Batch",
            color_discrete_map={k: CLASS_COLORS[k] for k in CLASS_ORDER},
        )
        fig2.update_layout(
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                categoryorder="array",
                categoryarray=_bar_sorted_categoryarray(cb_plot, "Batch", "condition_index_mean", largest_first=True),
            ),
        )
        fig2.update_traces(hovertemplate="Batch=%{x}<br>indice medio=%{y:.2f}<extra></extra>")
        children.append(dcc.Graph(figure=fig2))
    else:
        children.append(dcc.Graph(figure=fig_empty("Sin datos condition_index_by_batch")))
    return html.Div(children, className="tab-panel")


def tab_modelo() -> html.Div:
    mm = DATA["model_metrics"]
    conf = DATA["confusion"]
    kpis = DATA["kpis"]
    if mm.empty:
        return html.Div(
            [html.H3("Desempeno del modelo"), html.P("Sin dashboard_model_metrics.csv")],
            className="tab-panel",
        )
    ord_f1 = mm.sort_values(["f1_macro_mean", "model_name"], ascending=[True, True], na_position="last")
    fig_f1 = px.bar(
        ord_f1,
        x="f1_macro_mean",
        y="model_name",
        orientation="h",
        title="F1 macro en validacion cruzada temporal",
    )
    fig_f1.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=_bar_sorted_categoryarray(
                ord_f1, "model_name", "f1_macro_mean", largest_first=False
            ),
        ),
    )
    fig_f1.update_traces(hovertemplate="%{y}<br>F1 macro=%{x:.2f}<extra></extra>")
    ord_ba = mm.sort_values(["balanced_accuracy_mean", "model_name"], ascending=[True, True], na_position="last")
    fig_ba = px.bar(
        ord_ba,
        x="balanced_accuracy_mean",
        y="model_name",
        orientation="h",
        title="Exactitud balanceada en validacion cruzada temporal",
    )
    fig_ba.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=_bar_sorted_categoryarray(
                ord_ba, "model_name", "balanced_accuracy_mean", largest_first=False
            ),
        ),
    )
    fig_ba.update_traces(hovertemplate="%{y}<br>Exactitud balanceada=%{x:.2f}<extra></extra>")
    mat, labels = confusion_matrix_normalized(conf)
    fig_hm = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=[f"Pred {x}" for x in labels],
            y=[f"Real {x}" for x in labels],
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=np.round(mat, DASH_DECIMALS),
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>fraccion=%{z:.2f}<extra></extra>",
        )
    )
    fig_hm.update_layout(
        title="Matriz de confusion del modelo seleccionado (fraccion por fila: clase real)",
        paper_bgcolor="white",
    )
    best = str(kpi_lookup(kpis, "best_model_name") or "xgboost")
    sub_best = mm.loc[mm["model_name"] == best, "f1_macro_mean"]
    f1_best = float(sub_best.iloc[0]) if len(sub_best) else float("nan")
    f1_sv = float(mm.loc[mm["model_name"] == "soft_voting_ensemble", "f1_macro_mean"].iloc[0])
    diff = abs(f1_best - f1_sv)
    train_f1 = kpi_lookup(kpis, "training_f1_macro")
    note = html.Div(
        [
            html.P(
                "Esta pestaña compara los modelos entrenados mediante validacion cruzada temporal. "
                "Las metricas reportadas aqui son las que deben usarse para evaluar generalizacion; "
                "las metricas del entrenamiento final solo indican ajuste sobre el conjunto completo."
            ),
            html.P(
                f"El modelo seleccionado para despliegue reportado en los KPI es {best}. "
                "La eleccion prioriza F1 macro y exactitud balanceada en validacion cruzada temporal."
            ),
            html.P(
                f"La diferencia frente a soft_voting_ensemble en F1 macro es marginal (aprox. {diff:.2f})."
            ),
            html.P(
                f"El KPI training_f1_macro = {dash_format_number(train_f1)} corresponde al ajuste sobre las ventanas de entrenamiento "
                "final; no debe interpretarse como metrica de generalizacion."
            ),
        ],
        className="note",
    )
    tbl = dash_table.DataTable(
        columns=[{"name": MODEL_METRICS_COLUMN_LABELS.get(c, c), "id": c} for c in mm.columns],
        data=round_numeric_df(mm).to_dict("records"),
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "0.85rem"},
        style_header={"fontWeight": "600"},
    )
    return html.Div(
        [
            html.H3("Desempeno del modelo"),
            note,
            html.Div([dcc.Graph(figure=fig_f1), dcc.Graph(figure=fig_ba)], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1rem"}),
            dcc.Graph(figure=fig_hm),
            html.H4("Metricas por modelo (export)"),
            tbl,
        ],
        className="tab-panel",
    )


def tab_predicciones() -> html.Div:
    pred = DATA["predictions"]
    if pred.empty:
        return html.Div(
            [
                html.H3("Predicciones por ventana"),
                html.P(
                    "Cada punto representa una ventana temporal agregada. Las predicciones indican el batch clasificado por el modelo "
                    "y se complementan con confianza, margen entre clases e indices de condicion.",
                    className="note",
                ),
                html.P("Sin datos. Ejecute: python run_pipeline.py --stage dashboard_exports"),
                html.Div(
                    [
                        dcc.Dropdown(id="flt-batch", options=[{"label": "(todos)", "value": "(todos)"}], value="(todos)", style={"display": "none"}),
                        dcc.Dropdown(id="flt-pred", options=[{"label": "(todos)", "value": "(todos)"}], value="(todos)", style={"display": "none"}),
                        dcc.Dropdown(id="flt-ok", options=[{"label": "all", "value": "all"}], value="all", style={"display": "none"}),
                        dcc.RangeSlider(id="flt-wid", min=0, max=1, step=1, value=[0, 1], style={"display": "none"}),
                        dcc.Checklist(
                            id="pred-index-series",
                            options=[
                                    {"label": "Severidad por batch (CSV)", "value": "by_batch"},
                                    {"label": "EPI / salud relativa por batch (CSV)", "value": "health_batch"},
                            ],
                            value=list(PRED_INDEX_SERIES_DEFAULT),
                            style={"display": "none"},
                        ),
                    ]
                ),
                dcc.Graph(id="pred-graph-scatter"),
                html.Div(
                    [dcc.Graph(id="pred-graph-conf", style={"flex": "1"}), dcc.Graph(id="pred-graph-ci", style={"flex": "1"})],
                    style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
                ),
                dcc.Graph(id="pred-graph-count"),
                html.H4("Indices con linea base por crudo predicho"),
                html.P(
                    "El indice operacional usa la linea base del crudo predicho cuando la confianza del clasificador >= 0.80 "
                    "y el margen entre las dos clases mas probables >= 0.15; "
                    "si no, usa baseline global. EPI: puntos por clase predicha; severidad: bandas alineadas al complemento de EPI 60/80 "
                    "(0-20 / 20-40 / 40-100 en severidad), trazas segmentadas y media movil / suavizado. "
                    "Todo es exploratorio (no normativo).",
                    className="note",
                    style={"fontSize": "0.85rem"},
                ),
                dcc.Graph(id="op-health-graph", figure=fig_empty("Sin datos")),
                dcc.Graph(id="op-cond-graph", figure=fig_empty("Sin datos")),
                html.H4("Tabla filtrada"),
                html.Div(id="pred-table-container"),
            ],
            className="tab-panel",
        )
    batches = ["(todos)"] + [x for x in CLASS_ORDER if x in pred["Batch"].astype(str).unique()]
    preds = ["(todos)"] + sorted(pred["y_pred_label"].astype(str).unique().tolist())
    wmin, wmax = int(pred["window_id"].min()), int(pred["window_id"].max())
    return html.Div(
        [
            html.H3("Predicciones por ventana"),
            html.P(
                "Cada punto representa una ventana temporal agregada. Las predicciones indican el batch clasificado por el modelo "
                "y se complementan con confianza, margen entre clases e indices de condicion.",
                className="note",
                style={"marginBottom": "0.65rem"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Batch real"),
                            dcc.Dropdown(id="flt-batch", options=[{"label": x, "value": x} for x in batches], value="(todos)", clearable=False),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Clase predicha"),
                            dcc.Dropdown(id="flt-pred", options=[{"label": x, "value": x} for x in preds], value="(todos)", clearable=False),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Solo aciertos / errores"),
                            dcc.Dropdown(
                                id="flt-ok",
                                options=[
                                    {"label": "(todos)", "value": "all"},
                                    {"label": "Solo aciertos", "value": "True"},
                                    {"label": "Solo errores", "value": "False"},
                                ],
                                value="all",
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Rango de ventanas"),
                            dcc.RangeSlider(id="flt-wid", min=wmin, max=wmax, step=1, value=[wmin, wmax], marks=None, tooltip={"placement": "bottom"}),
                        ],
                        style={"minWidth": "280px", "flex": "1 1 280px"},
                    ),
                    html.Div(
                        [
                            html.Label("Series en grafico de indices"),
                            dcc.Checklist(
                                id="pred-index-series",
                                options=[
                                    {"label": "Severidad por batch (CSV)", "value": "by_batch"},
                                    {"label": "EPI / salud relativa por batch (CSV)", "value": "health_batch"},
                                ],
                                value=list(PRED_INDEX_SERIES_DEFAULT),
                                inline=True,
                            ),
                        ],
                        style={"flex": "1 1 320px"},
                    ),
                ],
                className="filter-row",
            ),
            html.Div(
                [
                    html.P(
                        "El filtro por rango usa el identificador de ventana (window_id). Las graficas de serie usan el tiempo de inicio de ventana (window_start) "
                        "en el eje horizontal cuando esta disponible. "
                        "La columna linea base usada refleja el batch historico real en este analisis. "
                        "En analisis historico se dispone del batch real. En operacion futura, la linea base por batch debe seleccionarse con la clase predicha "
                        "solo si la confianza es suficiente.",
                        className="note",
                        style={"fontSize": "0.85rem", "padding": "0.5rem", "margin": 0},
                    ),
                ],
                style={"marginBottom": "0.5rem"},
            ),
            dcc.Graph(id="pred-graph-scatter"),
            html.Div(
                [
                    dcc.Graph(id="pred-graph-conf", style={"flex": "1"}),
                    dcc.Graph(id="pred-graph-ci", style={"flex": "1"}),
                ],
                style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
            ),
            dcc.Graph(id="pred-graph-count"),
            html.H4("Indices con linea base por crudo predicho"),
            html.P(
                "El indice operacional usa la linea base del crudo predicho cuando la confianza del clasificador >= 0.80 "
                "y el margen entre las dos clases mas probables >= 0.15; "
                "si no, usa baseline global. EPI: puntos por clase predicha; severidad: bandas alineadas al complemento de EPI 60/80 "
                "(0-20 / 20-40 / 40-100 en severidad), trazas segmentadas y media movil / suavizado. "
                "Todo es exploratorio (no normativo).",
                className="note",
                style={"fontSize": "0.85rem"},
            ),
            dcc.Graph(id="op-health-graph", figure=fig_empty("Cargando...")),
            dcc.Graph(id="op-cond-graph", figure=fig_empty("Cargando...")),
            html.H4("Tabla filtrada"),
            html.Div(id="pred-table-container"),
        ],
        className="tab-panel",
    )


def tab_estadistica() -> html.Div:
    kpis = DATA["kpis"]
    pp = DATA["pairwise_perm"]
    pdisp = DATA["pairwise_disp"]
    pca_c = DATA["pca_centroids"]
    pca_pts = DATA["pca_projection"]
    cards = html.Div(
        [
            html.Div(
                [
                    html.Div("PERMANOVA R\u00b2", className="label"),
                    html.Div(dash_format_number(kpi_lookup(kpis, "permanova_r2")), className="value"),
                ],
                className="kpi-card",
            ),
            html.Div(
                [
                    html.Div("p-value PERMANOVA", className="label"),
                    html.Div(dash_format_number(kpi_lookup(kpis, "permanova_p_value")), className="value"),
                ],
                className="kpi-card",
            ),
            html.Div(
                [
                    html.Div("PERMDISP F", className="label"),
                    html.Div(dash_format_number(kpi_lookup(kpis, "permdisp_F")), className="value"),
                ],
                className="kpi-card",
            ),
            html.Div(
                [
                    html.Div("p-value PERMDISP", className="label"),
                    html.Div(dash_format_number(kpi_lookup(kpis, "permdisp_p_value")), className="value"),
                ],
                className="kpi-card",
            ),
            html.Div(
                [html.Div("Varianza explicada PC1", className="label"), html.Div(dash_format_number(kpi_lookup(kpis, "pca_explained_variance_PC1")), className="value")],
                className="kpi-card",
            ),
            html.Div(
                [html.Div("Varianza explicada PC2", className="label"), html.Div(dash_format_number(kpi_lookup(kpis, "pca_explained_variance_PC2")), className="value")],
                className="kpi-card",
            ),
        ],
        className="kpi-row",
    )
    children = [html.H3("Validacion estadistica"), cards]
    children.append(
        html.P(
            "Las pruebas estadisticas evaluan si las firmas vibratorias asociadas a CASTILLA, MEZCLA y RUBIALES presentan "
            "diferencias multivariadas. Estas pruebas no infieren composicion quimica del crudo.",
            className="note",
            style={"marginBottom": "0.65rem"},
        )
    )
    children.append(
        html.Div(
            [
                html.P("PERMANOVA evalua diferencias entre centroides multivariados."),
                html.P("PERMDISP evalua diferencias de dispersion interna."),
                html.P("Un resultado significativo no implica causalidad ni composicion quimica."),
            ],
            className="note",
        )
    )
    if not pp.empty and "r2" in pp.columns:
        pp2 = pp.copy()
        pp2["par"] = pp2["class_a"].astype(str) + " vs " + pp2["class_b"].astype(str)
        idx_min = pp2["r2"].astype(float).idxmin()
        par_min = str(pp2.loc[idx_min, "par"])
        r2_min = float(pp2.loc[idx_min, "r2"])
        children.append(html.P(f"Par con menor R2 pairwise PERMANOVA en este dataset: {par_min} (R2 ~ {r2_min:.2f})."))
        pp2s = pp2.sort_values(["r2", "par"], ascending=[False, True], na_position="last")
        fig_r2 = px.bar(pp2s, x="par", y="r2", title="R2 pairwise PERMANOVA")
        fig_r2.update_traces(marker_color="#4472a8", hovertemplate="%{x}<br>R2=%{y:.2f}<extra></extra>")
        fig_r2.update_layout(
            paper_bgcolor="white",
            height=380,
            xaxis_tickangle=-25,
            showlegend=False,
            xaxis=dict(
                categoryorder="array",
                categoryarray=_bar_sorted_categoryarray(pp2s, "par", "r2", largest_first=True),
            ),
        )
        children.append(dcc.Graph(figure=fig_r2))
    if not pdisp.empty and "mean_disp_a" in pdisp.columns:
        d2 = pdisp.copy()
        d2["par"] = d2["class_a"].astype(str) + " vs " + d2["class_b"].astype(str)
        d2["_disp_max"] = d2[["mean_disp_a", "mean_disp_b"]].max(axis=1)
        par_order = (
            d2.groupby("par", as_index=False)["_disp_max"]
            .max()
            .sort_values(["_disp_max", "par"], ascending=[False, True], na_position="last")["par"]
            .astype(str)
            .tolist()
        )
        d2m = d2.drop(columns="_disp_max").melt(
            id_vars=["par"],
            value_vars=["mean_disp_a", "mean_disp_b"],
            var_name="serie",
            value_name="dispersion",
        )
        fig_d = px.bar(
            d2m,
            x="par",
            y="dispersion",
            color="serie",
            barmode="group",
            title="Dispersion media por clase (pairwise PERMDISP)",
        )
        fig_d.update_layout(
            paper_bgcolor="white",
            height=400,
            xaxis_tickangle=-25,
            xaxis=dict(categoryorder="array", categoryarray=par_order),
        )
        fig_d.update_traces(hovertemplate="%{x}<br>%{fullData.name}<br>valor=%{y:.2f}<extra></extra>")
        children.append(dcc.Graph(figure=fig_d))
    _pca_ok = (
        not pca_pts.empty
        and {"PC1", "PC2", "Batch"}.issubset(set(pca_pts.columns))
        and pca_pts["PC1"].notna().any()
        and pca_pts["PC2"].notna().any()
    )
    if _pca_ok:
        batches_plot = [b for b in CLASS_ORDER if b in set(pca_pts["Batch"].astype(str))] + sorted(
            set(pca_pts["Batch"].astype(str)) - set(CLASS_ORDER)
        )
        cmap = {k: CLASS_COLORS.get(k, "#888") for k in batches_plot}
        fig_pca = px.scatter(
            pca_pts,
            x="PC1",
            y="PC2",
            color="Batch",
            category_orders={"Batch": batches_plot},
            color_discrete_map=cmap,
            title="Proyeccion PCA (PC1 vs PC2): todas las ventanas por Batch",
        )
        fig_pca.update_traces(
            marker=dict(size=6, opacity=0.55, line=dict(width=0)),
            hovertemplate="Batch=%{fullData.name}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
        )
        if not pca_c.empty and {"PC1_centroid", "PC2_centroid", "Batch"}.issubset(set(pca_c.columns)):
            fig_pca.add_trace(
                go.Scatter(
                    x=pca_c["PC1_centroid"].astype(float),
                    y=pca_c["PC2_centroid"].astype(float),
                    mode="markers+text",
                    text=pca_c["Batch"].astype(str),
                    textposition="top center",
                    name="Centroides (media por batch)",
                    marker=dict(
                        size=13,
                        symbol="diamond-open",
                        color=[CLASS_COLORS.get(str(b), "#333333") for b in pca_c["Batch"].astype(str)],
                        line=dict(width=2, color="#222222"),
                    ),
                    hovertemplate="%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra>Centroide</extra>",
                )
            )
        fig_pca.update_layout(paper_bgcolor="white", plot_bgcolor="white", legend_title_text="Batch")
        children.append(dcc.Graph(figure=fig_pca))
    elif not pca_c.empty:
        fig_pca = px.scatter(
            pca_c,
            x="PC1_centroid",
            y="PC2_centroid",
            text="Batch",
            color="Batch",
            color_discrete_map={k: CLASS_COLORS.get(k, "#888") for k in pca_c["Batch"].astype(str).unique()},
            title="Centroides PCA por Batch (sin archivo de proyeccion por ventana)",
        )
        fig_pca.update_traces(textposition="top center", hovertemplate="%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>")
        fig_pca.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        children.append(dcc.Graph(figure=fig_pca))
    children.append(html.H4("Pairwise PERMANOVA"))
    children.append(_df_table(pp))
    children.append(html.H4("Pairwise PERMDISP"))
    children.append(_df_table(pdisp))
    return html.Div(children, className="tab-panel")


def _df_table(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.P("Sin datos")
    rdf = round_numeric_df(df)
    return html.Div(
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in rdf.columns],
            data=rdf.to_dict("records"),
            page_size=8,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "0.8rem", "padding": "6px"},
        )
    )


def tab_interpretabilidad() -> html.Div:
    fi = DATA["feature_importance"]
    rc = DATA["rank_component"]
    rf = DATA["rank_family"]
    rp = DATA["rank_position"]
    rs = DATA["rank_statistic"]
    children = [
        html.H3("Interpretabilidad"),
        html.Div(
            [
                html.P(
                    "La interpretabilidad muestra que variables ayudan al modelo a distinguir los batches. "
                    "Estos resultados explican el comportamiento del clasificador; no son causalidad fisica ni pesos de condicion del activo."
                ),
                html.P(
                    "SHAP y permutation importance describen contribuciones al modelo entrenado; "
                    "no implican causalidad ni mecanismos fisicos directos."
                ),
                html.P(
                    "Los rankings deben leerse desde los datos exportados en cada ejecucion; no sustituyen revision tecnica ni inferencia causal."
                ),
            ],
            className="note",
        ),
    ]
    if not fi.empty and "consolidated_score" in fi.columns:
        top = fi.sort_values("consolidated_score", ascending=False).head(20).copy()
        top["consolidated_score"] = pd.to_numeric(top["consolidated_score"], errors="coerce").round(DASH_DECIMALS)
        top["label_short"] = top["feature"].astype(str).map(abbrev_label)
        # Eje Y: menor score abajo, mayor arriba (Plotly no garantiza orden de filas sin categoryarray).
        top_plot = top.sort_values(["consolidated_score", "feature"], ascending=[True, True], na_position="last").copy()
        top_plot["y_cat"] = _unique_bar_category_labels(top_plot, "label_short", "feature")
        fig_top = px.bar(
            top_plot,
            x="consolidated_score",
            y="y_cat",
            orientation="h",
            title="Top 20 variables del ranking consolidado de interpretabilidad",
            hover_data={"feature": True, "raw_variable": True},
        )
        fig_top.update_layout(
            paper_bgcolor="white",
            height=520,
            yaxis_title="",
            yaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=_bar_sorted_categoryarray(
                    top_plot, "y_cat", "consolidated_score", largest_first=False, tiebreak_cols=("feature",)
                ),
            ),
        )
        fig_top.update_traces(hovertemplate="%{customdata[0]}<br>consolidated_score=%{x:.2f}<extra></extra>")
        children.append(dcc.Graph(figure=fig_top))
        children.append(html.H4("Top 20 (tabla exportada)"))
        show_cols = [c for c in ["consolidated_rank", "feature", "raw_variable", "consolidated_score", "family", "component"] if c in top.columns]
        children.append(
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in show_cols],
                data=round_numeric_df(top[show_cols]).to_dict("records"),
                page_size=10,
                style_cell={"fontSize": "0.78rem", "padding": "5px"},
                style_table={"overflowX": "auto"},
            )
        )
    for title, df, xcol in [
        ("Ranking agregado de interpretabilidad por componente", rc, "share_percent"),
        ("Ranking agregado de interpretabilidad por familia", rf, "share_percent"),
        ("Ranking agregado de interpretabilidad por posicion", rp, "share_percent"),
        ("Ranking agregado de interpretabilidad por estadistico", rs, "share_percent"),
    ]:
        if not df.empty and xcol in df.columns and df.columns[0]:
            cat = df.columns[0]
            df_b = df.assign(_xv=pd.to_numeric(df[xcol], errors="coerce")).sort_values(
                ["_xv", cat], ascending=[False, True], na_position="last"
            )
            fig = px.bar(df_b.drop(columns="_xv"), x=cat, y=xcol, title=title, color=cat)
            fig.update_layout(
                showlegend=False,
                paper_bgcolor="white",
                height=340,
                xaxis=dict(
                    categoryorder="array",
                    categoryarray=_bar_sorted_categoryarray(df_b, cat, xcol, largest_first=True),
                ),
            )
            fig.update_traces(hovertemplate=f"%{{x}}<br>{xcol}=%{{y:.2f}}<extra></extra>")
            children.append(dcc.Graph(figure=fig))
    return html.Div(children, className="tab-panel")


def build_three_method_comparison_df() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    cb = DATA["cond_batch"]
    if not cb.empty and "Batch" in cb.columns and "condition_index_mean" in cb.columns:
        t = cb[["Batch", "condition_index_mean"]].copy()
        t["assessment_mode"] = "Fallback robusto"
        parts.append(t)
    cg = DATA["ci_g_batch"]
    if not cg.empty and "Batch" in cg.columns and "condition_index_mean" in cg.columns:
        t = cg[["Batch", "condition_index_mean"]].copy()
        t["assessment_mode"] = "Threshold global"
        parts.append(t)
    bb = DATA["ci_b_batch"]
    if not bb.empty and "Batch" in bb.columns and "condition_index_mean" in bb.columns:
        t = bb[["Batch", "condition_index_mean"]].copy()
        t["assessment_mode"] = "Threshold por batch"
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def fig_condition_indices_multi(dfp: pd.DataFrame, selected: list[str] | None) -> go.Figure:
    sel = list(selected or [])
    dfp = _sort_for_time_series_plots(dfp)
    xx = _plot_time_x(dfp)
    use_t = _plot_time_axis_usable(dfp)
    fig = go.Figure()
    series_map = [
        ("fallback", "condition_index", "Fallback robusto", "#9467bd"),
        ("global", "condition_index_thresholded_global", "Threshold global", "#2ca02c"),
        ("by_batch", "condition_index_thresholded_by_batch", "Severidad umbral por batch (CSV)", "#ff7f0e"),
        ("health_batch", "health_index_thresholded_by_batch", "EPI / salud relativa por batch (CSV)", "#1f77b4"),
    ]
    wid_cd = dfp["window_id"].values if "window_id" in dfp.columns else None
    for key, col, name, color in series_map:
        if key in sel and col in dfp.columns:
            ht = (
                "window_id=%{customdata}<br>tiempo=%{x|%Y-%m-%d %H:%M}<br>" + name + "=%{y:.2f}<extra></extra>"
                if use_t
                else "window_id=%{x}<br>" + name + "=%{y:.2f}<extra></extra>"
            )
            fig.add_trace(
                go.Scatter(
                    x=xx,
                    y=dfp[col],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1.8),
                    customdata=wid_cd,
                    hovertemplate=ht,
                )
            )
    if not fig.data:
        return fig_empty("Seleccione indices disponibles o verifique columnas threshold en predictions")
    fig.update_layout(
        title="Indices de condicion y EPI por ventana (tiempo de inicio)" if use_t else "Indices de condicion y EPI vs window_id",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
        xaxis=_plot_xaxis_layout(dfp),
    )
    return fig


def _epi_bpc_band_shapes() -> list[dict[str, Any]]:
    """Bandas visuales exploratorias estilo EPI (mayor Y = mejor)."""
    return [
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=60, fillcolor="rgba(214, 39, 40, 0.14)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=60, y1=80, fillcolor="rgba(230, 180, 50, 0.18)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=80, y1=100, fillcolor="rgba(44, 160, 44, 0.14)", line_width=0, layer="below"),
    ]


def _condition_index_bpc_severity_shapes() -> list[dict[str, Any]]:
    """Severidad en escala 0-100 como complemento de las bandas EPI 60/80: 100-EPI → cortes 20 y 40."""
    return [
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=20, fillcolor="rgba(44, 160, 44, 0.14)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=20, y1=40, fillcolor="rgba(230, 180, 50, 0.18)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=40, y1=100, fillcolor="rgba(214, 39, 40, 0.14)", line_width=0, layer="below"),
    ]


def _epi_bpc_y_column(pred_df: pd.DataFrame) -> str | None:
    if pred_df.empty:
        return None
    if "health_index_operational" in pred_df.columns:
        s = pd.to_numeric(pred_df["health_index_operational"], errors="coerce")
        if s.notna().any():
            return "health_index_operational"
    if "health_index_thresholded_by_batch" in pred_df.columns:
        return "health_index_thresholded_by_batch"
    return None


def fig_epi_bpc_picadora_style(pred_df: pd.DataFrame) -> go.Figure:
    """EPI_BPC: health operacional si existe; si no, health por batch thresholded. Bandas exploratorias tipo EPI picadora."""
    ycol = _epi_bpc_y_column(pred_df)
    if ycol is None:
        return fig_empty("Sin health_index_operational ni health_index_thresholded_by_batch en predicciones")
    d = _sort_for_time_series_plots(pred_df)
    xx = _plot_time_x(d)
    use_t = _plot_time_axis_usable(d)
    y_epi = pd.to_numeric(d[ycol], errors="coerce").clip(0, 100)
    ma60 = y_epi.rolling(60, min_periods=1).mean()
    sm4 = ma60.rolling(4, min_periods=1).mean()
    wid = d["window_id"].values
    yp = d["y_pred_label"].astype(str).values if "y_pred_label" in d.columns else np.repeat("?", len(d))
    conf = pd.to_numeric(d["confidence"], errors="coerce").values if "confidence" in d.columns else np.full(len(d), np.nan)
    marg = pd.to_numeric(d["margin_top2"], errors="coerce").values if "margin_top2" in d.columns else np.full(len(d), np.nan)
    ci_b = (
        pd.to_numeric(d["condition_index_thresholded_by_batch"], errors="coerce").values
        if "condition_index_thresholded_by_batch" in d.columns
        else np.full(len(d), np.nan)
    )
    cd = np.column_stack([wid, yp, conf, marg, ci_b])
    ht = (
        (
            "tiempo=%{x|%Y-%m-%d %H:%M}<br>window_id=%{customdata[0]}<br>y_pred_label=%{customdata[1]}<br>Confianza del clasificador=%{customdata[2]:.2f}<br>"
            "Margen entre las dos clases mas probables=%{customdata[3]:.2f}<br>EPI_BPC=%{y:.2f}<br>condition_index_thresholded_by_batch=%{customdata[4]:.2f}<extra></extra>"
        )
        if use_t
        else (
            "window_id=%{customdata[0]}<br>y_pred_label=%{customdata[1]}<br>Confianza del clasificador=%{customdata[2]:.2f}<br>"
            "Margen entre las dos clases mas probables=%{customdata[3]:.2f}<br>EPI_BPC=%{y:.2f}<br>condition_index_thresholded_by_batch=%{customdata[4]:.2f}<extra></extra>"
        )
    )
    fig = go.Figure()
    fig.update_layout(
        shapes=_epi_bpc_band_shapes(),
        title="EPI_BPC — salud relativa con linea base por crudo predicho",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=440,
        yaxis=dict(range=[0, 100], title="EPI_BPC / salud relativa (0-100)"),
        xaxis=_plot_xaxis_layout(d),
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01, bgcolor="rgba(255,255,255,0.92)", font=dict(size=11)),
        margin=dict(t=70, r=200, l=56, b=52),
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Rojo EPI < 60 (critico exploratorio)",
            marker=dict(size=11, color="rgba(214,39,40,0.55)", symbol="square"),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Amarillo 60 <= EPI <= 80 (observacion)",
            marker=dict(size=11, color="rgba(230,180,50,0.7)", symbol="square"),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Verde EPI > 80 (normal)",
            marker=dict(size=11, color="rgba(44,160,44,0.55)", symbol="square"),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    for tr in _segmented_scatter_lines_by_pred_batch(
        d,
        xx,
        y_epi,
        cd,
        ht,
        line_width=1.1,
        line_dash=None,
        legend_prefix="epi_inst",
        legend_label_base="EPI instantaneo (ventana)",
    ):
        fig.add_trace(tr)
    for tr in _segmented_scatter_lines_by_pred_batch(
        d,
        xx,
        ma60,
        cd,
        ht,
        line_width=2.1,
        line_dash=None,
        legend_prefix="epi_ma60",
        legend_label_base="EPI promedio movil (60 ventanas)",
    ):
        fig.add_trace(tr)
    for tr in _segmented_scatter_lines_by_pred_batch(
        d,
        xx,
        sm4,
        cd,
        ht,
        line_width=3.4,
        line_dash=None,
        legend_prefix="epi_sm4",
        legend_label_base="Suavizado (4 periodos)",
    ):
        fig.add_trace(tr)
    return fig


def fig_condition_index_bpc_bands(pred_df: pd.DataFrame) -> go.Figure:
    """Vista tecnica: severidad relativa con condition_index por batch (CSV); bandas exploratorias."""
    if pred_df.empty or "condition_index_thresholded_by_batch" not in pred_df.columns:
        return fig_empty("Sin condition_index_thresholded_by_batch en predicciones")
    d = _sort_for_time_series_plots(pred_df)
    xx = _plot_time_x(d)
    use_t = _plot_time_axis_usable(d)
    ci = pd.to_numeric(d["condition_index_thresholded_by_batch"], errors="coerce").clip(0, 100)
    ma60 = ci.rolling(60, min_periods=1).mean()
    sm4 = ma60.rolling(4, min_periods=1).mean()
    wid = d["window_id"].values
    yp = d["y_pred_label"].astype(str).values if "y_pred_label" in d.columns else np.repeat("?", len(d))
    conf = pd.to_numeric(d["confidence"], errors="coerce").values if "confidence" in d.columns else np.full(len(d), np.nan)
    marg = pd.to_numeric(d["margin_top2"], errors="coerce").values if "margin_top2" in d.columns else np.full(len(d), np.nan)
    hi_b = (
        pd.to_numeric(d["health_index_thresholded_by_batch"], errors="coerce").values
        if "health_index_thresholded_by_batch" in d.columns
        else np.full(len(d), np.nan)
    )
    cd = np.column_stack([wid, yp, conf, marg, hi_b])
    ht = (
        (
            "tiempo=%{x|%Y-%m-%d %H:%M}<br>window_id=%{customdata[0]}<br>y_pred_label=%{customdata[1]}<br>Confianza del clasificador=%{customdata[2]:.2f}<br>"
            "Margen entre las dos clases mas probables=%{customdata[3]:.2f}<br>condition_index_thr_batch=%{y:.2f}<br>health_index_thresholded_by_batch=%{customdata[4]:.2f}<extra></extra>"
        )
        if use_t
        else (
            "window_id=%{customdata[0]}<br>y_pred_label=%{customdata[1]}<br>Confianza del clasificador=%{customdata[2]:.2f}<br>"
            "Margen entre las dos clases mas probables=%{customdata[3]:.2f}<br>condition_index_thr_batch=%{y:.2f}<br>health_index_thresholded_by_batch=%{customdata[4]:.2f}<extra></extra>"
        )
    )
    fig = go.Figure()
    fig.update_layout(
        shapes=_condition_index_bpc_severity_shapes(),
        title="Condition Index — severidad relativa con linea base por crudo predicho",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=420,
        yaxis=dict(range=[0, 100], title="Condition index (severidad 0-100)"),
        xaxis=_plot_xaxis_layout(d),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            xref="paper",
            bgcolor="rgba(255,255,255,0.94)",
            font=dict(size=10),
            itemwidth=30,
        ),
        margin=dict(t=44, r=220, l=56, b=52),
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Verde 0-20 (baja severidad)",
            marker=dict(size=10, color="rgba(44,160,44,0.65)", symbol="square"),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Amarillo 20-40 (observacion)",
            marker=dict(size=10, color="rgba(230,180,50,0.75)", symbol="square"),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Rojo 40-100 (alta severidad)",
            marker=dict(size=10, color="rgba(214,39,40,0.65)", symbol="square"),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Lineas: color = crudo predicho (hover)",
            marker=dict(size=9, color="#666666", symbol="circle-open", line=dict(width=1.5, color="#666")),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    for tr in _segmented_scatter_lines_by_pred_batch(
        d,
        xx,
        ci,
        cd,
        ht,
        line_width=1.1,
        line_dash=None,
        legend_prefix="ci_inst",
        legend_label_base="Instantaneo",
    ):
        fig.add_trace(tr)
    for tr in _segmented_scatter_lines_by_pred_batch(
        d,
        xx,
        ma60,
        cd,
        ht,
        line_width=2.0,
        line_dash=None,
        legend_prefix="ci_ma60",
        legend_label_base="MA60",
        show_legend=False,
    ):
        fig.add_trace(tr)
    for tr in _segmented_scatter_lines_by_pred_batch(
        d,
        xx,
        sm4,
        cd,
        ht,
        line_width=3.2,
        line_dash=None,
        legend_prefix="ci_sm4",
        legend_label_base="Suavizado 4p",
        show_legend=False,
    ):
        fig.add_trace(tr)
    return fig


def fig_health_index_operational(dfp: pd.DataFrame) -> go.Figure:
    if dfp.empty or "health_index_operational" not in dfp.columns:
        return fig_empty("Sin series operacionales en predicciones exportadas (health_index_operational).")
    d = _sort_for_time_series_plots(dfp)
    xx = _plot_time_x(d)
    use_t = _plot_time_axis_usable(d)
    hi = pd.to_numeric(d["health_index_operational"], errors="coerce")
    ma60 = hi.rolling(60, min_periods=1).mean()
    sm4 = ma60.rolling(4, min_periods=1).mean()
    yp = d["y_pred_label"].astype(str) if "y_pred_label" in d.columns else pd.Series("?", index=d.index)
    bop = d["baseline_batch_operational"].astype(str) if "baseline_batch_operational" in d.columns else pd.Series("", index=d.index)
    bst = d["baseline_status"].astype(str) if "baseline_status" in d.columns else pd.Series("", index=d.index)
    colors = [CLASS_COLORS.get(str(x), "#888888") for x in yp]
    cd = np.column_stack([d["window_id"].values, yp.values, bop.values, bst.values])
    ht = (
        (
            "tiempo=%{x|%Y-%m-%d %H:%M}<br>window_id=%{customdata[0]}<br>y_pred=%{customdata[1]}<br>baseline_op=%{customdata[2]}<br>"
            "baseline_status=%{customdata[3]}<br>health=%{y:.2f}<extra></extra>"
        )
        if use_t
        else (
            "window_id=%{customdata[0]}<br>y_pred=%{customdata[1]}<br>baseline_op=%{customdata[2]}<br>"
            "baseline_status=%{customdata[3]}<br>health=%{y:.2f}<extra></extra>"
        )
    )
    fig = go.Figure()
    fig.update_layout(
        shapes=_epi_bpc_band_shapes(),  # mismos umbrales que EPI_BPC (tab Assessment)
        title="EPI_BPC operacional — salud relativa con linea base por crudo predicho",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=380,
        yaxis=dict(range=[0, 100], title="EPI_BPC operacional (0-100)"),
        xaxis=_plot_xaxis_layout(d),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=72),
    )
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=hi,
            mode="markers",
            name="instantaneo",
            marker=dict(color=colors, size=5, opacity=0.85),
            customdata=cd,
            hovertemplate=ht,
        )
    )
    fig.add_trace(
        go.Scatter(x=xx, y=ma60, mode="lines", name="media movil 60", line=dict(color="#333333", width=1.6))
    )
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=sm4,
            mode="lines",
            name="suavizado 4p de MA60",
            line=dict(color="#111111", width=1.4, dash="dash"),
        )
    )
    return fig


def fig_condition_index_operational(dfp: pd.DataFrame) -> go.Figure:
    """Bandas severidad = complemento EPI 60/80; trazas segmentadas; leyenda compacta a la derecha."""
    if dfp.empty or "condition_index_operational" not in dfp.columns:
        return fig_empty("Sin series operacionales en predicciones exportadas (condition_index_operational).")
    d = _sort_for_time_series_plots(dfp)
    xx = _plot_time_x(d)
    use_t = _plot_time_axis_usable(d)
    ci = pd.to_numeric(d["condition_index_operational"], errors="coerce").clip(0, 100)
    ma60 = ci.rolling(60, min_periods=1).mean()
    sm4 = ma60.rolling(4, min_periods=1).mean()
    wid = d["window_id"].values
    yp = d["y_pred_label"].astype(str).values if "y_pred_label" in d.columns else np.repeat("?", len(d))
    conf = pd.to_numeric(d["confidence"], errors="coerce").values if "confidence" in d.columns else np.full(len(d), np.nan)
    marg = pd.to_numeric(d["margin_top2"], errors="coerce").values if "margin_top2" in d.columns else np.full(len(d), np.nan)
    bop = (
        d["baseline_batch_operational"].astype(str).values
        if "baseline_batch_operational" in d.columns
        else np.repeat("", len(d))
    )
    bst = d["baseline_status"].astype(str).values if "baseline_status" in d.columns else np.repeat("", len(d))
    cd = np.column_stack([wid, yp, conf, marg, bop, bst])
    ht = (
        (
            "tiempo=%{x|%Y-%m-%d %H:%M}<br>window_id=%{customdata[0]}<br>y_pred_label=%{customdata[1]}<br>Confianza del clasificador=%{customdata[2]:.2f}<br>"
            "Margen entre las dos clases mas probables=%{customdata[3]:.2f}<br>condition_index_operational=%{y:.2f}<br>baseline_batch_operational=%{customdata[4]}<br>"
            "baseline_status=%{customdata[5]}<extra></extra>"
        )
        if use_t
        else (
            "window_id=%{customdata[0]}<br>y_pred_label=%{customdata[1]}<br>Confianza del clasificador=%{customdata[2]:.2f}<br>"
            "Margen entre las dos clases mas probables=%{customdata[3]:.2f}<br>condition_index_operational=%{y:.2f}<br>baseline_batch_operational=%{customdata[4]}<br>"
            "baseline_status=%{customdata[5]}<extra></extra>"
        )
    )
    fig = go.Figure()
    fig.update_layout(
        shapes=_condition_index_bpc_severity_shapes(),
        title="Condition index operacional — severidad relativa con linea base por crudo predicho",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=420,
        yaxis=dict(range=[0, 100], title="Severidad relativa operacional (0-100)"),
        xaxis=_plot_xaxis_layout(d),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            xref="paper",
            bgcolor="rgba(255,255,255,0.94)",
            font=dict(size=10),
            itemwidth=30,
        ),
        margin=dict(t=44, r=220, l=56, b=52),
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Verde 0-20 (baja severidad)",
            marker=dict(size=10, color="rgba(44,160,44,0.65)", symbol="square"),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Amarillo 20-40 (observacion)",
            marker=dict(size=10, color="rgba(230,180,50,0.75)", symbol="square"),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Rojo 40-100 (alta severidad)",
            marker=dict(size=10, color="rgba(214,39,40,0.65)", symbol="square"),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Lineas: color = crudo predicho (hover)",
            marker=dict(size=9, color="#666666", symbol="circle-open", line=dict(width=1.5, color="#666")),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    for tr in _segmented_scatter_lines_by_pred_batch(
        d,
        xx,
        ci,
        cd,
        ht,
        line_width=1.1,
        line_dash=None,
        legend_prefix="ci_op_inst",
        legend_label_base="Instantaneo",
    ):
        fig.add_trace(tr)
    for tr in _segmented_scatter_lines_by_pred_batch(
        d,
        xx,
        ma60,
        cd,
        ht,
        line_width=2.0,
        line_dash=None,
        legend_prefix="ci_op_ma60",
        legend_label_base="MA60",
        show_legend=False,
    ):
        fig.add_trace(tr)
    for tr in _segmented_scatter_lines_by_pred_batch(
        d,
        xx,
        sm4,
        cd,
        ht,
        line_width=3.2,
        line_dash=None,
        legend_prefix="ci_op_sm4",
        legend_label_base="Suavizado 4p",
        show_legend=False,
    ):
        fig.add_trace(tr)
    return fig


def _var_dropdown_options() -> list[dict[str, str]]:
    long_df = DATA.get("cond_contrib_long", pd.DataFrame())
    if long_df.empty or "raw_variable" not in long_df.columns:
        return [{"label": "(sin datos)", "value": ""}]
    vs = sorted(long_df["raw_variable"].astype(str).unique())
    return [{"label": abbrev_label(v, 58), "value": v} for v in vs]


def _resolve_batch_for_thresholds(row: pd.Series, mode: str) -> str:
    if mode == "global":
        return "GLOBAL"
    if mode == "real":
        return str(row.get("Batch", "GLOBAL"))
    return str(row.get("baseline_batch_operational", "GLOBAL"))


def _lookup_threshold_row(batch_eff: str, matched_col: str, thr_g: pd.DataFrame, thr_b: pd.DataFrame) -> pd.Series | None:
    if thr_g.empty or "matched_raw_column" not in thr_g.columns:
        return None
    mc = str(matched_col)
    glo = thr_g[thr_g["matched_raw_column"].astype(str) == mc]
    if batch_eff == "GLOBAL":
        return glo.iloc[0] if len(glo) else None
    if thr_b.empty or "Batch" not in thr_b.columns:
        return glo.iloc[0] if len(glo) else None
    sub = thr_b[(thr_b["Batch"].astype(str) == str(batch_eff)) & (thr_b["matched_raw_column"].astype(str) == mc)]
    if len(sub):
        return sub.iloc[0]
    return glo.iloc[0] if len(glo) else None


def _pred_x1_right_extension(xx: pd.Series, e: int) -> Any:
    """Borde derecho en eje X para el ultimo tramo de ventanas (e == n)."""
    last = xx.iloc[e - 1]
    if e < 2:
        return last
    prev = xx.iloc[e - 2]
    delta = last - prev
    try:
        td = pd.Timedelta(delta)
        ext = td if td > pd.Timedelta(0) else pd.Timedelta(seconds=30)
        return last + ext
    except (ValueError, TypeError):
        pass
    try:
        d = float(delta)
        if not np.isfinite(d) or abs(d) < 1e-15:
            d = 1.0
        return float(last) + d
    except (TypeError, ValueError):
        return last


def _variable_thr_row_equal(m: pd.DataFrame, i: int, j: int, tol: float = 1e-6) -> bool:
    """True si filas i y j comparten el mismo triple V0/H/HH (umbrales constantes en el tramo)."""
    for c in ("_v0", "_h", "_hh"):
        a = pd.to_numeric(m[c].iloc[i], errors="coerce")
        b = pd.to_numeric(m[c].iloc[j], errors="coerce")
        if not (np.isfinite(a) and np.isfinite(b)):
            return False
        if abs(float(a) - float(b)) > tol:
            return False
    return True


def _variable_umbrales_zone_shapes(xx: pd.Series, m: pd.DataFrame, yv: pd.Series, *, alpha: float = 0.16) -> list[dict[str, Any]]:
    """Franjas horizontales por zona V0/H/HH en cada tramo de tiempo con umbrales constantes (no por batch)."""
    shapes: list[dict[str, Any]] = []
    n = len(m)
    if n == 0 or len(xx) != n:
        return shapes
    vnum = pd.to_numeric(yv, errors="coerce")
    v0a = pd.to_numeric(m["_v0"], errors="coerce")
    ha = pd.to_numeric(m["_h"], errors="coerce")
    hha = pd.to_numeric(m["_hh"], errors="coerce")
    stack = pd.concat([vnum, v0a, ha, hha], axis=1)
    y_lo = float(stack.min().min())
    y_hi = float(stack.max().max())
    if not (np.isfinite(y_lo) and np.isfinite(y_hi)):
        return shapes
    span = y_hi - y_lo
    pad = span * 0.04 if span > 1e-12 else 0.01
    y_min = y_lo - pad
    y_max = y_hi + pad
    zone_colors = [
        _hex_to_rgba("#2ca02c", alpha),
        _hex_to_rgba("#e6b800", alpha),
        _hex_to_rgba("#ff7f0e", alpha * 0.95),
        _hex_to_rgba("#d62728", alpha),
    ]
    s = 0
    while s < n:
        e = s + 1
        while e < n and _variable_thr_row_equal(m, s, e):
            e += 1
        x0 = xx.iloc[s]
        x1 = xx.iloc[e] if e < n else _pred_x1_right_extension(xx, e)
        v0f = float(pd.to_numeric(m["_v0"].iloc[s], errors="coerce"))
        hf = float(pd.to_numeric(m["_h"].iloc[s], errors="coerce"))
        hhf = float(pd.to_numeric(m["_hh"].iloc[s], errors="coerce"))
        if not (np.isfinite(v0f) and np.isfinite(hf) and np.isfinite(hhf)):
            s = e
            continue
        brks = sorted({v0f, hf, hhf})
        edges = [y_min] + brks + [y_max]
        nband = len(edges) - 1
        for bi in range(nband):
            cidx = min(bi, len(zone_colors) - 1)
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=x0,
                    x1=x1,
                    y0=edges[bi],
                    y1=edges[bi + 1],
                    fillcolor=zone_colors[cidx],
                    line_width=0,
                    layer="below",
                )
            )
        s = e
    return shapes


def _segmented_scatter_lines_by_pred_batch(
    d: pd.DataFrame,
    xx: pd.Series,
    y: pd.Series,
    cd: np.ndarray,
    hovertemplate: str,
    *,
    line_width: float,
    line_dash: str | None = None,
    legend_prefix: str,
    legend_label_base: str,
    show_legend: bool = True,
) -> list[go.Scatter]:
    """Tramos contiguos por y_pred_label; color CLASS_COLORS; punto duplicado en fronteras."""
    n = len(d)
    if n == 0:
        return []
    ypl = d["y_pred_label"].astype(str) if "y_pred_label" in d.columns else pd.Series("?", index=d.index)
    traces: list[go.Scatter] = []
    seen_batch: set[str] = set()
    s = 0
    while s < n:
        batch = str(ypl.iloc[s])
        e = s + 1
        while e < n and str(ypl.iloc[e]) == batch:
            e += 1
        lo = s - 1 if s > 0 else s
        xx_seg = xx.iloc[lo:e]
        y_seg = y.iloc[lo:e]
        cd_seg = cd[lo:e]
        col = CLASS_COLORS.get(batch, "#555555")
        show_leg = show_legend and (batch not in seen_batch)
        seen_batch.add(batch)
        line_kw: dict[str, Any] = dict(color=col, width=line_width)
        if line_dash:
            line_kw["dash"] = line_dash
        traces.append(
            go.Scatter(
                x=xx_seg,
                y=y_seg,
                mode="lines",
                name=f"{legend_label_base} ({batch})",
                legendgroup=f"{legend_prefix}_{batch}",
                showlegend=show_leg,
                line=line_kw,
                customdata=cd_seg,
                hovertemplate=hovertemplate,
            )
        )
        s = e
    return traces


def _value_traces_colored_by_pred_batch(
    m: pd.DataFrame,
    xx: pd.Series,
    yv: pd.Series,
    cd: np.ndarray,
    ht_val: str,
) -> list[go.Scatter]:
    """Una traza continua por tramo de y_pred_label; color por crudo predicho (CLASS_COLORS)."""
    return _segmented_scatter_lines_by_pred_batch(
        m,
        xx,
        yv,
        cd,
        ht_val,
        line_width=1.8,
        line_dash=None,
        legend_prefix="val_pred",
        legend_label_base="valor",
    )


def build_variable_threshold_figure(raw_var: str, mode: str) -> go.Figure:
    """Umbrales V0/H/HH solo como franjas de fondo (shapes); valor segmentado por crudo predicho."""
    long_df = DATA.get("cond_contrib_long", pd.DataFrame())
    pred_op = DATA.get("predictions_operational", pd.DataFrame())
    thr_g = DATA.get("ath_global", pd.DataFrame())
    thr_b = DATA.get("ath_by_batch", pd.DataFrame())
    mode = (mode or "predicted").lower()
    if not raw_var or long_df.empty or pred_op.empty:
        return fig_empty("Seleccione variable o cargue datos long / predictions")
    sub = long_df[long_df["raw_variable"].astype(str) == str(raw_var)].copy()
    if sub.empty:
        return fig_empty("Variable sin filas en contributions long")
    mcol = str(sub["matched_raw_column"].iloc[0])
    cols_pred = [
        "window_id",
        "Batch",
        "y_pred_label",
        "confidence",
        "margin_top2",
        "baseline_batch_operational",
        "baseline_status",
    ]
    have = [c for c in cols_pred if c in pred_op.columns]
    if "window_start" in pred_op.columns and "window_start" not in have:
        have.append("window_start")
    # No sustituir window_start del long por el de pred: si falta match en pred, se pierde tiempo y Plotly colapsa el eje X.
    if "window_start" in sub.columns:
        have_no_ws = [c for c in have if c != "window_start"]
        m = sub.merge(pred_op[have_no_ws], on="window_id", how="left")
    else:
        m = sub.merge(pred_op[have], on="window_id", how="left")
    v0l: list[float] = []
    hl: list[float] = []
    hhl: list[float] = []
    for _, r in m.iterrows():
        bk = _resolve_batch_for_thresholds(r, mode)
        rowt = _lookup_threshold_row(bk, mcol, thr_g, thr_b)
        if rowt is None:
            v0l.append(float("nan"))
            hl.append(float("nan"))
            hhl.append(float("nan"))
        else:
            v0l.append(float(rowt["v0_estimated"]))
            hl.append(float(rowt["h_estimated"]))
            hhl.append(float(rowt["hh_estimated"]))
    m = m.assign(_v0=v0l, _h=hl, _hh=hhl)
    m = _sort_for_time_series_plots(m)
    xx = _plot_time_x(m)
    use_t = _plot_time_axis_usable(m)
    yv = pd.to_numeric(m["value"], errors="coerce")
    if "condition_score" in m.columns:
        cs = pd.to_numeric(m["condition_score"], errors="coerce")
    else:
        cs = pd.Series(np.nan, index=m.index)

    cd = np.column_stack(
        [
            m["window_id"].values,
            m["y_pred_label"].astype(str).values,
            m["baseline_batch_operational"].astype(str).values,
            yv.values,
            m["_v0"].values,
            m["_h"].values,
            m["_hh"].values,
            cs.values,
        ]
    )
    ht_val = (
        (
            "tiempo=%{x|%Y-%m-%d %H:%M}<br>window_id=%{customdata[0]}<br>y_pred=%{customdata[1]}<br>baseline_op=%{customdata[2]}<br>"
            "value=%{customdata[3]:.2f}<br>V0=%{customdata[4]:.2f}<br>H=%{customdata[5]:.2f}<br>HH=%{customdata[6]:.2f}"
            "<br>condition_score=%{customdata[7]:.2f}<extra></extra>"
        )
        if use_t
        else (
            "window_id=%{customdata[0]}<br>y_pred=%{customdata[1]}<br>baseline_op=%{customdata[2]}<br>"
            "value=%{customdata[3]:.2f}<br>V0=%{customdata[4]:.2f}<br>H=%{customdata[5]:.2f}<br>HH=%{customdata[6]:.2f}"
            "<br>condition_score=%{customdata[7]:.2f}<extra></extra>"
        )
    )
    fig = go.Figure()
    for tr in _value_traces_colored_by_pred_batch(m, xx, yv, cd, ht_val):
        fig.add_trace(tr)
    lab = abbrev_label(raw_var, 64)
    fig.update_layout(
        title=f"Senal vibracional vs V0/H/HH dinamicos (modo {mode}) — {lab}",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=440,
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=72, b=56),
        xaxis=_plot_xaxis_layout(m),
        shapes=_variable_umbrales_zone_shapes(xx, m, yv, alpha=0.16),
    )
    return fig


def operational_variable_section() -> list[Any]:
    opts = _var_dropdown_options()
    default_v = opts[0]["value"] if opts and opts[0].get("value") else ""
    return [
        html.Hr(),
        html.H4("Variable individual con V0/H/HH dinamicos"),
        html.P(
            "Esta grafica muestra la variable cruda y sus umbrales V0/H/HH seleccionados segun el modo de linea base: "
            "predicted batch, real batch historico o global. "
            "La linea del valor se segmenta y colorea por crudo predicho (y_pred_label), misma paleta que las clases. "
            "Los umbrales V0/H/HH no se dibujan como lineas; el fondo sombreado horizontal marca las zonas entre esos umbrales vigentes en cada tramo (exploratorio, no normativo). "
            "Pase el cursor sobre la serie para ver V0, H y HH numericos.",
            className="note",
        ),
        html.P(
            "Los umbrales son exploratorios (P40/P75/P99), no limites normativos ni diagnostico de falla.",
            className="note",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Variable (raw_variable)"),
                        dcc.Dropdown(id="var-thresh-raw", options=opts, value=default_v, clearable=False),
                    ],
                    style={"flex": "2", "minWidth": "280px"},
                ),
                html.Div(
                    [
                        html.Label("Linea base umbrales"),
                        dcc.Dropdown(
                            id="var-thresh-mode",
                            options=[
                                {"label": "Predicted batch", "value": "predicted"},
                                {"label": "Real batch historico", "value": "real"},
                                {"label": "Global", "value": "global"},
                            ],
                            value="predicted",
                            clearable=False,
                        ),
                    ],
                    style={"flex": "1", "minWidth": "220px"},
                ),
            ],
            className="filter-row var-thresh-controls",
        ),
        html.Div(dcc.Graph(id="var-thresh-graph", style={"minHeight": "400px"}), className="var-thresh-graph-wrap"),
    ]


def _last_window_top_contributors(top_df: pd.DataFrame) -> pd.DataFrame:
    if top_df.empty or "window_id" not in top_df.columns:
        return pd.DataFrame()
    wid_max = int(pd.to_numeric(top_df["window_id"], errors="coerce").max())
    return top_df[top_df["window_id"] == wid_max].copy()


def fig_top_contributors_last_window(top_df: pd.DataFrame) -> go.Figure:
    sub = _last_window_top_contributors(top_df)
    if sub.empty:
        return fig_empty("Sin datos de top contributors (dashboard_condition_contributions_top_by_window.csv)")
    for c in ("weighted_score", "condition_score", "weight_normalized"):
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce").round(DASH_DECIMALS)
    tie = tuple(c for c in ("raw_variable",) if c in sub.columns)
    sub = sub.sort_values(["weighted_score", *tie], ascending=[True] * (1 + len(tie)), na_position="last")
    sub["lab"] = sub["raw_variable"].astype(str).map(lambda x: abbrev_label(x, 52))
    sub["y_cat"] = _unique_bar_category_labels(sub, "lab", "raw_variable")
    fig = px.bar(
        sub,
        x="weighted_score",
        y="y_cat",
        orientation="h",
        title="Top 5 contribuyentes de condicion (ultima ventana historica): weighted_score",
        hover_data={
            "raw_variable": True,
            "condition_score": True,
            "weight_normalized": True,
            "component": True,
            "family": True,
        },
    )
    fig.update_layout(
        paper_bgcolor="white",
        height=max(320, 50 + 48 * len(sub)),
        yaxis_title="",
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=_bar_sorted_categoryarray(
                sub, "y_cat", "weighted_score", largest_first=False, tiebreak_cols=tie
            ),
        ),
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<br>weighted_score=%{x:.2f}<extra></extra>")
    return fig


def fig_group_weighted_last_window(top_df: pd.DataFrame, col: str, title: str) -> go.Figure:
    sub = _last_window_top_contributors(top_df)
    if sub.empty or col not in sub.columns:
        return fig_empty(f"Sin datos agrupados por {col}")
    g = (
        sub.groupby(col, as_index=False)["weighted_score"]
        .sum()
        .assign(_s=lambda d: pd.to_numeric(d["weighted_score"], errors="coerce"))
        .sort_values(["_s", col], ascending=[True, True], na_position="last")
        .drop(columns="_s")
    )
    g["weighted_score"] = pd.to_numeric(g["weighted_score"], errors="coerce").round(DASH_DECIMALS)
    fig = px.bar(g, x="weighted_score", y=col, orientation="h", title=title)
    fig.update_layout(
        paper_bgcolor="white",
        height=340,
        yaxis_title="",
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=_bar_sorted_categoryarray(g, col, "weighted_score", largest_first=False),
        ),
    )
    return fig


def fig_variables_long_last_window(long_df: pd.DataFrame) -> go.Figure:
    if long_df.empty or "window_id" not in long_df.columns or "weighted_score" not in long_df.columns:
        return fig_empty("Sin datos long para descomposicion por variable")
    wmax = int(pd.to_numeric(long_df["window_id"], errors="coerce").max())
    chunk = long_df.loc[pd.to_numeric(long_df["window_id"], errors="coerce") == wmax]
    s = chunk.groupby("raw_variable", as_index=False)["weighted_score"].sum()
    s["_w"] = pd.to_numeric(s["weighted_score"], errors="coerce")
    s = s.nlargest(15, "_w", keep="all").drop(columns="_w")
    if s.empty:
        return fig_empty("Sin filas long en ultima ventana")
    s = s.copy()
    s = s.sort_values(["weighted_score", "raw_variable"], ascending=[True, True], na_position="last")
    s["lab"] = s["raw_variable"].astype(str).map(lambda x: abbrev_label(x, 48))
    s["y_cat"] = _unique_bar_category_labels(s, "lab", "raw_variable")
    s["weighted_score"] = pd.to_numeric(s["weighted_score"], errors="coerce").round(DASH_DECIMALS)
    fig = px.bar(
        s,
        x="weighted_score",
        y="y_cat",
        orientation="h",
        title="Contribucion por variable (ultima ventana; suma weighted_score)",
        hover_data={"raw_variable": True},
    )
    fig.update_layout(
        paper_bgcolor="white",
        height=400,
        yaxis_title="",
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=_bar_sorted_categoryarray(
                s, "y_cat", "weighted_score", largest_first=False, tiebreak_cols=("raw_variable",)
            ),
        ),
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<br>weighted_score=%{x:.2f}<extra></extra>")
    return fig


def fig_health_series_predictions(pred: pd.DataFrame) -> go.Figure:
    if pred.empty or "health_index_thresholded_by_batch" not in pred.columns:
        return fig_empty("Sin health_index_thresholded_by_batch en dashboard_predictions.csv")
    dfp = _sort_for_time_series_plots(pred)
    xx = _plot_time_x(dfp)
    use_t = _plot_time_axis_usable(dfp)
    ht = (
        "tiempo=%{x|%Y-%m-%d %H:%M}<br>window_id=%{customdata}<br>health_index=%{y:.2f}<extra></extra>"
        if use_t
        else "window_id=%{x}<br>health_index=%{y:.2f}<extra></extra>"
    )
    wid_cd = dfp["window_id"].values if "window_id" in dfp.columns else None
    fig = go.Figure(
        go.Scatter(
            x=xx,
            y=dfp["health_index_thresholded_by_batch"],
            mode="lines",
            name="EPI / salud relativa por batch (CSV)",
            line=dict(color="#1f77b4", width=1.5),
            customdata=wid_cd,
            hovertemplate=ht,
        )
    )
    fig.update_layout(
        title="EPI / salud relativa por batch vs tiempo (inicio de ventana; CSV)" if use_t else "EPI / salud relativa por batch vs window_id (CSV)",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=300,
        yaxis=dict(range=[0, 100]),
        xaxis=_plot_xaxis_layout(dfp),
    )
    return fig


def estado_actual_del_activo_section() -> list[Any]:
    """Bloques UI para Tab 6 — estado actual y alertas exploratorias (Etapa 10C)."""
    cur = DATA.get("cond_current", pd.DataFrame())
    alt = DATA.get("cond_alerts", pd.DataFrame())
    tr = DATA.get("cond_trend", pd.DataFrame())
    top = DATA.get("cond_contrib_top", pd.DataFrame())
    long_df = DATA.get("cond_contrib_long", pd.DataFrame())
    pred = DATA.get("predictions", pd.DataFrame())
    js = DATA.get("asset_state_json")

    cards_src: list[html.Div] = []
    if not cur.empty:
        r0 = cur.iloc[0]
        specs: list[tuple[str, str, Any]] = [
            ("Batch real", str(r0.get("Batch", "")), r0.get("Batch")),
            ("Crudo predicho", str(r0.get("y_pred_label", "")), r0.get("y_pred_label")),
            ("Confianza del clasificador", dash_format_number(r0.get("confidence")), r0.get("confidence")),
            ("Indice de condicion", dash_format_number(r0.get("condition_index")), r0.get("condition_index")),
            ("EPI_BPC / salud relativa", dash_format_number(r0.get("health_index")), r0.get("health_index")),
            ("Estado exploratorio", str(r0.get("condition_state", "")), r0.get("condition_state")),
        ]
        if not tr.empty and "trend_direction" in tr.columns:
            specs.append(("Tendencia reciente", str(tr.iloc[0]["trend_direction"]), tr.iloc[0]["trend_direction"]))
        alert_lab_map = {
            "n_active_attention_variables": "Alertas exploratorias attention",
            "n_active_high_variables": "Alertas exploratorias high",
        }
        for col in ("n_active_attention_variables", "n_active_high_variables"):
            if col in r0.index:
                specs.append((alert_lab_map[col], str(r0.get(col, "")), r0.get(col)))
        cards_src = [
            html.Div(
                [html.Div(lab, className="label"), html.Div(val, className="value")],
                className="kpi-card",
                style=_activo_kpi_card_style(lab, val, raw),
            )
            for lab, val, raw in specs
        ]
    cards_row = html.Div(cards_src, className="kpi-row") if cards_src else html.P("Sin dashboard_condition_current_state.csv.", className="note")

    alt_path = DASHBOARD_DATA / "dashboard_condition_alerts_active.csv"
    if not alt_path.is_file():
        alert_block: Any = html.P("Archivo de alertas exploratorias no encontrado.", className="note")
    elif alt.empty:
        alert_block = html.P("Sin alertas exploratorias activas.", className="note")
    else:
        alert_block = _df_table(alt)

    if js is not None:
        json_block: Any = html.Details(
            [
                html.Summary("current_asset_state.json (salida operativa; ultima ventana historica exportada)"),
                html.Pre(json.dumps(json_round_floats(js), indent=2, ensure_ascii=False), className="json-preview"),
            ],
            style={"marginTop": "0.75rem"},
        )
    else:
        json_block = html.P("current_asset_state.json no disponible o JSON invalido.", className="note")

    methodology = html.Div(
        [
            html.P("Notas metodologicas (exploratorias, no normativas):"),
            html.Ul(
                [
                    html.Li("EPI_BPC = 100 - condition_index."),
                    html.Li("Mayor EPI_BPC indica mejor condicion relativa."),
                    html.Li("Mayor condition_index indica mayor severidad relativa."),
                    html.Li(
                        "V0/H/HH se estimaron como P40/P75/P99 del historico disponible; las bandas son visuales y exploratorias."
                    ),
                    html.Li(
                        "El estado exploratorio (condition_state) resume bandas sobre condition_index: normal si < 20; "
                        "attention si 20 <= condition_index < 40; high si >= 40. "
                        "Sobre EPI_BPC: normal si > 80; attention si 60 <= EPI <= 80; high si < 60."
                    ),
                    html.Li("Las alertas no son alarmas normativas."),
                    html.Li("Las alertas exploratorias por variable usan condition_score individual, no solo el indice global."),
                    html.Li(
                        "La tendencia reciente resume la pendiente del indice sobre ventanas pasadas; no es RUL ni vida util remanente."
                    ),
                    html.Li("Una variable puede estar en attention aunque el estado global sea normal."),
                    html.Li("Nada de lo anterior constituye diagnostico normativo."),
                ]
            ),
        ],
        className="note",
    )

    return [
        html.Hr(),
        html.H4("Estado actual del activo"),
        html.P(
            "Esta seccion apoya decisiones tecnicas con datos ya exportados; no sustituye revision humana ni procedimientos de planta.",
            className="note",
        ),
        cards_row,
        methodology,
        html.H5("Contribuyentes principales al indice (ultima ventana)"),
        dcc.Graph(figure=fig_top_contributors_last_window(top)),
        html.Div(
            [
                dcc.Graph(
                    figure=fig_group_weighted_last_window(
                        top, "component", "Contribucion por componente (suma weighted_score, ultima ventana)"
                    ),
                    style={"flex": "1"},
                ),
                dcc.Graph(
                    figure=fig_group_weighted_last_window(
                        top, "family", "Contribucion por familia (suma weighted_score, ultima ventana)"
                    ),
                    style={"flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
        ),
        html.H5("Descomposicion por variable (long, ultima ventana)"),
        dcc.Graph(figure=fig_variables_long_last_window(long_df)),
        html.H5("EPI / salud relativa temporal (CSV de predicciones)"),
        dcc.Graph(figure=fig_health_series_predictions(pred)),
        html.H5("Alertas exploratorias"),
        alert_block,
        html.H5("Estado completo (CSV)"),
        _df_table(cur),
        html.H5("Resumen de tendencia"),
        _df_table(tr),
        html.H5("Top contributors ultima ventana (tabla)"),
        _df_table(_last_window_top_contributors(top)),
        html.H5("Salida operativa (JSON)"),
        json_block,
    ]


def tab_assessment() -> html.Div:
    cb = DATA["cond_batch"]
    cg = DATA["ci_g_batch"]
    bb = DATA["ci_b_batch"]
    pred = DATA["predictions"]
    pred_op = DATA.get("predictions_operational", pd.DataFrame())
    comp = build_three_method_comparison_df()
    children = [
        html.H3("Assessment ponderado"),
        html.Div(
            [
                html.P(
                    "El assessment ponderado estima una condicion relativa del activo usando pesos de variables vibracionales "
                    "y lineas base exploratorias. Esta capa es independiente de la importancia del modelo ML."
                ),
                html.P(
                    "Los pesos ponderados no son importancia ML; describen aporte al indice exploratorio de condicion."
                ),
                html.P(
                    "Los umbrales V0/H/HH se estimaron como P40/P75/P99 sobre el historico exportado; "
                    "no son limites normativos de alarma."
                ),
                html.P(
                    "El modo por batch usa el batch real en este analisis historico. En operacion futura se debe "
                    "seleccionar la linea base por batch predicho solo si la confianza del clasificador y el margen entre las dos clases mas probables son suficientes."
                ),
                html.P(
                    "El indice es exploratorio y comparativo entre ventanas y batches; no constituye diagnostico normativo."
                ),
                html.P(
                    "Las bandas de color en indices operacionales siguen al crudo predicho por el modelo cuando "
                    "la confianza es suficiente; si no, se usa baseline global y el assessment operacional se interpreta como mas incierto. "
                    "Las bandas son visuales y exploratorias; las alertas no son alarmas normativas."
                ),
                html.P(
                    "En analisis historico se puede comparar contra el batch real; en operacion el baseline operacional se deriva de "
                    "la clase predicha con la compuerta de confianza configurada en la app."
                ),
            ],
            className="note",
        ),
    ]
    children.extend(
        [
            html.Hr(),
            html.H4("EPI_BPC de condicion relativa"),
            html.P("EPI_BPC = 100 - condition_index.", className="note"),
            html.P("Mayor EPI_BPC indica mejor condicion relativa.", className="note"),
            html.P("Mayor condition_index indica mayor severidad relativa.", className="note"),
            html.P(
                "El EPI_BPC usa la linea base del crudo predicho cuando la confianza del clasificador es suficiente; "
                "si no, se usa baseline global o se marca el assessment como incierto en sentido operacional.",
                className="note",
            ),
            dcc.Graph(figure=fig_epi_bpc_picadora_style(pred_op)),
            html.P(
                "Las bandas visuales del grafico son exploratorias. Los umbrales V0/H/HH por variable se estimaron con "
                "P40/P75/P99 del historico disponible y no son limites normativos.",
                className="note",
            ),
            html.H4("Indice de severidad (condition_index)"),
            html.P(
                "Serie con condition_index_thresholded_by_batch del CSV (linea base por batch del historico exportado). "
                "Bandas de severidad alineadas al complemento de EPI 60/80 en escala 0-100: verde 0-20, amarillo 20-40, rojo 40-100. "
                "Lectura exploratoria, no diagnostico de falla.",
                className="note",
                style={"fontSize": "0.82rem"},
            ),
            dcc.Graph(figure=fig_condition_index_bpc_bands(pred_op)),
        ]
    )
    children.extend(
        [
            html.Hr(),
            html.H4("Lineas base por crudo predicho"),
            html.P(
                "Los umbrales mostrados en la siguiente seccion dependen de la linea base seleccionada "
                "(clase predicha con compuerta de confianza, batch historico o global). Son exploratorios (P40/P75/P99).",
                className="note",
            ),
        ]
    )
    children.extend(operational_variable_section())
    children.extend(estado_actual_del_activo_section())
    children.extend(
        [
            html.H4("A. Assessment original (robust_percentile_fallback)"),
            html.P("Fuente: dashboard_condition_index_by_batch.csv. method = robust_percentile_fallback.", className="note"),
        ]
    )
    if not cb.empty and "condition_index_mean" in cb.columns:
        cb_a = cb.assign(_m=pd.to_numeric(cb["condition_index_mean"], errors="coerce")).sort_values(
            ["_m", "Batch"], ascending=[False, True], na_position="last"
        )
        fig_a = px.bar(
            cb_a,
            x="Batch",
            y="condition_index_mean",
            title="Indice medio por Batch (fallback)",
            color="Batch",
            color_discrete_map={k: CLASS_COLORS[k] for k in CLASS_ORDER},
        )
        fig_a.update_layout(
            showlegend=False,
            paper_bgcolor="white",
            xaxis=dict(
                categoryorder="array",
                categoryarray=_bar_sorted_categoryarray(cb_a, "Batch", "condition_index_mean", largest_first=True),
            ),
        )
        fig_a.update_traces(hovertemplate="Batch=%{x}<br>indice medio=%{y:.2f}<extra></extra>")
        children.append(dcc.Graph(figure=fig_a))
    else:
        children.append(dcc.Graph(figure=fig_empty("Sin dashboard_condition_index_by_batch.csv")))
    children.append(html.H5("Tabla (original)"))
    children.append(_df_table(cb))

    children.extend(
        [
            html.H4("B. Assessment con umbrales globales data-driven"),
            html.P("Fuente: dashboard_condition_index_thresholded_global_by_batch.csv.", className="note"),
        ]
    )
    if not cg.empty and "condition_index_mean" in cg.columns:
        cg_b = cg.assign(_m=pd.to_numeric(cg["condition_index_mean"], errors="coerce")).sort_values(
            ["_m", "Batch"], ascending=[False, True], na_position="last"
        )
        fig_b = px.bar(
            cg_b,
            x="Batch",
            y="condition_index_mean",
            title="Indice medio por Batch (umbrales globales)",
            color="Batch",
            color_discrete_map={k: CLASS_COLORS[k] for k in CLASS_ORDER},
        )
        fig_b.update_layout(
            showlegend=False,
            paper_bgcolor="white",
            xaxis=dict(
                categoryorder="array",
                categoryarray=_bar_sorted_categoryarray(cg_b, "Batch", "condition_index_mean", largest_first=True),
            ),
        )
        fig_b.update_traces(hovertemplate="Batch=%{x}<br>indice medio=%{y:.2f}<extra></extra>")
        children.append(dcc.Graph(figure=fig_b))
    else:
        children.append(dcc.Graph(figure=fig_empty("Sin datos threshold global por batch")))
    children.append(html.H5("Tabla (global)"))
    children.append(_df_table(cg))

    children.extend(
        [
            html.H4("C. Assessment con lineas base por batch data-driven"),
            html.P("Fuente: dashboard_condition_index_thresholded_by_batch_by_batch.csv.", className="note"),
        ]
    )
    if not bb.empty and "condition_index_mean" in bb.columns:
        bb_c = bb.assign(_m=pd.to_numeric(bb["condition_index_mean"], errors="coerce")).sort_values(
            ["_m", "Batch"], ascending=[False, True], na_position="last"
        )
        fig_c = px.bar(
            bb_c,
            x="Batch",
            y="condition_index_mean",
            title="Indice medio por Batch (linea base por batch)",
            color="Batch",
            color_discrete_map={k: CLASS_COLORS[k] for k in CLASS_ORDER},
        )
        fig_c.update_layout(
            showlegend=False,
            paper_bgcolor="white",
            xaxis=dict(
                categoryorder="array",
                categoryarray=_bar_sorted_categoryarray(bb_c, "Batch", "condition_index_mean", largest_first=True),
            ),
        )
        fig_c.update_traces(hovertemplate="Batch=%{x}<br>indice medio=%{y:.2f}<extra></extra>")
        children.append(dcc.Graph(figure=fig_c))
    else:
        children.append(dcc.Graph(figure=fig_empty("Sin datos threshold por batch")))
    children.append(html.H5("Tabla (por batch)"))
    children.append(_df_table(bb))

    children.append(html.H4("D. Comparacion de los tres metodos (agrupado por Batch)"))
    if not comp.empty:
        batch_order = (
            comp.assign(_m=pd.to_numeric(comp["condition_index_mean"], errors="coerce"))
            .groupby("Batch", as_index=False)["_m"]
            .mean()
            .sort_values(["_m", "Batch"], ascending=[False, True], na_position="last")["Batch"]
            .astype(str)
            .tolist()
        )
        fig_d = px.bar(
            comp,
            x="Batch",
            y="condition_index_mean",
            color="assessment_mode",
            barmode="group",
            title="condition_index_mean: fallback vs global vs por batch",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_d.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend_title_text="Modo",
            xaxis=dict(categoryorder="array", categoryarray=batch_order),
        )
        fig_d.update_traces(hovertemplate="Batch=%{x}<br>%{fullData.name}<br>indice medio=%{y:.2f}<extra></extra>")
        children.append(dcc.Graph(figure=fig_d))
    else:
        children.append(dcc.Graph(figure=fig_empty("Sin tablas suficientes para comparacion agrupada")))

    children.append(html.H4("E. Serie temporal (predictions)"))
    if not pred.empty and "window_id" in pred.columns:
        dfp = _sort_for_time_series_plots(pred)
        sel_e = list(PRED_INDEX_SERIES_DEFAULT)
        if "health_index_thresholded_by_batch" not in dfp.columns:
            sel_e = [s for s in sel_e if s != "health_batch"]
        fig_e = fig_condition_indices_multi(dfp, sel_e)
        children.append(dcc.Graph(figure=fig_e))
    else:
        children.append(dcc.Graph(figure=fig_empty("Sin dashboard_predictions.csv")))

    children.append(html.H4("F. Top variables ponderadas (con umbrales si existen)"))
    sw_df = DATA["sw_thr"]
    tw_df = DATA["top_weighted"]
    if not sw_df.empty:
        preferred_cols = [
            "weight_variable",
            "component",
            "family",
            "position",
            "weight_normalized",
            "v0",
            "h",
            "hh",
            "threshold_method",
        ]
        show_w = [c for c in preferred_cols if c in sw_df.columns]
        if "weight_normalized" in sw_df.columns:
            t15 = (
                sw_df.assign(_wn=pd.to_numeric(sw_df["weight_normalized"], errors="coerce"))
                .sort_values("_wn", ascending=False, na_position="last")
                .head(15)
                .copy()
            )
            t15["lab"] = t15["weight_variable"].astype(str).map(abbrev_label)
            t15["weight_normalized"] = pd.to_numeric(t15["weight_normalized"], errors="coerce").round(DASH_DECIMALS)
            t15_plot = t15.sort_values(["weight_normalized", "lab"], ascending=[True, True], na_position="last")
            id_w = "weight_variable" if "weight_variable" in t15_plot.columns else None
            if id_w is None:
                t15_plot = t15_plot.copy()
                t15_plot["_rid"] = np.arange(len(t15_plot), dtype=int).astype(str)
                id_w = "_rid"
            t15_plot["y_cat"] = _unique_bar_category_labels(t15_plot, "lab", id_w)
            fig_w = px.bar(
                t15_plot,
                x="weight_normalized",
                y="y_cat",
                orientation="h",
                title="Pesos normalizados (top 15, con umbrales)",
                **(
                    {"hover_data": {"weight_variable": True}}
                    if "weight_variable" in t15_plot.columns
                    else {}
                ),
            )
            fig_w.update_layout(
                paper_bgcolor="white",
                height=420,
                yaxis_title="",
                yaxis=dict(
                    type="category",
                    categoryorder="array",
                    categoryarray=_bar_sorted_categoryarray(
                        t15_plot,
                        "y_cat",
                        "weight_normalized",
                        largest_first=False,
                        tiebreak_cols=(("weight_variable",) if "weight_variable" in t15_plot.columns else ()),
                    ),
                ),
            )
            if "weight_variable" in t15_plot.columns:
                fig_w.update_traces(hovertemplate="%{customdata[0]}<br>peso normalizado=%{x:.2f}<extra></extra>")
            else:
                fig_w.update_traces(hovertemplate="%{y}<br>peso normalizado=%{x:.2f}<extra></extra>")
            children.append(dcc.Graph(figure=fig_w))
        children.append(
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in (show_w or list(sw_df.columns))],
                data=round_numeric_df(sw_df[show_w or list(sw_df.columns)]).to_dict("records"),
                page_size=12,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "0.78rem", "padding": "5px"},
            )
        )
    elif not tw_df.empty and "share_percent" in tw_df.columns:
        children.append(
            html.P(
                "dashboard_sensor_weights_with_thresholds.csv no disponible; se muestra dashboard_top_weighted_variables.csv.",
                className="note",
            )
        )
        t2 = tw_df.assign(_sp=pd.to_numeric(tw_df["share_percent"], errors="coerce")).nlargest(15, "_sp", keep="all").drop(columns="_sp")
        t2["lab"] = t2["weight_variable"].astype(str).map(abbrev_label)
        t2["share_percent"] = pd.to_numeric(t2["share_percent"], errors="coerce").round(DASH_DECIMALS)
        t2_plot = t2.sort_values(["share_percent", "lab"], ascending=[True, True], na_position="last")
        id_w2 = "weight_variable" if "weight_variable" in t2_plot.columns else None
        if id_w2 is None:
            t2_plot = t2_plot.copy()
            t2_plot["_rid"] = np.arange(len(t2_plot), dtype=int).astype(str)
            id_w2 = "_rid"
        t2_plot["y_cat"] = _unique_bar_category_labels(t2_plot, "lab", id_w2)
        fig_w = px.bar(
            t2_plot,
            x="share_percent",
            y="y_cat",
            orientation="h",
            title="Variables ponderadas (share %, top 15)",
            **({"hover_data": {"weight_variable": True}} if "weight_variable" in t2_plot.columns else {}),
        )
        fig_w.update_layout(
            paper_bgcolor="white",
            height=420,
            yaxis_title="",
            yaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=_bar_sorted_categoryarray(
                    t2_plot,
                    "y_cat",
                    "share_percent",
                    largest_first=False,
                    tiebreak_cols=(("weight_variable",) if "weight_variable" in t2_plot.columns else ()),
                ),
            ),
        )
        if "weight_variable" in t2_plot.columns:
            fig_w.update_traces(hovertemplate="%{customdata[0]}<br>share_percent=%{x:.2f}<extra></extra>")
        else:
            fig_w.update_traces(hovertemplate="%{y}<br>share_percent=%{x:.2f}<extra></extra>")
        children.append(dcc.Graph(figure=fig_w))
        children.append(_df_table(tw_df))
    else:
        children.append(html.P("Sin datos de pesos ponderados.", className="note"))

    children.append(
        html.Div(
            [
                html.H4("G. Texto metodologico (obligatorio)"),
                html.Ul(
                    [
                        html.Li("Los pesos ponderados no son importancia ML."),
                        html.Li("Los umbrales V0/H/HH fueron estimados con P40/P75/P99."),
                        html.Li("El modo by_batch usa Batch real en analisis historico."),
                        html.Li(
                            "En operacion futura se debe seleccionar baseline por batch predicho solo si la confianza del clasificador "
                            "y el margen entre las dos clases mas probables son suficientes."
                        ),
                        html.Li("El indice es exploratorio/comparativo, no diagnostico normativo."),
                        html.Li(
                            "Las bandas operacionales siguen al crudo predicho cuando la confianza y el margen superan los umbrales de la app; "
                            "V0/H/HH por variable siguen siendo exploratorios (P40/P75/P99), no normativos."
                        ),
                    ]
                ),
            ],
            className="note",
        )
    )
    return html.Div(children, className="tab-panel")


def tab_advertencias() -> html.Div:
    readme = DATA["readme"]
    bt = DATA["batch_transitions"]
    summary = DATA["ath_summary"]
    md = (
        (readme[:4000] + "\n\n[...]\n") if len(readme) > 4000 else readme
    ) if readme else "_README_dashboard_data.md no encontrado._"
    warn = """
**Advertencias metodologicas (lectura rapida)**

**A. Alcance de datos**

1. El dashboard usa la ultima exportacion historica disponible en `data/dashboard/`.
2. No representa necesariamente una lectura en vivo ni telemetria continua garantizada.

**B. Modelo**

1. Las metricas de generalizacion son las de validacion cruzada temporal; no sustituyen el criterio tecnico de despliegue.
2. Las metricas del entrenamiento final sobre todo el conjunto describen ajuste interno, no generalizacion.
3. El modelo no debe usarse para control automatico sin revision tecnica y procedimientos de planta.

**C. Interpretabilidad**

1. SHAP y permutation importance no implican causalidad ni mecanismos fisicos directos.

**D. Assessment**

1. Los pesos ponderados no son importancia ML del clasificador.
2. V0/H/HH son exploratorios (P40/P75/P99); las bandas y alertas no son normativas.
3. El assessment incluye `robust_percentile_fallback` y variantes con umbrales data-driven cuando los CSV estan disponibles.
4. `condition_state` resume bandas visuales sobre `condition_index`: normal si < 20; attention si 20 <= condition_index < 40; high si >= 40.
5. Sobre **EPI_BPC** (= 100 - condition_index, alineado con health index en exportaciones): normal si > 80; attention si 60 <= EPI <= 80; high si < 60.
6. Las alertas en CSV son exploratorias; no son alarmas normativas.
7. En analisis historico suele compararse con batch real; en operacion futura la linea base por batch debe alinearse con la clase predicha solo si la confianza y el margen entre las dos clases mas probables son suficientes; si no, usar baseline global o marcar incertidumbre.
8. Las bandas operacionales en graficos siguen al crudo predicho cuando la compuerta de confianza de la app se cumple; si no, se usa baseline global.

**E. Batch MEZCLA**

1. MEZCLA se trata como clase operacional independiente (no se interpreta como mezcla fisica de Castilla y Rubiales).

**F. Tendencia**

1. La tendencia es una pendiente o resumen reciente del indice sobre ventanas historicas exportadas; no es RUL ni vida util remanente ni pronostico de falla.

**Umbrales V0 / H / HH (exploratorios)**

1. Los umbrales V0/H/HH globales y por batch son exploratorios.
2. Fueron estimados desde el historico disponible: V0 = P40, H = P75, HH = P99.
3. No son limites normativos de alarma.
4. El modo by_batch usa batch real en el analisis historico.
"""
    js = DATA.get("asset_state_json")
    json_panel = (
        html.Details(
            [
                html.Summary("current_asset_state.json (expandir / colapsar)"),
                html.Pre(
                    json.dumps(json_round_floats(js), indent=2, ensure_ascii=False) if js is not None else "(no disponible)",
                    className="json-preview",
                ),
            ],
            style={"marginTop": "1rem"},
        )
    )

    return html.Div(
        [
            html.H3("Datos y advertencias metodologicas"),
            dcc.Markdown(warn),
            json_panel,
            html.H4("Resumen de umbrales (dashboard_assessment_thresholds_summary.csv)"),
            _df_table(summary),
            html.H4("README datos dashboard (extracto)"),
            dcc.Markdown(md),
            html.H4("Transiciones de batch"),
            _df_table(bt),
        ],
        className="tab-panel",
    )


app = Dash(
    __name__,
    assets_folder="assets",
    suppress_callback_exceptions=True,
    meta_tags=[
        {"http-equiv": "Cache-Control", "content": "no-cache, no-store, must-revalidate"},
        {"http-equiv": "Pragma", "content": "no-cache"},
        {"http-equiv": "Expires", "content": "0"},
    ],
)
app.title = f"Clasificacion BPC — Dashboard v{DASHBOARD_APP_VERSION}"
app.server.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
server = app.server


def _dash_asset(filename: str) -> str:
    """URL de archivo estatico en dashboard/assets/ (compatible con prefijos de ruta)."""
    return app.get_asset_url(filename)


@app.server.after_request
def _no_cache_headers(response):
    """Evita que el navegador cachee respuestas del dashboard (recarga con datos actuales)."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    html.Img(
                        src=_dash_asset("logo_utp.png"),
                        alt="Universidad Tecnológica de Pereira (UTP)",
                        className="header-logo header-logo-utp",
                    ),
                    className="header-logo-slot header-logo-slot-left",
                ),
                html.Div(
                    [
                        html.H1(
                            [
                                "Clasificacion de crudo y monitoreo de condicion — BPC ",
                                html.Span(
                                    f"(version dashboard {DASHBOARD_APP_VERSION})",
                                    style={"fontSize": "0.55em", "color": "#555", "fontWeight": "500"},
                                ),
                            ]
                        ),
                        html.P(
                            "Dashboard de decision support basado en firma vibratoria, clasificacion de batch y assessment exploratorio de condicion del activo.",
                            style={"color": "#333", "margin": "0.25rem 0 0 0", "fontSize": "0.95rem", "maxWidth": "62rem"},
                        ),
                        html.P(
                            "Los resultados se generan a partir de archivos derivados en data/dashboard. "
                            "La app no reentrena modelos ni recalcula analisis (solo lectura al arrancar este proceso).",
                            style={"color": "#555", "margin": "0.35rem 0 0 0"},
                        ),
                        html.P(
                            f"Version app {DASHBOARD_APP_VERSION} | mtime dashboard/app.py UTC: {DASH_APP_BUILD}. "
                            f"Datos CSV/JSON leidos al arrancar este proceso: {DATA_LOADED_AT}. "
                            "Si ves una version vieja: cierra otras ventanas del mismo puerto, mata procesos Python viejos en ese puerto, "
                            "reinicia el servidor y recarga (Ctrl+F5).",
                            style={"color": "#666", "fontSize": "0.78rem", "margin": "0.4rem 0 0 0"},
                        ),
                    ],
                    className="header-text-block",
                ),
                html.Div(
                    html.Img(
                        src=_dash_asset("logo_idc.png"),
                        alt="IDC Ingeniería de Confiabilidad",
                        className="header-logo header-logo-idc",
                    ),
                    className="header-logo-slot header-logo-slot-right",
                ),
            ],
            className="app-header app-header-with-logos",
        ),
        *build_alerts(),
        dcc.Tabs(
            id="main-tabs",
            value="tab-res",
            children=[
                dcc.Tab(label="1. Resumen ejecutivo", value="tab-res", children=tab_resumen()),
                dcc.Tab(label="2. Desempeno del modelo", value="tab-mod", children=tab_modelo()),
                dcc.Tab(label="3. Predicciones por ventana", value="tab-pre", children=tab_predicciones()),
                dcc.Tab(label="4. Validacion estadistica", value="tab-sta", children=tab_estadistica()),
                dcc.Tab(label="5. Interpretabilidad", value="tab-int", children=tab_interpretabilidad()),
                dcc.Tab(label="6. Assessment ponderado", value="tab-ass", children=tab_assessment()),
                dcc.Tab(label="7. Datos y advertencias metodologicas", value="tab-wrn", children=tab_advertencias()),
            ],
        ),
    ],
    className="dash-app",
)


def _filter_predictions(batch: str, pred_label: str, ok: str, w_range: list) -> pd.DataFrame:
    dfp = DATA.get("predictions_operational", pd.DataFrame()).copy()
    if dfp.empty:
        return dfp
    if not w_range or len(w_range) != 2:
        return dfp
    lo, hi = int(w_range[0]), int(w_range[1])
    dfp = dfp[(dfp["window_id"] >= lo) & (dfp["window_id"] <= hi)]
    if batch and batch != "(todos)":
        dfp = dfp[dfp["Batch"].astype(str) == batch]
    if pred_label and pred_label != "(todos)":
        dfp = dfp[dfp["y_pred_label"].astype(str) == pred_label]
    if ok == "True":
        dfp = dfp[dfp["is_correct"].astype(str).str.lower().isin(["true", "1"])]
    elif ok == "False":
        dfp = dfp[dfp["is_correct"].astype(str).str.lower().isin(["false", "0"])]
    return dfp


@callback(
    Output("pred-graph-scatter", "figure"),
    Output("pred-graph-conf", "figure"),
    Output("pred-graph-ci", "figure"),
    Output("pred-graph-count", "figure"),
    Output("pred-table-container", "children"),
    Output("op-health-graph", "figure"),
    Output("op-cond-graph", "figure"),
    Input("flt-batch", "value"),
    Input("flt-pred", "value"),
    Input("flt-ok", "value"),
    Input("flt-wid", "value"),
    Input("pred-index-series", "value"),
)
def update_predictions(batch, pred_label, ok, w_range, index_series):
    empty = fig_empty("Sin datos de predicciones")
    if DATA["predictions"].empty:
        return empty, empty, empty, empty, html.P("Ejecute: python run_pipeline.py --stage dashboard_exports"), empty, empty
    dfp = _filter_predictions(batch or "(todos)", pred_label or "(todos)", ok or "all", w_range or [0, 1])
    if dfp.empty:
        empty_f = fig_empty("Sin filas con los filtros seleccionados")
        return empty_f, empty_f, empty_f, empty_f, html.P("Sin filas."), empty_f, empty_f
    dfp = _sort_for_time_series_plots(dfp)
    use_t = _plot_time_axis_usable(dfp)
    xcol = "_plot_time" if use_t else "window_id"
    color_map = {k: CLASS_COLORS.get(k, "#888") for k in dfp["y_pred_label"].astype(str).unique()}
    fig_sc = px.scatter(
        dfp,
        x=xcol,
        y="y_pred_label",
        color="y_pred_label",
        color_discrete_map=color_map,
        title="Clase predicha vs tiempo (inicio de ventana)" if use_t else "Clase predicha vs ventana (window_id)",
    )
    fig_sc.update_layout(paper_bgcolor="white", height=360, xaxis=_plot_xaxis_layout(dfp))
    if use_t and "window_id" in dfp.columns:
        fig_sc.update_traces(
            customdata=dfp["window_id"],
            hovertemplate="tiempo=%{x|%Y-%m-%d %H:%M}<br>window_id=%{customdata}<br>clase=%{y}<extra></extra>",
        )
    else:
        fig_sc.update_traces(hovertemplate="window_id=%{x}<br>clase=%{y}<extra></extra>")
    fig_cf = px.line(
        dfp,
        x=xcol,
        y="confidence",
        title="Confianza del clasificador vs tiempo (inicio de ventana)" if use_t else "Confianza del clasificador vs window_id",
    )
    fig_cf.update_traces(line=dict(color="#444"))
    fig_cf.update_layout(paper_bgcolor="white", height=300, xaxis=_plot_xaxis_layout(dfp))
    if use_t and "window_id" in dfp.columns:
        fig_cf.update_traces(
            customdata=dfp["window_id"],
            hovertemplate="tiempo=%{x|%Y-%m-%d %H:%M}<br>window_id=%{customdata}<br>Confianza del clasificador=%{y:.2f}<extra></extra>",
        )
    else:
        fig_cf.update_traces(hovertemplate="window_id=%{x}<br>Confianza del clasificador=%{y:.2f}<extra></extra>")
    sel = [s for s in (list(index_series) if index_series else list(PRED_INDEX_SERIES_DEFAULT)) if s in PRED_INDEX_SERIES_ALLOWED]
    if not sel:
        sel = list(PRED_INDEX_SERIES_DEFAULT)
    if "health_index_thresholded_by_batch" not in dfp.columns:
        sel = [s for s in sel if s != "health_batch"]
    if not sel:
        sel = ["by_batch"]
    fig_ci = fig_condition_indices_multi(dfp, sel)
    vc = dfp["y_pred_label"].astype(str).value_counts().reindex(CLASS_ORDER, fill_value=0).reset_index()
    vc.columns = ["clase", "n"]
    vc_ct = vc.assign(_n=pd.to_numeric(vc["n"], errors="coerce")).sort_values(
        ["_n", "clase"], ascending=[False, True], na_position="last"
    )
    fig_ct = px.bar(
        vc_ct,
        x="clase",
        y="n",
        title="Conteo de clases predichas (filtro activo)",
        color="clase",
        color_discrete_map=CLASS_COLORS,
    )
    fig_ct.update_layout(
        showlegend=False,
        paper_bgcolor="white",
        height=320,
        xaxis=dict(
            categoryorder="array",
            categoryarray=_bar_sorted_categoryarray(vc_ct, "clase", "n", largest_first=True),
        ),
    )
    fig_ct.update_traces(hovertemplate="clase=%{x}<br>n=%{y:.0f}<extra></extra>")
    show = [
        "window_id",
        "window_start",
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
        "condition_index_thresholded_global",
        "condition_index_thresholded_by_batch",
        "baseline_batch_used",
        "health_index_thresholded_by_batch",
        "baseline_batch_operational",
        "baseline_status",
        "health_index_operational",
        "condition_index_operational",
    ]
    cols = [c for c in show if c in dfp.columns]
    tbl = dash_table.DataTable(
        columns=[{"name": PREDICTION_TABLE_COLUMN_LABELS.get(c, c), "id": c} for c in cols],
        data=round_numeric_df(dfp[cols].head(200)).to_dict("records"),
        page_size=15,
        style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
        style_cell={"fontSize": "0.75rem", "padding": "4px"},
        tooltip_header=_prediction_table_tooltip_header_for_datatable(cols),
    )
    fig_op_h = fig_health_index_operational(dfp)
    fig_op_c = fig_condition_index_operational(dfp)
    return fig_sc, fig_cf, fig_ci, fig_ct, tbl, fig_op_h, fig_op_c


@callback(Output("var-thresh-graph", "figure"), Input("var-thresh-raw", "value"), Input("var-thresh-mode", "value"))
def update_var_thresh_graph(raw_var: str | None, mode: str | None) -> go.Figure:
    return build_variable_threshold_figure(str(raw_var or ""), str(mode or "predicted"))


if __name__ == "__main__":
    print("=== Clasificacion BPC / dashboard ===")
    print(f"Dashboard app version: {DASHBOARD_APP_VERSION}")
    for msg in DATA.get("_health_warnings") or []:
        print("[dashboard health]", msg)
    if not (DATA.get("_health_warnings") or []):
        print("[dashboard health] OK: sin advertencias de validacion threshold.")
    for msg in DATA.get("_condition_state_warnings") or []:
        print("[condition_state]", msg)
    if not (DATA.get("_condition_state_warnings") or []):
        print("[condition_state] OK: artefactos de estado exportados presentes y validaciones basicas cumplen.")
    for msg in DATA.get("_operational_10d_warnings") or []:
        print("[operational baseline]", msg)
    if not (DATA.get("_operational_10d_warnings") or []):
        print("[operational baseline] OK: columnas operacionales listas para graficos con linea base predicha.")
    port_env = os.environ.get("PORT")
    if port_env is not None:
        host = "0.0.0.0"
        port = int(port_env)
    else:
        host = "127.0.0.1"
        port = int(os.environ.get("DASH_PORT", "8050"))
    print(f"Abrir: http://{host}:{port}/  (local: si 8050 esta ocupado, use DASH_PORT=8051 python dashboard/app.py)")
    print(f"Version app.py (mtime UTC): {DASH_APP_BUILD} | datos cargados: {DATA_LOADED_AT} | app v{DASHBOARD_APP_VERSION}")
    app.run(debug=False, host=host, port=port)
