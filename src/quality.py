"""Controles de calidad de datos: integridad, duplicados, rangos y consistencia temporal por batch."""

from __future__ import annotations

import bisect
from typing import Any

import numpy as np
import pandas as pd

from src.config import TARGET_COL, TIME_COL


def _series_to_ns_int64(series: pd.Series) -> np.ndarray:
    """Convierte timestamps a entero int64 de nanosegundos de forma estable."""
    s_dt = pd.to_datetime(series, errors="coerce")
    return np.asarray(s_dt, dtype="datetime64[ns]").astype(np.int64)


def summarize_dataset(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Resume dimensiones, tiempo, clases y duplicados en formato metric/value."""
    df = df.sort_values(time_col).reset_index(drop=True)
    n_rows = int(len(df))
    n_columns = int(len(df.columns))
    feature_cols = [c for c in df.columns if c not in {time_col, target_col}]
    n_features = int(len(feature_cols))
    start_time = df[time_col].min()
    end_time = df[time_col].max()
    duration_hours = float((end_time - start_time).total_seconds() / 3600.0)
    classes = sorted(df[target_col].dropna().astype(str).unique().tolist())
    n_classes = int(len(classes))
    class_names = ",".join(classes)
    duplicated_rows = int(df.duplicated().sum())
    duplicated_timestamps = int(df.duplicated(subset=[time_col]).sum())

    metrics: list[dict[str, Any]] = [
        {"metric": "n_rows", "value": n_rows},
        {"metric": "n_columns", "value": n_columns},
        {"metric": "n_features", "value": n_features},
        {"metric": "start_time", "value": str(start_time)},
        {"metric": "end_time", "value": str(end_time)},
        {"metric": "duration_hours", "value": round(duration_hours, 6)},
        {"metric": "n_classes", "value": n_classes},
        {"metric": "class_names", "value": class_names},
        {"metric": "duplicated_rows", "value": duplicated_rows},
        {"metric": "duplicated_timestamps", "value": duplicated_timestamps},
    ]
    return pd.DataFrame(metrics)


def compute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta y porcentaje de valores faltantes por columna, ordenado por severidad."""
    n = len(df)
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_percent = float(100.0 * missing_count / n) if n else 0.0
        rows.append(
            {
                "column": col,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("missing_percent", ascending=False).reset_index(drop=True)


def compute_class_distribution(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Distribucion de frecuencias de la variable objetivo por clase."""
    counts = df[target_col].value_counts(dropna=False)
    total = int(counts.sum())
    rows = []
    for cls, cnt in counts.items():
        pct = float(100.0 * int(cnt) / total) if total else 0.0
        rows.append({"class": cls, "count": int(cnt), "percent": pct})
    out = pd.DataFrame(rows)
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def detect_timestamp_gaps(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    expected_seconds: float = 1.0,
    tolerance_seconds: float = 0.5,
) -> pd.DataFrame:
    """Detecta intervalos entre muestras consecutivas mayores al umbral esperado."""
    df = df.sort_values(time_col).reset_index(drop=True)
    threshold = float(expected_seconds + tolerance_seconds)
    prev_ts = df[time_col].shift(1)
    curr_ts = df[time_col]
    delta = (curr_ts - prev_ts).dt.total_seconds()
    mask = delta > threshold
    idx = df.index[mask.fillna(False)]
    rows: list[dict[str, Any]] = []
    for i in idx:
        rows.append(
            {
                "previous_timestamp": prev_ts.loc[i],
                "current_timestamp": curr_ts.loc[i],
                "delta_seconds": float(delta.loc[i]),
            }
        )
    cols = ["previous_timestamp", "current_timestamp", "delta_seconds"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def detect_batch_transitions(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Identifica cambios de Batch entre filas consecutivas ordenadas por tiempo."""
    df = df.sort_values(time_col).reset_index(drop=True)
    cols = [
        "transition_id",
        "timestamp",
        "previous_batch",
        "current_batch",
        "previous_timestamp",
        "delta_seconds_since_previous_sample",
    ]
    prev_batch = df[target_col].shift(1)
    prev_ts = df[time_col].shift(1)
    curr_batch = df[target_col]
    curr_ts = df[time_col]
    delta_sec = (curr_ts - prev_ts).dt.total_seconds()
    mask = prev_batch.notna() & (curr_batch != prev_batch)
    idx = df.index[mask]
    if len(idx) == 0:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for tid, i in enumerate(idx, start=1):
        rows.append(
            {
                "transition_id": int(tid),
                "timestamp": curr_ts.loc[i],
                "previous_batch": prev_batch.loc[i],
                "current_batch": curr_batch.loc[i],
                "previous_timestamp": prev_ts.loc[i],
                "delta_seconds_since_previous_sample": float(delta_sec.loc[i]),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def flag_transition_neighborhoods(
    df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    time_col: str = TIME_COL,
    buffer_seconds: float = 60.0,
) -> pd.DataFrame:
    """Marca filas cercanas a transiciones de Batch dentro de un buffer temporal."""
    out = df.copy()
    out = out.sort_values(time_col).reset_index(drop=True)

    if transitions_df is None or transitions_df.empty:
        out["near_transition"] = False
        out["transition_id_nearest"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["seconds_to_nearest_transition"] = np.nan
        return out

    trans_times = pd.to_datetime(transitions_df["timestamp"], errors="coerce")
    trans_ids = transitions_df["transition_id"].astype("Int64")
    trans_times_ns = _series_to_ns_int64(trans_times)
    order = np.argsort(trans_times_ns)
    trans_times_ns = trans_times_ns[order]
    trans_ids_arr = trans_ids.iloc[order].to_numpy()

    row_times_ns = _series_to_ns_int64(out[time_col])
    ns_buffer = int(buffer_seconds * 1_000_000_000)

    near_flags: list[bool] = []
    nearest_ids: list[Any] = []
    nearest_signed_sec: list[float] = []

    for te in row_times_ns:
        pos = bisect.bisect_left(trans_times_ns, te)
        best_tid: int | None = None
        best_abs: int | None = None
        best_signed_ns: int | None = None

        for cand in (pos - 1, pos):
            if cand < 0 or cand >= len(trans_times_ns):
                continue
            diff = int(te - int(trans_times_ns[cand]))
            ad = abs(diff)
            tid = int(trans_ids_arr[cand])
            if (
                best_abs is None
                or ad < best_abs
                or (ad == best_abs and best_tid is not None and tid < best_tid)
            ):
                best_abs = ad
                best_tid = tid
                best_signed_ns = diff

        if best_abs is None or best_signed_ns is None or best_tid is None:
            near_flags.append(False)
            nearest_ids.append(pd.NA)
            nearest_signed_sec.append(float("nan"))
            continue

        signed_sec = float(best_signed_ns) / 1_000_000_000.0
        is_near = best_abs <= ns_buffer
        near_flags.append(bool(is_near))
        nearest_ids.append(int(best_tid) if is_near else pd.NA)
        nearest_signed_sec.append(signed_sec if is_near else float("nan"))

    out["near_transition"] = near_flags
    out["transition_id_nearest"] = pd.array(nearest_ids, dtype="Int64")
    out["seconds_to_nearest_transition"] = nearest_signed_sec
    return out


def summarize_numeric_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Estadisticos descriptivos basicos por columna numerica de features."""
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        rows.append(
            {
                "column": col,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if series.notna().sum() > 1 else float("nan"),
                "min": float(series.min()),
                "p01": float(series.quantile(0.01)),
                "p05": float(series.quantile(0.05)),
                "median": float(series.median()),
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def validate_batch_values(
    df: pd.DataFrame,
    expected_classes: list[str],
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Contrasta valores observados en Batch frente a una lista de clases esperadas."""
    counts = df[target_col].value_counts(dropna=False)
    expected_set = {str(x) for x in expected_classes}
    rows = []
    for val, cnt in counts.items():
        if pd.isna(val):
            is_expected = False
        else:
            is_expected = str(val) in expected_set
        rows.append(
            {
                "batch_value": val,
                "count": int(cnt),
                "is_expected": bool(is_expected),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
