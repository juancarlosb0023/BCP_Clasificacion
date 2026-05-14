"""Ventaneo temporal, agregaciones y construccion de variables de entrada para el modelado."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import TARGET_COL, TIME_COL, WINDOW_SECONDS

_STAT_SUFFIXES: tuple[str, ...] = ("__median", "__mean", "__iqr", "__p95", "__p05")
_STAT_NAMES: tuple[str, ...] = ("median", "mean", "iqr", "p95", "p05")
_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = ("__std", "__var", "skew", "kurt")


def assign_time_windows(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    window_seconds: int = WINDOW_SECONDS,
) -> pd.DataFrame:
    """Asigna window_id por ventanas fijas desde el primer timestamp (sin solapamiento)."""
    out = df.copy()
    out = out.sort_values(time_col).reset_index(drop=True)
    min_ts = out[time_col].min()
    delta_sec = (out[time_col] - min_ts).dt.total_seconds()
    out["window_id"] = np.floor(delta_sec / float(window_seconds)).astype(np.int64)
    return out


def majority_label(
    labels: pd.Series,
    min_majority: float = 0.8,
) -> tuple[str | None, float]:
    """Devuelve la clase mayoritaria si alcanza la fraccion minima; si no, None y la fraccion."""
    if labels.empty:
        return None, 0.0
    vc = labels.value_counts(dropna=False)
    top_count = int(vc.iloc[0])
    frac = float(top_count / len(labels))
    top_label = vc.index[0]
    if frac >= min_majority:
        if pd.isna(top_label):
            return None, frac
        return str(top_label), frac
    return None, frac


def compute_iqr(series: pd.Series) -> float:
    """IQR = p75 - p25 ignorando NaN."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    q75 = float(s.quantile(0.75))
    q25 = float(s.quantile(0.25))
    return float(q75 - q25)


def compute_window_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "window_id",
) -> pd.DataFrame:
    """Agrega estadisticos por ventana para cada feature vibracional."""
    grouped = df.groupby(group_col, sort=True)
    aggs: dict[str, pd.NamedAgg] = {}
    for col in feature_cols:
        aggs[f"{col}__median"] = pd.NamedAgg(column=col, aggfunc="median")
        aggs[f"{col}__mean"] = pd.NamedAgg(column=col, aggfunc="mean")
        aggs[f"{col}__iqr"] = pd.NamedAgg(column=col, aggfunc=compute_iqr)
        aggs[f"{col}__p95"] = pd.NamedAgg(column=col, aggfunc=lambda s: float(s.quantile(0.95)))
        aggs[f"{col}__p05"] = pd.NamedAgg(column=col, aggfunc=lambda s: float(s.quantile(0.05)))
    return grouped.agg(**aggs).reset_index()


def build_window_metadata(
    df_with_windows: pd.DataFrame,
    target_col: str = TARGET_COL,
    time_col: str = TIME_COL,
    group_col: str = "window_id",
    min_majority: float = 0.8,
) -> pd.DataFrame:
    """Construye metadatos por ventana (tiempos, muestras, etiqueta y flags de transicion)."""
    has_near = "near_transition" in df_with_windows.columns
    rows: list[dict[str, Any]] = []
    for wid, g in df_with_windows.groupby(group_col, sort=True):
        window_start = g[time_col].min()
        window_end = g[time_col].max()
        n_samples = int(len(g))
        _maj_label, maj_frac = majority_label(g[target_col], min_majority=min_majority)
        mode_label = g[target_col].value_counts(dropna=False).index[0]
        batch_value = mode_label if pd.isna(mode_label) else str(mode_label)
        is_ambiguous = _maj_label is None
        if has_near:
            n_near = int(g["near_transition"].fillna(False).astype(bool).sum())
            near_frac = float(n_near / n_samples) if n_samples else 0.0
            has_tr = bool(n_near > 0)
            row = {
                "window_id": wid,
                "window_start": window_start,
                "window_end": window_end,
                "n_samples": n_samples,
                "Batch": batch_value,
                "label_majority_fraction": float(maj_frac),
                "is_ambiguous_label": bool(is_ambiguous),
                "n_near_transition_samples": n_near,
                "has_near_transition": has_tr,
                "near_transition_fraction": near_frac,
            }
        else:
            row = {
                "window_id": wid,
                "window_start": window_start,
                "window_end": window_end,
                "n_samples": n_samples,
                "Batch": batch_value,
                "label_majority_fraction": float(maj_frac),
                "is_ambiguous_label": bool(is_ambiguous),
                "n_near_transition_samples": 0,
                "has_near_transition": False,
                "near_transition_fraction": 0.0,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_windowed_model_features(
    windowed_df: pd.DataFrame,
    raw_feature_cols: list[str],
) -> None:
    """Comprueba que existan exactamente 120 features modelables y solo estadisticos permitidos."""
    model_cols = get_model_feature_columns(windowed_df)
    if len(model_cols) != 120:
        raise ValueError(
            f"Se esperaban 120 columnas modelables (24 features x 5 stats); "
            f"se encontraron {len(model_cols)}."
        )
    for col in model_cols:
        low = col.lower()
        if any(bad in low for bad in _FORBIDDEN_SUBSTRINGS):
            raise ValueError(
                f"Columna modelable no permitida en esta etapa (contiene estadistico prohibido): {col}"
            )
    for raw in raw_feature_cols:
        for stat in _STAT_NAMES:
            name = f"{raw}__{stat}"
            if name not in windowed_df.columns:
                raise ValueError(
                    f"Falta la columna agregada esperada '{name}' para la feature base."
                )


def build_windowed_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
    time_col: str = TIME_COL,
    window_seconds: int = WINDOW_SECONDS,
    min_majority: float = 0.8,
    drop_ambiguous: bool = True,
) -> pd.DataFrame:
    """Pipeline completo: ventanas, metadata, agregados y filtrado de ventanas ambiguas."""
    dfw = assign_time_windows(df, time_col=time_col, window_seconds=window_seconds)
    meta = build_window_metadata(
        dfw,
        target_col=target_col,
        time_col=time_col,
        min_majority=min_majority,
    )
    stats = compute_window_stats(dfw, feature_cols, group_col="window_id")
    merged = meta.merge(stats, on="window_id", how="inner")
    n_before = int(len(merged))
    n_ambiguous = int(merged["is_ambiguous_label"].sum())
    if drop_ambiguous:
        merged = merged.loc[~merged["is_ambiguous_label"]].copy()
    if merged.empty:
        raise ValueError(
            "No quedaron ventanas despues del filtrado. Revise min_majority o los datos de entrada."
        )
    merged = merged.sort_values("window_id").reset_index(drop=True)
    merged.attrs["n_ambiguous_windows_dropped"] = n_ambiguous
    merged.attrs["n_windows_before_drop"] = n_before
    merged.attrs["n_windows_after_drop"] = int(len(merged))
    merged.attrs["window_seconds"] = int(window_seconds)
    merged.attrs["min_majority"] = float(min_majority)
    _validate_windowed_model_features(merged, raw_feature_cols=feature_cols)
    return merged


def get_model_feature_columns(windowed_df: pd.DataFrame) -> list[str]:
    """Lista columnas de agregacion (__median, __mean, etc.) excluyendo metadatos."""
    cols: list[str] = []
    for c in windowed_df.columns:
        if "__" not in c:
            continue
        if any(c.endswith(suf) for suf in _STAT_SUFFIXES):
            stat = c.rsplit("__", 1)[-1]
            if stat in _STAT_NAMES:
                cols.append(c)
    return sorted(cols)


def build_feature_schema(
    windowed_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Resume dimensiones y columnas del dataset ventaneado para auditoria."""
    _ = metadata_df  # reservado para extensiones futuras
    model_cols = get_model_feature_columns(windowed_df)
    meta_cols = sorted(
        [c for c in windowed_df.columns if c not in set(model_cols)]
    )
    count_by_stat: dict[str, int] = {
        stat: sum(1 for c in model_cols if c.endswith(f"__{stat}")) for stat in _STAT_NAMES
    }
    return {
        "n_windows": int(len(windowed_df)),
        "n_columns_total": int(len(windowed_df.columns)),
        "n_model_features": int(len(model_cols)),
        "window_seconds": int(windowed_df.attrs.get("window_seconds", WINDOW_SECONDS)),
        "stats_used": list(_STAT_NAMES),
        "metadata_columns": meta_cols,
        "model_feature_columns": model_cols,
        "count_by_stat": count_by_stat,
    }
