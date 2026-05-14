"""Interpretabilidad global del modelo final (permutacion, SHAP, rankings agregados)."""

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src import statistical_validation as sval
from src.schema import infer_component, infer_position, infer_signal_family, infer_unit

_STAT_SUFFIXES: frozenset[str] = frozenset({"median", "mean", "iqr", "p95", "p05"})


def load_model_artifacts(models_dir: str | Path) -> dict[str, Any]:
    """Carga modelo final, encoder, listas de columnas y metadatos desde outputs/models."""
    root = Path(models_dir)
    model = joblib.load(root / "model_final_xgboost.pkl")
    label_encoder = joblib.load(root / "label_encoder.pkl")
    with (root / "feature_columns.json").open(encoding="utf-8") as f:
        feature_columns: list[str] = json.load(f)
    with (root / "class_names.json").open(encoding="utf-8") as f:
        class_names: list[str] = json.load(f)
    with (root / "final_model_metadata.json").open(encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)
    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "class_names": class_names,
        "metadata": metadata,
    }


def prepare_interpretability_dataset(
    windowed_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "Batch",
    exclude_transition_windows: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepara X, y y metadata para interpretabilidad (mismas reglas que modelado)."""
    df_interpretability = sval.get_analysis_dataset(
        windowed_df,
        target_col=target_col,
        exclude_transition_windows=exclude_transition_windows,
    )
    missing = [c for c in feature_columns if c not in df_interpretability.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas de features en el dataset: {len(missing)} (ej. {missing[:2]})."
        )
    X = df_interpretability[feature_columns].copy()
    if X.shape[1] != 120:
        raise ValueError(f"X debe tener 120 columnas; se encontraron {X.shape[1]}.")
    if X.isna().any().any():
        raise ValueError("Hay valores NaN en X.")
    y = df_interpretability[target_col].astype(str).copy()
    if int(y.nunique()) != 3:
        raise ValueError(f"Se esperaban 3 clases; se encontraron {int(y.nunique())}.")
    return X, y, df_interpretability


def parse_windowed_feature_name(feature_name: str) -> dict[str, str]:
    """Parsea nombre ventaneado <raw>__<statistic> hacia componentes de negocio."""
    if "__" not in feature_name:
        raise ValueError(f"Nombre de feature sin sufijo __statistic: {feature_name[:80]}")
    raw_variable, statistic = feature_name.rsplit("__", 1)
    if statistic not in _STAT_SUFFIXES:
        statistic = "unknown"
    base_for_infer = raw_variable if raw_variable else feature_name
    return {
        "feature": feature_name,
        "raw_variable": raw_variable,
        "statistic": statistic,
        "family": infer_signal_family(base_for_infer),
        "component": infer_component(base_for_infer),
        "position": infer_position(base_for_infer),
        "unit": infer_unit(base_for_infer),
    }


def build_windowed_feature_metadata(feature_columns: list[str]) -> pd.DataFrame:
    """Tabla de metadatos por feature ventaneada (120 filas)."""
    rows = [parse_windowed_feature_name(f) for f in feature_columns]
    meta = pd.DataFrame(rows)
    if len(meta) != 120:
        raise ValueError(f"Se esperaban 120 filas de metadata; se obtuvieron {len(meta)}.")
    return meta


def compute_permutation_importance_table(
    model: Any,
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    feature_columns: list[str],
    scoring: str = "f1_macro",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Importancia por permutacion (sklearn) sobre matriz densa compatible con XGBoost."""
    X_np = X[feature_columns].to_numpy(dtype=np.float64)
    perm = permutation_importance(
        model,
        X_np,
        y_encoded,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )
    ranks = pd.Series(perm.importances_mean).rank(ascending=False, method="min").astype(int)
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
            "perm_importance_rank": ranks.to_numpy(),
        }
    )


def compute_shap_global_importance(
    model: Any,
    X: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
    max_samples: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Any]:
    """
    SHAP global (TreeExplainer): importancia media de |SHAP| y, si aplica, por clase.
    El segundo valor es un dict con arrays auxiliares para graficos, o None si falla.
    """
    rng = np.random.default_rng(random_state)
    n_take = min(max_samples, len(X))
    idx = rng.choice(len(X), size=n_take, replace=False)
    X_s = X.iloc[idx][feature_columns].to_numpy(dtype=np.float64)
    empty_extra: dict[str, Any] = {
        "status": "failed",
        "warning": "",
        "values": None,
        "X": None,
        "explainer": None,
    }

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_s)
    except Exception as exc:  # noqa: BLE001
        empty_extra["warning"] = f"TreeExplainer: {exc}"
        n_feat = len(feature_columns)
        df_fail = pd.DataFrame(
            {
                "feature": feature_columns,
                "shap_importance_mean_abs": np.full(n_feat, np.nan),
                "shap_rank": np.full(n_feat, np.nan),
                "shap_CASTILLA": np.full(n_feat, np.nan),
                "shap_MEZCLA": np.full(n_feat, np.nan),
                "shap_RUBIALES": np.full(n_feat, np.nan),
            }
        )
        return df_fail, empty_extra

    try:
        if isinstance(sv.values, list):
            vals = np.stack([np.asarray(a) for a in sv.values], axis=-1)
        else:
            vals = np.asarray(sv.values)
        shap_warning = ""

        if vals.ndim == 3:
            n_s, n_f, n_c = X_s.shape[0], len(feature_columns), len(class_names)
            if vals.shape == (n_c, n_s, n_f):
                vals = np.transpose(vals, (1, 2, 0))
            if vals.shape[0] != n_s or vals.shape[1] != n_f:
                raise ValueError(
                    f"Forma SHAP 3D inesperada: {vals.shape}; esperado ({n_s}, {n_f}, {n_c})."
                )
            mean_abs_global = np.abs(vals).mean(axis=(0, 2))
            per_class = {}
            for c in range(vals.shape[2]):
                lab = class_names[c] if c < len(class_names) else str(c)
                per_class[f"shap_{lab}"] = np.abs(vals[:, :, c]).mean(axis=0)
        elif vals.ndim == 2:
            mean_abs_global = np.abs(vals).mean(axis=0)
            n_feat = vals.shape[1]
            per_class = {f"shap_{lab}": np.full(n_feat, np.nan) for lab in class_names}
            shap_warning = "SHAP sin eje de clase; columnas por clase en NaN."
        else:
            raise ValueError(f"Forma SHAP no soportada: {vals.shape}")

        rank = pd.Series(mean_abs_global).rank(ascending=False, method="min")
        out = pd.DataFrame(
            {
                "feature": feature_columns,
                "shap_importance_mean_abs": mean_abs_global,
                "shap_rank": rank.astype("int64").to_numpy(),
            }
        )
        for lab in class_names:
            col = f"shap_{lab}"
            if col in per_class:
                out[col] = per_class[col]
            else:
                out[col] = np.nan

        extra: dict[str, Any] = {
            "status": "ok",
            "warning": shap_warning,
            "values": vals,
            "X": X_s,
            "explainer": explainer,
            "explanation": sv,
        }
        return out, extra
    except Exception as exc:  # noqa: BLE001
        empty_extra["warning"] = f"procesamiento_shap: {exc}"
        n_feat = len(feature_columns)
        df_fail2 = pd.DataFrame(
            {
                "feature": feature_columns,
                "shap_importance_mean_abs": np.full(n_feat, np.nan),
                "shap_rank": np.full(n_feat, np.nan),
                "shap_CASTILLA": np.full(n_feat, np.nan),
                "shap_MEZCLA": np.full(n_feat, np.nan),
                "shap_RUBIALES": np.full(n_feat, np.nan),
            }
        )
        return df_fail2, empty_extra


def build_consolidated_ranking(
    permutation_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    feature_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Une permutacion, SHAP y metadata; score consolidado como media de ranks normalizados."""
    n = len(permutation_df)
    if n < 2:
        denom = 1.0
    else:
        denom = float(n - 1)
    perm = permutation_df.copy()
    perm["perm_score_norm"] = 1.0 - (perm["perm_importance_rank"].astype(float) - 1.0) / denom

    shap = shap_df.copy()
    if shap["shap_importance_mean_abs"].notna().any() and shap["shap_rank"].notna().all():
        shap["shap_rank_norm"] = 1.0 - (shap["shap_rank"].astype(float) - 1.0) / denom
    else:
        shap["shap_rank_norm"] = np.nan

    merged = feature_metadata_df.merge(perm, on="feature", how="left").merge(
        shap, on="feature", how="left"
    )
    if merged["shap_rank_norm"].notna().any():
        merged["consolidated_score"] = merged[["perm_score_norm", "shap_rank_norm"]].mean(
            axis=1, skipna=True
        )
        mask = merged["shap_rank_norm"].isna()
        merged.loc[mask, "consolidated_score"] = merged.loc[mask, "perm_score_norm"]
    else:
        merged["consolidated_score"] = merged["perm_score_norm"]

    merged = merged.sort_values("consolidated_score", ascending=False, na_position="last")
    merged["consolidated_rank"] = np.arange(1, len(merged) + 1)
    return merged[
        [
            "feature",
            "raw_variable",
            "statistic",
            "family",
            "component",
            "position",
            "unit",
            "perm_importance_mean",
            "perm_importance_std",
            "perm_importance_rank",
            "shap_importance_mean_abs",
            "shap_rank",
            "consolidated_score",
            "consolidated_rank",
        ]
    ]


def aggregate_importance(
    ranking_df: pd.DataFrame,
    group_col: str,
    score_col: str = "consolidated_score",
) -> pd.DataFrame:
    """Agrega score consolidado por grupo (suma, media, conteo y participacion)."""
    sub = ranking_df[[group_col, score_col]].dropna(subset=[score_col])
    agg = sub.groupby(group_col, as_index=False).agg(
        importance_sum=(score_col, "sum"),
        importance_mean=(score_col, "mean"),
        n_features=(score_col, "count"),
    )
    total = float(agg["importance_sum"].sum())
    agg["share_percent"] = np.where(
        total > 0, 100.0 * agg["importance_sum"].astype(float) / total, 0.0
    )
    return agg.sort_values("importance_sum", ascending=False).reset_index(drop=True)


def plot_top_importance(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    output_path: str | Path,
    title: str,
    top_n: int = 20,
) -> None:
    """Barras horizontales de top importancias."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_ok = df.dropna(subset=[value_col])
    if df_ok.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.05, 0.5, "Sin datos para graficar", fontsize=11)
        ax.axis("off")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    sub = df_ok.nlargest(top_n, value_col).sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.22)))
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub[value_col].to_numpy(), color="steelblue", alpha=0.85)
    labels = sub[label_col].astype(str).to_list()
    if labels:
        max_len = 55
        labels = [s if len(s) <= max_len else s[: max_len - 3] + "..." for s in labels]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_importance(
    agg_df: pd.DataFrame,
    group_col: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Barras horizontales de importancia agregada por grupo."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = agg_df.sort_values("importance_sum", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(sub) * 0.35)))
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub["importance_sum"].to_numpy(), color="coral", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub[group_col].astype(str).to_list())
    ax.set_xlabel("importance_sum (consolidated_score)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap_summary_beeswarm(
    model: Any,
    X_sample: pd.DataFrame,
    feature_columns: list[str],
    output_path: str | Path,
) -> str:
    """
    Intenta beeswarm SHAP; si falla (p. ej. multiclase), grafica barras top SHAP.
    Devuelve mensaje de advertencia vacio o texto descriptivo.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X_np = X_sample[feature_columns].to_numpy(dtype=np.float64)
    warn_msg = ""
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_np)
        vals = np.array(sv.values)
        if vals.ndim == 3:
            mean_abs = np.abs(vals).mean(axis=(0, 2))
            top_idx = np.argsort(mean_abs)[-20:][::-1]
            fig, ax = plt.subplots(figsize=(8, 6))
            names = [feature_columns[i][:50] for i in top_idx]
            ax.barh(np.arange(len(top_idx)), mean_abs[top_idx][::-1], color="teal", alpha=0.85)
            ax.set_yticks(np.arange(len(top_idx)))
            ax.set_yticklabels(names[::-1], fontsize=6)
            ax.set_title("SHAP top 20 (mean |value|) - alternativa a beeswarm multiclase")
            ax.set_xlabel("mean |SHAP|")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            warn_msg = "beeswarm_multiclase_alternativa_barras_top20"
            return warn_msg
        shap.plots.beeswarm(sv, max_display=20, show=False)
        fig = plt.gcf()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return warn_msg
    except Exception as exc:  # noqa: BLE001
        warn_msg = f"shap_plot_fallback: {exc}"
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer(X_np)
            vals = np.asarray(sv.values)
            if vals.ndim == 3:
                mean_abs = np.abs(vals).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(vals).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-20:][::-1]
            fig, ax = plt.subplots(figsize=(8, 6))
            names = [feature_columns[i][:50] for i in top_idx]
            ax.barh(np.arange(len(top_idx)), mean_abs[top_idx][::-1], color="gray", alpha=0.85)
            ax.set_yticks(np.arange(len(top_idx)))
            ax.set_yticklabels(names[::-1], fontsize=6)
            ax.set_title("SHAP top 20 (fallback)")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        except Exception as exc2:  # noqa: BLE001
            warn_msg = f"{warn_msg}; segundo_fallo: {exc2}"
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.1, 0.5, "SHAP figure no disponible", fontsize=12)
            ax.axis("off")
            fig.savefig(path, dpi=150)
            plt.close(fig)
        return warn_msg
