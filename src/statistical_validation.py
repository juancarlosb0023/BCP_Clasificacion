"""Pruebas y analisis estadisticos para validar separabilidad entre batches (Kruskal, PCA, PERMANOVA)."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

_STAT_SUFFIXES: tuple[str, ...] = (
    "__median",
    "__mean",
    "__iqr",
    "__p95",
    "__p05",
)


def get_model_feature_columns(windowed_df: pd.DataFrame) -> list[str]:
    """Devuelve las 120 columnas modelables (stats por feature vibracional)."""
    cols: list[str] = []
    for c in windowed_df.columns:
        if "__" not in c:
            continue
        if any(c.endswith(suf) for suf in _STAT_SUFFIXES):
            stat = c.rsplit("__", 1)[-1]
            if stat in {"median", "mean", "iqr", "p95", "p05"}:
                cols.append(c)
    cols_sorted = sorted(cols)
    if len(cols_sorted) != 120:
        raise ValueError(
            f"Se esperaban 120 features modelables; se encontraron {len(cols_sorted)}."
        )
    return cols_sorted


def get_analysis_dataset(
    windowed_df: pd.DataFrame,
    target_col: str = "Batch",
    exclude_transition_windows: bool = True,
) -> pd.DataFrame:
    """Prepara copia del dataset ventaneado, excluyendo ventanas cercanas a transicion."""
    out = windowed_df.copy()
    if exclude_transition_windows and "has_near_transition" in out.columns:
        mask_nt = out["has_near_transition"].fillna(False).astype(bool)
        out = out.loc[~mask_nt].copy()
    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError("El dataset de analisis quedo vacio tras los filtros.")
    n_classes = int(out[target_col].nunique(dropna=False))
    if n_classes < 2:
        raise ValueError(
            f"Se requieren al menos 2 clases en '{target_col}'; se encontraron {n_classes}."
        )
    counts = out[target_col].value_counts(dropna=False)
    if int(counts.min()) < 1:
        raise ValueError("Hay clases sin ventanas tras el filtrado.")
    return out


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Escala columnas con StandardScaler sin mutar el DataFrame de entrada."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy(dtype=float))
    return X_scaled, scaler


def run_kruskal_by_feature(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Batch",
) -> pd.DataFrame:
    """Kruskal-Wallis por feature entre clases de Batch."""
    rows: list[dict[str, Any]] = []
    labels = sorted(df[target_col].dropna().astype(str).unique().tolist())
    n_groups = int(len(labels))
    n_total = int(len(df))
    for feat in feature_cols:
        row: dict[str, Any] = {
            "feature": feat,
            "statistic": np.nan,
            "p_value": np.nan,
            "n_groups": n_groups,
            "n_total": n_total,
            "error": "",
        }
        try:
            groups = [
                df.loc[df[target_col].astype(str) == lab, feat].dropna().to_numpy()
                for lab in labels
            ]
            if any(len(g) == 0 for g in groups):
                row["error"] = "Grupo sin valores para la feature."
                rows.append(row)
                continue
            stat, pval = kruskal(*groups)
            row["statistic"] = float(stat)
            row["p_value"] = float(pval)
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_fdr_correction(
    results_df: pd.DataFrame,
    p_col: str = "p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Correccion FDR Benjamini-Hochberg; conserva NaN en p originales."""
    out = results_df.copy()
    mask = out[p_col].notna()
    out["p_value_fdr"] = np.nan
    out["reject_fdr_0_05"] = False
    if mask.sum() == 0:
        return out
    pvals = out.loc[mask, p_col].astype(float).to_numpy()
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    out.loc[mask, "p_value_fdr"] = p_adj
    out.loc[mask, "reject_fdr_0_05"] = reject
    return out


def compute_pca_projection(
    X_scaled: np.ndarray,
    y: pd.Series,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA]:
    """Proyeccion PCA 2D con varianza explicada en el objeto PCA."""
    pca = PCA(n_components=n_components, random_state=random_state)
    pcs = pca.fit_transform(X_scaled)
    projection_df = pd.DataFrame(
        {
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1] if n_components > 1 else np.zeros(len(pcs)),
            "Batch": y.astype(str).to_numpy(),
        }
    )
    return projection_df, pca


def compute_pca_centroids(projection_df: pd.DataFrame) -> pd.DataFrame:
    """Centroides por clase en el espacio PC1-PC2."""
    rows: list[dict[str, Any]] = []
    for batch, g in projection_df.groupby("Batch", sort=True):
        rows.append(
            {
                "Batch": batch,
                "PC1_centroid": float(g["PC1"].mean()),
                "PC2_centroid": float(g["PC2"].mean()),
                "n_windows": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def compute_pairwise_centroid_distances(centroids_df: pd.DataFrame) -> pd.DataFrame:
    """Distancias euclidianas entre centroides de PCA por par de clases."""
    rows: list[dict[str, Any]] = []
    df = centroids_df.set_index("Batch")
    for a, b in itertools.combinations(sorted(df.index.astype(str)), 2):
        v1 = np.array(
            [df.loc[a, "PC1_centroid"], df.loc[a, "PC2_centroid"]], dtype=float
        )
        v2 = np.array(
            [df.loc[b, "PC1_centroid"], df.loc[b, "PC2_centroid"]], dtype=float
        )
        dist = float(np.linalg.norm(v1 - v2))
        rows.append(
            {"class_a": a, "class_b": b, "centroid_distance_pca": dist}
        )
    return pd.DataFrame(rows)


def _pseudo_f_permanova(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Calcula pseudo-F y sumas de cuadrados para PERMANOVA euclidea."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    labels = np.unique(y)
    G = int(len(labels))
    n = int(X.shape[0])
    if G < 2 or n <= G:
        return float("nan"), float("nan"), float("nan")
    global_centroid = X.mean(axis=0)
    ss_total = float(np.sum(np.linalg.norm(X - global_centroid, axis=1) ** 2))
    ss_between = 0.0
    ss_within = 0.0
    for lab in labels:
        mask = y == lab
        Xg = X[mask]
        ng = int(Xg.shape[0])
        if ng == 0:
            continue
        cg = Xg.mean(axis=0)
        ss_between += float(ng * np.sum((cg - global_centroid) ** 2))
        ss_within += float(np.sum(np.linalg.norm(Xg - cg, axis=1) ** 2))
    if ss_within <= 0:
        pseudo_f = float("inf") if ss_between > 0 else float("nan")
    else:
        pseudo_f = float((ss_between / (G - 1)) / (ss_within / (n - G)))
    r2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
    return pseudo_f, ss_between, r2


def permanova(
    X_scaled: np.ndarray,
    y: pd.Series,
    n_permutations: int = 999,
    random_state: int = 42,
) -> dict[str, Any]:
    """PERMANOVA global con distancia euclidea en espacio escalado."""
    X = np.asarray(X_scaled, dtype=float)
    y_arr = np.asarray(y.astype(str))
    labels = np.unique(y_arr)
    G = int(len(labels))
    n = int(len(y_arr))
    if G < 2:
        raise ValueError("PERMANOVA requiere al menos 2 grupos.")
    counts = {lab: int(np.sum(y_arr == lab)) for lab in labels}
    if any(c < 2 for c in counts.values()):
        raise ValueError("Cada grupo debe tener al menos 2 muestras para PERMANOVA.")
    F_obs, _, r2_obs = _pseudo_f_permanova(X, y_arr)
    rng = np.random.default_rng(random_state)
    ge = 1
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_arr)
        Fp, _, _ = _pseudo_f_permanova(X, y_perm)
        if not np.isnan(Fp) and not np.isnan(F_obs) and Fp >= F_obs:
            ge += 1
        if np.isnan(F_obs) and np.isnan(Fp):
            ge += 1
    p_value = float(ge / (n_permutations + 1))
    return {
        "pseudo_F": float(F_obs) if not np.isnan(F_obs) else float("nan"),
        "p_value": p_value,
        "r2": float(r2_obs),
        "n_permutations": int(n_permutations),
        "n_samples": n,
        "n_groups": G,
    }


def _distances_to_centroids(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distancia euclidea de cada muestra al centroide de su grupo."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    dists = np.zeros(X.shape[0], dtype=float)
    for lab in np.unique(y):
        mask = y == lab
        Xg = X[mask]
        c = Xg.mean(axis=0)
        dists[mask] = np.linalg.norm(Xg - c, axis=1)
    return dists


def _f_oneway_distances(dists: np.ndarray, y: np.ndarray) -> float:
    """Estadistico F de ANOVA one-way sobre distancias por grupo."""
    groups = [dists[y == lab] for lab in np.unique(y)]
    if any(len(g) < 2 for g in groups):
        return float("nan")
    res = f_oneway(*groups)
    return float(res.statistic)


def permdisp(
    X_scaled: np.ndarray,
    y: pd.Series,
    n_permutations: int = 999,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """PERMDISP global: ANOVA one-way sobre distancias al centroide del grupo."""
    X = np.asarray(X_scaled, dtype=float)
    y_arr = np.asarray(y.astype(str))
    labels = np.unique(y_arr)
    G = int(len(labels))
    n = int(len(y_arr))
    if G < 2:
        raise ValueError("PERMDISP requiere al menos 2 grupos.")
    counts = {lab: int(np.sum(y_arr == lab)) for lab in labels}
    if any(c < 2 for c in counts.values()):
        raise ValueError("Cada grupo debe tener al menos 2 muestras para PERMDISP.")
    dists = _distances_to_centroids(X, y_arr)
    F_obs = _f_oneway_distances(dists, y_arr)
    rng = np.random.default_rng(random_state)
    ge = 1
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_arr)
        d_perm = _distances_to_centroids(X, y_perm)
        Fp = _f_oneway_distances(d_perm, y_perm)
        if not np.isnan(Fp) and not np.isnan(F_obs) and Fp >= F_obs:
            ge += 1
        if np.isnan(F_obs) and np.isnan(Fp):
            ge += 1
    p_value = float(ge / (n_permutations + 1))
    dispersion_df = pd.DataFrame({"Batch": y_arr, "distance_to_centroid": dists})
    summary = {
        "F": float(F_obs) if not np.isnan(F_obs) else float("nan"),
        "p_value": p_value,
        "n_permutations": int(n_permutations),
        "n_samples": n,
        "n_groups": G,
    }
    return summary, dispersion_df


def pairwise_permanova(
    X_scaled: np.ndarray,
    y: pd.Series,
    n_permutations: int = 999,
    random_state: int = 42,
) -> pd.DataFrame:
    """PERMANOVA por pares de clases con correccion FDR."""
    y_all = y.astype(str)
    classes = sorted(y_all.unique().tolist())
    rows: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(itertools.combinations(classes, 2)):
        mask = y_all.isin([a, b])
        Xp = np.asarray(X_scaled, dtype=float)[mask]
        yp = y_all[mask].to_numpy()
        if len(np.unique(yp)) < 2:
            continue
        if np.sum(yp == a) < 2 or np.sum(yp == b) < 2:
            raise ValueError(
                f"PERMANOVA pairwise requiere al menos 2 muestras por clase en el par {a} vs {b}."
            )
        res = permanova(
            Xp,
            pd.Series(yp),
            n_permutations=n_permutations,
            random_state=random_state + idx,
        )
        rows.append(
            {
                "class_a": a,
                "class_b": b,
                "pseudo_F": res["pseudo_F"],
                "p_value": res["p_value"],
                "r2": res["r2"],
                "n_samples": res["n_samples"],
                "n_permutations": res["n_permutations"],
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != 3:
        raise ValueError(
            f"Se esperaban exactamente 3 pares de clases; se obtuvieron {len(out)}."
        )
    out = apply_fdr_correction(out, p_col="p_value")
    return out


def pairwise_permdisp(
    X_scaled: np.ndarray,
    y: pd.Series,
    n_permutations: int = 999,
    random_state: int = 42,
) -> pd.DataFrame:
    """PERMDISP por pares con correccion FDR."""
    y_all = y.astype(str)
    classes = sorted(y_all.unique().tolist())
    rows: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(itertools.combinations(classes, 2)):
        mask = y_all.isin([a, b])
        Xp = np.asarray(X_scaled, dtype=float)[mask]
        yp = y_all[mask].to_numpy()
        if np.sum(yp == a) < 2 or np.sum(yp == b) < 2:
            raise ValueError(
                f"PERMDISP pairwise requiere al menos 2 muestras por clase en el par {a} vs {b}."
            )
        summ, disp = permdisp(
            Xp,
            pd.Series(yp),
            n_permutations=n_permutations,
            random_state=random_state + idx,
        )
        mean_a = float(disp.loc[disp["Batch"] == a, "distance_to_centroid"].mean())
        mean_b = float(disp.loc[disp["Batch"] == b, "distance_to_centroid"].mean())
        rows.append(
            {
                "class_a": a,
                "class_b": b,
                "F": summ["F"],
                "p_value": summ["p_value"],
                "mean_disp_a": mean_a,
                "mean_disp_b": mean_b,
                "n_samples": summ["n_samples"],
                "n_permutations": summ["n_permutations"],
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != 3:
        raise ValueError(
            f"Se esperaban exactamente 3 pares de clases; se obtuvieron {len(out)}."
        )
    out = apply_fdr_correction(out, p_col="p_value")
    return out


def plot_pca_batches(
    projection_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Scatter PC1 vs PC2 por Batch con centroides marcados."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for batch, g in projection_df.groupby("Batch"):
        ax.scatter(g["PC1"], g["PC2"], s=12, alpha=0.5, label=str(batch))
    ax.scatter(
        centroids_df["PC1_centroid"],
        centroids_df["PC2_centroid"],
        s=120,
        marker="X",
        c="black",
        label="centroid",
        zorder=5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=1.0, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_kruskal_features(
    kruskal_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 20,
) -> None:
    """Barras horizontales de las features con menor p_value_fdr (top_n)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = kruskal_df.copy()
    df = df.loc[df["p_value_fdr"].notna()].sort_values("p_value_fdr", ascending=True)
    pcol = "p_value_fdr"
    if df.empty:
        df = kruskal_df.loc[kruskal_df["p_value"].notna()].sort_values(
            "p_value", ascending=True
        )
        pcol = "p_value"
    df = df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(df))))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, -np.log10(df[pcol].clip(lower=1e-300).astype(float)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontsize=7)
    ax.set_xlabel("-log10(p)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permdisp_boxplot(
    dispersion_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Boxplot de distancia al centroide por Batch."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        dispersion_df.loc[dispersion_df["Batch"] == lab, "distance_to_centroid"].to_numpy()
        for lab in sorted(dispersion_df["Batch"].astype(str).unique())
    ]
    labels = sorted(dispersion_df["Batch"].astype(str).unique())
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("distance_to_centroid")
    ax.set_xlabel("Batch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
