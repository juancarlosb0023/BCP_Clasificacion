"""Estrategias de particion de datos, validacion cruzada y protocolos sin fugas temporales."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def create_temporal_groups(
    windowed_df: pd.DataFrame,
    group_size_windows: int = 30,
    window_id_col: str = "window_id",
) -> pd.Series:
    """Agrupa ventanas contiguas por bloques de window_id (sin reordenar el DataFrame)."""
    if window_id_col not in windowed_df.columns:
        raise ValueError(
            f"No existe la columna '{window_id_col}' requerida para grupos temporales."
        )
    wid = windowed_df[window_id_col].astype(int)
    grp = (wid // int(group_size_windows)).astype(int)
    return pd.Series(grp.to_numpy(), index=windowed_df.index, name="temporal_group_id")


def get_stratified_group_kfold(
    n_splits: int = 5,
    random_state: int = 42,
) -> StratifiedGroupKFold:
    """Particion estratificada por grupos temporales (sin mezclar grupos entre train y test)."""
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def summarize_cv_groups(
    y: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    """Resume composicion de clases por grupo temporal."""
    df = pd.DataFrame({"y": y.astype(str), "group_id": groups.astype(int).to_numpy()})
    rows: list[dict[str, Any]] = []
    for gid, g in df.groupby("group_id", sort=True):
        vc = g["y"].value_counts()
        dominant = str(vc.idxmax()) if len(vc) else ""
        rows.append(
            {
                "group_id": int(gid),
                "n_samples": int(len(g)),
                "count_CASTILLA": int(vc.get("CASTILLA", 0)),
                "count_MEZCLA": int(vc.get("MEZCLA", 0)),
                "count_RUBIALES": int(vc.get("RUBIALES", 0)),
                "dominant_class": dominant,
            }
        )
    return pd.DataFrame(rows)


def validate_cv_splits(
    cv: StratifiedGroupKFold,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    """Verifica particiones CV: grupos disjuntos y cobertura de clases por fold."""
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        g_tr = set(groups.iloc[train_idx].astype(int).unique().tolist())
        g_te = set(groups.iloc[test_idx].astype(int).unique().tolist())
        overlap = int(len(g_tr & g_te))
        if overlap != 0:
            raise ValueError(
                f"Fuga de grupos en fold {fold}: {overlap} grupos compartidos entre train y test."
            )
        y_tr = y.iloc[train_idx].astype(str)
        y_te = y.iloc[test_idx].astype(str)
        train_classes = sorted(y_tr.unique().tolist())
        test_classes = sorted(y_te.unique().tolist())
        train_all = set(train_classes) >= {"CASTILLA", "MEZCLA", "RUBIALES"}
        test_all = set(test_classes) >= {"CASTILLA", "MEZCLA", "RUBIALES"}
        warn_parts: list[str] = []
        if not train_all:
            warn_parts.append("train_sin_3_clases")
        if not test_all:
            warn_parts.append("test_sin_3_clases")
        rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_groups": int(len(g_tr)),
                "n_test_groups": int(len(g_te)),
                "train_classes": ",".join(train_classes),
                "test_classes": ",".join(test_classes),
                "train_group_overlap_with_test": overlap,
                "class_coverage_warning": "|".join(warn_parts) if warn_parts else "",
            }
        )
    return pd.DataFrame(rows)
