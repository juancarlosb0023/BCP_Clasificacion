"""Punto de entrada del pipeline del Proyecto 4 (BPC)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.config import (
    CLASSES,
    DATA_DASHBOARD_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
    SENSOR_WEIGHTS_FILE,
    TABLES_DIR,
    TARGET_COL,
    TIME_COL,
    WINDOW_SECONDS,
)
from src.data_loading import load_raw_data, validate_required_columns
from src.exports import (
    export_json,
    export_table,
    run_dashboard_exports,
    save_json,
    save_pickle,
)
from src import feature_engineering
from src import quality
from src.schema import build_variable_metadata, get_feature_columns
from src import statistical_validation as sval
from src import validation as tval
from src import modeling as mdl
from src import evaluation as evl
from src import interpretability as intr
from src import assessment as asm
from src.static_dashboard_publish import run_static_dashboard_publish


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline Proyecto 4 BPC.")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Etapa: check_data, quality, features, stats, modeling, final_model, interpretability, assessment, assessment_thresholds, condition_state, dashboard_exports, static_dashboard_publish.",
    )
    return parser.parse_args(argv)


def _rel_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _run_check_data() -> None:
    df = load_raw_data()
    meta = build_variable_metadata(df)
    features = get_feature_columns(df)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "variable_metadata.csv"
    meta.to_csv(out_path, index=False)

    n_rows = len(df)
    n_cols = len(df.columns)
    n_features = len(features)
    classes = sorted(df[TARGET_COL].dropna().astype(str).unique().tolist())
    t_min = df[TIME_COL].min()
    t_max = df[TIME_COL].max()
    by_family = meta["family"].value_counts().to_dict()
    by_component = meta["component"].value_counts().to_dict()

    print(f"Filas: {n_rows}")
    print(f"Columnas totales: {n_cols}")
    print(f"Features detectadas: {n_features}")
    print(f"Clases en {TARGET_COL}: {classes}")
    print(f"Rango temporal: {t_min} -> {t_max}")
    print("Conteo por familia:")
    for k, v in sorted(by_family.items(), key=lambda kv: kv[0]):
        print(f"  {k}: {v}")
    print("Conteo por componente:")
    for k, v in sorted(by_component.items(), key=lambda kv: kv[0]):
        print(f"  {k}: {v}")
    print(f"Metadata guardada en: {_rel_path(out_path)}")


def _run_quality() -> None:
    df = load_raw_data()
    feature_cols = get_feature_columns(df)

    summary_df = quality.summarize_dataset(df)
    missing_df = quality.compute_missing_values(df)
    class_df = quality.compute_class_distribution(df)
    gaps_df = quality.detect_timestamp_gaps(df)
    transitions_df = quality.detect_batch_transitions(df)
    df_flagged = quality.flag_transition_neighborhoods(
        df, transitions_df, buffer_seconds=60.0
    )
    numeric_df = quality.summarize_numeric_features(df, feature_cols)
    batch_val_df = quality.validate_batch_values(df, CLASSES)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": TABLES_DIR / "data_quality_summary.csv",
        "missing": TABLES_DIR / "missing_values.csv",
        "class_dist": TABLES_DIR / "class_distribution.csv",
        "gaps": TABLES_DIR / "timestamp_gaps.csv",
        "transitions": TABLES_DIR / "batch_transitions.csv",
        "numeric": TABLES_DIR / "numeric_feature_summary.csv",
        "batch_val": TABLES_DIR / "batch_values_validation.csv",
        "flagged": DATA_PROCESSED_DIR / "data_with_transition_flags.csv",
    }

    export_table(summary_df, paths["summary"])
    export_table(missing_df, paths["missing"])
    export_table(class_df, paths["class_dist"])
    export_table(gaps_df, paths["gaps"])
    export_table(transitions_df, paths["transitions"])
    export_table(numeric_df, paths["numeric"])
    export_table(batch_val_df, paths["batch_val"])
    export_table(df_flagged, paths["flagged"])

    n_rows = int(len(df))
    n_cols = int(len(df.columns))
    n_features = int(len(feature_cols))
    t_min = df[TIME_COL].min()
    t_max = df[TIME_COL].max()
    n_classes = int(df[TARGET_COL].nunique(dropna=False))
    class_lines = class_df.apply(
        lambda r: f"  {r['class']}: count={int(r['count'])}, pct={round(float(r['percent']), 4)}",
        axis=1,
    ).tolist()
    total_missing = int(missing_df["missing_count"].sum())
    dup_ts = int(df.duplicated(subset=[TIME_COL]).sum())
    n_gaps = int(len(gaps_df))
    n_trans = int(len(transitions_df))
    n_near = int(df_flagged["near_transition"].sum())

    print("Resumen de calidad (ASCII)")
    print(f"Filas: {n_rows}")
    print(f"Columnas: {n_cols}")
    print(f"Features: {n_features}")
    print(f"Rango temporal: {t_min} -> {t_max}")
    print(f"Numero de clases: {n_classes}")
    print("Distribucion de clases:")
    for line in class_lines:
        print(line)
    print(f"Total missing values (todas las columnas): {total_missing}")
    print(f"Timestamps duplicados (filas extra): {dup_ts}")
    print(f"Gaps temporales detectados: {n_gaps}")
    print(f"Transiciones entre batches: {n_trans}")
    print(f"Filas near_transition (buffer 60s): {n_near}")
    print("Archivos exportados:")
    for key in (
        "summary",
        "missing",
        "class_dist",
        "gaps",
        "transitions",
        "numeric",
        "batch_val",
        "flagged",
    ):
        print(f"  {_rel_path(paths[key])}")


def _load_dataframe_for_features() -> pd.DataFrame:
    """Carga datos con flags de transicion si existen; si no, datos crudos validados."""
    flagged_path = DATA_PROCESSED_DIR / "data_with_transition_flags.csv"
    if flagged_path.is_file():
        df = pd.read_csv(flagged_path)
        validate_required_columns(df)
        df = df.copy()
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        if df[TIME_COL].isna().any():
            bad = int(df[TIME_COL].isna().sum())
            raise ValueError(
                f"Hay {bad} valores en '{TIME_COL}' no convertibles a datetime en "
                f"{flagged_path.name}."
            )
        if "near_transition" in df.columns:
            df["near_transition"] = df["near_transition"].astype(bool)
        return df.sort_values(TIME_COL).reset_index(drop=True)
    return load_raw_data()


def _run_features() -> None:
    df = _load_dataframe_for_features()
    exclude_cols = {
        "near_transition",
        "transition_id_nearest",
        "seconds_to_nearest_transition",
    }
    feature_cols = [
        c for c in get_feature_columns(df) if c not in exclude_cols
    ]

    windowed_df = feature_engineering.build_windowed_dataset(
        df,
        feature_cols,
        target_col=TARGET_COL,
        time_col=TIME_COL,
        window_seconds=WINDOW_SECONDS,
        min_majority=0.8,
        drop_ambiguous=True,
    )

    schema = feature_engineering.build_feature_schema(windowed_df)
    model_cols = feature_engineering.get_model_feature_columns(windowed_df)

    n_ambig = int(windowed_df.attrs.get("n_ambiguous_windows_dropped", 0))
    n_with_tr = int(windowed_df["has_near_transition"].sum())
    total_windows = int(len(windowed_df))
    n_cols = int(len(windowed_df.columns))
    n_model = int(len(model_cols))

    counts = windowed_df["Batch"].value_counts(dropna=False)
    total = int(counts.sum())
    class_rows = []
    for cls, cnt in counts.items():
        pct = float(100.0 * int(cnt) / total) if total else 0.0
        class_rows.append({"class": cls, "count": int(cnt), "percent": pct})
    class_dist_df = pd.DataFrame(class_rows).sort_values(
        "count", ascending=False
    ).reset_index(drop=True)

    summary_df = pd.DataFrame(
        [
            {
                "n_windows_total_after_drop": total_windows,
                "n_columns_total": n_cols,
                "n_model_features": n_model,
                "n_ambiguous_windows_dropped": n_ambig,
                "n_windows_with_transition": n_with_tr,
                "window_seconds": WINDOW_SECONDS,
                "min_majority": 0.8,
                "start_time": str(windowed_df["window_start"].min()),
                "end_time": str(windowed_df["window_end"].max()),
            }
        ]
    )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "windowed": DATA_PROCESSED_DIR / "bpc_windowed_features.csv",
        "schema": DATA_PROCESSED_DIR / "feature_schema.json",
        "summary": TABLES_DIR / "windowed_dataset_summary.csv",
        "class_dist": TABLES_DIR / "windowed_class_distribution.csv",
    }

    export_table(windowed_df, out_paths["windowed"])
    export_json(schema, out_paths["schema"])
    export_table(summary_df, out_paths["summary"])
    export_table(class_dist_df, out_paths["class_dist"])

    print("Resumen de features (ASCII)")
    print(f"Ventanas finales: {total_windows}")
    print(f"Columnas totales: {n_cols}")
    print(f"Features modelables: {n_model}")
    print(f"Ventanas ambiguas descartadas: {n_ambig}")
    print(f"Ventanas con transicion: {n_with_tr}")
    print("Distribucion de clases por ventana:")
    for _, row in class_dist_df.iterrows():
        print(
            f"  {row['class']}: count={int(row['count'])}, "
            f"pct={round(float(row['percent']), 4)}"
        )
    print("Archivos exportados:")
    for p in out_paths.values():
        print(f"  {_rel_path(p)}")


def _run_stats() -> None:
    windowed_path = DATA_PROCESSED_DIR / "bpc_windowed_features.csv"
    if not windowed_path.is_file():
        print(
            "ERROR: No existe data/processed/bpc_windowed_features.csv. "
            "Ejecute primero: python run_pipeline.py --stage features"
        )
        sys.exit(1)

    windowed_df = pd.read_csv(windowed_path)
    if TARGET_COL not in windowed_df.columns:
        raise ValueError(f"Falta la columna objetivo '{TARGET_COL}' en el dataset ventaneado.")

    n_windows_original = int(len(windowed_df))
    n_transition_excluded = 0
    if "has_near_transition" in windowed_df.columns:
        n_transition_excluded = int(windowed_df["has_near_transition"].sum())

    feature_cols = sval.get_model_feature_columns(windowed_df)
    analysis_df = sval.get_analysis_dataset(
        windowed_df, target_col=TARGET_COL, exclude_transition_windows=True
    )
    counts = analysis_df[TARGET_COL].value_counts(dropna=False)
    if int(counts.min()) < 1:
        raise ValueError("Quedo menos de 1 ventana por clase tras excluir transiciones.")

    X = analysis_df[feature_cols]
    y = analysis_df[TARGET_COL].astype(str)
    X_scaled, _scaler = sval.scale_features(X)

    kruskal_df = sval.run_kruskal_by_feature(analysis_df, feature_cols, target_col=TARGET_COL)
    kruskal_df = sval.apply_fdr_correction(kruskal_df, p_col="p_value")

    projection_df, pca = sval.compute_pca_projection(
        X_scaled, y, n_components=2, random_state=RANDOM_STATE
    )
    centroids_df = sval.compute_pca_centroids(projection_df)
    centroid_dists = sval.compute_pairwise_centroid_distances(centroids_df)

    perm_res = sval.permanova(
        X_scaled, y, n_permutations=999, random_state=RANDOM_STATE
    )
    permdisp_summ, dispersion_df = sval.permdisp(
        X_scaled, y, n_permutations=999, random_state=RANDOM_STATE
    )
    pairwise_perm = sval.pairwise_permanova(
        X_scaled, y, n_permutations=999, random_state=RANDOM_STATE
    )
    pairwise_disp = sval.pairwise_permdisp(
        X_scaled, y, n_permutations=999, random_state=RANDOM_STATE
    )

    ev = pca.explained_variance_ratio_
    pca_pc1 = float(ev[0]) if len(ev) > 0 else float("nan")
    pca_pc2 = float(ev[1]) if len(ev) > 1 else float("nan")

    summary_row = {
        "n_windows_original": n_windows_original,
        "n_windows_used": int(len(analysis_df)),
        "n_transition_windows_excluded": n_transition_excluded,
        "n_model_features": int(len(feature_cols)),
        "n_classes": int(y.nunique()),
        "permanova_pseudo_F": perm_res["pseudo_F"],
        "permanova_p_value": perm_res["p_value"],
        "permanova_r2": perm_res["r2"],
        "permdisp_F": permdisp_summ["F"],
        "permdisp_p_value": permdisp_summ["p_value"],
        "pca_explained_variance_PC1": pca_pc1,
        "pca_explained_variance_PC2": pca_pc2,
    }
    summary_df = pd.DataFrame([summary_row])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "kruskal": TABLES_DIR / "kruskal_results.csv",
        "pca_proj": TABLES_DIR / "pca_projection.csv",
        "pca_centroids": TABLES_DIR / "pca_centroids.csv",
        "centroid_dists": TABLES_DIR / "pairwise_centroid_distances.csv",
        "permanova": TABLES_DIR / "permanova_results.csv",
        "permdisp": TABLES_DIR / "permdisp_results.csv",
        "permdisp_dist": TABLES_DIR / "permdisp_distances.csv",
        "pair_perm": TABLES_DIR / "pairwise_permanova_fdr_results.csv",
        "pair_disp": TABLES_DIR / "pairwise_permdisp_fdr_results.csv",
        "stats_summary": TABLES_DIR / "statistical_validation_summary.csv",
        "fig_pca": FIGURES_DIR / "pca_batches.png",
        "fig_kruskal": FIGURES_DIR / "top_kruskal_features.png",
        "fig_disp": FIGURES_DIR / "permdisp_boxplot.png",
    }

    export_table(kruskal_df, paths["kruskal"])
    export_table(projection_df, paths["pca_proj"])
    export_table(centroids_df, paths["pca_centroids"])
    export_table(centroid_dists, paths["centroid_dists"])
    export_table(pd.DataFrame([perm_res]), paths["permanova"])
    export_table(pd.DataFrame([permdisp_summ]), paths["permdisp"])
    export_table(dispersion_df, paths["permdisp_dist"])
    export_table(pairwise_perm, paths["pair_perm"])
    export_table(pairwise_disp, paths["pair_disp"])
    export_table(summary_df, paths["stats_summary"])

    sval.plot_pca_batches(projection_df, centroids_df, paths["fig_pca"])
    sval.plot_top_kruskal_features(kruskal_df, paths["fig_kruskal"], top_n=20)
    sval.plot_permdisp_boxplot(dispersion_df, paths["fig_disp"])

    classes_used = sorted(y.unique().tolist())
    print("Resumen estadistico (ASCII)")
    print(f"Ventanas originales: {n_windows_original}")
    print(f"Ventanas usadas: {len(analysis_df)}")
    print(f"Ventanas de transicion excluidas: {n_transition_excluded}")
    print(f"Features modelables: {len(feature_cols)}")
    print(f"Clases usadas: {classes_used}")
    print(
        f"PERMANOVA pseudo-F: {perm_res['pseudo_F']}, "
        f"p-value: {perm_res['p_value']}, R2: {perm_res['r2']}"
    )
    print(f"PERMDISP F: {permdisp_summ['F']}, p-value: {permdisp_summ['p_value']}")
    print(f"PCA var explicada PC1: {pca_pc1}, PC2: {pca_pc2}")
    print("Archivos exportados:")
    for p in paths.values():
        print(f"  {_rel_path(p)}")


def _run_modeling() -> None:
    windowed_path = DATA_PROCESSED_DIR / "bpc_windowed_features.csv"
    if not windowed_path.is_file():
        print(
            "ERROR: No existe data/processed/bpc_windowed_features.csv. "
            "Ejecute primero: python run_pipeline.py --stage features"
        )
        sys.exit(1)

    windowed_df = pd.read_csv(windowed_path)
    X, y, df_model = mdl.prepare_modeling_dataset(
        windowed_df, target_col=TARGET_COL, exclude_transition_windows=True
    )
    if X.shape[1] != 120:
        raise ValueError(f"Se esperaban 120 features en X; se encontraron {X.shape[1]}.")
    if int(y.nunique()) != 3:
        raise ValueError("Se esperaban 3 clases en y.")
    if len(X) != 1451:
        raise ValueError(
            f"Se esperaban 1451 ventanas tras excluir transicion; se encontraron {len(X)}."
        )

    groups = tval.create_temporal_groups(df_model, group_size_windows=30)
    cv = tval.get_stratified_group_kfold(n_splits=5, random_state=RANDOM_STATE)
    cv_group_df = tval.summarize_cv_groups(y, groups)
    cv_split_df = tval.validate_cv_splits(cv, X, y, groups)

    models = mdl.build_models(random_state=RANDOM_STATE)
    if len(models) != 5:
        raise ValueError(f"Se esperaban 5 modelos; se definieron {len(models)}.")

    (
        metrics_fold,
        metrics_sum,
        oof_df,
        report_df,
    ) = evl.evaluate_models_cv(
        models,
        X,
        y,
        groups,
        cv,
        df_meta=df_model,
        random_state=RANDOM_STATE,
    )

    conf_long = evl.build_confusion_matrices(
        oof_df, sorted(y.unique().tolist())
    )
    best_name = evl.select_best_model_from_summary(
        metrics_sum, primary_metric="f1_macro", secondary_metric="balanced_accuracy"
    )
    f1_mean = float(
        metrics_sum.loc[
            (metrics_sum["model_name"] == best_name)
            & (metrics_sum["metric"] == "f1_macro"),
            "mean",
        ].iloc[0]
    )
    bal_mean = float(
        metrics_sum.loc[
            (metrics_sum["model_name"] == best_name)
            & (metrics_sum["metric"] == "balanced_accuracy"),
            "mean",
        ].iloc[0]
    )
    best_sel = pd.DataFrame(
        [
            {
                "best_model_name": best_name,
                "primary_metric": "f1_macro",
                "secondary_metric": "balanced_accuracy",
                "f1_macro_mean": f1_mean,
                "balanced_accuracy_mean": bal_mean,
            }
        ]
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "cv_split": TABLES_DIR / "cv_split_summary.csv",
        "cv_group": TABLES_DIR / "cv_group_summary.csv",
        "metrics_fold": TABLES_DIR / "classifier_metrics_by_fold.csv",
        "metrics_sum": TABLES_DIR / "classifier_metrics_summary.csv",
        "oof": TABLES_DIR / "oof_predictions.csv",
        "report": TABLES_DIR / "classification_report_by_model.csv",
        "conf": TABLES_DIR / "confusion_matrix_by_model.csv",
        "best": TABLES_DIR / "best_model_selection.csv",
        "fig_f1": FIGURES_DIR / "model_comparison_f1_macro.png",
        "fig_bal": FIGURES_DIR / "model_comparison_balanced_accuracy.png",
        "fig_conf": FIGURES_DIR / "confusion_matrix_best_model.png",
    }

    export_table(cv_split_df, paths["cv_split"])
    export_table(cv_group_df, paths["cv_group"])
    export_table(metrics_fold, paths["metrics_fold"])
    export_table(metrics_sum, paths["metrics_sum"])
    export_table(oof_df, paths["oof"])
    export_table(report_df, paths["report"])
    export_table(conf_long, paths["conf"])
    export_table(best_sel, paths["best"])

    evl.plot_model_metric_comparison(metrics_sum, "f1_macro", paths["fig_f1"])
    evl.plot_model_metric_comparison(
        metrics_sum, "balanced_accuracy", paths["fig_bal"]
    )
    evl.plot_confusion_matrix_for_model(conf_long, best_name, paths["fig_conf"])

    n_groups = int(groups.nunique())
    print("Resumen de modelado (ASCII)")
    print(f"Ventanas usadas: {len(X)}")
    print(f"Features modelables: {X.shape[1]}")
    print(f"Clases: {sorted(y.unique().tolist())}")
    print(f"Grupos temporales: {n_groups}")
    print("Folds: 5")
    print(f"Modelos evaluados: {', '.join(models.keys())}")
    print(f"Mejor modelo: {best_name}")
    print(f"f1_macro medio (mejor): {round(f1_mean, 6)}")
    print(f"balanced_accuracy medio (mejor): {round(bal_mean, 6)}")
    print("Archivos exportados:")
    for p in paths.values():
        print(f"  {_rel_path(p)}")


def _run_final_model() -> None:
    windowed_path = DATA_PROCESSED_DIR / "bpc_windowed_features.csv"
    if not windowed_path.is_file():
        print(
            "ERROR: No existe data/processed/bpc_windowed_features.csv. "
            "Ejecute primero: python run_pipeline.py --stage features"
        )
        sys.exit(1)

    windowed_df = pd.read_csv(windowed_path)
    n_transition_excluded = 0
    if "has_near_transition" in windowed_df.columns:
        n_transition_excluded = int(
            windowed_df["has_near_transition"].fillna(False).astype(bool).sum()
        )

    X, y, df_model = mdl.prepare_modeling_dataset(
        windowed_df, target_col=TARGET_COL, exclude_transition_windows=True
    )
    if X.shape[1] != 120:
        raise ValueError(f"Se esperaban 120 features en X; se encontraron {X.shape[1]}.")
    if int(y.nunique()) != 3:
        raise ValueError("Se esperaban 3 clases en y.")
    if len(X) != 1451:
        raise ValueError(
            f"Se esperaban 1451 ventanas tras excluir transicion; se encontraron {len(X)}."
        )

    feature_columns = list(X.columns)
    model, label_encoder, train_pred = mdl.train_final_model(
        X, y, random_state=RANDOM_STATE
    )
    class_names = [str(c) for c in label_encoder.classes_]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    paths_models = {
        "model": MODELS_DIR / "model_final_xgboost.pkl",
        "encoder": MODELS_DIR / "label_encoder.pkl",
        "features": MODELS_DIR / "feature_columns.json",
        "classes": MODELS_DIR / "class_names.json",
        "meta": MODELS_DIR / "final_model_metadata.json",
    }
    save_pickle(model, paths_models["model"])
    save_pickle(label_encoder, paths_models["encoder"])
    save_json(feature_columns, paths_models["features"])
    save_json(class_names, paths_models["classes"])

    metadata = {
        "project_name": "Proyecto4_BPC",
        "model_name": "xgboost",
        "model_version": "v1.0",
        "random_state": RANDOM_STATE,
        "n_training_windows": int(len(X)),
        "n_model_features": int(X.shape[1]),
        "class_names": class_names,
        "window_seconds": int(WINDOW_SECONDS),
        "transition_windows_excluded": True,
        "n_transition_windows_excluded": n_transition_excluded,
        "source_dataset": "data/processed/bpc_windowed_features.csv",
        "selected_from_stage": "modeling",
        "selection_primary_metric": "f1_macro",
        "selection_secondary_metric": "balanced_accuracy",
        "cv_f1_macro_mean": 0.9357428378591859,
        "cv_balanced_accuracy_mean": 0.9345908484289833,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(metadata, paths_models["meta"])

    meta_cols = [
        c
        for c in (
            "window_id",
            "window_start",
            "window_end",
            "n_samples",
            TARGET_COL,
            "label_majority_fraction",
            "has_near_transition",
        )
        if c in df_model.columns
    ]
    meta_df = df_model[meta_cols].reset_index(drop=True)
    prob_cols = [c for c in train_pred.columns if c.startswith("prob_")]
    pred_part = train_pred[
        ["y_true_label", "y_pred_label"] + prob_cols + ["confidence", "margin_top2"]
    ].reset_index(drop=True)
    pred_part["is_correct"] = pred_part["y_true_label"] == pred_part["y_pred_label"]
    predictions_out = pd.concat([meta_df, pred_part], axis=1)
    pred_csv = TABLES_DIR / "final_model_predictions.csv"
    export_table(predictions_out, pred_csv)

    y_true_enc = train_pred["y_true"].to_numpy()
    y_pred_enc = train_pred["y_pred"].to_numpy()
    train_acc = float(accuracy_score(y_true_enc, y_pred_enc))
    train_bal = float(balanced_accuracy_score(y_true_enc, y_pred_enc))
    train_f1 = float(f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))
    train_prec = float(
        precision_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
    )
    train_rec = float(
        recall_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
    )
    train_mcc = float(matthews_corrcoef(y_true_enc, y_pred_enc))
    train_kappa = float(cohen_kappa_score(y_true_enc, y_pred_enc))
    mean_conf = float(train_pred["confidence"].mean())
    median_conf = float(train_pred["confidence"].median())
    mean_marg = float(train_pred["margin_top2"].mean())
    median_marg = float(train_pred["margin_top2"].median())

    summary_row = {
        "n_training_windows": int(len(X)),
        "n_model_features": int(X.shape[1]),
        "n_classes": int(len(class_names)),
        "training_accuracy": train_acc,
        "training_balanced_accuracy": train_bal,
        "training_f1_macro": train_f1,
        "training_precision_macro": train_prec,
        "training_recall_macro": train_rec,
        "training_mcc": train_mcc,
        "training_cohen_kappa": train_kappa,
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "mean_margin_top2": mean_marg,
        "median_margin_top2": median_marg,
    }
    summary_csv = TABLES_DIR / "final_model_training_summary.csv"
    export_table(pd.DataFrame([summary_row]), summary_csv)

    print("Resumen modelo final (ASCII)")
    print("Modelo final: xgboost (model_final_xgboost.pkl)")
    print(f"Ventanas de entrenamiento: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Clases: {class_names}")
    print(f"training_f1_macro: {round(train_f1, 6)}")
    print(f"training_balanced_accuracy: {round(train_bal, 6)}")
    print(f"mean_confidence: {round(mean_conf, 6)}")
    print("Artefactos guardados:")
    for p in list(paths_models.values()) + [pred_csv, summary_csv]:
        print(f"  {_rel_path(p)}")


def _run_interpretability() -> None:
    def _ascii_console(s: str) -> str:
        return s.encode("ascii", "replace").decode("ascii")

    windowed_path = DATA_PROCESSED_DIR / "bpc_windowed_features.csv"
    required_models = [
        MODELS_DIR / "model_final_xgboost.pkl",
        MODELS_DIR / "label_encoder.pkl",
        MODELS_DIR / "feature_columns.json",
        MODELS_DIR / "class_names.json",
    ]
    if not windowed_path.is_file():
        print(
            "ERROR: No existe data/processed/bpc_windowed_features.csv. "
            "Ejecute primero: python run_pipeline.py --stage features"
        )
        sys.exit(1)
    missing = [p for p in required_models if not p.is_file()]
    if missing:
        print(
            "ERROR: Faltan artefactos del modelo final. "
            "Ejecute primero: python run_pipeline.py --stage final_model"
        )
        sys.exit(1)

    art = intr.load_model_artifacts(MODELS_DIR)
    model = art["model"]
    label_encoder = art["label_encoder"]
    feature_columns: list[str] = art["feature_columns"]
    class_names: list[str] = art["class_names"]

    windowed_df = pd.read_csv(windowed_path)
    X, y, _df_i = intr.prepare_interpretability_dataset(
        windowed_df,
        feature_columns,
        target_col=TARGET_COL,
        exclude_transition_windows=True,
    )
    if len(X) != 1451:
        raise ValueError(
            f"Se esperaban 1451 ventanas para interpretabilidad; se encontraron {len(X)}."
        )

    y_enc = label_encoder.transform(y.astype(str))

    meta_df = intr.build_windowed_feature_metadata(feature_columns)
    perm_df = intr.compute_permutation_importance_table(
        model,
        X,
        y_enc,
        feature_columns,
        scoring="f1_macro",
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    shap_df, shap_extra = intr.compute_shap_global_importance(
        model,
        X,
        feature_columns,
        class_names,
        max_samples=1000,
        random_state=RANDOM_STATE,
    )
    shap_status = str(shap_extra.get("status", "failed"))
    shap_warn_parts: list[str] = []
    w0 = str(shap_extra.get("warning") or "").strip()
    if w0:
        shap_warn_parts.append(w0)

    ranking_df = intr.build_consolidated_ranking(perm_df, shap_df, meta_df)

    rank_raw = intr.aggregate_importance(ranking_df, "raw_variable")
    rank_comp = intr.aggregate_importance(ranking_df, "component")
    rank_fam = intr.aggregate_importance(ranking_df, "family")
    rank_pos = intr.aggregate_importance(ranking_df, "position")
    rank_stat = intr.aggregate_importance(ranking_df, "statistic")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths_tbl = {
        "perm": TABLES_DIR / "permutation_importance.csv",
        "shap": TABLES_DIR / "shap_global_importance.csv",
        "rank": TABLES_DIR / "ranking_features.csv",
        "rraw": TABLES_DIR / "rank_raw_variable.csv",
        "rcomp": TABLES_DIR / "rank_component.csv",
        "rfam": TABLES_DIR / "rank_family.csv",
        "rpos": TABLES_DIR / "rank_position.csv",
        "rstat": TABLES_DIR / "rank_statistic.csv",
        "summ": TABLES_DIR / "interpretability_summary.csv",
    }
    export_table(perm_df, paths_tbl["perm"])
    export_table(shap_df, paths_tbl["shap"])
    export_table(ranking_df, paths_tbl["rank"])
    export_table(rank_raw, paths_tbl["rraw"])
    export_table(rank_comp, paths_tbl["rcomp"])
    export_table(rank_fam, paths_tbl["rfam"])
    export_table(rank_pos, paths_tbl["rpos"])
    export_table(rank_stat, paths_tbl["rstat"])

    paths_fig = {
        "p_perm": FIGURES_DIR / "top_20_permutation_importance.png",
        "p_shap": FIGURES_DIR / "top_20_shap_importance.png",
        "p_comp": FIGURES_DIR / "importance_by_component.png",
        "p_fam": FIGURES_DIR / "importance_by_family.png",
        "p_pos": FIGURES_DIR / "importance_by_position.png",
        "p_stat": FIGURES_DIR / "importance_by_statistic.png",
        "p_bee": FIGURES_DIR / "shap_summary_beeswarm.png",
    }
    intr.plot_top_importance(
        perm_df,
        "perm_importance_mean",
        "feature",
        paths_fig["p_perm"],
        "Top 20 permutation importance (f1_macro)",
    )
    intr.plot_top_importance(
        shap_df,
        "shap_importance_mean_abs",
        "feature",
        paths_fig["p_shap"],
        "Top 20 SHAP mean |value|",
    )
    intr.plot_group_importance(
        rank_comp, "component", paths_fig["p_comp"], "Importancia por componente"
    )
    intr.plot_group_importance(
        rank_fam, "family", paths_fig["p_fam"], "Importancia por familia"
    )
    intr.plot_group_importance(
        rank_pos, "position", paths_fig["p_pos"], "Importancia por posicion"
    )
    intr.plot_group_importance(
        rank_stat, "statistic", paths_fig["p_stat"], "Importancia por estadistico"
    )

    n_plot = min(500, len(X))
    X_plot = X.sample(n=n_plot, random_state=RANDOM_STATE)
    plot_warn = intr.plot_shap_summary_beeswarm(
        model, X_plot, feature_columns, paths_fig["p_bee"]
    )
    if plot_warn:
        shap_warn_parts.append(plot_warn)

    row_best = ranking_df.loc[ranking_df["consolidated_rank"] == 1]
    if row_best.empty:
        row_best = ranking_df.iloc[[0]]
    row1 = row_best.iloc[0]
    best_f = str(row1["feature"])
    best_feature_raw = str(row1["raw_variable"])
    top_raw_var = str(rank_raw.iloc[0]["raw_variable"]) if len(rank_raw) else ""
    top_comp = str(rank_comp.iloc[0]["component"]) if len(rank_comp) else ""
    top_fam = str(rank_fam.iloc[0]["family"]) if len(rank_fam) else ""
    top_pos = str(rank_pos.iloc[0]["position"]) if len(rank_pos) else ""
    top_stat = str(rank_stat.iloc[0]["statistic"]) if len(rank_stat) else ""

    summary_row = {
        "n_windows_used": int(len(X)),
        "n_model_features": int(X.shape[1]),
        "n_classes": int(len(class_names)),
        "best_feature_consolidated": best_f,
        "best_feature_raw_variable": best_feature_raw,
        "top_raw_variable": top_raw_var,
        "top_component": top_comp,
        "top_family": top_fam,
        "top_position": top_pos,
        "top_statistic": top_stat,
        "permutation_n_repeats": 10,
        "shap_max_samples": 1000,
        "shap_status": shap_status,
        "shap_warning": " | ".join(shap_warn_parts) if shap_warn_parts else "",
    }
    export_table(pd.DataFrame([summary_row]), paths_tbl["summ"])

    print("Resumen interpretabilidad (ASCII)")
    print(f"Ventanas usadas: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Clases: {class_names}")
    print(f"Top feature consolidado: {_ascii_console(best_f)}")
    print(f"Raw variable de la mejor feature: {_ascii_console(best_feature_raw)}")
    print(f"Top variable base agregada: {_ascii_console(top_raw_var)}")
    print(f"Top componente: {_ascii_console(top_comp)}")
    print(f"Top familia: {_ascii_console(top_fam)}")
    print(f"Top posicion: {_ascii_console(top_pos)}")
    print(f"Top estadistico: {_ascii_console(top_stat)}")
    print(f"Estado SHAP: {shap_status}")
    print("Archivos exportados:")
    for p in list(paths_tbl.values()) + list(paths_fig.values()):
        print(f"  {_rel_path(p)}")


def _run_assessment() -> None:
    def _ascii_console(s: str) -> str:
        return str(s).encode("ascii", "replace").decode("ascii")

    wpath = DATA_RAW_DIR / SENSOR_WEIGHTS_FILE
    feat_path = DATA_PROCESSED_DIR / "bpc_windowed_features.csv"
    meta_path = DATA_PROCESSED_DIR / "variable_metadata.csv"
    if not wpath.is_file():
        print(
            "ERROR: Falta data/raw/Pesos_Ponderados_Proyecto4.xlsx. "
            "Copie el archivo de pesos ponderados en data/raw y reintente."
        )
        sys.exit(1)
    if not feat_path.is_file():
        print(
            "ERROR: No existe data/processed/bpc_windowed_features.csv. "
            "Ejecute primero: python run_pipeline.py --stage features"
        )
        sys.exit(1)
    if not meta_path.is_file():
        print(
            "ERROR: No existe data/processed/variable_metadata.csv. "
            "Ejecute primero: python run_pipeline.py --stage check_data"
        )
        sys.exit(1)

    weights_raw = asm.load_sensor_weights(wpath)
    weights_clean = asm.clean_sensor_weights(weights_raw)
    var_meta = pd.read_csv(meta_path)
    mapping, map_warns = asm.build_weight_mapping(weights_clean, var_meta)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "clean": TABLES_DIR / "sensor_weights_clean.csv",
        "map": TABLES_DIR / "sensor_weights_mapping.csv",
        "win": TABLES_DIR / "condition_index_by_window.csv",
        "long": TABLES_DIR / "condition_scores_long.csv",
        "batch": TABLES_DIR / "condition_index_by_batch.csv",
        "glob": TABLES_DIR / "condition_index_summary.csv",
        "topw": TABLES_DIR / "top_weighted_variables.csv",
        "asum": TABLES_DIR / "assessment_summary.csv",
        "fig_batch": FIGURES_DIR / "condition_index_by_batch.png",
        "fig_ts": FIGURES_DIR / "condition_index_time_series.png",
        "fig_top": FIGURES_DIR / "top_weighted_variables.png",
    }
    export_table(weights_clean, paths["clean"])
    export_table(mapping, paths["map"])
    for w in map_warns:
        print(f"Aviso mapping: {_ascii_console(w)}")

    windowed = pd.read_csv(feat_path)
    try:
        cond_win = asm.compute_condition_index_by_window(windowed, mapping)
        scores_long = asm.compute_condition_scores_long(windowed, mapping)
        by_batch = asm.summarize_condition_by_batch(cond_win)
        glob_summ = asm.summarize_condition_global(cond_win)
        top_w = asm.summarize_top_weighted_variables(mapping)
    except Exception as exc:
        n_unm = int((mapping["match_status"] == "unmatched").sum())
        matched_ok = mapping[
            (mapping["match_status"].isin(["exact", "fuzzy", "manual_semantic"]))
            & (mapping["matched_raw_column"].astype(str).str.len() > 0)
        ]
        vc = matched_ok["matched_raw_column"].value_counts()
        duped = vc[vc > 1]
        print(f"ERROR en calculo de indice de condicion: {_ascii_console(str(exc))}")
        print(f"Pesos no mapeados (mapping): {n_unm}")
        if len(duped):
            print(f"Duplicados matched_raw_column: {_ascii_console(str(duped.to_dict()))}")
        print(f"Archivos parciales exportados: {_rel_path(paths['clean'])}, {_rel_path(paths['map'])}")
        raise

    export_table(cond_win, paths["win"])
    export_table(scores_long, paths["long"])
    export_table(by_batch, paths["batch"])
    export_table(glob_summ, paths["glob"])
    export_table(top_w, paths["topw"])

    n_mapped = int(
        (
            mapping["match_status"].isin(["exact", "fuzzy", "manual_semantic"])
            & (mapping["matched_raw_column"].astype(str).str.len() > 0)
        ).sum()
    )
    n_unm = int((mapping["match_status"] == "unmatched").sum())
    matched_ok = mapping[
        (mapping["match_status"].isin(["exact", "fuzzy", "manual_semantic"]))
        & (mapping["matched_raw_column"].astype(str).str.len() > 0)
    ]
    vc = matched_ok["matched_raw_column"].value_counts()
    n_dup = int((vc[vc > 1] - 1).sum()) if len(vc) else 0
    wsum = float(weights_clean["weight_normalized"].sum())
    n_scored = int(cond_win["condition_index"].notna().sum())
    med_aw = float(cond_win["condition_index_available_weight"].median())

    top_var = str(top_w.iloc[0]["weight_variable"]) if len(top_w) else ""
    top_comp = str(top_w.iloc[0]["component"]) if len(top_w) else ""
    top_fam = str(top_w.iloc[0]["family"]) if len(top_w) else ""
    meth_mode = cond_win["assessment_method"].dropna().mode()
    assessment_meth = str(meth_mode.iloc[0]) if len(meth_mode) else "unknown"

    asum = {
        "n_weight_rows": int(len(weights_clean)),
        "n_weights_mapped": n_mapped,
        "n_weights_unmatched": n_unm,
        "n_duplicate_matches": n_dup,
        "normalized_weight_sum": wsum,
        "n_windows_scored": n_scored,
        "median_available_weight": med_aw,
        "top_weighted_variable": top_var,
        "top_weighted_component": top_comp,
        "top_weighted_family": top_fam,
        "assessment_method": assessment_meth,
    }
    export_table(pd.DataFrame([asum]), paths["asum"])

    asm.plot_condition_index_by_batch(cond_win, paths["fig_batch"])
    asm.plot_condition_index_time_series(cond_win, paths["fig_ts"])
    asm.plot_top_weighted_variables(top_w, paths["fig_top"])

    ci_mean = float(cond_win["condition_index"].mean())
    ci_med = float(cond_win["condition_index"].median())

    print("Resumen assessment (ASCII)")
    print("Nota: pesos ponderados son funcion de condicion del activo, no importancia ML.")
    print(f"Filas de pesos: {len(weights_clean)}")
    print(f"Pesos mapeados: {n_mapped}")
    print(f"Pesos no mapeados: {n_unm}")
    print(f"Suma pesos normalizados: {round(wsum, 6)}")
    print(f"Ventanas evaluadas: {len(cond_win)}")
    print(f"Indice condicion promedio: {round(ci_mean, 4)}")
    print(f"Indice condicion mediano: {round(ci_med, 4)}")
    print(f"Top variable ponderada: {_ascii_console(top_var)}")
    print(f"Top componente ponderado: {_ascii_console(top_comp)}")
    print(f"Top familia ponderada: {_ascii_console(top_fam)}")
    print("Archivos exportados:")
    for p in paths.values():
        print(f"  {_rel_path(p)}")


def _run_assessment_thresholds() -> None:
    feat_path = DATA_PROCESSED_DIR / "bpc_windowed_features.csv"
    clean_path = TABLES_DIR / "sensor_weights_clean.csv"
    map_path = TABLES_DIR / "sensor_weights_mapping.csv"

    if not feat_path.is_file():
        print(
            "ERROR: No existe data/processed/bpc_windowed_features.csv. "
            "Ejecute primero: python run_pipeline.py --stage features"
        )
        sys.exit(1)
    if not clean_path.is_file() or not map_path.is_file():
        print("Ejecutar primero: python run_pipeline.py --stage assessment")
        sys.exit(1)

    windowed_df = pd.read_csv(feat_path)
    weights_clean = pd.read_csv(clean_path)
    mapping = pd.read_csv(map_path)

    th_g = asm.estimate_thresholds_global(windowed_df, mapping)
    th_b = asm.estimate_thresholds_by_batch(windowed_df, mapping)
    w_thr = asm.merge_weights_with_thresholds(weights_clean, mapping, th_g)
    cond_g = asm.compute_condition_index_with_global_thresholds(windowed_df, w_thr)
    cond_b = asm.compute_condition_index_with_batch_thresholds(windowed_df, mapping, th_b)
    batch_g = asm.summarize_condition_by_batch_thresholded(
        cond_g, "data_driven_global_percentile_thresholds"
    )
    batch_b = asm.summarize_condition_by_batch_thresholded(
        cond_b, "data_driven_batch_percentile_thresholds"
    )

    n_warn_g = int((th_g["threshold_warning"].fillna("").astype(str).str.strip().str.len() > 0).sum())
    n_warn_b = int((th_b["threshold_warning"].fillna("").astype(str).str.strip().str.len() > 0).sum())

    s_g = cond_g["condition_index"].dropna()
    s_b = cond_b["condition_index"].dropna()
    summary_rows = [
        ("n_variables_thresholded", 24.0),
        ("n_batches_thresholded", 3.0),
        ("n_thresholds_global_rows", float(len(th_g))),
        ("n_thresholds_by_batch_rows", float(len(th_b))),
        ("n_threshold_warnings_global", float(n_warn_g)),
        ("n_threshold_warnings_by_batch", float(n_warn_b)),
        ("threshold_method_global", "global_percentile_p05_p95_p99"),
        ("threshold_method_by_batch", "batch_percentile_p05_p95_p99"),
        ("condition_index_mean_thresholded_global", float(s_g.mean()) if len(s_g) else float("nan")),
        ("condition_index_median_thresholded_global", float(s_g.median()) if len(s_g) else float("nan")),
        ("condition_index_p95_thresholded_global", float(s_g.quantile(0.95)) if len(s_g) else float("nan")),
        ("condition_index_min_thresholded_global", float(s_g.min()) if len(s_g) else float("nan")),
        ("condition_index_max_thresholded_global", float(s_g.max()) if len(s_g) else float("nan")),
        ("condition_index_mean_thresholded_by_batch", float(s_b.mean()) if len(s_b) else float("nan")),
        ("condition_index_median_thresholded_by_batch", float(s_b.median()) if len(s_b) else float("nan")),
        ("condition_index_p95_thresholded_by_batch", float(s_b.quantile(0.95)) if len(s_b) else float("nan")),
        ("condition_index_min_thresholded_by_batch", float(s_b.min()) if len(s_b) else float("nan")),
        ("condition_index_max_thresholded_by_batch", float(s_b.max()) if len(s_b) else float("nan")),
        (
            "note",
            "Exploratory thresholds estimated from available historical windowed data; "
            "not normative alarm limits.",
        ),
    ]
    summ_df = pd.DataFrame(summary_rows, columns=["metric", "value"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "th_g": TABLES_DIR / "assessment_thresholds_global.csv",
        "th_b": TABLES_DIR / "assessment_thresholds_by_batch.csv",
        "w_thr": TABLES_DIR / "sensor_weights_with_thresholds.csv",
        "win_g": TABLES_DIR / "condition_index_by_window_thresholded_global.csv",
        "bat_g": TABLES_DIR / "condition_index_by_batch_thresholded_global.csv",
        "win_b": TABLES_DIR / "condition_index_by_window_thresholded_by_batch.csv",
        "bat_b": TABLES_DIR / "condition_index_by_batch_thresholded_by_batch.csv",
        "summ": TABLES_DIR / "assessment_thresholds_summary.csv",
        "fig_bg": FIGURES_DIR / "condition_index_by_batch_thresholded_global.png",
        "fig_bb": FIGURES_DIR / "condition_index_by_batch_thresholded_by_batch.png",
        "fig_tg": FIGURES_DIR / "condition_index_time_series_thresholded_global.png",
        "fig_tb": FIGURES_DIR / "condition_index_time_series_thresholded_by_batch.png",
    }
    export_table(th_g, paths["th_g"])
    export_table(th_b, paths["th_b"])
    export_table(w_thr, paths["w_thr"])
    export_table(cond_g, paths["win_g"])
    export_table(batch_g, paths["bat_g"])
    export_table(cond_b, paths["win_b"])
    export_table(batch_b, paths["bat_b"])
    export_table(summ_df, paths["summ"])
    asm.plot_condition_index_by_batch(cond_g, paths["fig_bg"])
    asm.plot_condition_index_by_batch(cond_b, paths["fig_bb"])
    asm.plot_condition_index_time_series(cond_g, paths["fig_tg"])
    asm.plot_condition_index_time_series(cond_b, paths["fig_tb"])

    if len(th_g) != 24:
        raise ValueError(f"assessment_thresholds_global: se esperaban 24 filas; hay {len(th_g)}.")
    if len(th_b) != 72:
        raise ValueError(f"assessment_thresholds_by_batch: se esperaban 72 filas; hay {len(th_b)}.")
    if len(w_thr) != 24:
        raise ValueError(f"sensor_weights_with_thresholds: se esperaban 24 filas; hay {len(w_thr)}.")
    for _, r in w_thr.iterrows():
        if pd.isna(r["v0"]) or pd.isna(r["h"]) or pd.isna(r["hh"]):
            raise ValueError("sensor_weights_with_thresholds: v0/h/hh no deben ser nulos.")
        if not (float(r["v0"]) < float(r["h"]) < float(r["hh"])):
            raise ValueError(f"V0<H<HH falla en merge row weight_id={r.get('weight_id')}")
    for _, r in th_g.iterrows():
        if not (float(r["v0_estimated"]) < float(r["h_estimated"]) < float(r["hh_estimated"])):
            raise ValueError(f"V0<H<HH falla en global thresholds weight_id={r.get('weight_id')}")
    for _, r in th_b.iterrows():
        if not (float(r["v0_estimated"]) < float(r["h_estimated"]) < float(r["hh_estimated"])):
            raise ValueError(
                f"V0<H<HH falla en by_batch Batch={r.get('Batch')} raw={r.get('matched_raw_column')}"
            )
    if len(cond_g) != 1466:
        raise ValueError(f"condition_index global ventanas: se esperaban 1466; hay {len(cond_g)}.")
    if len(cond_b) != 1466:
        raise ValueError(f"condition_index by_batch ventanas: se esperaban 1466; hay {len(cond_b)}.")
    if len(batch_g) != 3:
        raise ValueError(f"resumen batch global: se esperaban 3 filas; hay {len(batch_g)}.")
    if len(batch_b) != 3:
        raise ValueError(f"resumen batch by_batch: se esperaban 3 filas; hay {len(batch_b)}.")
    if (cond_g["assessment_method"] != "data_driven_global_percentile_thresholds").any():
        raise ValueError("assessment_method global incorrecto.")
    if (cond_b["assessment_method"] != "data_driven_batch_percentile_thresholds").any():
        raise ValueError("assessment_method by_batch incorrecto.")
    mx_g = float(cond_g["condition_index"].max())
    mn_g = float(cond_g["condition_index"].min())
    mx_b = float(cond_b["condition_index"].max())
    mn_b = float(cond_b["condition_index"].min())
    if mn_g < -1e-6 or mx_g > 100.0 + 1e-6:
        raise ValueError(f"condition_index global fuera de [0,100]: min={mn_g}, max={mx_g}")
    if mn_b < -1e-6 or mx_b > 100.0 + 1e-6:
        raise ValueError(f"condition_index by_batch fuera de [0,100]: min={mn_b}, max={mx_b}")

    print("Resumen assessment_thresholds (ASCII)")
    print("Umbrales exploratorios (no normativos); orientacion cost; V0<H<HH tras ajustes.")
    print(f"Variables con umbrales globales: {len(th_g)}")
    print(f"Filas umbrales por batch: {len(th_b)}")
    print(f"Warnings globales (no vacio): {n_warn_g}")
    print(f"Warnings por batch (no vacio): {n_warn_b}")
    print(f"Indice promedio (global thresholds): {round(float(s_g.mean()), 6)}")
    print(f"Indice promedio (batch thresholds): {round(float(s_b.mean()), 6)}")
    print(f"Indice mediano (global thresholds): {round(float(s_g.median()), 6)}")
    print(f"Indice mediano (batch thresholds): {round(float(s_b.median()), 6)}")
    print("Archivos exportados:")
    for p in paths.values():
        print(f"  {_rel_path(p)}")


def _run_dashboard_exports() -> None:
    def _ascii_console(s: str) -> str:
        return str(s).encode("ascii", "replace").decode("ascii")

    try:
        info = run_dashboard_exports()
    except FileNotFoundError as e:
        print(_ascii_console(str(e)))
        sys.exit(1)

    print("Resumen dashboard_exports (ASCII)")
    print(f"Carpeta dashboard: {_rel_path(DATA_DASHBOARD_DIR)}")
    print("Archivos dashboard exportados:")
    for name in sorted(
        [
            "dashboard_kpis.csv",
            "dashboard_model_metrics.csv",
            "dashboard_confusion_best_model.csv",
            "dashboard_predictions.csv",
            "dashboard_feature_importance.csv",
            "dashboard_rank_raw_variable.csv",
            "dashboard_rank_component.csv",
            "dashboard_rank_family.csv",
            "dashboard_rank_position.csv",
            "dashboard_rank_statistic.csv",
            "dashboard_top_weighted_variables.csv",
            "dashboard_condition_index_by_batch.csv",
            "dashboard_assessment_thresholds_summary.csv",
            "dashboard_assessment_thresholds_global.csv",
            "dashboard_assessment_thresholds_by_batch.csv",
            "dashboard_sensor_weights_with_thresholds.csv",
            "dashboard_condition_index_thresholded_global_by_window.csv",
            "dashboard_condition_index_thresholded_global_by_batch.csv",
            "dashboard_condition_index_thresholded_by_batch_by_window.csv",
            "dashboard_condition_index_thresholded_by_batch_by_batch.csv",
            "dashboard_condition_contributions_long.csv",
            "dashboard_condition_contributions_top_by_window.csv",
            "dashboard_condition_current_state.csv",
            "dashboard_condition_alerts_active.csv",
            "dashboard_condition_trend_summary.csv",
            "current_asset_state.json",
            "dashboard_pairwise_permanova.csv",
            "dashboard_pairwise_permdisp.csv",
            "dashboard_pca_centroids.csv",
            "dashboard_pca_projection.csv",
            "dashboard_batch_transitions.csv",
            "README_dashboard_data.md",
        ]
    ):
        print(f"  data/dashboard/{name}")
    print(f"Mejor modelo: {_ascii_console(str(info['best_model_name']))}")
    print(f"F1 macro CV: {round(float(info['f1_macro_cv']), 6)}")
    print(f"Balanced accuracy CV: {round(float(info['balanced_accuracy_cv']), 6)}")
    print(f"PERMANOVA R2: {round(float(info['permanova_r2']), 6)}")
    print(f"Top componente interpretabilidad: {_ascii_console(str(info['top_component']))}")
    print(f"Top familia interpretabilidad: {_ascii_console(str(info['top_family']))}")
    if info.get("condition_index_mean") is not None:
        print(f"Condition index mean original: {round(float(info['condition_index_mean']), 6)}")
    else:
        print("Condition index mean original: n/a")
    print(f"Assessment method original: {_ascii_console(str(info['assessment_method']))}")
    print(f"Threshold method global: {_ascii_console(str(info.get('threshold_method_global', '')))}")
    print(f"Threshold method by batch: {_ascii_console(str(info.get('threshold_method_by_batch', '')))}")
    m_g = info.get("condition_index_mean_thresholded_global")
    m_b = info.get("condition_index_mean_thresholded_by_batch")
    if m_g is not None:
        print(f"Condition index mean thresholded global: {round(float(m_g), 6)}")
    else:
        print("Condition index mean thresholded global: n/a")
    if m_b is not None:
        print(f"Condition index mean thresholded by batch: {round(float(m_b), 6)}")
    else:
        print("Condition index mean thresholded by batch: n/a")

    print("Estado condition_state (ultima ventana exportada, ASCII):")
    print(f"  window_id: {info.get('current_window_id')}")
    print(f"  Batch real: {_ascii_console(str(info.get('current_batch_real', '')))}")
    print(f"  Batch predicho: {_ascii_console(str(info.get('current_batch_predicted', '')))}")
    ci = info.get("current_condition_index")
    hi = info.get("current_health_index")
    if ci is not None:
        print(f"  condition_index: {round(float(ci), 6)}")
    else:
        print("  condition_index: n/a")
    if hi is not None:
        print(f"  health_index: {round(float(hi), 6)}")
    else:
        print("  health_index: n/a")
    print(f"  condition_state: {_ascii_console(str(info.get('current_condition_state', '')))}")
    print(f"  trend_direction: {_ascii_console(str(info.get('trend_direction', '')))}")
    print(f"  alertas attention: {info.get('n_attention_alerts', 0)}")
    print(f"  alertas high: {info.get('n_high_alerts', 0)}")


def _run_condition_state() -> None:
    """Etapa 8C: contribuciones por variable, alertas exploratorias y estado actual (solo lectura de salidas previas)."""

    def _ascii(s: str) -> str:
        return str(s).encode("ascii", "replace").decode("ascii")

    paths = {
        "windowed": DATA_PROCESSED_DIR / "bpc_windowed_features.csv",
        "mapping": TABLES_DIR / "sensor_weights_mapping.csv",
        "thresholds_by_batch": TABLES_DIR / "assessment_thresholds_by_batch.csv",
        "weights_thr": TABLES_DIR / "sensor_weights_with_thresholds.csv",
        "cond_win_thr": TABLES_DIR / "condition_index_by_window_thresholded_by_batch.csv",
        "cond_batch_thr": TABLES_DIR / "condition_index_by_batch_thresholded_by_batch.csv",
        "predictions": TABLES_DIR / "final_model_predictions.csv",
    }
    hints = {
        "windowed": "python run_pipeline.py --stage features",
        "mapping": "python run_pipeline.py --stage assessment",
        "thresholds_by_batch": "python run_pipeline.py --stage assessment_thresholds",
        "weights_thr": "python run_pipeline.py --stage assessment_thresholds",
        "cond_win_thr": "python run_pipeline.py --stage assessment_thresholds",
        "cond_batch_thr": "python run_pipeline.py --stage assessment_thresholds",
        "predictions": "python run_pipeline.py --stage final_model",
    }
    missing = {k: v for k, v in paths.items() if not v.is_file()}
    if missing:
        print("ERROR condition_state: faltan archivos requeridos.")
        for k, p in missing.items():
            print(f"  {k}: {p.as_posix()}")
            print(f"    Ejecute: {hints[k]}")
        sys.exit(1)

    windowed_df = pd.read_csv(paths["windowed"], encoding="utf-8")
    mapping_df = pd.read_csv(paths["mapping"], encoding="utf-8")
    thresholds_by_batch_df = pd.read_csv(paths["thresholds_by_batch"], encoding="utf-8")
    pd.read_csv(paths["weights_thr"], encoding="utf-8")
    condition_index_df = pd.read_csv(paths["cond_win_thr"], encoding="utf-8")
    pd.read_csv(paths["cond_batch_thr"], encoding="utf-8")
    final_predictions_df = pd.read_csv(paths["predictions"], encoding="utf-8")

    contrib = asm.compute_condition_contributions_long(
        windowed_df, mapping_df, thresholds_by_batch_df, value_stat="mean", batch_col="Batch"
    )
    top_by_win = asm.summarize_top_contributors_by_window(contrib, top_n=5)
    current = asm.build_condition_current_state(
        condition_index_df, contrib, final_predictions_df, top_n=10
    )
    alerts = asm.build_condition_alerts_active(contrib, condition_index_df, last_n_windows=3)
    trend = asm.compute_condition_trend_summary(condition_index_df, window_minutes=60)
    payload = asm.build_current_asset_state_payload(current, alerts, trend, top_by_win)

    if len(contrib) != 35184:
        raise ValueError(f"condition_contributions_long: se esperaban 35184 filas; hay {len(contrib)}")
    if len(top_by_win) != 7330:
        raise ValueError(f"condition_contributions_top_by_window: se esperaban 7330 filas; hay {len(top_by_win)}")
    if len(current) != 1:
        raise ValueError(f"condition_current_state: se esperaba 1 fila; hay {len(current)}")
    if len(trend) < 1:
        raise ValueError("condition_trend_summary sin filas.")

    cs_arr = pd.to_numeric(contrib["condition_score"], errors="coerce")
    if ((cs_arr < 0) | (cs_arr > 100)).any():
        raise ValueError("condition_score fuera de [0,100].")
    hs_arr = pd.to_numeric(contrib["health_score"], errors="coerce")
    if ((hs_arr < 0) | (hs_arr > 100)).any():
        raise ValueError("health_score fuera de [0,100].")
    if (pd.to_numeric(contrib["weighted_score"], errors="coerce") < 0).any():
        raise ValueError("weighted_score negativo.")
    ci0 = float(current["condition_index"].iloc[0])
    hi0 = float(current["health_index"].iloc[0])
    if not (0 <= ci0 <= 100 and 0 <= hi0 <= 100):
        raise ValueError("condition_index o health_index en estado actual fuera de [0,100].")

    out_long = TABLES_DIR / "condition_contributions_long.csv"
    out_top = TABLES_DIR / "condition_contributions_top_by_window.csv"
    out_cur = TABLES_DIR / "condition_current_state.csv"
    out_alt = TABLES_DIR / "condition_alerts_active.csv"
    out_tr = TABLES_DIR / "condition_trend_summary.csv"
    export_table(contrib, out_long)
    export_table(top_by_win, out_top)
    export_table(current, out_cur)
    export_table(alerts, out_alt)
    export_table(trend, out_tr)
    asm.export_current_asset_state_json(payload, REPORTS_DIR / "current_asset_state.json")

    if not out_alt.is_file():
        raise ValueError("condition_alerts_active.csv no se genero.")
    if not (REPORTS_DIR / "current_asset_state.json").is_file():
        raise ValueError("current_asset_state.json no se genero.")

    last_wid = int(current["window_id"].iloc[0])
    cont_last = contrib[contrib["window_id"].astype(int) == last_wid]
    asm.plot_condition_top_contributors_last_window(
        cont_last, FIGURES_DIR / "condition_top_contributors_last_window.png"
    )
    asm.plot_condition_health_index_time_series(
        condition_index_df, FIGURES_DIR / "condition_health_index_time_series.png"
    )
    asm.plot_condition_component_contribution_last_window(
        cont_last, FIGURES_DIR / "condition_component_contribution_last_window.png"
    )

    n_att_alerts = int((alerts["alert_level"] == "attention").sum()) if len(alerts) else 0
    n_hi_alerts = int((alerts["alert_level"] == "high").sum()) if len(alerts) else 0

    print("Resumen condition_state (ASCII)")
    print(f"Filas condition_contributions_long: {len(contrib)}")
    print(f"Filas condition_contributions_top_by_window: {len(top_by_win)}")
    print(f"Ultima window_id: {last_wid}")
    print(f"Batch actual: {_ascii(str(current['Batch'].iloc[0]))}")
    ypl = current["y_pred_label"].iloc[0]
    print(f"y_pred_label (si existe): {_ascii(str(ypl))}")
    print(f"condition_index actual: {round(ci0, 6)}")
    print(f"health_index actual: {round(hi0, 6)}")
    print(f"condition_state: {_ascii(str(current['condition_state'].iloc[0]))}")
    print(f"Alertas exploratorias attention: {n_att_alerts}")
    print(f"Alertas exploratorias high: {n_hi_alerts}")
    print(f"trend_direction: {_ascii(str(trend['trend_direction'].iloc[0]))}")
    print("Archivos exportados:")
    print(f"  {_rel_path(out_long)}")
    print(f"  {_rel_path(out_top)}")
    print(f"  {_rel_path(out_cur)}")
    print(f"  {_rel_path(out_alt)}")
    print(f"  {_rel_path(out_tr)}")
    print(f"  {_rel_path(REPORTS_DIR / 'current_asset_state.json')}")
    print(f"  {_rel_path(FIGURES_DIR / 'condition_top_contributors_last_window.png')}")
    print(f"  {_rel_path(FIGURES_DIR / 'condition_health_index_time_series.png')}")
    print(f"  {_rel_path(FIGURES_DIR / 'condition_component_contribution_last_window.png')}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.stage is None:
        print(
            "Pipeline inicializado. Use --stage check_data para validar carga de datos. "
            "Use --stage quality para reportes de calidad. "
            "Use --stage features para ventaneo y features. "
            "Use --stage stats para validacion estadistica. "
            "Use --stage modeling para validacion temporal y modelos. "
            "Use --stage final_model para entrenar y persistir el modelo final. "
            "Use --stage interpretability para SHAP y ranking de features. "
            "Use --stage assessment para pesos ponderados e indice de condicion. "
            "Use --stage assessment_thresholds para umbrales V0/H/HH data-driven y indices thresholded. "
            "Use --stage condition_state para contribuciones, alertas exploratorias y estado actual del activo. "
            "Use --stage dashboard_exports para tablas consolidadas en data/dashboard. "
            "Use --stage static_dashboard_publish para copiar el dashboard estatico a docs/ (GitHub Pages)."
        )
        return
    if args.stage == "check_data":
        _run_check_data()
        return
    if args.stage == "quality":
        _run_quality()
        return
    if args.stage == "features":
        _run_features()
        return
    if args.stage == "stats":
        _run_stats()
        return
    if args.stage == "modeling":
        _run_modeling()
        return
    if args.stage == "final_model":
        _run_final_model()
        return
    if args.stage == "interpretability":
        _run_interpretability()
        return
    if args.stage == "assessment":
        _run_assessment()
        return
    if args.stage == "assessment_thresholds":
        _run_assessment_thresholds()
        return
    if args.stage == "condition_state":
        _run_condition_state()
        return
    if args.stage == "dashboard_exports":
        _run_dashboard_exports()
        return
    if args.stage == "static_dashboard_publish":
        run_static_dashboard_publish()
        return
    print(f"Etapa desconocida: {args.stage}")
    sys.exit(1)


if __name__ == "__main__":
    main()
