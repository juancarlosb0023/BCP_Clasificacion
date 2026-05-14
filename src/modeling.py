"""Entrenamiento, ajuste de hiperparametros y comparacion de modelos de clasificacion supervisada."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src import statistical_validation as sval


def get_model_feature_columns(windowed_df: pd.DataFrame) -> list[str]:
    """Delega en la validacion estadistica para mantener una sola definicion de features."""
    return sval.get_model_feature_columns(windowed_df)


def _logistic_regression_for_pipeline(random_state: int) -> LogisticRegression:
    """LogisticRegression compatible con sklearn antiguo (multi_class) y reciente (sin multi_class)."""
    base = dict(max_iter=5000, class_weight="balanced", random_state=random_state)
    try:
        return LogisticRegression(multi_class="auto", **base)
    except TypeError:
        return LogisticRegression(**base)


def build_models(random_state: int = 42) -> dict[str, Pipeline | RandomForestClassifier | XGBClassifier | VotingClassifier]:
    """Construye los cinco modelos base y el ensamble por voto suave."""
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    _logistic_regression_for_pipeline(random_state),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": rf,
        "xgboost": xgb,
        "soft_voting_ensemble": VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                )),
                (
                    "xgb",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ],
            voting="soft",
            n_jobs=-1,
        ),
    }


def prepare_modeling_dataset(
    windowed_df: pd.DataFrame,
    target_col: str = "Batch",
    exclude_transition_windows: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepara X, y y DataFrame auxiliar con metadata (p. ej. window_id)."""
    feature_cols = get_model_feature_columns(windowed_df)
    df_model = sval.get_analysis_dataset(
        windowed_df,
        target_col=target_col,
        exclude_transition_windows=exclude_transition_windows,
    )
    missing = [c for c in feature_cols if c not in df_model.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features en el dataset de modelado: {missing[:5]}...")
    X = df_model[feature_cols].copy()
    if X.shape[1] != 120:
        raise ValueError(f"X debe tener 120 columnas modelables; se encontraron {X.shape[1]}.")
    if X.isna().any().any():
        raise ValueError("Hay valores NaN en X; revise el dataset ventaneado.")
    y = df_model[target_col].astype(str).copy()
    n_classes = int(y.nunique())
    if n_classes != 3:
        raise ValueError(f"Se esperaban 3 clases; se encontraron {n_classes}.")
    return X, y, df_model


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Codifica etiquetas de texto a enteros con orden estable."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, le


def compute_sample_weights(y_encoded: np.ndarray) -> np.ndarray:
    """Pesos balanceados por frecuencia de clase."""
    return compute_sample_weight("balanced", y_encoded)


def build_final_xgboost_model(random_state: int = 42) -> XGBClassifier:
    """Instancia XGBClassifier con la misma configuracion que en la etapa de modelado comparativo."""
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )


def validate_feature_schema(
    X: pd.DataFrame,
    feature_columns: list[str],
    expected_n_features: int = 120,
) -> None:
    """Comprueba cardinal de features y que todas las columnas requeridas existan en X."""
    if len(feature_columns) != expected_n_features:
        raise ValueError(
            f"Se esperaban {expected_n_features} nombres en feature_columns; "
            f"se recibieron {len(feature_columns)}."
        )
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(
            f"Faltan {len(missing)} columnas en X respecto a feature_columns "
            f"(ejemplos: {missing[:3]})."
        )


def _proba_per_class_name(
    model: XGBClassifier,
    proba: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict[str, np.ndarray]:
    """Mapea predict_proba a vectores por nombre de clase segun el LabelEncoder."""
    cls_arr = np.asarray(model.classes_)
    out: dict[str, np.ndarray] = {}
    for cname in label_encoder.classes_:
        yi = int(label_encoder.transform([str(cname)])[0])
        col_j = int(np.where(cls_arr == yi)[0][0])
        out[str(cname)] = proba[:, col_j]
    return out


def _confidence_and_margin(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maxima probabilidad y margen entre las dos clases mas probables por fila."""
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    confidence = sorted_p[:, 0]
    margin_top2 = sorted_p[:, 0] - sorted_p[:, 1]
    return confidence, margin_top2


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[XGBClassifier, LabelEncoder, pd.DataFrame]:
    """Entrena XGBoost sobre todo el conjunto modelable y devuelve predicciones en entrenamiento."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    sw = compute_sample_weight("balanced", y_enc)
    model = build_final_xgboost_model(random_state=random_state)
    X_np = X.to_numpy(dtype=np.float64)
    model.fit(X_np, y_enc, sample_weight=sw)
    y_pred_enc = model.predict(X_np).astype(int)
    proba = model.predict_proba(X_np)
    per_cls = _proba_per_class_name(model, proba, le)
    confidence, margin_top2 = _confidence_and_margin(proba)
    rows: list[dict[str, Any]] = []
    for i in range(len(X)):
        row: dict[str, Any] = {
            "row_index": int(i),
            "y_true": int(y_enc[i]),
            "y_pred": int(y_pred_enc[i]),
            "y_true_label": str(le.inverse_transform([y_enc[i]])[0]),
            "y_pred_label": str(le.inverse_transform([y_pred_enc[i]])[0]),
            "confidence": float(confidence[i]),
            "margin_top2": float(margin_top2[i]),
        }
        for cname in le.classes_:
            row[f"prob_{cname}"] = float(per_cls[str(cname)][i])
        rows.append(row)
    training_predictions_df = pd.DataFrame(rows)
    return model, le, training_predictions_df


def predict_with_final_model(
    model: XGBClassifier,
    X: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    """Inferencia ordenando columnas y devolviendo etiquetas y probabilidades por clase."""
    validate_feature_schema(X, feature_columns)
    X_ord = X[feature_columns]
    X_np = X_ord.to_numpy(dtype=np.float64)
    y_pred_enc = model.predict(X_np).astype(int)
    proba = model.predict_proba(X_np)
    le_tmp = LabelEncoder()
    le_tmp.fit(class_names)
    per_cls = _proba_per_class_name(model, proba, le_tmp)
    confidence, margin_top2 = _confidence_and_margin(proba)
    out: dict[str, Any] = {
        "y_pred_encoded": y_pred_enc,
        "y_pred_label": [str(le_tmp.inverse_transform([int(v)])[0]) for v in y_pred_enc],
        "confidence": confidence,
        "margin_top2": margin_top2,
    }
    for cname in class_names:
        out[f"prob_{cname}"] = per_cls[str(cname)]
    return pd.DataFrame(out, index=X.index)
