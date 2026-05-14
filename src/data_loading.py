"""Lectura de tablas y metadatos desde `data/raw` y `data/processed` sin alterar los archivos fuente."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    RAW_DATA_FILE,
    SENSOR_WEIGHTS_FILE,
    TARGET_COL,
    TIME_COL,
)
from src import assessment as _assessment


def validate_required_columns(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
) -> None:
    """Comprueba que existan las columnas minimas requeridas."""
    missing = [c for c in (time_col, target_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en el DataFrame: {missing}. "
            f"Se esperaban '{time_col}' y '{target_col}'."
        )


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el CSV principal desde `data/raw`, valida y ordena por tiempo."""
    csv_path = Path(path) if path is not None else DATA_RAW_DIR / RAW_DATA_FILE
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo de datos: {csv_path}. "
            "Coloque el CSV en data/raw o pase una ruta explicita."
        )
    df = pd.read_csv(csv_path)
    validate_required_columns(df)
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    if df[TIME_COL].isna().any():
        bad = int(df[TIME_COL].isna().sum())
        raise ValueError(
            f"Hay {bad} valores en '{TIME_COL}' que no se pudieron convertir a datetime."
        )
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def load_sensor_weights(path: str | Path | None = None) -> pd.DataFrame:
    """Carga la tabla de pesos ponderados desde Excel en `data/raw` (deteccion robusta de encabezado)."""
    xlsx_path = Path(path) if path is not None else DATA_RAW_DIR / SENSOR_WEIGHTS_FILE
    if not xlsx_path.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo de pesos: {xlsx_path}. "
            "Coloque el Excel en data/raw o pase una ruta explicita."
        )
    return _assessment.load_sensor_weights(xlsx_path)
