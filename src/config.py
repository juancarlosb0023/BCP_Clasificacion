"""Parametros del proyecto, constantes y rutas relativas al repositorio (pathlib)."""

from __future__ import annotations

from pathlib import Path

# Raiz del proyecto (carpeta que contiene `src/` y `data/`).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DATA_DASHBOARD_DIR: Path = PROJECT_ROOT / "data" / "dashboard"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

RAW_DATA_FILE: str = "Data_BPC_processed.csv"
SENSOR_WEIGHTS_FILE: str = "Pesos_Ponderados_Proyecto4.xlsx"

TARGET_COL: str = "Batch"
TIME_COL: str = "Timestamp"

CLASSES: list[str] = ["CASTILLA", "MEZCLA", "RUBIALES"]

WINDOW_SECONDS: int = 60
BASE_WINDOW_STATS: list[str] = ["median", "mean", "iqr", "p95", "p05"]
RANDOM_STATE: int = 42
