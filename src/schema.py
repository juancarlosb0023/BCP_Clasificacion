"""Esquema y metadatos de variables vibracionales del dataset BPC."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.config import TARGET_COL, TIME_COL


def _normalize_name(name: str) -> str:
    """Minusculas y sin marcas diacriticas para reglas robustas de coincidencia."""
    folded = unicodedata.normalize("NFD", name.casefold())
    return "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")


def get_feature_columns(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
) -> list[str]:
    """Devuelve las columnas de entrada excluyendo timestamp y objetivo."""
    skip = {time_col, target_col}
    return [c for c in df.columns if c not in skip]


def infer_signal_family(column_name: str) -> str:
    """Clasifica la familia de senal a partir del nombre de columna."""
    n = _normalize_name(column_name)
    if "desplazamiento pico a pico" in n:
        return "displacement_pp"
    if "aceleracion rms" in n:
        return "acceleration_rms"
    if "velocidad rms" in n:
        return "velocity_rms"
    return "unknown"


def infer_component(column_name: str) -> str:
    """Identifica el componente instrumentado (motor, bomba, variador)."""
    upper = column_name.upper()
    if "MOTOR" in upper:
        return "motor"
    if "BOMBA" in upper:
        return "bomba"
    if "VARIADOR" in upper:
        return "variador"
    return "unknown"


def infer_position(column_name: str) -> str:
    """Detecta la posicion LL, LA, LM o LB en el nombre de columna."""
    name = column_name.strip()
    for pos in ("LM", "LA", "LB", "LL"):
        if name.startswith(f"{pos} ") or name.startswith(f"{pos}\\"):
            return pos.lower()
    for match in re.finditer(r"\b(LL|LA|LM|LB)\b", name):
        return match.group(1).lower()
    return "unknown"


def infer_unit(column_name: str) -> str:
    """Infiere la unidad fisica asociada al tipo de medicion en el nombre."""
    family = infer_signal_family(column_name)
    if family == "displacement_pp":
        return "um"
    if family == "acceleration_rms":
        return "g"
    if family == "velocity_rms":
        return "mm/s"
    return "unknown"


def build_variable_metadata(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Construye una tabla de metadatos por columna feature."""
    features = get_feature_columns(df, time_col=time_col, target_col=target_col)
    rows: list[dict[str, object]] = []
    for col in features:
        rows.append(
            {
                "raw_column": col,
                "family": infer_signal_family(col),
                "component": infer_component(col),
                "position": infer_position(col),
                "unit": infer_unit(col),
                "is_feature": True,
            }
        )
    return pd.DataFrame(rows)
