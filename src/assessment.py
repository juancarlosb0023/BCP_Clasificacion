"""Assessment de condicion del activo con pesos ponderados (separado de interpretabilidad ML)."""

from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import WINDOW_SECONDS


def normalize_text_for_matching(text: str) -> str:
    """Minusculas, sin tildes, caracteres especiales a espacios, espacios colapsados."""
    s = str(text).strip().lower()
    folded = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_weight_variable_key(text: str) -> str:
    """
    Normaliza etiqueta VARIABLE del Excel para lookup en diccionario Proyecto 4.
    Trata signos micro como 'u' (µm / um equivalentes) y reutiliza normalize_text_for_matching.
    """
    s = str(text).strip()
    s = s.replace("\u00b5", "u").replace("\u03bc", "u")
    return normalize_text_for_matching(s)


def _flex_match_column(norm_headers: dict[str, str], patterns: list[str]) -> str | None:
    """Encuentra nombre de columna original cuyo encabezado normalizado coincide con patrones."""
    for pat in patterns:
        pn = normalize_text_for_matching(pat)
        for nk, orig in norm_headers.items():
            if len(pn) <= 2:
                if nk == pn:
                    return orig
            elif nk == pn or pn in nk or nk in pn:
                return orig
    return None


def detect_weights_header_row(
    excel_path: str | Path,
    sheet_name: str | int = 0,
    max_scan_rows: int = 10,
) -> int:
    """
    Encuentra la fila de encabezado del Excel de pesos (titulo opcional arriba).
    Requiere celdas equivalentes a VARIABLE y a Pesos/peso en la misma fila.
    """
    p = Path(excel_path)
    preview = pd.read_excel(p, sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    n = len(preview)
    for i in range(n):
        row = preview.iloc[i]
        tokens: list[str] = []
        for val in row:
            if pd.isna(val):
                continue
            nv = normalize_text_for_matching(str(val))
            if nv:
                tokens.append(nv)
        has_var = any(t in ("variable", "variables") for t in tokens)
        has_peso = any(t in ("pesos", "peso") for t in tokens)
        if has_var and has_peso:
            return int(i)
    raise ValueError(
        "No se pudo detectar fila de encabezado con VARIABLE y Pesos en el Excel de pesos. "
        f"Revise las primeras {max_scan_rows} filas."
    )


def load_sensor_weights(path: str | Path) -> pd.DataFrame:
    """
    Carga la primera hoja del Excel de pesos ponderados.
    Detecta la fila de encabezado aunque exista una fila de titulo previa.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    header_row = detect_weights_header_row(p, sheet_name=0, max_scan_rows=10)
    print(f"Pesos Excel: fila de encabezado detectada indice {header_row}.")
    df = pd.read_excel(p, sheet_name=0, header=header_row)
    df = df.dropna(axis=0, how="all")
    norm_map: dict[str, str] = {}
    for c in df.columns:
        norm_map[normalize_text_for_matching(str(c))] = str(c)
    col_var = _flex_match_column(norm_map, ["variable", "nombre variable", "sensor", "descripcion"])
    col_peso = _flex_match_column(norm_map, ["pesos", "peso", "ponderacion"])
    if col_var is None or col_peso is None:
        raise ValueError(
            "Tras detectar fila de encabezado, no se pudieron ubicar columnas VARIABLE y Pesos."
        )
    var_series = df[col_var].map(lambda x: str(x).strip() if pd.notna(x) else "")
    peso_num = pd.to_numeric(df[col_peso], errors="coerce")
    keep = (var_series.ne("")) & (var_series.str.lower() != "nan") & peso_num.notna()
    df = df.loc[keep].copy()
    return df.reset_index(drop=True)


def clean_sensor_weights(weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza columnas del Excel de pesos y normaliza pesos a suma ~1.
    No modifica el archivo fuente.
    """
    norm_map: dict[str, str] = {}
    for c in weights_df.columns:
        norm_map[normalize_text_for_matching(str(c))] = str(c)

    col_id = _flex_match_column(norm_map, ["id", "codigo", "cod", "no", "numero", "id variable"])
    col_var = _flex_match_column(norm_map, ["variable", "nombre variable", "sensor", "descripcion"])
    col_peso = _flex_match_column(norm_map, ["pesos", "peso", "ponderacion"])
    col_div = _flex_match_column(norm_map, ["division", "división", "div"])
    col_alarma = _flex_match_column(norm_map, ["alarma", "high", "limite alarma"])
    col_crit = _flex_match_column(
        norm_map, ["critico", "critico high high", "high high", "critico high", "limite critico"]
    )
    col_h = _flex_match_column(norm_map, ["h"])
    col_hh = _flex_match_column(norm_map, ["hh"])
    col_v0 = _flex_match_column(norm_map, ["v0", "v 0", "referencia"])

    if col_var is None or col_peso is None:
        raise ValueError(
            "No se pudieron detectar columnas obligatorias (variable, pesos) en el Excel de pesos."
        )

    out = pd.DataFrame(
        {
            "weight_id": weights_df[col_id] if col_id else np.nan,
            "weight_variable": weights_df[col_var].astype(str),
            "weight_raw": pd.to_numeric(weights_df[col_peso], errors="coerce"),
        }
    )
    if col_div and col_div in weights_df.columns:
        out["weight_normalized"] = pd.to_numeric(weights_df[col_div], errors="coerce")
    else:
        s = out["weight_raw"].sum()
        out["weight_normalized"] = np.where(s > 0, out["weight_raw"] / s, np.nan)

    def _num_col(name: str | None) -> pd.Series:
        if name and name in weights_df.columns:
            return pd.to_numeric(weights_df[name], errors="coerce")
        return pd.Series(np.nan, index=weights_df.index)

    out["alarm_high"] = _num_col(col_alarma)
    out["alarm_high_high"] = _num_col(col_crit)
    out["h"] = _num_col(col_h)
    out["hh"] = _num_col(col_hh)
    out["v0"] = _num_col(col_v0)

    wsum = float(out["weight_normalized"].sum())
    if not np.isfinite(wsum) or wsum <= 0:
        raise ValueError("weight_normalized no suma un valor finito positivo.")
    if abs(wsum - 1.0) > 0.05:
        out["weight_normalized"] = out["weight_normalized"] / wsum
        wsum = float(out["weight_normalized"].sum())
    if abs(wsum - 1.0) > 0.02:
        raise ValueError(
            f"Suma de weight_normalized fuera de tolerancia (~1): {wsum}. Revise columna Division o pesos."
        )

    return out.reset_index(drop=True)


def build_expected_weight_to_dataset_mapping() -> dict[str, str]:
    """
    Correspondencia semantica deterministica Proyecto 4: VARIABLE (Excel) -> raw_column (dataset).
    Claves del dict son normalize_weight_variable_key(excel_label).
    """
    pairs: list[tuple[str, str]] = [
        (
            "Desplazamiento Pico a Pico VIX1 \u2014 LL Motor [\u00b5m]",
            "LL MOTOR\\VIX1-74447\\Desplazamiento Pico a Pico [\u00b5m]",
        ),
        (
            "Desplazamiento Pico a Pico VIY1 \u2014 LL Motor [\u00b5m]",
            "LL MOTOR\\VIY1-74448\\Desplazamiento Pico a Pico [\u00b5m]",
        ),
        (
            "Desplazamiento Pico a Pico VIX2 \u2014 LA Motor [\u00b5m]",
            "LA MOTOR\\VIX2-74445\\Desplazamiento Pico a Pico [\u00b5m]",
        ),
        (
            "Desplazamiento Pico a Pico VIY2 \u2014 LA Motor [\u00b5m]",
            "LA MOTOR\\VIY2-74446\\Desplazamiento Pico a Pico [\u00b5m]",
        ),
        (
            "Desplazamiento Pico a Pico VIX3 \u2014 LA Bomba [\u00b5m]",
            "LA BOMBA\\VIX3-74442\\Desplazamiento Pico a Pico [\u00b5m]",
        ),
        (
            "Desplazamiento Pico a Pico VIY3 \u2014 LA Bomba [\u00b5m]",
            "LA BOMBA\\VIY3-74441\\Desplazamiento Pico a Pico [\u00b5m]",
        ),
        (
            "Desplazamiento Pico a Pico VIX4 \u2014 LL Bomba [\u00b5m]",
            "LL BOMBA\\VIX4-74443\\Desplazamiento Pico a Pico [\u00b5m]",
        ),
        (
            "Desplazamiento Pico a Pico VIY4 \u2014 LL Bomba [\u00b5m]",
            "LL BOMBA\\VIY4-74444\\Desplazamiento Pico a Pico [\u00b5m]",
        ),
        (
            "Aceleraci\u00f3n RMS \u2014 LA Motor Horizontal [g]",
            "LA MOTOR\\AC LA H MOTOR\\Aceleraci\u00f3n Rms [g]",
        ),
        (
            "Velocidad RMS \u2014 LA Motor Horizontal [mm/s]",
            "LA MOTOR\\AC LA H MOTOR\\Velocidad Rms [mm/s]",
        ),
        (
            "Aceleraci\u00f3n RMS \u2014 LA Motor Axial [g]",
            "LA MOTOR\\AC LA A MOTOR\\Aceleraci\u00f3n Rms [g]",
        ),
        (
            "Velocidad RMS \u2014 LA Motor Axial [mm/s]",
            "LA MOTOR\\AC LA A MOTOR\\Velocidad Rms [mm/s]",
        ),
        (
            "Aceleraci\u00f3n RMS \u2014 LA Bomba Axial [g]",
            "LA BOMBA\\AC LA A BOMBA\\Aceleraci\u00f3n Rms [g]",
        ),
        (
            "Velocidad RMS \u2014 LA Bomba Axial [mm/s]",
            "LA BOMBA\\AC LA A BOMBA\\Velocidad Rms [mm/s]",
        ),
        (
            "Aceleraci\u00f3n RMS \u2014 LA Bomba Horizontal [g]",
            "LA BOMBA\\AC LA H BOMBA\\Aceleraci\u00f3n Rms [g]",
        ),
        (
            "Velocidad RMS \u2014 LA Bomba Horizontal [mm/s]",
            "LA BOMBA\\AC LA H BOMBA\\Velocidad Rms [mm/s]",
        ),
        (
            "Aceleraci\u00f3n RMS \u2014 LM Variador Horizontal [g]",
            "LM VARIADOR\\AC LM H VARIADOR\\Aceleraci\u00f3n Rms [g]",
        ),
        (
            "Velocidad RMS \u2014 LM Variador Horizontal [mm/s]",
            "LM VARIADOR\\AC LM H VARIADOR\\Velocidad Rms [mm/s]",
        ),
        (
            "Aceleraci\u00f3n RMS \u2014 LB Variador Horizontal [g]",
            "LB VARIADOR\\AC LB H VARIADOR\\Aceleraci\u00f3n Rms [g]",
        ),
        (
            "Velocidad RMS \u2014 LB Variador Horizontal [mm/s]",
            "LB VARIADOR\\AC LB H VARIADOR\\Velocidad Rms [mm/s]",
        ),
        (
            "Aceleraci\u00f3n RMS \u2014 LL Motor Horizontal [g]",
            "LL MOTOR\\AC LL H MOTOR\\Aceleraci\u00f3n Rms [g]",
        ),
        (
            "Velocidad RMS \u2014 LL Motor Horizontal [mm/s]",
            "LL MOTOR\\AC LL H MOTOR\\Velocidad Rms [mm/s]",
        ),
        (
            "Aceleraci\u00f3n RMS \u2014 LL Bomba Horizontal [g]",
            "LL BOMBA\\AC LL H BOMBA\\Aceleraci\u00f3n Rms [g]",
        ),
        (
            "Velocidad RMS \u2014 LL Bomba Horizontal [mm/s]",
            "LL BOMBA\\AC LL H BOMBA\\Velocidad Rms [mm/s]",
        ),
    ]
    out: dict[str, str] = {}
    for excel_label, raw_col in pairs:
        k = normalize_weight_variable_key(excel_label)
        if k in out and out[k] != raw_col:
            raise ValueError(
                f"Clave duplicada en diccionario pesos esperado: {k!r} -> {out[k]!r} vs {raw_col!r}"
            )
        out[k] = raw_col
    return out


def build_weight_mapping(
    weights_clean_df: pd.DataFrame,
    variable_metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Mapea variables de pesos a raw_column del metadata.
    Prioridad: diccionario determinista Proyecto 4; luego exacto/fuzzy sobre metadata.
    Devuelve (mapping_df, warnings).
    """
    warnings: list[str] = []
    n_exp = 24
    if len(weights_clean_df) != n_exp:
        warnings.append(f"Se esperaban {n_exp} filas de pesos; hay {len(weights_clean_df)}.")

    meta = variable_metadata_df.copy()
    if "is_feature" in meta.columns:
        isf = meta["is_feature"]
        mask = (isf == True) | (isf.astype(str).str.lower() == "true")  # noqa: E712
        meta = meta.loc[mask].copy()
    raw_cols = meta["raw_column"].astype(str).tolist()
    raw_set = set(raw_cols)
    norm_raw = {normalize_weight_variable_key(rc): rc for rc in raw_cols}

    expected = build_expected_weight_to_dataset_mapping()

    rows: list[dict[str, Any]] = []
    for _, wr in weights_clean_df.iterrows():
        wv = str(wr["weight_variable"])
        nk = normalize_weight_variable_key(wv)
        matched: str | None = None
        status = "unmatched"
        score = 0.0
        mapping_rule = "none"
        match_notes = ""

        expected_rc = expected.get(nk)
        if expected_rc is not None:
            if expected_rc in raw_set:
                matched = expected_rc
                status = "manual_semantic"
                score = 1.0
                mapping_rule = "manual_semantic"
                match_notes = "mapped_by_expected_project4_dictionary"
            else:
                warnings.append(
                    "Peso: raw_column esperado por diccionario no esta en metadata: "
                    f"{expected_rc.encode('ascii', 'replace').decode('ascii')[:120]}"
                )
                if nk in norm_raw:
                    matched = norm_raw[nk]
                    status = "exact"
                    score = 1.0
                    mapping_rule = "fallback_metadata_exact"
                    match_notes = "expected_raw_missing_used_metadata_exact"
                else:
                    close = difflib.get_close_matches(
                        nk, list(norm_raw.keys()), n=1, cutoff=0.55
                    )
                    if close:
                        matched = norm_raw[close[0]]
                        status = "fuzzy"
                        score = float(
                            difflib.SequenceMatcher(None, nk, close[0]).ratio()
                        )
                        mapping_rule = "fallback_metadata_fuzzy"
                        match_notes = "expected_raw_missing_used_fuzzy"
                    else:
                        mapping_rule = "unmatched"
                        match_notes = "expected_raw_missing_no_fallback"
        else:
            if nk in norm_raw:
                matched = norm_raw[nk]
                status = "exact"
                score = 1.0
                mapping_rule = "metadata_exact"
                match_notes = "excel_label_not_in_dictionary_matched_metadata_exact"
            else:
                close = difflib.get_close_matches(
                    nk, list(norm_raw.keys()), n=1, cutoff=0.55
                )
                if close:
                    matched = norm_raw[close[0]]
                    status = "fuzzy"
                    score = float(
                        difflib.SequenceMatcher(None, nk, close[0]).ratio()
                    )
                    mapping_rule = "metadata_fuzzy"
                    match_notes = "excel_label_not_in_dictionary_used_fuzzy"
                else:
                    mapping_rule = "unmatched"
                    match_notes = "excel_label_not_in_dictionary_no_metadata_match"

        mrow = None
        if matched:
            subm = meta[meta["raw_column"] == matched]
            if len(subm) > 0:
                mrow = subm.iloc[0]
        rows.append(
            {
                "weight_id": wr.get("weight_id", np.nan),
                "weight_variable": wv,
                "matched_raw_column": matched if matched else "",
                "match_status": status,
                "match_score": score,
                "mapping_rule": mapping_rule,
                "match_notes": match_notes,
                "family": mrow["family"] if mrow is not None else np.nan,
                "component": mrow["component"] if mrow is not None else np.nan,
                "position": mrow["position"] if mrow is not None else np.nan,
                "unit": mrow["unit"] if mrow is not None else np.nan,
                "weight_raw": wr["weight_raw"],
                "weight_normalized": wr["weight_normalized"],
                "alarm_high": wr.get("alarm_high", np.nan),
                "alarm_high_high": wr.get("alarm_high_high", np.nan),
                "h": wr.get("h", np.nan),
                "hh": wr.get("hh", np.nan),
                "v0": wr.get("v0", np.nan),
            }
        )

    mapping = pd.DataFrame(rows)
    unmatched = int((mapping["match_status"] == "unmatched").sum())
    if unmatched:
        warnings.append(f"Variables de peso sin match en metadata: {unmatched}.")

    matched_ok = mapping[mapping["matched_raw_column"].astype(str).str.len() > 0]
    dup = matched_ok["matched_raw_column"].value_counts()
    duped = dup[dup > 1]
    if len(duped):
        warnings.append(
            "Duplicados en matched_raw_column: "
            f"{duped.to_dict().__repr__().encode('ascii', 'replace').decode('ascii')}"
        )

    return mapping, warnings


def _univariate_score_detail(
    value: float,
    v0: float | None,
    h: float | None,
    hh: float | None,
    fallback_p05: float,
    fallback_p95: float,
) -> tuple[float, bool]:
    """Score 0-100 y bandera de uso de percentiles robustos como umbrales."""
    used_fallback = False
    v0f = float(v0) if v0 is not None and np.isfinite(v0) else None
    hf = float(h) if h is not None and np.isfinite(h) else None
    hhf = float(hh) if hh is not None and np.isfinite(hh) else None

    if v0f is None or hf is None or hhf is None:
        used_fallback = True
        v0f = float(fallback_p05) if v0f is None else v0f
        hf = float(fallback_p95) if hf is None else hf
        if hhf is None:
            hhf = hf + max(hf - v0f, 1e-9)

    eps = 1e-9
    val = float(value)
    if not np.isfinite(val):
        return float("nan"), used_fallback

    if val <= v0f:
        return 0.0, used_fallback
    if val <= hf:
        den = max(hf - v0f, eps)
        return float(50.0 * (val - v0f) / den), used_fallback
    if val <= hhf:
        den = max(hhf - hf, eps)
        return float(50.0 + 50.0 * (val - hf) / den), used_fallback
    return 100.0, used_fallback


def compute_univariate_condition_score(
    value: float,
    v0: float | None,
    h: float | None,
    hh: float | None,
    fallback_p05: float,
    fallback_p95: float,
) -> float:
    """Indicador exploratorio 0-100 (mayor valor ~ peor condicion con umbrales dados)."""
    s, _ = _univariate_score_detail(value, v0, h, hh, fallback_p05, fallback_p95)
    return s


def compute_condition_index_by_window(
    windowed_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Indice de condicion por ventana usando estadistico __mean y pesos normalizados.
    Indice exploratorio; no implica deteccion de fallas mecanicas.
    """
    meta_cols = [
        c
        for c in (
            "window_id",
            "window_start",
            "window_end",
            "Batch",
            "has_near_transition",
        )
        if c in windowed_df.columns
    ]
    base = windowed_df[meta_cols].copy()

    used = mapping_df[mapping_df["match_status"].isin(["exact", "fuzzy", "manual_semantic"])].copy()
    used = used[used["matched_raw_column"].astype(str).str.len() > 0]

    p05_map: dict[str, float] = {}
    p95_map: dict[str, float] = {}
    for _, mr in used.iterrows():
        rc = str(mr["matched_raw_column"])
        cmean = f"{rc}__mean"
        if cmean not in windowed_df.columns:
            continue
        ser = pd.to_numeric(windowed_df[cmean], errors="coerce")
        p05_map[cmean] = float(ser.quantile(0.05))
        p95_map[cmean] = float(ser.quantile(0.95))

    n_win = len(windowed_df)
    ci_list: list[float] = []
    aw_list: list[float] = []
    nu_list: list[int] = []
    nm_list: list[int] = []
    meth_list: list[str] = []

    for i in range(n_win):
        scores: list[float] = []
        weights: list[float] = []
        fb_flags: list[bool] = []
        n_used = 0
        n_fail = 0
        for _, mr in used.iterrows():
            rc = str(mr["matched_raw_column"])
            cmean = f"{rc}__mean"
            if cmean not in windowed_df.columns:
                n_fail += 1
                continue
            val = windowed_df.iloc[i][cmean]
            if pd.isna(val):
                n_fail += 1
                continue
            v0 = mr["v0"] if "v0" in mr else np.nan
            h = mr["h"] if "h" in mr else np.nan
            hh = mr["hh"] if "hh" in mr else np.nan
            p05 = p05_map.get(cmean, float("nan"))
            p95 = p95_map.get(cmean, float("nan"))
            sc, fb = _univariate_score_detail(
                float(val),
                float(v0) if pd.notna(v0) else None,
                float(h) if pd.notna(h) else None,
                float(hh) if pd.notna(hh) else None,
                p05,
                p95,
            )
            if not np.isfinite(sc):
                continue
            w = float(mr["weight_normalized"])
            scores.append(sc)
            weights.append(w)
            fb_flags.append(fb)
            n_used += 1

        wsum = float(np.sum(weights)) if weights else 0.0
        if wsum > 0:
            ci = float(np.dot(weights, scores) / wsum)
        else:
            ci = float("nan")
        ci_list.append(ci)
        aw_list.append(wsum)
        nu_list.append(n_used)
        nm_list.append(n_fail)
        if not fb_flags:
            meth_list.append("no_data")
        elif all(fb_flags):
            meth_list.append("robust_percentile_fallback")
        elif not any(fb_flags):
            meth_list.append("asset_thresholds")
        else:
            meth_list.append("mixed")

    out = base.copy()
    out["condition_index"] = ci_list
    out["condition_index_available_weight"] = aw_list
    out["n_variables_used"] = nu_list
    out["n_variables_unmatched"] = nm_list
    out["assessment_method"] = meth_list
    return out


def compute_condition_scores_long(
    windowed_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Formato largo: una fila por ventana y variable mapeada con score y peso."""
    used = mapping_df[mapping_df["match_status"].isin(["exact", "fuzzy", "manual_semantic"])].copy()
    used = used[used["matched_raw_column"].astype(str).str.len() > 0]

    p05_map: dict[str, float] = {}
    p95_map: dict[str, float] = {}
    for _, mr in used.iterrows():
        rc = str(mr["matched_raw_column"])
        cmean = f"{rc}__mean"
        if cmean not in windowed_df.columns:
            continue
        ser = pd.to_numeric(windowed_df[cmean], errors="coerce")
        p05_map[cmean] = float(ser.quantile(0.05))
        p95_map[cmean] = float(ser.quantile(0.95))

    rows: list[dict[str, Any]] = []
    for i in range(len(windowed_df)):
        wid = windowed_df.iloc[i].get("window_id", i)
        batch = windowed_df.iloc[i].get("Batch", np.nan)
        for _, mr in used.iterrows():
            rc = str(mr["matched_raw_column"])
            cmean = f"{rc}__mean"
            if cmean not in windowed_df.columns:
                continue
            val = windowed_df.iloc[i][cmean]
            if pd.isna(val):
                continue
            p05 = p05_map[cmean]
            p95 = p95_map[cmean]
            v0 = mr["v0"] if "v0" in mr else np.nan
            h = mr["h"] if "h" in mr else np.nan
            hh = mr["hh"] if "hh" in mr else np.nan
            sc, _ = _univariate_score_detail(
                float(val),
                float(v0) if pd.notna(v0) else None,
                float(h) if pd.notna(h) else None,
                float(hh) if pd.notna(hh) else None,
                p05,
                p95,
            )
            wn = float(mr["weight_normalized"])
            rows.append(
                {
                    "window_id": wid,
                    "Batch": batch,
                    "raw_variable": rc,
                    "value_mean": float(val),
                    "condition_score": sc,
                    "weight_normalized": wn,
                    "weighted_score": sc * wn,
                    "component": mr.get("component", np.nan),
                    "family": mr.get("family", np.nan),
                    "position": mr.get("position", np.nan),
                }
            )
    return pd.DataFrame(rows)


def summarize_condition_by_batch(condition_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen del indice de condicion por Batch."""

    def _q05(s: pd.Series) -> float:
        return float(s.quantile(0.05))

    def _q95(s: pd.Series) -> float:
        return float(s.quantile(0.95))

    agg = condition_df.groupby("Batch", as_index=False).agg(
        n_windows=("condition_index", "count"),
        condition_index_mean=("condition_index", "mean"),
        condition_index_median=("condition_index", "median"),
        condition_index_p05=("condition_index", _q05),
        condition_index_p95=("condition_index", _q95),
        condition_index_min=("condition_index", "min"),
        condition_index_max=("condition_index", "max"),
    )
    return agg


def summarize_condition_global(condition_df: pd.DataFrame) -> pd.DataFrame:
    """Metricas globales en formato metric,value."""
    s = condition_df["condition_index"].dropna()
    rows = [
        ("n_windows", float(len(condition_df))),
        ("condition_index_mean", float(s.mean()) if len(s) else float("nan")),
        ("condition_index_median", float(s.median()) if len(s) else float("nan")),
        ("condition_index_p95", float(s.quantile(0.95)) if len(s) else float("nan")),
        (
            "n_variables_used_median",
            float(condition_df["n_variables_used"].median()),
        ),
        (
            "condition_index_available_weight_median",
            float(condition_df["condition_index_available_weight"].median()),
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_top_weighted_variables(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Variables ponderadas ordenadas por peso normalizado."""
    m = mapping_df[mapping_df["match_status"].isin(["exact", "fuzzy", "manual_semantic"])].copy()
    m = m[m["matched_raw_column"].astype(str).str.len() > 0]
    m = m.sort_values("weight_normalized", ascending=False).reset_index(drop=True)
    total = float(m["weight_normalized"].sum()) if len(m) else 0.0
    if total > 0:
        m["share_percent"] = 100.0 * m["weight_normalized"] / total
    else:
        m["share_percent"] = np.nan
    return m[
        [
            "weight_variable",
            "matched_raw_column",
            "component",
            "family",
            "position",
            "weight_raw",
            "weight_normalized",
            "share_percent",
        ]
    ]


def _feature_stat_col(matched_raw_column: str, value_stat: str) -> str:
    return f"{str(matched_raw_column)}__{value_stat}"


def _matched_mapping_rows(mapping_df: pd.DataFrame) -> pd.DataFrame:
    used = mapping_df[mapping_df["match_status"].isin(["exact", "fuzzy", "manual_semantic"])].copy()
    return used[used["matched_raw_column"].astype(str).str.len() > 0]


def _enforce_v0_h_hh_order(
    v0: float,
    h: float,
    hh: float,
    *,
    std_series: float,
) -> tuple[float, float, float, str]:
    """
    Garantiza V0 < H < HH (orientacion cost). Aplica epsilon si hace falta.
    No elimina variables; solo ajusta umbrales y deja nota en threshold_warning.
    """
    parts: list[str] = []
    v, hm, hx = float(v0), float(h), float(hh)
    if not (np.isfinite(v) and np.isfinite(hm) and np.isfinite(hx)):
        parts.append("non_finite_quantiles")
        return v, hm, hx, ";".join(parts)

    if std_series < 1e-18 or abs(hx - v) < 1e-18:
        parts.append("near_constant_series")
    scale = max(abs(v), abs(hm), abs(hx), 1e-12)
    eps = scale * 1e-6 + 1e-12

    if v >= hm:
        parts.append("v0_ge_h_epsilon_applied")
        hm = v + eps
    if hm >= hx:
        parts.append("h_ge_hh_epsilon_applied")
        hx = hm + eps
    if v >= hm:
        v = hm - 2 * eps
        parts.append("v0_shifted_below_h")
    if v >= hm:
        hm = v + eps
    if hm >= hx:
        hx = hm + eps
    for _ in range(30):
        if v < hm < hx:
            break
        hx = hm + eps
        if v >= hm:
            v = hm - 2 * eps
    if not (v < hm < hx):
        parts.append("forced_symmetric_spacing")
        mid = (float(v0) + float(h) + float(hh)) / 3.0
        span = max(abs(float(hh) - float(v0)), eps * 100, 1e-9)
        v = mid - span
        hm = mid
        hx = mid + span
    return v, hm, hx, ";".join(parts)


def _threshold_row_from_series(
    ser: pd.Series,
    mr: pd.Series,
    *,
    value_stat: str,
    threshold_method: str,
    v0_q: float,
    h_q: float,
    hh_q: float,
    batch: str | None,
) -> dict[str, Any]:
    ser = pd.to_numeric(ser, errors="coerce").dropna()
    n_windows = int(len(ser))
    std_s = float(ser.std(ddof=0)) if n_windows else 0.0
    if n_windows == 0:
        raise ValueError(
            f"Sin valores para umbral: matched_raw_column={mr.get('matched_raw_column')!r} "
            f"batch={batch!r}"
        )
    min_v = float(ser.min())
    max_v = float(ser.max())
    p01 = float(ser.quantile(0.01))
    p05 = float(ser.quantile(0.05))
    med = float(ser.median())
    p95 = float(ser.quantile(0.95))
    p99 = float(ser.quantile(0.99))
    v0_raw = float(ser.quantile(v0_q))
    h_raw = float(ser.quantile(h_q))
    hh_raw = float(ser.quantile(hh_q))
    v0_e, h_e, hh_e, warn = _enforce_v0_h_hh_order(v0_raw, h_raw, hh_raw, std_series=std_s)
    core: dict[str, Any] = {
        "weight_id": mr.get("weight_id", np.nan),
        "weight_variable": str(mr["weight_variable"]),
        "matched_raw_column": str(mr["matched_raw_column"]),
        "component": mr.get("component", np.nan),
        "family": mr.get("family", np.nan),
        "position": mr.get("position", np.nan),
        "unit": mr.get("unit", np.nan),
        "value_stat": value_stat,
        "orientation": "cost",
        "v0_estimated": v0_e,
        "h_estimated": h_e,
        "hh_estimated": hh_e,
        "min_value": min_v,
        "p01": p01,
        "p05": p05,
        "median_value": med,
        "p95": p95,
        "p99": p99,
        "max_value": max_v,
        "n_windows": n_windows,
        "threshold_method": threshold_method,
        "threshold_warning": warn,
    }
    if batch is not None:
        return {"Batch": batch, **core}
    return core


def estimate_thresholds_global(
    windowed_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    value_stat: str = "mean",
    v0_quantile: float = 0.05,
    h_quantile: float = 0.95,
    hh_quantile: float = 0.99,
) -> pd.DataFrame:
    """Umbrales V0/H/HH globales por variable (percentiles sobre todas las ventanas)."""
    used = _matched_mapping_rows(mapping_df)
    rows: list[dict[str, Any]] = []
    for _, mr in used.iterrows():
        ccol = _feature_stat_col(str(mr["matched_raw_column"]), value_stat)
        if ccol not in windowed_df.columns:
            raise ValueError(f"Falta columna de feature '{ccol}' en windowed_df.")
        rows.append(
            _threshold_row_from_series(
                windowed_df[ccol],
                mr,
                value_stat=value_stat,
                threshold_method="global_percentile_p05_p95_p99",
                v0_q=v0_quantile,
                h_q=h_quantile,
                hh_q=hh_quantile,
                batch=None,
            )
        )
    return pd.DataFrame(rows)


def estimate_thresholds_by_batch(
    windowed_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    value_stat: str = "mean",
    target_col: str = "Batch",
    v0_quantile: float = 0.05,
    h_quantile: float = 0.95,
    hh_quantile: float = 0.99,
) -> pd.DataFrame:
    """Umbrales V0/H/HH por Batch y variable (percentiles dentro de cada batch)."""
    if target_col not in windowed_df.columns:
        raise ValueError(f"Falta columna '{target_col}' en windowed_df.")
    batches = sorted(windowed_df[target_col].dropna().astype(str).unique().tolist())
    if len(batches) != 3:
        raise ValueError(
            f"Se esperaban 3 batches en '{target_col}'; se encontraron {len(batches)}: {batches}"
        )
    used = _matched_mapping_rows(mapping_df)
    rows: list[dict[str, Any]] = []
    for batch in batches:
        sub = windowed_df[windowed_df[target_col].astype(str) == batch]
        for _, mr in used.iterrows():
            ccol = _feature_stat_col(str(mr["matched_raw_column"]), value_stat)
            if ccol not in windowed_df.columns:
                raise ValueError(f"Falta columna de feature '{ccol}' en windowed_df.")
            rows.append(
                _threshold_row_from_series(
                    sub[ccol],
                    mr,
                    value_stat=value_stat,
                    threshold_method="batch_percentile_p05_p95_p99",
                    v0_q=v0_quantile,
                    h_q=h_quantile,
                    hh_q=hh_quantile,
                    batch=batch,
                )
            )
    return pd.DataFrame(rows)


def merge_weights_with_thresholds(
    weights_clean_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    thresholds_global_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combina pesos limpios, mapping y umbrales globales estimados.
    Salida: 24 filas con v0/h/hh no vacios (exploratorios, no normativos).
    """
    used = _matched_mapping_rows(mapping_df)[
        [
            "weight_id",
            "weight_variable",
            "matched_raw_column",
            "component",
            "family",
            "position",
            "unit",
        ]
    ].copy()
    wc = weights_clean_df[["weight_id", "weight_raw", "weight_normalized"]].copy()
    used["weight_id"] = pd.to_numeric(used["weight_id"], errors="coerce")
    wc["weight_id"] = pd.to_numeric(wc["weight_id"], errors="coerce")
    th = thresholds_global_df[
        [
            "weight_id",
            "v0_estimated",
            "h_estimated",
            "hh_estimated",
        ]
    ].copy()
    th["weight_id"] = pd.to_numeric(th["weight_id"], errors="coerce")
    out = used.merge(wc, on="weight_id", how="inner").merge(th, on="weight_id", how="inner")
    if len(out) != 24:
        raise ValueError(f"merge_weights_with_thresholds: se esperaban 24 filas; hay {len(out)}.")
    out = out.rename(
        columns={"v0_estimated": "v0", "h_estimated": "h", "hh_estimated": "hh"}
    )
    out["orientation"] = "cost"
    out["threshold_method"] = "global_percentile_p05_p95_p99"
    for _, r in out.iterrows():
        if not (float(r["v0"]) < float(r["h"]) < float(r["hh"])):
            raise ValueError(
                f"V0<H<HH falla tras merge para weight_id={r.get('weight_id')}: "
                f"v0={r['v0']}, h={r['h']}, hh={r['hh']}"
            )
    cols = [
        "weight_id",
        "weight_variable",
        "matched_raw_column",
        "component",
        "family",
        "position",
        "unit",
        "weight_raw",
        "weight_normalized",
        "orientation",
        "v0",
        "h",
        "hh",
        "threshold_method",
    ]
    return out[cols].reset_index(drop=True)


def compute_condition_index_with_global_thresholds(
    windowed_df: pd.DataFrame,
    weights_with_thresholds_df: pd.DataFrame,
    value_stat: str = "mean",
) -> pd.DataFrame:
    """
    Indice por ventana usando umbrales globales data-driven (no normativos).
    """
    meta_cols = [
        c
        for c in (
            "window_id",
            "window_start",
            "window_end",
            "Batch",
            "has_near_transition",
        )
        if c in windowed_df.columns
    ]
    base = windowed_df[meta_cols].copy()
    wdf = weights_with_thresholds_df.copy()
    n_win = len(windowed_df)
    ci_list: list[float] = []
    aw_list: list[float] = []
    nu_list: list[int] = []
    nm_list: list[int] = []
    for i in range(n_win):
        scores: list[float] = []
        weights: list[float] = []
        n_used = 0
        n_fail = 0
        for _, wr in wdf.iterrows():
            rc = str(wr["matched_raw_column"])
            ccol = _feature_stat_col(rc, value_stat)
            if ccol not in windowed_df.columns:
                n_fail += 1
                continue
            val = windowed_df.iloc[i][ccol]
            if pd.isna(val):
                n_fail += 1
                continue
            v0 = float(wr["v0"])
            h = float(wr["h"])
            hh = float(wr["hh"])
            sc = compute_univariate_condition_score(float(val), v0, h, hh, v0, h)
            if not np.isfinite(sc):
                n_fail += 1
                continue
            wn = float(wr["weight_normalized"])
            scores.append(sc)
            weights.append(wn)
            n_used += 1
        wsum = float(np.sum(weights)) if weights else 0.0
        ci = float(np.dot(weights, scores) / wsum) if wsum > 0 else float("nan")
        ci_list.append(ci)
        aw_list.append(wsum)
        nu_list.append(n_used)
        nm_list.append(n_fail)
    out = base.copy()
    out["condition_index"] = ci_list
    out["condition_index_available_weight"] = aw_list
    out["n_variables_used"] = nu_list
    out["n_variables_unmatched"] = nm_list
    out["assessment_method"] = "data_driven_global_percentile_thresholds"
    return out


def compute_condition_index_with_batch_thresholds(
    windowed_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    thresholds_by_batch_df: pd.DataFrame,
    value_stat: str = "mean",
    batch_col: str = "Batch",
) -> pd.DataFrame:
    """
    Indice por ventana usando umbrales por batch (modo historico: Batch real por ventana).
    """
    meta_cols = [
        c
        for c in (
            "window_id",
            "window_start",
            "window_end",
            "Batch",
            "has_near_transition",
        )
        if c in windowed_df.columns
    ]
    base = windowed_df[meta_cols].copy()
    used = _matched_mapping_rows(mapping_df)[
        ["matched_raw_column", "weight_normalized"]
    ].copy()
    th = (
        thresholds_by_batch_df[
            [
                "Batch",
                "matched_raw_column",
                "v0_estimated",
                "h_estimated",
                "hh_estimated",
                "p05",
                "p95",
            ]
        ]
        .drop_duplicates(subset=["Batch", "matched_raw_column"], keep="first")
        .copy()
    )
    key = th.set_index([th["Batch"].astype(str), th["matched_raw_column"].astype(str)])
    n_win = len(windowed_df)
    ci_list: list[float] = []
    aw_list: list[float] = []
    nu_list: list[int] = []
    nm_list: list[int] = []
    for i in range(n_win):
        b = str(windowed_df.iloc[i][batch_col])
        scores: list[float] = []
        weights: list[float] = []
        n_used = 0
        n_fail = 0
        for _, mr in used.iterrows():
            rc = str(mr["matched_raw_column"])
            ccol = _feature_stat_col(rc, value_stat)
            if ccol not in windowed_df.columns:
                n_fail += 1
                continue
            val = windowed_df.iloc[i][ccol]
            if pd.isna(val):
                n_fail += 1
                continue
            try:
                tr = key.loc[(b, rc)]
            except KeyError:
                n_fail += 1
                continue
            if isinstance(tr, pd.DataFrame):
                tr = tr.iloc[0]
            v0 = float(tr["v0_estimated"])
            h = float(tr["h_estimated"])
            hh = float(tr["hh_estimated"])
            fb_p05 = float(tr["p05"])
            fb_p95 = float(tr["p95"])
            sc = compute_univariate_condition_score(float(val), v0, h, hh, fb_p05, fb_p95)
            if not np.isfinite(sc):
                n_fail += 1
                continue
            wn = float(mr["weight_normalized"])
            scores.append(sc)
            weights.append(wn)
            n_used += 1
        wsum = float(np.sum(weights)) if weights else 0.0
        ci = float(np.dot(weights, scores) / wsum) if wsum > 0 else float("nan")
        ci_list.append(ci)
        aw_list.append(wsum)
        nu_list.append(n_used)
        nm_list.append(n_fail)
    out = base.copy()
    out["baseline_batch_used"] = out[batch_col].astype(str)
    out["condition_index"] = ci_list
    out["condition_index_available_weight"] = aw_list
    out["n_variables_used"] = nu_list
    out["n_variables_unmatched"] = nm_list
    out["assessment_method"] = "data_driven_batch_percentile_thresholds"
    return out


def summarize_condition_by_batch_thresholded(
    condition_df: pd.DataFrame,
    assessment_method: str,
) -> pd.DataFrame:
    """Resumen del indice por Batch con metodo de assessment explicito."""
    agg = summarize_condition_by_batch(condition_df)
    agg["assessment_method"] = assessment_method
    return agg


def plot_condition_index_by_batch(condition_df: pd.DataFrame, output_path: str | Path) -> None:
    """Boxplot del indice de condicion por Batch."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = condition_df.dropna(subset=["condition_index", "Batch"])
    if sub.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.1, 0.5, "Sin datos para graficar", fontsize=11)
        ax.axis("off")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    batches = sorted(sub["Batch"].astype(str).unique().tolist())
    data = [sub.loc[sub["Batch"].astype(str) == b, "condition_index"].to_numpy() for b in batches]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=batches)
    ax.set_ylabel("condition_index")
    ax.set_title("Indice de condicion por Batch (exploratorio)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_condition_index_time_series(condition_df: pd.DataFrame, output_path: str | Path) -> None:
    """Serie del indice por window_id, coloreado por Batch si hay pocas clases."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = condition_df.copy()
    if "window_id" not in sub.columns or sub.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.1, 0.5, "Sin window_id", fontsize=11)
        ax.axis("off")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    sub = sub.sort_values("window_id")
    sub = sub.dropna(subset=["condition_index"])
    fig, ax = plt.subplots(figsize=(10, 4))
    batches = sorted(sub["Batch"].astype(str).unique().tolist()) if "Batch" in sub.columns else []
    if len(batches) <= 5 and "Batch" in sub.columns:
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
        for i, b in enumerate(batches):
            sb = sub[sub["Batch"].astype(str) == b]
            ax.plot(
                sb["window_id"].to_numpy(),
                sb["condition_index"].to_numpy(),
                ".",
                alpha=0.35,
                label=str(b),
                color=colors[i % len(colors)],
            )
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.plot(
            sub["window_id"].to_numpy(),
            sub["condition_index"].to_numpy(),
            ".",
            alpha=0.4,
            color="steelblue",
        )
    ax.set_xlabel("window_id")
    ax.set_ylabel("condition_index")
    ax.set_title("Indice de condicion vs ventana (exploratorio)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_condition_contributions_long(
    windowed_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    thresholds_by_batch_df: pd.DataFrame,
    value_stat: str = "mean",
    batch_col: str = "Batch",
) -> pd.DataFrame:
    """
    Descomposicion del indice por variable (modo by_batch, umbrales ya estimados).
    Una fila por ventana x variable mapeada.
    """
    used = _matched_mapping_rows(mapping_df)
    th = thresholds_by_batch_df.drop_duplicates(
        subset=["Batch", "matched_raw_column"], keep="first"
    ).copy()
    th["_b"] = th["Batch"].astype(str)
    th["_rc"] = th["matched_raw_column"].astype(str)
    key_cols = [
        "v0_estimated",
        "h_estimated",
        "hh_estimated",
        "p05",
        "p95",
    ]
    if "orientation" in th.columns:
        key_cols.append("orientation")
    th_idx = th.set_index(["_b", "_rc"])

    rows: list[dict[str, Any]] = []
    for i in range(len(windowed_df)):
        wr = windowed_df.iloc[i]
        b = str(wr[batch_col])
        wid = wr.get("window_id", i)
        wst = wr.get("window_start", np.nan)
        wen = wr.get("window_end", np.nan)
        for _, mr in used.iterrows():
            rc = str(mr["matched_raw_column"])
            ccol = _feature_stat_col(rc, value_stat)
            if ccol not in windowed_df.columns:
                continue
            val = wr[ccol]
            if pd.isna(val):
                continue
            try:
                tr = th_idx.loc[(b, rc)]
            except KeyError:
                continue
            if isinstance(tr, pd.DataFrame):
                tr = tr.iloc[0]
            v0 = float(tr["v0_estimated"])
            h = float(tr["h_estimated"])
            hh = float(tr["hh_estimated"])
            p05 = float(tr["p05"])
            p95 = float(tr["p95"])
            if not (np.isfinite(v0) and np.isfinite(h) and np.isfinite(hh)):
                raise ValueError(f"Umbrales no finitos: Batch={b!r} matched_raw_column={rc!r}")
            orient = str(tr["orientation"]) if "orientation" in tr.index else "cost"
            sc = compute_univariate_condition_score(float(val), v0, h, hh, p05, p95)
            if not np.isfinite(sc):
                continue
            wn = float(mr["weight_normalized"])
            hs = 100.0 - float(sc)
            rows.append(
                {
                    "window_id": wid,
                    "window_start": wst,
                    "window_end": wen,
                    "Batch": b,
                    "baseline_batch_used": b,
                    "raw_variable": rc,
                    "weight_variable": str(mr["weight_variable"]),
                    "matched_raw_column": rc,
                    "component": mr.get("component", np.nan),
                    "family": mr.get("family", np.nan),
                    "position": mr.get("position", np.nan),
                    "unit": mr.get("unit", np.nan),
                    "value_stat": value_stat,
                    "value": float(val),
                    "v0": v0,
                    "h": h,
                    "hh": hh,
                    "orientation": orient,
                    "condition_score": float(sc),
                    "health_score": float(hs),
                    "weight_normalized": wn,
                    "weighted_score": float(sc) * wn,
                    "weighted_health_score": wn * float(hs),
                    "assessment_method": "data_driven_batch_percentile_thresholds",
                }
            )
    return pd.DataFrame(rows)


def summarize_top_contributors_by_window(
    contributions_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Top variables por ventana segun weighted_score."""
    if contributions_df.empty:
        return pd.DataFrame(
            columns=[
                "window_id",
                "rank",
                "raw_variable",
                "component",
                "family",
                "position",
                "condition_score",
                "weight_normalized",
                "weighted_score",
                "share_of_condition_index_percent",
            ]
        )
    out_rows: list[dict[str, Any]] = []
    for wid, grp in contributions_df.groupby("window_id", sort=False):
        g2 = grp.sort_values("weighted_score", ascending=False, kind="mergesort").head(top_n).copy()
        den = float(pd.to_numeric(grp["weighted_score"], errors="coerce").sum())
        if not np.isfinite(den) or den == 0.0:
            den = np.nan
        for rank, (_, r) in enumerate(g2.iterrows(), start=1):
            ws = float(r["weighted_score"])
            share = (100.0 * ws / den) if den and np.isfinite(den) else np.nan
            out_rows.append(
                {
                    "window_id": wid,
                    "rank": rank,
                    "raw_variable": str(r["raw_variable"]),
                    "component": r.get("component", np.nan),
                    "family": r.get("family", np.nan),
                    "position": r.get("position", np.nan),
                    "condition_score": float(r["condition_score"]),
                    "weight_normalized": float(r["weight_normalized"]),
                    "weighted_score": ws,
                    "share_of_condition_index_percent": share,
                }
            )
    return pd.DataFrame(out_rows)


def classify_condition_state(condition_index: float) -> str:
    """Bandas exploratorias sobre el indice agregado (no normativas).

    Cortes 20/40: complemento en severidad de las bandas EPI 60/80 en escala 0-100
    (zona EPI verde 80-100 corresponde a condition 0-20, etc.).
    """
    x = float(condition_index)
    if not np.isfinite(x):
        return "unknown"
    if x < 20.0:
        return "normal"
    if x < 40.0:
        return "attention"
    return "high"


def classify_variable_alert(condition_score: float) -> str:
    """Bandas exploratorias por variable (no alarmas normativas)."""
    x = float(condition_score)
    if not np.isfinite(x):
        return "unknown"
    if x < 50.0:
        return "normal"
    if x < 80.0:
        return "attention"
    return "high"


def build_condition_current_state(
    condition_index_df: pd.DataFrame,
    contributions_df: pd.DataFrame,
    final_predictions_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Una fila: ultima ventana por window_id."""
    if condition_index_df.empty:
        return pd.DataFrame()
    cidx = condition_index_df.sort_values("window_id")
    last_wid = int(cidx["window_id"].max())
    row_ci = cidx[cidx["window_id"].astype(int) == last_wid].iloc[0]
    cont_last = contributions_df[contributions_df["window_id"].astype(int) == last_wid].copy()
    if cont_last.empty:
        raise ValueError("Sin filas de contributions para la ultima ventana.")
    cont_last = cont_last.sort_values("weighted_score", ascending=False)
    top_part = cont_last.nlargest(top_n, "weighted_score", keep="all")
    by_comp = top_part.groupby("component", as_index=False)["weighted_score"].sum()
    by_fam = top_part.groupby("family", as_index=False)["weighted_score"].sum()
    top_comp = str(by_comp.sort_values("weighted_score", ascending=False).iloc[0]["component"])
    top_fam = str(by_fam.sort_values("weighted_score", ascending=False).iloc[0]["family"])
    top_var = str(top_part.iloc[0]["raw_variable"])
    top_ws = float(top_part.iloc[0]["weighted_score"])
    n_att = int(((cont_last["condition_score"] >= 50.0) & (cont_last["condition_score"] < 80.0)).sum())
    n_hi = int((cont_last["condition_score"] >= 80.0).sum())
    ci = float(row_ci["condition_index"])
    hi = 100.0 - ci
    pred_cols = ["window_id", "y_pred_label", "confidence", "margin_top2"]
    pred_sub = final_predictions_df[[c for c in pred_cols if c in final_predictions_df.columns]].copy()
    pred_sub["_wid"] = pred_sub["window_id"].astype(int)
    pred_row = pred_sub[pred_sub["_wid"] == last_wid]
    y_pred = pred_row["y_pred_label"].iloc[0] if len(pred_row) and "y_pred_label" in pred_row.columns else np.nan
    conf = pred_row["confidence"].iloc[0] if len(pred_row) and "confidence" in pred_row.columns else np.nan
    marg = pred_row["margin_top2"].iloc[0] if len(pred_row) and "margin_top2" in pred_row.columns else np.nan
    note = (
        "Exploratory condition state based on data-driven batch thresholds; "
        "not a normative fault diagnosis."
    )
    meth = str(row_ci.get("assessment_method", "data_driven_batch_percentile_thresholds"))
    out = {
        "window_id": last_wid,
        "window_start": row_ci.get("window_start", np.nan),
        "window_end": row_ci.get("window_end", np.nan),
        "Batch": str(row_ci.get("Batch", "")),
        "baseline_batch_used": str(row_ci.get("baseline_batch_used", row_ci.get("Batch", ""))),
        "y_pred_label": y_pred,
        "confidence": conf,
        "margin_top2": marg,
        "condition_index": ci,
        "health_index": hi,
        "condition_state": classify_condition_state(ci),
        "assessment_method": meth,
        "top_component_by_contribution": top_comp,
        "top_family_by_contribution": top_fam,
        "top_variable_by_contribution": top_var,
        "top_variable_weighted_score": top_ws,
        "n_active_attention_variables": n_att,
        "n_active_high_variables": n_hi,
        "note": note,
    }
    return pd.DataFrame([out])


def build_condition_alerts_active(
    contributions_df: pd.DataFrame,
    condition_index_df: pd.DataFrame,
    last_n_windows: int = 3,
) -> pd.DataFrame:
    """
    Alertas exploratorias: variables con score elevado en la ultima ventana y persistencia.
    """
    cols = [
        "alert_id",
        "window_id",
        "raw_variable",
        "component",
        "family",
        "position",
        "unit",
        "condition_score",
        "weighted_score",
        "alert_level",
        "persisted_last_n_windows",
        "persistence_count",
        "baseline_batch_used",
        "assessment_method",
        "note",
    ]
    if contributions_df.empty or condition_index_df.empty:
        return pd.DataFrame(columns=cols)
    last_wid = int(condition_index_df["window_id"].max())
    ordered = condition_index_df.sort_values("window_id")
    tail_ids = ordered["window_id"].astype(int).tail(last_n_windows).tolist()
    last_rows = contributions_df[contributions_df["window_id"].astype(int) == last_wid]
    cand = last_rows[last_rows["condition_score"] >= 50.0].copy()
    if cand.empty:
        return pd.DataFrame(columns=cols)
    note = (
        "Exploratory alert from percentile-based condition score; not a normative alarm."
    )
    alerts: list[dict[str, Any]] = []
    for idx, (_, r) in enumerate(cand.iterrows(), start=1):
        rv = str(r["raw_variable"])
        b = str(r.get("baseline_batch_used", r.get("Batch", "")))
        pers = 0
        for tw in tail_ids:
            sub = contributions_df[
                (contributions_df["window_id"].astype(int) == int(tw))
                & (contributions_df["raw_variable"].astype(str) == rv)
            ]
            if len(sub) == 0:
                continue
            sc = float(sub.iloc[0]["condition_score"])
            if np.isfinite(sc) and sc >= 50.0:
                pers += 1
        n_tail = len(tail_ids)
        persisted = bool(pers >= n_tail and n_tail > 0)
        lvl = classify_variable_alert(float(r["condition_score"]))
        alerts.append(
            {
                "alert_id": f"ALT-{last_wid:05d}-{idx:03d}",
                "window_id": last_wid,
                "raw_variable": rv,
                "component": r.get("component", np.nan),
                "family": r.get("family", np.nan),
                "position": r.get("position", np.nan),
                "unit": r.get("unit", np.nan),
                "condition_score": float(r["condition_score"]),
                "weighted_score": float(r["weighted_score"]),
                "alert_level": lvl,
                "persisted_last_n_windows": persisted,
                "persistence_count": pers,
                "baseline_batch_used": b,
                "assessment_method": str(r.get("assessment_method", "data_driven_batch_percentile_thresholds")),
                "note": note,
            }
        )
    return pd.DataFrame(alerts, columns=cols)


def compute_condition_trend_summary(
    condition_index_df: pd.DataFrame,
    window_minutes: int = 60,
) -> pd.DataFrame:
    """
    Tendencia simple del indice en las ultimas ventanas (1 ventana ~ 60 s).
    No usar como RUL ni vida util remanente.
    """
    if condition_index_df.empty:
        return pd.DataFrame(
            columns=[
                "n_windows_used",
                "first_window_id",
                "last_window_id",
                "condition_index_first",
                "condition_index_last",
                "condition_index_delta",
                "slope_per_window",
                "slope_per_hour",
                "rolling_mean_last",
                "rolling_mean_previous",
                "trend_direction",
            ]
        )
    df = condition_index_df.sort_values("window_id").dropna(subset=["condition_index"]).copy()
    n_tail = min(int(window_minutes), len(df))
    seg = df.tail(n_tail).reset_index(drop=True)
    n = len(seg)
    wid = seg["window_id"].astype(float).to_numpy()
    ci = seg["condition_index"].astype(float).to_numpy()
    first_w = int(seg["window_id"].iloc[0])
    last_w = int(seg["window_id"].iloc[-1])
    c0 = float(ci[0])
    c1 = float(ci[-1])
    delta = c1 - c0
    if n >= 2:
        slope_w, _intercept = np.polyfit(wid, ci, 1)
        slope_w = float(slope_w)
    else:
        slope_w = 0.0
    sec_per_step = float(WINDOW_SECONDS)
    slope_h = slope_w * (3600.0 / sec_per_step) if sec_per_step > 0 else 0.0
    mid = n // 2
    if n >= 2:
        rm_prev = float(np.mean(ci[:mid])) if mid > 0 else float(np.mean(ci))
        rm_last = float(np.mean(ci[mid:])) if mid < n else float(np.mean(ci))
    else:
        rm_prev = float("nan")
        rm_last = float(ci[0])
    if slope_h > 1.0:
        direction = "increasing"
    elif slope_h < -1.0:
        direction = "decreasing"
    else:
        direction = "stable"
    return pd.DataFrame(
        [
            {
                "n_windows_used": n,
                "first_window_id": first_w,
                "last_window_id": last_w,
                "condition_index_first": c0,
                "condition_index_last": c1,
                "condition_index_delta": delta,
                "slope_per_window": slope_w,
                "slope_per_hour": float(slope_h),
                "rolling_mean_last": rm_last,
                "rolling_mean_previous": rm_prev,
                "trend_direction": direction,
            }
        ]
    )


def build_current_asset_state_payload(
    current_state_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    top_contributors_df: pd.DataFrame,
) -> dict[str, Any]:
    """Payload JSON consumible por sistemas externos (exploratorio)."""
    if current_state_df.empty:
        raise ValueError("current_state_df vacio.")
    cs = current_state_df.iloc[0].to_dict()
    last_wid = int(cs["window_id"])
    drivers = top_contributors_df[top_contributors_df["window_id"].astype(int) == last_wid].copy()
    drivers = drivers.sort_values("rank")
    top_list = drivers.head(10).to_dict(orient="records")
    alerts_list = alerts_df.to_dict(orient="records") if len(alerts_df) else []
    trend_dict = trend_df.iloc[0].to_dict() if len(trend_df) else {}
    return {
        "asset_id": "BPC_ESTACION_BOMBEO",
        "state_timestamp": datetime.now(timezone.utc).isoformat(),
        "current_window": {
            "window_id": cs.get("window_id"),
            "window_start": cs.get("window_start"),
            "window_end": cs.get("window_end"),
        },
        "batch_real": cs.get("Batch"),
        "batch_predicted": cs.get("y_pred_label"),
        "classification_confidence": cs.get("confidence"),
        "classification_margin_top2": cs.get("margin_top2"),
        "baseline_used": cs.get("baseline_batch_used"),
        "condition_index": cs.get("condition_index"),
        "health_index": cs.get("health_index"),
        "condition_state": cs.get("condition_state"),
        "assessment_method": "data_driven_batch_percentile_thresholds",
        "top_condition_drivers": top_list,
        "alerts_active": alerts_list,
        "trend": trend_dict,
        "methodological_note": (
            "Exploratory condition assessment based on data-driven percentile thresholds; "
            "not a normative fault diagnosis."
        ),
    }


def export_current_asset_state_json(payload: dict[str, Any], output_path: str | Path) -> None:
    from src.exports import save_json

    save_json(payload, output_path)


def plot_condition_top_contributors_last_window(
    contributions_last_window: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 12,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if contributions_last_window.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.1, 0.5, "Sin datos ultima ventana", fontsize=10)
        ax.axis("off")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    sub = contributions_last_window.nlargest(top_n, "weighted_score").sort_values(
        "weighted_score", ascending=True
    )
    lab = sub["raw_variable"].astype(str).str.slice(0, 48)
    fig, ax = plt.subplots(figsize=(8, max(3, top_n * 0.22)))
    y = np.arange(len(sub))
    ax.barh(y, sub["weighted_score"].to_numpy(), color="steelblue", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("weighted_score (exploratorio)")
    ax.set_title("Top aportes al indice (ultima ventana)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_condition_health_index_time_series(
    condition_index_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if condition_index_df.empty or "window_id" not in condition_index_df.columns:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.1, 0.5, "Sin serie", fontsize=10)
        ax.axis("off")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    sub = condition_index_df.sort_values("window_id").dropna(subset=["condition_index"])
    hi = 100.0 - pd.to_numeric(sub["condition_index"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub["window_id"].to_numpy(), hi.to_numpy(), color="seagreen", linewidth=0.8)
    ax.set_xlabel("window_id")
    ax.set_ylabel("health_index = 100 - condition_index")
    ax.set_title("Serie health index (exploratorio)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_condition_component_contribution_last_window(
    contributions_last_window: pd.DataFrame,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if contributions_last_window.empty or "component" not in contributions_last_window.columns:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.1, 0.5, "Sin datos por componente", fontsize=10)
        ax.axis("off")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    g = (
        contributions_last_window.groupby("component", as_index=False)["weighted_score"]
        .sum()
        .sort_values("weighted_score", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(g["component"].astype(str), g["weighted_score"].to_numpy(), color="coral", alpha=0.85)
    ax.set_xlabel("suma weighted_score")
    ax.set_title("Contribucion por componente (ultima ventana)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_weighted_variables(
    top_weights_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Barras horizontales de pesos normalizados."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if top_weights_df.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.1, 0.5, "Sin datos", fontsize=11)
        ax.axis("off")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    sub = top_weights_df.head(top_n).sort_values("weight_normalized", ascending=True)
    labels = sub["weight_variable"].astype(str).str.slice(0, 50).to_list()
    fig, ax = plt.subplots(figsize=(8, max(3, top_n * 0.25)))
    y = np.arange(len(sub))
    ax.barh(y, sub["weight_normalized"].to_numpy(), color="darkseagreen", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("weight_normalized")
    ax.set_title("Top variables ponderadas (funcion de condicion)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
