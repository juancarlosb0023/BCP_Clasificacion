# Dashboard visual (Dash) — Proyecto 4 BPC

## Requisitos

1. Instalar dependencias del proyecto:

```text
pip install -r requirements.txt
```

2. Generar CSV en `data/dashboard/` (solo lectura de `outputs/tables/`; no entrena modelos). Para incluir estado de condicion y JSON del activo, ejecute antes la etapa 8C y luego exportes:

```text
python run_pipeline.py --stage condition_state
python run_pipeline.py --stage dashboard_exports
```

Si solo necesita tablas previas al estado actual, basta con:

```text
python run_pipeline.py --stage dashboard_exports
```

3. Ejecutar la aplicacion desde la **raiz del subproyecto** `Proyecto4_BPC_Clasificacion`:

```text
python dashboard/app.py
```

4. Abrir en el navegador:

```text
http://127.0.0.1:8050
```

## Notas

- La app **solo** lee archivos en `data/dashboard/*.csv` y `data/dashboard/current_asset_state.json` (y el README de esa carpeta).
- No invoca `src/modeling.py`, `src/interpretability.py` ni `src/assessment.py`.
- Si faltan CSV criticos, la pantalla indica ejecutar `dashboard_exports` primero.

## Estado actual del activo

En la pestana **6. Assessment ponderado** y en advertencias se usan artefactos de las etapas **8C / 9C** (solo lectura):

| Archivo | Rol |
|---------|-----|
| `dashboard_condition_current_state.csv` | Una fila con el estado de la **ultima ventana historica** exportada (indices, banda `condition_state`, batches, confianza). |
| `dashboard_condition_alerts_active.csv` | Alertas **exploratorias** por variable (0 o mas filas); no son alarmas normativas. |
| `dashboard_condition_contributions_top_by_window.csv` | Top contribuyentes de condicion por ventana (p. ej. top 5 internos en pipeline); el dashboard filtra la **ultima** `window_id`. |
| `dashboard_condition_contributions_long.csv` | Serie larga para descomposicion por variable en la ultima ventana. |
| `dashboard_condition_trend_summary.csv` | Resumen de tendencia reciente del indice (pendiente sobre ventanas pasadas, no pronostico). |
| `current_asset_state.json` | Vista consolidada para **salida operativa** / integraciones; sigue siendo dato historico exportado, **no** telemetria en vivo. |

**Health index:** donde exista en CSV, `health_index = 100 - condition_index` (misma convencion que en `dashboard_predictions.csv` para `health_index_thresholded_by_batch`).

**Estado global vs alertas por variable:** el `condition_state` resume una banda visual sobre el **indice global** de la ventana. Una variable puede generar una alerta exploratoria **attention** por su `condition_score` aunque el estado global sea **normal**.

### Advertencias

- Todo lo anterior es **exploratorio** y de **apoyo a la decision**; no es **diagnostico normativo**.
- **No** se calcula RUL ni vida util remanente.
- **No** usar el dashboard para **control automatico** sin revision tecnica y procedimientos de planta.

## Lineas base de condicion

El dashboard compara **tres vistas** del indice de condicion (solo lectura de CSV ya exportados):

1. **Assessment original** (`robust_percentile_fallback`): indices y medias por batch segun `dashboard_condition_index_by_batch.csv` y `condition_index` en `dashboard_predictions.csv`.
2. **Threshold global (data-driven)**: indices recalibrados con umbrales globales estimados del historico; ver `dashboard_condition_index_thresholded_global_by_batch.csv`, `dashboard_assessment_thresholds_global.csv` y columnas `condition_index_thresholded_global` / `assessment_method_thresholded_global` en predicciones.
3. **Threshold por batch (data-driven)**: linea base diferenciada por batch operacional **real** del historico; ver `dashboard_condition_index_thresholded_by_batch_by_batch.csv` y columnas `condition_index_thresholded_by_batch`, `assessment_method_thresholded_by_batch` y `baseline_batch_used` en predicciones.

Los niveles **V0 / H / HH** en tablas de pesos son **exploratorios**: en este proyecto se estimaron como percentiles **P05 / P95 / P99** sobre el historico disponible. **No** deben interpretarse como limites normativos de alarma ni como evidencia de falla real.

**Baseline por crudo predicho (implementado en Etapa 10D):** en la pestana **Predicciones**, la app calcula `baseline_batch_operational` a partir de `y_pred_label` solo si `confidence >= 0.80` y `margin_top2 >= 0.15`; si no, usa `GLOBAL` y marca el assessment operacional como incierto (`baseline_status`). Ver tambien la seccion siguiente.

## Bandas dinamicas por crudo predicho (Etapa 10D)

Sin reentrenar modelos ni SHAP, el dashboard deriva en memoria columnas operacionales desde `dashboard_predictions.csv` y grafica **Health Index** y **Condition Index** con bandas de color **exploratorias** (no normativas):

| Columna | Regla resumida |
|---------|----------------|
| `baseline_batch_operational` | `y_pred_label` si pasa la compuerta de confianza; si no, `GLOBAL`. |
| `baseline_status` | `predicted_batch_baseline` o `global_fallback_due_to_low_confidence`. |
| `health_index_operational` | Por batch predicho: `health_index_thresholded_by_batch`; si `GLOBAL`: `100 - condition_index_thresholded_global`. |
| `condition_index_operational` | Por batch predicho: `condition_index_thresholded_by_batch`; si `GLOBAL`: `condition_index_thresholded_global`. |

En **Assessment**, la grafica *Variable individual con V0/H/HH por crudo predicho* une `dashboard_condition_contributions_long.csv` con predicciones y umbrales (`dashboard_assessment_thresholds_by_batch.csv` / `dashboard_assessment_thresholds_global.csv`). El selector permite **Predicted batch** (operacional), **Real batch historico** o **Global**.

**Historico vs operacion:** en analisis historico suele compararse con **Batch real**; en lectura operacional se usa el baseline derivado del modelo con la compuerta anterior, o el fallback global.

## EPI_BPC

**EPI_BPC** es el indicador **gerencial** de condicion relativa del activo en el dashboard, alineado con el **health index** (convencion del proyecto: **health = 100 - condition** donde aplica en las series exportadas).

- **Definicion operacional:** cuando existen columnas de baseline predicho (Etapa 10D), el grafico usa **`health_index_operational`** (linea base dinamica segun `y_pred_label` si la confianza es suficiente; si no, fallback global). Si no hay serie operacional, usa **`health_index_thresholded_by_batch`** del CSV.
- **Interpretacion:** mayor **EPI_BPC** indica **mejor** condicion relativa en la escala 0-100 (exploratoria), frente a severidad del **condition index**.
- **Bandas visuales** en la figura tipo picadora (solo apoyo visual, **no** alarmas normativas):
  - **Verde:** EPI > 80 (normal exploratorio)
  - **Amarillo:** 60 <= EPI <= 80 (observacion)
  - **Rojo:** EPI < 60 (critico exploratorio)
- La **linea base** de los indices subyacentes depende del **crudo predicho** cuando la clasificacion cumple la compuerta de confianza de la app; en caso contrario se usa baseline **global**.
- Los umbrales **V0/H/HH** del historico (P05/P95/P99) **no** son limites normativos de alarma.

La pestana **6. Assessment** incluye la figura principal y una vista tecnica complementaria del **condition index** por batch; la pestana **1. Resumen** muestra un resumen compacto del **EPI_BPC**.
