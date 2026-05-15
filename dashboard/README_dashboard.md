# Dashboard Dash — Clasificacion de crudo y monitoreo de condicion (BPC)

## Proposito

Herramienta de **decision support** que lee archivos ya exportados en `data/dashboard/` (CSV y JSON). Integra:

- **Clasificacion de batch** (CASTILLA, MEZCLA, RUBIALES) a partir de firma vibratoria en ventanas agregadas.
- **Metricas del modelo** desde validacion cruzada temporal (referencia de generalizacion).
- **Assessment exploratorio** de condicion relativa del activo, con **EPI_BPC**, bandas visuales y alertas **exploratorias** (no normativas).

La aplicacion **no** reentrena modelos, **no** recalcula SHAP, estadistica ni assessment: solo presenta lo exportado por el pipeline.

## Requisitos y ejecucion

1. Instalar dependencias del proyecto:

```text
pip install -r requirements.txt
```

2. Generar CSV en `data/dashboard/` (el pipeline copia/agrega desde `outputs/tables/`; no entrena en esta etapa). Para incluir estado de condicion y JSON del activo:

```text
python run_pipeline.py --stage condition_state
python run_pipeline.py --stage dashboard_exports
```

Si solo necesita tablas previas al bloque de estado actual:

```text
python run_pipeline.py --stage dashboard_exports
```

3. Ejecutar desde la **raiz** del subproyecto `Proyecto4_BPC_Clasificacion`:

```text
python dashboard/app.py
```

4. Abrir en el navegador (puerto por defecto 8050; si esta ocupado, use `DASH_PORT`):

```text
http://127.0.0.1:8050
```

```text
set DASH_PORT=8051
python dashboard/app.py
```

## EPI_BPC y condition_index

- **EPI_BPC** y el **health index** exportado siguen la convencion **EPI_BPC = 100 - condition_index** cuando aplica en las series (p. ej. `health_index_thresholded_by_batch` en `dashboard_predictions.csv`).
- **Mayor EPI_BPC** indica **mejor** condicion relativa frente a la linea base usada.
- **Mayor condition_index** indica **mayor severidad relativa**.

**Bandas exploratorias (no normativas):**

- Sobre **condition_index**: normal si **< 20**; attention si **20 a 40**; high si **>= 40**.
- Sobre **EPI_BPC** (0-100): normal si **> 80**; attention si **60 a 80**; high si **< 60**.

## Lineas base dinamicas (predicciones operacionales)

En memoria, el dashboard deriva columnas operacionales desde `dashboard_predictions.csv` para alinear umbrales y series con el **crudo predicho** cuando la **confianza** y el **margen entre las dos clases mas probables** superan los umbrales configurados en la app; si no, usa baseline **GLOBAL** y marca el estado operacional como mas **incierto**.

| Columna (export o derivada) | Lectura resumida |
|-----------------------------|------------------|
| `baseline_batch_operational` | Linea base efectiva segun clase predicha y compuerta de confianza, o `GLOBAL`. |
| `baseline_status` | Indica si la linea base sigue al crudo predicho o a fallback global. |
| `health_index_operational` / `condition_index_operational` | Series operacionales coherentes con esa linea base. |

En **analisis historico** suele disponerse del **batch real**; en **operacion futura** la linea base por batch debe alinearse con la **clase predicha** solo si la confianza (y el margen) son suficientes.

## V0 / H / HH

Se estimaron como percentiles **P40, P75 y P99** del historico disponible. Son **exploratorios**; **no** son limites normativos de alarma.

## Estado actual y alertas exploratorias

Artefactos tipicos (solo lectura):

| Archivo | Rol |
|---------|-----|
| `dashboard_condition_current_state.csv` | Ultima **ventana historica exportada**: indices, banda de estado exploratorio, batches, confianza. |
| `dashboard_condition_alerts_active.csv` | Alertas **exploratorias** por variable; **no** son alarmas normativas. |
| `dashboard_condition_contributions_top_by_window.csv` | Contribuyentes al indice; el dashboard filtra la ultima `window_id`. |
| `dashboard_condition_trend_summary.csv` | Resumen de **tendencia reciente** (pendiente sobre ventanas pasadas); **no** es RUL ni vida util remanente. |
| `current_asset_state.json` | Vista consolidada para integraciones; sigue siendo dato **exportado**, no telemetria en vivo garantizada. |

**Estado global vs alertas por variable:** el estado exploratorio resume el **condition_index** global de la ventana. Una variable puede aparecer en alerta **attention** por su `condition_score` aunque el estado global sea **normal**.

## Interpretabilidad y pesos ponderados

- **SHAP** y **permutation importance** explican el **clasificador**; **no** implican causalidad fisica.
- Los **pesos ponderados** del assessment describen aporte al **indice de condicion**; **no** son importancia ML del modelo.

## Batch MEZCLA

**MEZCLA** se modela como **clase operacional independiente**. No debe interpretarse como una composicion fisica garantizada de otros batches.

## Limitaciones

- Los datos son la **ultima exportacion** disponible; no se garantiza lectura en vivo.
- Las metricas de **generalizacion** son las de **validacion cruzada temporal**; las del entrenamiento final sobre todo el conjunto miden **ajuste**, no generalizacion.
- **No** usar el dashboard para **control automatico** de la bomba sin revision tecnica y procedimientos de planta.
- **No** se calcula **RUL** ni **vida util remanente**.
- La **tendencia** mostrada es estadistica reciente sobre el historico exportado, **no** pronostico de falla.
