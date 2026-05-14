# Importacion Power BI — Proyecto 4 BPC (Clasificacion)

Este documento complementa `dashboard_manifest.json` (contrato de columnas y relaciones sugeridas).

## Prerrequisito

Generar o actualizar los CSV/JSON en `data/dashboard/`:

```text
python run_pipeline.py --stage dashboard_exports
```

Opcionalmente, regenerar el contrato:

```text
python run_pipeline.py --stage powerbi_contract
```

## Que archivos importar

Importar como **texto/CSV** (o carpeta) todos los archivos listados en `dashboard_manifest.json` (`tables[].filename` con `status: present`), mas este README si se desea documentacion en el proyecto PBIX.

**JSON:** `current_asset_state.json` puede importarse con **Obtener datos > JSON** y expandir registros en Power Query.

## Tablas principales (hechos y tiempo)

- `dashboard_predictions.csv` — **hecho principal**: una fila por `window_id`, probabilidades, clase predicha, indices de condicion.
- `dashboard_condition_contributions_long.csv` — detalle granular por ventana y variable (grande; usar filtros o agregaciones).

## Tablas auxiliares / dimensiones

- `dashboard_condition_index_thresholded_*_by_window.csv` — extension del hecho por ventana (umbrales global / por batch).
- `dashboard_condition_index_*_by_batch.csv` — resumenes por `Batch`.
- `dashboard_kpis.csv`, `dashboard_assessment_thresholds_summary.csv` — parametros y notas.
- `dashboard_model_metrics.csv`, `dashboard_feature_importance.csv`, `dashboard_rank_*.csv` — interpretabilidad y comparacion de modelos.
- `dashboard_pairwise_permanova.csv`, `dashboard_pairwise_permdisp.csv`, `dashboard_pca_centroids.csv` — estadistica multivariante.
- `dashboard_batch_transitions.csv` — contexto de cambios de lote.
- `dashboard_condition_current_state.csv` — **una fila** snapshot de ultima ventana exportada.
- `dashboard_condition_trend_summary.csv` — tendencia reciente del indice.
- `dashboard_condition_alerts_active.csv` — cero o mas alertas exploratorias.
- `dashboard_sensor_weights_with_thresholds.csv`, `dashboard_assessment_thresholds_*.csv` — umbrales y pesos.

## Advertencias

- Umbrales e indices son **exploratorios**; no son limites normativos de alarma.
- No publicar datos sensibles en repositorios publicos.
