# Datos derivados para dashboard (Proyecto 4 BPC)

## Origen

Los archivos CSV en esta carpeta son **derivados**: se generan leyendo `outputs/tables/` y
`outputs/reports/` (JSON de estado) y `data/processed/` **sin** recalcular features, modelos, SHAP, estadistica inferencial ni assessment;
los artefactos **thresholded** (Etapa 8B) y **condition_state** (Etapa 8C) solo se **copian** desde `outputs/` si ya existen.

## Uso

No editar manualmente estos archivos en entornos productivos.

Regenerar tras actualizar etapas previas:

```text
python run_pipeline.py --stage dashboard_exports
```

## Archivos

| Archivo | Contenido breve |
|---------|-----------------|
| dashboard_kpis.csv | KPIs consolidados (metric, value, category, description) |
| dashboard_model_metrics.csv | Metricas CV por modelo (filas = modelos) |
| dashboard_confusion_best_model.csv | Matriz de confusion del mejor modelo |
| dashboard_predictions.csv | Predicciones finales + indice de condicion por ventana + umbrales globales/por batch + health por batch derivado (100 - indice) |
| dashboard_feature_importance.csv | Ranking consolidado de features (interpretabilidad) |
| dashboard_rank_*.csv | Rankings agregados por variable, componente, familia, posicion, estadistico |
| dashboard_top_weighted_variables.csv | Variables con mayor peso en assessment |
| dashboard_condition_index_by_batch.csv | Indice de condicion resumido por Batch |
| dashboard_pairwise_permanova.csv | Comparaciones pairwise PERMANOVA (FDR) |
| dashboard_pairwise_permdisp.csv | Comparaciones pairwise PERMDISP (FDR) |
| dashboard_pca_centroids.csv | Centroides PCA por clase |
| dashboard_pca_projection.csv | Proyeccion PCA (PC1, PC2) por ventana y batch |
| dashboard_batch_transitions.csv | Transiciones de operacion entre batches |
| dashboard_assessment_thresholds_summary.csv | KPIs y notas de umbrales V0/H/HH exploratorios (Etapa 8B) |
| dashboard_assessment_thresholds_global.csv | Umbrales globales por variable (P40/P75/P99) |
| dashboard_assessment_thresholds_by_batch.csv | Umbrales por Batch y variable |
| dashboard_sensor_weights_with_thresholds.csv | Pesos normalizados con V0/H/HH globales estimados |
| dashboard_condition_index_thresholded_global_by_window.csv | Indice por ventana con baseline global |
| dashboard_condition_index_thresholded_global_by_batch.csv | Resumen por Batch (baseline global) |
| dashboard_condition_index_thresholded_by_batch_by_window.csv | Indice por ventana con baseline por Batch |
| dashboard_condition_index_thresholded_by_batch_by_batch.csv | Resumen por Batch (baseline por Batch) |
| dashboard_condition_contributions_long.csv | Contribuciones por variable y ventana (Etapa 8C, exploratorio) |
| dashboard_condition_contributions_top_by_window.csv | Top contribuciones agregadas por ventana |
| dashboard_condition_current_state.csv | Estado de condicion de la ultima ventana historica exportada |
| dashboard_condition_alerts_active.csv | Alertas exploratorias activas asociadas a esa ventana (0 o mas filas) |
| dashboard_condition_trend_summary.csv | Resumen de tendencia reciente del indice de condicion |
| current_asset_state.json | Ultimo estado del activo serializado (lectura de `outputs/reports/`) |

## Advertencias

- Las metricas de **generalizacion** relevantes para comparar modelos son las de **validacion cruzada**
  (`dashboard_model_metrics.csv`), no las de entrenamiento del modelo final (pueden ser optimistas).
- El **assessment** de condicion usa en este proyecto principalmente `robust_percentile_fallback`
  cuando los umbrales H, HH y V0 del Excel de pesos estan vacios.
- Los **pesos ponderados** del activo **no** representan importancia ML ni causalidad.
- **SHAP** y permutation importance **no** implican causalidad.
- **MEZCLA** se trata como clase operacional independiente (no es solo transicion).

### Etapa 8C — Estado de condicion y alertas (exploratorio)

- Las **alertas** en `dashboard_condition_alerts_active.csv` son **exploratorias**; **no** son alarmas normativas ni sustituyen procedimientos de operacion.
- **condition_state** resume **bandas visuales** sobre **condition_index**: **normal** si < 20, **attention** si 20 <= indice < 40, **high** si >= 40 (umbrales no normativos). Sobre **EPI_BPC** (= health_index = 100 - condition_index): **normal** si EPI > 80, **attention** si 60 <= EPI <= 80, **high** si EPI < 60.
- **health_index** = 100 − **condition_index** (misma convencion en series y KPIs cuando aplica).
- **No** se calcula **RUL** ni vida util remanente en estos artefactos.
- **current_asset_state.json** representa el **ultimo estado historico disponible** en los datos exportados, **no** una lectura en vivo del activo.

### Umbrales V0 / H / HH (Etapa 8B, exploratorios)

- Los umbrales V0/H/HH globales y por batch son **exploratorios**; no son limites normativos de alarma ni detectan falla real por si solos.
- Se estimaron con percentiles del historico disponible en ventanas: **V0 = P40, H = P75, HH = P99** (por variable; por batch dentro de cada Batch).
- El modo **by_batch** en analisis historico usa el **Batch real** de cada ventana.
- En operacion futura, si se usara batch **predicho**, solo seria aceptable con **confianza y margen** del modelo suficientes; si no, conviene baseline **global** o marcar el assessment como **incierto**.
