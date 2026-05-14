# Proyecto 4 — Clasificación BPC Estación de Bombeo

## Objetivo

Desarrollar un modelo de clasificación supervisada para identificar el tipo de crudo trasegado por una BPC entre CASTILLA, RUBIALES y MEZCLA, usando variables vibracionales del conjunto motor–bomba–variador.

## Estructura del proyecto

- **data/raw**: ubicación de los datos fuente originales (por ejemplo, el dataset con las 24 variables vibracionales, `Timestamp`, `Batch` y archivos de pesos asociados). No deben alterarse en el flujo de trabajo.
- **data/processed**: datos derivados tras limpieza, unión, agregaciones o ventaneo; es el lugar habitual para tablas intermedias y listas para modelado.
- **data/dashboard**: datos preparados específicamente para alimentar visualizaciones o la aplicación de dashboard (formato reducido, agregados, etc.).
- **notebooks**: exploración, experimentación y el informe final en formato notebook.
- **src**: código Python reutilizable del pipeline (carga, calidad, features, modelado, evaluación, exportaciones).
- **outputs**: artefactos generados (figuras, tablas, modelos serializados, informes estáticos) organizados por tipo.
- **dashboard**: código y assets de la aplicación Dash (o front relacionado) cuando se implemente el tablero.
- **documentation**: documentación complementaria del proyecto (notas técnicas, diagramas, referencias).

## Regla sobre datos fuente

Los archivos en `data/raw` no deben modificarse. Toda transformación debe generar archivos derivados en `data/processed` o `data/dashboard`.

## Flujo general del proyecto

1. Carga y validación del dataset.
2. Calidad del dato.
3. Ventaneo temporal y extracción de features.
4. Validación estadística.
5. Modelado comparativo.
6. Interpretabilidad.
7. Assessment ponderado.
8. Exportaciones para dashboard (`python run_pipeline.py --stage dashboard_exports`).
9. Dashboard visual en Dash (`python dashboard/app.py` — ver `dashboard/README_dashboard.md`).
10. Notebook e informe final.
