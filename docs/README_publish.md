# Publicacion GitHub Pages (dashboard estatico existente)

## 1. Contenido de la carpeta `docs/`

- `index.html`: copia del dashboard HTML estatico del proyecto (sin redisenar).
- `assets/`: hojas de estilo y JavaScript referenciados por ese HTML (copia).
- `data/`: archivos de datos minimos que el dashboard carga por `fetch` (copia desde `data/dashboard/`).
- `README_publish.md`: esta guia.

## 2. Origen de `docs/index.html`

Es una copia del archivo HTML principal del dashboard estatico ya presente en el repositorio
(ruta detectada al ejecutar la etapa `static_dashboard_publish`).

## 3. Diseno y logica

No se modifica el diseno ni la logica del dashboard en esta etapa: solo se copian archivos
existentes a la estructura esperada por GitHub Pages.

## 4. Regenerar la publicacion

Desde la raiz del proyecto:

```text
python run_pipeline.py --stage dashboard_exports
python run_pipeline.py --stage powerbi_contract
python run_pipeline.py --stage static_dashboard_publish
```

Si `run_pipeline.py` no define la etapa `powerbi_contract`, omita ese comando y ejecute las otras dos.

## 5. Conectar el repositorio remoto (si aun no esta)

```text
git init
git branch -M main
git remote add origin https://github.com/juancarlosb0023/BCP_Clasificacion.git
```

Si ya existe `origin`:

```text
git remote set-url origin https://github.com/juancarlosb0023/BCP_Clasificacion.git
```

## 6. Subir a GitHub

```text
git status
git add docs src/static_dashboard_publish.py run_pipeline.py
git commit -m "Publish existing static dashboard with GitHub Pages"
git push -u origin main
```

## 7. Activar GitHub Pages

En GitHub: **Repository -> Settings -> Pages -> Build and deployment**

- **Source**: Deploy from a branch
- **Branch**: `main`
- **Folder**: `/docs`
- **Save**

## 8. URL esperada

https://juancarlosb0023.github.io/BCP_Clasificacion/

## 9. Seguridad / privacidad

Si el repositorio es publico, todo lo que este dentro de `docs/` (incluido `docs/data/`) sera publico.
No publique datos sensibles, credenciales, informacion operacional confidencial ni archivos privados.
