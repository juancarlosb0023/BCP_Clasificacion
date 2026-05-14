"""Preparacion de la carpeta docs/ para GitHub Pages (solo copias, sin transformar el HTML)."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from src.config import DATA_DASHBOARD_DIR, PROJECT_ROOT

FORBIDDEN_SUFFIXES = (".pkl", ".joblib", ".ipynb")
FORBIDDEN_NAMES = {".env", ".git", "thumbs.db"}
SKIP_COPY_NAMES = {"desktop.ini"}


def find_existing_dashboard_html(project_root: Path) -> tuple[Path | None, list[str]]:
    """
    Localiza el HTML principal del dashboard estatico.
    Orden de prioridad: rutas explicitas; luego cualquier index.html bajo dashboard/;
    por ultimo docs/index.html.
    """
    warnings: list[str] = []
    ordered = [
        project_root / "outputs" / "dashboard" / "index.html",
        project_root / "dashboard" / "index.html",
        project_root / "dashboard" / "static" / "index.html",
        project_root / "dashboard" / "html" / "index.html",
        project_root / "public" / "index.html",
        project_root / "static_dashboard" / "index.html",
    ]
    docs_html = project_root / "docs" / "index.html"
    all_candidates: list[Path] = []
    for p in ordered:
        if p.is_file():
            all_candidates.append(p.resolve())
    dash_root = project_root / "dashboard"
    if dash_root.is_dir():
        for p in sorted(dash_root.rglob("index.html")):
            try:
                rel = p.relative_to(project_root)
            except ValueError:
                continue
            if str(rel).startswith("docs" + "/"):
                continue
            rp = p.resolve()
            if rp.is_file() and rp not in all_candidates:
                all_candidates.append(rp)
    if docs_html.is_file() and docs_html.resolve() not in all_candidates:
        all_candidates.append(docs_html.resolve())

    if not all_candidates:
        return None, warnings

    chosen: Path | None = None
    for p in ordered:
        if p.is_file():
            chosen = p.resolve()
            break
    if chosen is None and dash_root.is_dir():
        for p in sorted(dash_root.rglob("index.html")):
            try:
                rel = p.relative_to(project_root)
            except ValueError:
                continue
            if str(rel).startswith("docs" + "/"):
                continue
            if p.is_file():
                chosen = p.resolve()
                break
    if chosen is None and docs_html.is_file():
        chosen = docs_html.resolve()

    if chosen is None:
        return None, warnings

    if len(all_candidates) > 1:
        warnings.append(
            "Multiples HTML candidatos; se publica el de mayor prioridad segun reglas del proyecto: "
            + ", ".join(_rel_posix(project_root, Path(p)) for p in all_candidates)
        )
    if chosen == docs_html.resolve() and len(all_candidates) == 1:
        warnings.append(
            "Unico HTML estatico encontrado: docs/index.html (bundle ya preparado en el repo). "
            "Se regenera docs/ desde ese archivo y desde data/dashboard/."
        )
    return Path(chosen), warnings


def _rel_posix(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def prepare_docs_directory(project_root: Path) -> Path:
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "data").mkdir(parents=True, exist_ok=True)
    (docs_dir / "assets").mkdir(parents=True, exist_ok=True)
    for junk in docs_dir.rglob("desktop.ini"):
        try:
            junk.unlink()
        except OSError:
            pass
    return docs_dir


def _parse_html_local_refs(html_text: str) -> list[str]:
    refs: list[str] = []
    for m in re.finditer(
        r"""<(?:link|script)[^>]+?(?:href|src)\s*=\s*["']([^"']+)["']""",
        html_text,
        flags=re.IGNORECASE,
    ):
        u = m.group(1).strip()
        if u.startswith(("http://", "https://", "//", "data:")):
            continue
        refs.append(u)
    return refs


def _parse_js_data_paths(js_text: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r'["\'](\./data/[^"\']+)["\']', js_text):
        out.append(m.group(1).strip())
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def copy_existing_dashboard_files(source_html: Path, docs_dir: Path) -> list[str]:
    """Copia index.html y recursos referenciados con ruta relativa desde el directorio del HTML fuente."""
    copied: list[str] = []
    src_dir = source_html.parent
    dest_index = docs_dir / "index.html"
    src_res = source_html.resolve()
    dst_res = dest_index.resolve()
    if src_res != dst_res:
        shutil.copy2(source_html, dest_index)
    copied.append(_rel_posix(PROJECT_ROOT, dest_index))

    html_text = source_html.read_text(encoding="utf-8", errors="replace")
    for ref in _parse_html_local_refs(html_text):
        rel = ref.lstrip("./")
        src_file = (src_dir / ref).resolve()
        try:
            src_file.relative_to(src_dir.resolve())
        except ValueError:
            continue
        if not src_file.is_file():
            continue
        if src_file.name.lower() in SKIP_COPY_NAMES:
            continue
        dest_file = (docs_dir / rel).resolve()
        try:
            dest_file.relative_to(docs_dir.resolve())
        except ValueError:
            continue
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        if src_file.resolve() == dest_file.resolve():
            copied.append(_rel_posix(PROJECT_ROOT, dest_file))
            continue
        shutil.copy2(src_file, dest_file)
        copied.append(_rel_posix(PROJECT_ROOT, dest_file))
    return copied


def copy_required_assets(source_html: Path, docs_dir: Path) -> tuple[list[str], list[str]]:
    """
    Asegura assets locales (p. ej. app.js, style.css) copiando la carpeta assets/
    del directorio del HTML fuente si existe.
    """
    warnings: list[str] = []
    copied: list[str] = []
    src_assets = source_html.parent / "assets"
    if not src_assets.is_dir():
        return copied, warnings
    dest_assets = docs_dir / "assets"
    dest_assets.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_assets.rglob("*")):
        if not f.is_file():
            continue
        if f.name.lower() in SKIP_COPY_NAMES:
            continue
        rel = f.relative_to(src_assets)
        dest = dest_assets / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if f.resolve() == dest.resolve():
            copied.append(_rel_posix(PROJECT_ROOT, dest))
            continue
        shutil.copy2(f, dest)
        copied.append(_rel_posix(PROJECT_ROOT, dest))
    return copied, warnings


def copy_required_dashboard_data(project_root: Path, docs_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Copia CSV/JSON requeridos por el JS del dashboard desde data/dashboard/."""
    warnings: list[str] = []
    copied: list[str] = []
    missing_required: list[str] = []
    data_dir = DATA_DASHBOARD_DIR
    app_js = docs_dir / "assets" / "app.js"
    required_names: list[str] = []
    if app_js.is_file():
        js_text = app_js.read_text(encoding="utf-8", errors="replace")
        for rel in _parse_js_data_paths(js_text):
            name = Path(rel).name
            if name and name not in required_names:
                required_names.append(name)
    if not required_names:
        required_names = [
            "current_asset_state.json",
            "dashboard_condition_alerts_active.csv",
            "dashboard_condition_trend_summary.csv",
            "dashboard_condition_contributions_top_by_window.csv",
        ]
    optional_extra = (
        "README_powerbi_import.md",
        "dashboard_manifest.json",
        "README_dashboard_data.md",
    )
    dest_data = docs_dir / "data"
    dest_data.mkdir(parents=True, exist_ok=True)
    for name in required_names:
        src = data_dir / name
        if not src.is_file():
            msg = f"Falta en data/dashboard (requerido por el dashboard): {name}"
            warnings.append(msg)
            missing_required.append(name)
            continue
        dest = dest_data / name
        if src.resolve() == dest.resolve():
            copied.append(_rel_posix(PROJECT_ROOT, dest))
            continue
        shutil.copy2(src, dest)
        copied.append(_rel_posix(PROJECT_ROOT, dest))
    for name in optional_extra:
        src = data_dir / name
        if src.is_file():
            dest = dest_data / name
            if src.resolve() == dest.resolve():
                copied.append(_rel_posix(PROJECT_ROOT, dest))
                continue
            shutil.copy2(src, dest)
            copied.append(_rel_posix(PROJECT_ROOT, dest))
    return copied, warnings, missing_required


def write_publish_readme(docs_dir: Path) -> Path:
    body = """# Publicacion GitHub Pages (dashboard estatico existente)

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
"""
    path = docs_dir / "README_publish.md"
    path.write_text(body, encoding="utf-8")
    return path


def validate_index_exists(docs_dir: Path) -> None:
    if not (docs_dir / "index.html").is_file():
        print("ERROR: Falta docs/index.html tras la publicacion estatica.")
        sys.exit(1)


def validate_no_forbidden_files(docs_dir: Path) -> list[str]:
    """Advertencias por archivos no deseables; termina el proceso si hay extensiones sensibles."""
    warnings: list[str] = []
    if not docs_dir.is_dir():
        return warnings
    fatal: list[str] = []
    for p in docs_dir.rglob("*"):
        if p.is_dir():
            if p.name in (".git", "__pycache__"):
                fatal.append(f"Carpeta no permitida en docs/: {_rel_posix(PROJECT_ROOT, p)}")
            continue
        name = p.name.lower()
        if name in FORBIDDEN_NAMES:
            fatal.append(f"Archivo no permitido en docs/: {_rel_posix(PROJECT_ROOT, p)}")
        if name in SKIP_COPY_NAMES:
            warnings.append(f"Archivo no deseado (Windows): {_rel_posix(PROJECT_ROOT, p)}")
        for suf in FORBIDDEN_SUFFIXES:
            if name.endswith(suf):
                fatal.append(f"Archivo sensible no permitido en docs/: {_rel_posix(PROJECT_ROOT, p)}")
    if fatal:
        print("ERROR: validacion docs/ fallo (archivos prohibidos):")
        for line in fatal:
            print(f"  {line}")
        sys.exit(1)
    return warnings


def _scan_windows_paths_in_index(docs_dir: Path) -> list[str]:
    idx = docs_dir / "index.html"
    if not idx.is_file():
        return []
    text = idx.read_text(encoding="utf-8", errors="replace")
    warns: list[str] = []
    if re.search(r"[A-Za-z]:\\", text):
        warns.append(
            "docs/index.html contiene posibles rutas absolutas de Windows; "
            "revise manualmente o reemplace por rutas relativas."
        )
    if "file://" in text.lower():
        warns.append("docs/index.html contiene referencias file://; pueden fallar en GitHub Pages.")
    return warns


def run_static_dashboard_publish() -> None:
    print("Static dashboard publish")
    source_html, find_warns = find_existing_dashboard_html(PROJECT_ROOT)
    if source_html is None:
        print(
            "No se encontro un HTML estatico existente para publicar. "
            "Genere primero el dashboard HTML o indique manualmente la ruta del archivo HTML principal."
        )
        sys.exit(1)

    print(f"HTML fuente detectado: {_rel_posix(PROJECT_ROOT, source_html)}")
    print("Carpeta destino: docs/")

    docs_dir = prepare_docs_directory(PROJECT_ROOT)
    copied: list[str] = []
    warnings = list(find_warns)

    copied.extend(copy_existing_dashboard_files(source_html, docs_dir))
    assets_copied, aw = copy_required_assets(source_html, docs_dir)
    copied.extend(assets_copied)
    warnings.extend(aw)

    data_copied, dw, missing_data = copy_required_dashboard_data(PROJECT_ROOT, docs_dir)
    copied.extend(data_copied)
    warnings.extend(dw)
    if missing_data:
        print("ERROR: faltan archivos de datos requeridos para el dashboard estatico:")
        for name in missing_data:
            print(f"  {name}")
        print(
            "Ejecute antes: python run_pipeline.py --stage dashboard_exports "
            "(y las etapas previas que generen esos CSV/JSON en data/dashboard/)."
        )
        sys.exit(1)

    readme_path = write_publish_readme(docs_dir)
    copied.append(_rel_posix(PROJECT_ROOT, readme_path))

    warnings.extend(validate_no_forbidden_files(docs_dir))
    warnings.extend(_scan_windows_paths_in_index(docs_dir))

    validate_index_exists(docs_dir)

    seen: set[str] = set()
    copied_u: list[str] = []
    for c in copied:
        if c not in seen:
            seen.add(c)
            copied_u.append(c)

    print("Archivos copiados:")
    for line in sorted(copied_u):
        print(f"  {line}")
    print("Advertencias:")
    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print("  (ninguna)")
    print("Estado: OK")


__all__ = [
    "run_static_dashboard_publish",
    "find_existing_dashboard_html",
    "prepare_docs_directory",
    "copy_existing_dashboard_files",
    "copy_required_assets",
    "copy_required_dashboard_data",
    "write_publish_readme",
    "validate_no_forbidden_files",
    "validate_index_exists",
]
