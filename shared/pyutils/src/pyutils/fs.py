"""Resolución de rutas locales o de objeto (`gs://`) a un sistema de archivos Arrow."""

from pathlib import Path

import pyarrow.fs as pafs


def resolve_fs(root: str | Path) -> tuple[pafs.FileSystem, str]:
    """Devuelve el sistema de archivos y la ruta base dentro de él.

    Una ruta con esquema (`gs://...`) la resuelve `FileSystem.from_uri`; una
    sin esquema es local y se devuelve absoluta.
    """
    if isinstance(root, str) and "://" in root:
        return pafs.FileSystem.from_uri(root)
    return pafs.LocalFileSystem(), str(Path(root).resolve())
