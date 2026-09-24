"""Emisión de hallazgos: fila persistida en Parquet más línea de log (§9.4 del TRD-L1)."""

import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.fs as pafs
import pyarrow.parquet as pq

from dq.finding import Finding, Severity, to_table

logger = logging.getLogger("dq")

_LOG_LEVELS = {
    Severity.INFO: logging.INFO,
    Severity.WARNING: logging.WARNING,
    Severity.ERROR: logging.ERROR,
}


def _resolve(root: str | Path) -> tuple[pafs.FileSystem, str]:
    """Devuelve el sistema de archivos y la ruta base dentro de él.

    `gs://bucket/prefijo` se resuelve con `FileSystem.from_uri` (sin tocar la
    red); cualquier otra cosa es una ruta local.
    """
    if isinstance(root, str) and root.startswith("gs://"):
        return pafs.FileSystem.from_uri(root)
    return pafs.LocalFileSystem(), str(Path(root).resolve())


def _detected_date(finding: Finding) -> str:
    return datetime.fromtimestamp(finding.detected_at / 1_000_000, UTC).strftime(
        "%Y-%m-%d"
    )


def _log(finding: Finding) -> None:
    logger.log(
        _LOG_LEVELS[finding.severity],
        "check_type=%s status=%s stage=%s provider=%s market=%s asset=%s "
        "year=%d month=%d finding_id=%s",
        finding.check_type,
        finding.status,
        finding.stage,
        finding.provider,
        finding.market,
        finding.asset,
        finding.year,
        finding.month,
        finding.finding_id,
    )


def _write(findings: list[Finding], fs: pafs.FileSystem, base: str) -> list[str]:
    """Escribe los hallazgos bajo `base` en cualquier `pyarrow.fs.FileSystem`."""
    by_date: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        by_date[_detected_date(finding)].append(finding)

    paths = []
    for date, group in by_date.items():
        directory = f"{base}/detected_date={date}"
        if not isinstance(fs, pafs.GcsFileSystem):
            # En un almacén de objetos los directorios no existen: crearlos
            # solo agrega objetos vacíos y puede exigir permisos de bucket.
            fs.create_dir(directory, recursive=True)
        path = f"{directory}/{group[0].run_id}-{uuid.uuid4()}.parquet"
        pq.write_table(
            to_table(group),
            path,
            filesystem=fs,
            compression="zstd",
            compression_level=3,
            write_statistics=True,
        )
        paths.append(path)
        for finding in group:
            _log(finding)
    return paths


def emit_findings(findings: list[Finding], root: str | Path) -> list[str]:
    """Persiste los hallazgos y deja una línea de log por cada uno.

    `root` es una ruta local o `gs://<bucket>/<prefijo>`; la disposición es la
    misma en ambos. Escribe un Parquet (ZSTD-3) por fecha de detección bajo
    `<root>/detected_date=YYYY-MM-DD/`, con nombre único para no sobrescribir
    nunca (append-only). Devuelve las rutas escritas; en GCS son
    `<bucket>/<prefijo>/...`, sin el esquema `gs://`.

    Autenticación en GCS: Application Default Credentials. En Cloud Run son
    las de la service account del Job; en local, las de
    `gcloud auth application-default login`. No hay credenciales en código.
    """
    fs, base = _resolve(root)
    return _write(findings, fs, base)
