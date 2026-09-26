"""Núcleo de una unidad de L1: un día o un mes de punta a punta (§8.1 del TRD-L1)."""

import logging
from dataclasses import dataclass
from pathlib import Path

from dq import Finding, Severity, Stage, Status, emit_findings

from l1_ingest.checks import CheckResult
from l1_ingest.conform import TimestampUnitError, conform
from l1_ingest.download import BASE_URL, ChecksumError, fetch, source_url
from l1_ingest.integrity import check_agg_trade_id, ensure_order
from l1_ingest.manifest import ManifestEntry, write_manifest
from l1_ingest.parse import open_zip_batches, read_zip
from l1_ingest.stream import NotStreamable, stream_partition
from l1_ingest.write import (
    CONSOLIDATED,
    content_hash,
    day_filename,
    partition_path,
    write_partition,
)

logger = logging.getLogger(__name__)

LAYER = "l1"


@dataclass(frozen=True)
class Unit:
    """Un día (si `day` viene) o un mes completo de un activo."""

    year: int
    month: int
    day: int | None = None
    provider: str = "binance"
    market: str = "spot"
    asset: str = "BTCUSDT"

    @property
    def stage(self) -> Stage:
        return Stage.PROVISIONAL if self.day is not None else Stage.CANONICAL

    def __str__(self) -> str:
        day = "" if self.day is None else f"-{self.day:02d}"
        return (
            f"{self.provider}/{self.market}/{self.asset}/"
            f"{self.year:04d}-{self.month:02d}{day}"
        )


@dataclass(frozen=True)
class RunContext:
    mode: str
    run_id: str
    image_version: str
    landing_root: str | Path
    dq_root: str | Path
    manifest_root: str | Path
    source_base_url: str = BASE_URL


@dataclass(frozen=True)
class Result:
    path: str
    content_hash: str
    findings: list[Finding]


def _finding(check: CheckResult, unit: Unit, ctx: RunContext) -> Finding:
    return Finding(
        layer=LAYER,
        mode=ctx.mode,
        check_type=check.check_type,
        severity=check.severity,
        stage=unit.stage,
        status=check.status,
        provider=unit.provider,
        market=unit.market,
        asset=unit.asset,
        year=unit.year,
        month=unit.month,
        metric_value=check.metric_value,
        details=check.details,
        run_id=ctx.run_id,
        image_version=ctx.image_version,
    )


def _emit(checks: list[CheckResult], unit: Unit, ctx: RunContext) -> list[Finding]:
    findings = [_finding(c, unit, ctx) for c in checks]
    emit_findings(findings, ctx.dq_root)
    return findings


def _materialized(data: bytes, path: str) -> tuple[list[CheckResult], str]:
    """Ruta O(mes) para datos desordenados: hay que ordenar la unidad completa."""
    table, _ = read_zip(data)
    table, time_check = conform(table)
    table, reorder = ensure_order(table)
    write_partition(table, path)
    return [time_check, reorder, *check_agg_trade_id(table)], content_hash(table)


def process_unit(unit: Unit, ctx: RunContext) -> Result:
    """Procesa la unidad; todo lo crudo vive en RAM y solo se persiste el resultado.

    El CSV nunca existe entero: se descomprime y decodifica por bloques desde el
    ZIP y cada lote se suelta al escribirse, así que en RAM conviven solo el ZIP
    comprimido y un lote. La ruta materializada (datos desordenados) es O(mes).

    Si aborta por checksum o por unidad temporal, emite antes el hallazgo
    error/fail y relanza. Un fallo de red no es un chequeo: se relanza sin más.
    """
    logger.info("inicio unidad=%s modo=%s run_id=%s", unit, ctx.mode, ctx.run_id)
    url = source_url(unit.asset, unit.year, unit.month, unit.day, ctx.source_base_url)
    try:
        download = fetch(url)
    except ChecksumError as exc:
        fail = CheckResult(
            "checksum_fail", Severity.ERROR, Status.FAIL, 1.0, {"error": str(exc)}
        )
        _emit([fail], unit, ctx)
        raise

    checks = [
        CheckResult(
            "checksum_fail",
            Severity.INFO,
            Status.PASS,
            0.0,
            {"sha256": download.sha256, "source_url": download.source_url},
        )
    ]
    write_manifest(
        [
            ManifestEntry(
                provider=unit.provider,
                market=unit.market,
                asset=unit.asset,
                year=unit.year,
                month=unit.month,
                granularity="monthly" if unit.day is None else "daily",
                source_url=download.source_url,
                sha256=download.sha256,
                downloaded_at=int(download.downloaded_at.timestamp() * 1_000_000),
                file_bytes=download.file_bytes,
                image_version=ctx.image_version,
            )
        ],
        ctx.manifest_root,
        ctx.run_id,
    )

    filename = CONSOLIDATED if unit.day is None else day_filename(unit.day)
    path = partition_path(
        ctx.landing_root,
        unit.provider,
        unit.market,
        unit.asset,
        unit.year,
        unit.month,
        filename,
    )
    try:
        with open_zip_batches(download.data) as (header, batches):
            checks.append(header)
            try:
                rest, digest = stream_partition(batches, path)
            except NotStreamable:
                rest = None
            # El generador suspendido retendría el lector de Arrow y sus bloques.
            del batches
        if rest is None:
            # Fuera del `except`: su traceback retendría el generador y el lote.
            logger.warning("unidad=%s sin orden creciente: ruta materializada", unit)
            rest, digest = _materialized(download.data, path)
    except TimestampUnitError as exc:
        _emit([*checks, exc.check], unit, ctx)
        raise
    checks += rest
    findings = _emit(checks, unit, ctx)
    logger.info("fin unidad=%s ruta=%s content_hash=%s", unit, path, digest)
    return Result(path, digest, findings)
