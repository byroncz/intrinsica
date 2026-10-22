"""Backfill del linaje de L2 (ITSC-368): publica `content_hash.json` en cada partición que no lo tiene.

Desde la 0.11.0 L2 deja, junto a `events.parquet` y `carry_over.parquet` de cada (θ, mes), un
objeto `content_hash.json` con los dos `content_hash` lógicos; L3 lo lee en vez de recalcularlo
y falla con `l2_lineage_missing` si falta. Las particiones publicadas antes no lo tienen: este
script las cubre sin reprocesar L2. Debe correr **antes** del apply de la imagen de L3 que lo
exige (runbook de L3, "Cambios requeridos").

Por cada partición con sus dos Parquet y sin `content_hash.json`:

  1. Si L2 dejó hallazgos `events_summary` de ese (θ, mes), usa los hashes de la **última
     corrida** (la de mayor `detected_at`): es el mismo valor que `ContentHasher` da sobre los
     archivos, porque L2 los calculó al escribirlos.
  2. Si no hay hallazgo, recalcula los dos hashes con `ContentHasher` leyendo los archivos
     (una partición sin hallazgo no se deja sin linaje).

Antes de escribir nada, recalcula con `ContentHasher` una muestra de las particiones que
saldrían de un hallazgo y aborta (código 1, sin escribir) si alguna difiere. La muestra es de
al menos `--sample-fraction` (5 %) e incluye siempre el primer mes de la serie, 2021-01 y las
particiones cuyas corridas no coinciden entre sí; el resto sale al azar con `--seed`.

Las que ya tienen `content_hash.json` no se tocan (la cuenta del job solo puede crear el
objeto, no pisarlo): el script es idempotente y relanzarlo completa lo que falte. Una partición
con solo uno de los dos Parquet no es de L3 todavía: se reporta y se salta.

Args (todos opcionales):
  --events-root gs://<proyecto>-dc-events/l2   (por defecto, el del proyecto según
                                                OPS_RESULTS_URI; también una ruta local)
  --dq-root gs://<proyecto>-dq-findings/l2     (ídem)
  --provider binance --market spot --asset BTCUSDT
  --sample-fraction 0.05 --seed 0 --must-verify 2021-01
  --workers 8
  --dry-run                                    (verifica la muestra y cuenta, sin escribir)

Última línea (siempre): `particiones: N; escritas: W (hallazgo: H, recalculadas: R); ya
existían: E; sin hallazgo: S; incompletas: I; muestra: M; discrepancias: D; errores: X`.
Sale con 1 si hay discrepancias, errores de escritura o ninguna partición. `escritas` cuenta lo
que este run creó; `sin hallazgo` es cuántas había que recalcular.

Dónde corre: con `ops-script`, que da permiso de lectura sobre el lago y de **creación** solo de
`l2/**/content_hash.json` en `dc-events`; o en local con `uv run`.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pyutils import ContentHasher, lineage, resolve_fs

EVENTS = "events.parquet"
CARRY_OVER = "carry_over.parquet"
LINEAGE = lineage.LINEAGE
FILE = re.compile(
    r"/theta=(0\.\d{8})/year=(\d{4})/month=(\d{2})/"
    r"(events\.parquet|carry_over\.parquet|content_hash\.json)$"
)
MAX_PRINTED = 20

Key = tuple[str, int, int]  # (θ como en la partición, año, mes)


def default_root(suffix: str) -> str:
    match = re.match(r"gs://(.+)-ops/", os.environ.get("OPS_RESULTS_URI", ""))
    if match is None:
        sys.exit(f"sin --{suffix}-root y sin OPS_RESULTS_URI: no sé de qué proyecto es")
    return f"gs://{match[1]}-{suffix}/l2"


def series(provider: str, market: str, asset: str) -> str:
    return f"provider={provider}/market={market}/asset={asset}"


def partition_uri(root: str, parts: str, key: Key, name: str) -> str:
    theta, year, month = key
    return f"{root}/{parts}/theta={theta}/year={year:04d}/month={month:02d}/{name}"


def list_partitions(root: str, parts: str) -> dict[Key, set[str]]:
    """Los archivos de cada (θ, mes) bajo `root`, en un solo listado recursivo."""
    fs, base = resolve_fs(root)
    found: dict[Key, set[str]] = defaultdict(set)
    selector = pafs.FileSelector(
        f"{base}/{parts}", recursive=True, allow_not_found=True
    )
    for info in fs.get_file_info(selector):
        match = FILE.search(info.path) if info.type == pafs.FileType.File else None
        if match:
            theta, year, month, name = match.groups()
            found[theta, int(year), int(month)].add(name)
    return found


def read_findings(
    dq_root: str, provider: str, market: str, asset: str
) -> dict[Key, tuple[dict, bool]]:
    """El `events_summary` de la última corrida de cada (θ, mes) y si las corridas discrepan.

    La última es la de mayor `detected_at` (con `run_id` para desempatar). Discrepan si
    alguna dejó otro par de hashes que la última.
    """
    fs, base = resolve_fs(dq_root)
    # Se lista con selector y no se abre el prefijo con `ds.dataset(base)`: para saber si `base`
    # es archivo o carpeta, pyarrow pide los metadatos del objeto `l2` sin barra, y la condición
    # IAM de ops-script solo da permiso bajo `l2/` (403, ITSC-370).
    selector = pafs.FileSelector(f"{base}/", recursive=True, allow_not_found=True)
    paths = sorted(
        info.path
        for info in fs.get_file_info(selector)
        if info.type == pafs.FileType.File and info.path.endswith(".parquet")
    )
    if not paths:  # sin hallazgos no hay ni esquema contra el cual filtrar
        return {}
    dataset = ds.dataset(paths, filesystem=fs, format="parquet")
    keep = (
        (pc.field("check_type") == "events_summary")
        & (pc.field("provider") == provider)
        & (pc.field("market") == market)
        & (pc.field("asset") == asset)
    )
    table = dataset.to_table(
        columns=["detected_at", "run_id", "year", "month", "details"], filter=keep
    )
    latest: dict[Key, tuple[tuple, dict]] = {}
    hashes: dict[Key, set[tuple[str, str]]] = defaultdict(set)
    for row in table.to_pylist():
        details = json.loads(row["details"])
        key = (f"0.{details['theta']:08d}", row["year"], row["month"])
        order = (row["detected_at"], row["run_id"])
        hashes[key].add(
            (details["events_content_hash"], details["carry_over_content_hash"])
        )
        if key not in latest or order > latest[key][0]:
            latest[key] = (order, details)
    return {key: (latest[key][1], len(hashes[key]) > 1) for key in latest}


def file_hash(path: str) -> str:
    """El `content_hash` de un Parquet leyéndolo un row group a la vez."""
    fs, resolved = resolve_fs(path)
    with closing(pq.ParquetFile(resolved, filesystem=fs)) as parquet:
        hasher = ContentHasher(parquet.schema_arrow)
        for i in range(parquet.num_row_groups):
            table = parquet.read_row_group(i)
            for batch in table.to_batches():
                hasher.update(batch)
            del table
        return hasher.hexdigest()


def partition_hashes(root: str, parts: str, key: Key) -> tuple[str, str]:
    return (
        file_hash(partition_uri(root, parts, key, EVENTS)),
        file_hash(partition_uri(root, parts, key, CARRY_OVER)),
    )


def month_key(text: str) -> tuple[int, int]:
    year, month = text.split("-")
    return int(year), int(month)


def choose_sample(
    keys: list[Key],
    forced: set[tuple[int, int]],
    disagree: set[Key],
    fraction: float,
    seed: int,
) -> list[Key]:
    """La muestra a verificar: los meses y las particiones obligadas y, hasta completar
    `fraction` de `keys` (al menos una si hay), las que salgan al azar.
    """
    must = [k for k in keys if (k[1], k[2]) in forced or k in disagree]
    wanted = max(math.ceil(fraction * len(keys)), len(must)) if keys else 0
    rest = sorted(set(keys) - set(must))
    extra = random.Random(seed).sample(rest, min(len(rest), wanted - len(must)))
    return sorted([*must, *extra])


@dataclass
class Report:
    """Lo que el script cuenta; `line()` es su última línea."""

    complete: int = 0
    existing: int = 0
    incomplete: int = 0
    no_finding: int = 0
    sample: int = 0
    mismatches: int = 0
    from_finding: int = 0
    recomputed: int = 0
    errors: int = 0

    def line(self) -> str:
        return (
            f"particiones: {self.complete}; escritas: "
            f"{self.from_finding + self.recomputed} (hallazgo: {self.from_finding}, "
            f"recalculadas: {self.recomputed}); ya existían: {self.existing}; sin hallazgo: "
            f"{self.no_finding}; incompletas: {self.incomplete}; muestra: {self.sample}; "
            f"discrepancias: {self.mismatches}; errores: {self.errors}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events-root")
    parser.add_argument("--dq-root")
    parser.add_argument("--provider", default="binance")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--must-verify", action="append", default=None, metavar="AAAA-MM"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    ns = parser.parse_args(argv)
    if not 0 < ns.sample_fraction <= 1:
        sys.exit("--sample-fraction debe estar en (0, 1]")

    root = (ns.events_root or default_root("dc-events")).rstrip("/")
    dq_root = (ns.dq_root or default_root("dq-findings")).rstrip("/")
    parts = series(ns.provider, ns.market, ns.asset)

    listed = list_partitions(root, parts)
    findings = read_findings(dq_root, ns.provider, ns.market, ns.asset)
    complete = sorted(k for k, names in listed.items() if {EVENTS, CARRY_OVER} <= names)
    incomplete = sorted(set(listed) - set(complete))
    todo = [k for k in complete if LINEAGE not in listed[k]]
    from_finding = [k for k in todo if k in findings]
    to_recompute = [k for k in todo if k not in findings]
    report = Report(
        complete=len(complete),
        existing=len(complete) - len(todo),
        incomplete=len(incomplete),
        no_finding=len(to_recompute),
    )
    print(
        f"serie: {parts}; completas: {report.complete}; con linaje: {report.existing}; "
        f"por escribir: {len(todo)} (con hallazgo: {len(from_finding)}, sin hallazgo: "
        f"{len(to_recompute)}); incompletas: {report.incomplete}",
        flush=True,
    )
    for key in incomplete[:MAX_PRINTED]:
        print(f"INCOMPLETA θ={key[0]} {key[1]}-{key[2]:02d}: {sorted(listed[key])}")
    if not complete:
        print(f"FALLO sin_particiones: nada que cubrir bajo {root}/{parts}")
        print(report.line())
        return 1

    # La muestra: el primer mes de la serie, 2021-01 (o lo que pida --must-verify) y las
    # particiones cuyas corridas no coinciden, más lo que falte hasta el porcentaje.
    must = {month_key(m) for m in (ns.must_verify or ["2021-01"])}
    must.add(min((year, month) for _, year, month in complete))
    disagree = {k for k in from_finding if findings[k][1]}
    sample = choose_sample(from_finding, must, disagree, ns.sample_fraction, ns.seed)
    report.sample = len(sample)

    def published(key: Key) -> tuple[str, str]:
        details = findings[key][0]
        return details["events_content_hash"], details["carry_over_content_hash"]

    def verify(key: Key) -> tuple[Key, tuple[str, str]]:
        return key, partition_hashes(root, parts, key)

    with ThreadPoolExecutor(ns.workers) as pool:
        bad = [
            (key, published(key), got)
            for key, got in pool.map(verify, sample)
            if got != published(key)
        ]
        report.mismatches = len(bad)
        for key, said, got in bad[:MAX_PRINTED]:
            print(
                f"DISCREPANCIA θ={key[0]} {key[1]}-{key[2]:02d}: el hallazgo dice "
                f"{said}, los archivos dan {got}"
            )
        if bad:
            print(
                f"FALLO muestra_difiere: {len(bad)} de {len(sample)} particiones; no se "
                "escribió nada. Los hallazgos no describen los archivos actuales: "
                "revisa si L2 reprocesó el mes sin dejar hallazgo."
            )
            print(report.line())
            return 1
        print(
            f"muestra verificada: {len(sample)} particiones sin diferencias", flush=True
        )

        # Lo que no tiene hallazgo se calcula entero: nunca se deja sin linaje.
        plan = {key: published(key) for key in from_finding}
        plan |= zip(
            to_recompute,
            pool.map(lambda k: partition_hashes(root, parts, k), to_recompute),
            strict=True,
        )
        if ns.dry_run:
            print(f"--dry-run: se escribirían {len(plan)} content_hash.json")
            report.from_finding, report.recomputed = (
                len(from_finding),
                len(to_recompute),
            )
            print(report.line())
            return 0

        errors: list[tuple[Key, Exception]] = []

        def write(key: Key) -> Key | None:
            path = partition_uri(root, parts, key, LINEAGE)
            try:
                fs, resolved = resolve_fs(path)
                if fs.get_file_info(resolved).type == pafs.FileType.File:
                    return None  # otro lo publicó mientras tanto: no se pisa
                events, carry = plan[key]
                lineage.write(path, lineage.document(events, carry, datetime.now(UTC)))
            except Exception as error:  # noqa: BLE001 - se cuenta y se reporta
                errors.append((key, error))
                return None
            return key

        done = {key for key in pool.map(write, sorted(plan)) if key is not None}
    for key, error in errors[:MAX_PRINTED]:
        print(
            f"ERROR θ={key[0]} {key[1]}-{key[2]:02d}: {type(error).__name__}: {error}"
        )
    report.from_finding = sum(1 for key in from_finding if key in done)
    report.recomputed = sum(1 for key in to_recompute if key in done)
    report.errors = len(errors)
    print(report.line())
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
