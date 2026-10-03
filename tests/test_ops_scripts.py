"""Los tres scripts de operación de docs/runbooks/ops-scripts/ contra un lago real.

El lago sale de correr L2 de verdad sobre tres meses de la landing, así la
definición de la costura del seam-check se prueba contra lo que L2 escribe, no
contra lo que se cree que escribe.
"""

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from l2_dc_events.cli import main as l2_main

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "shared/dc_core/tests/fixtures"
SCRIPTS = ROOT / "docs/runbooks/ops-scripts"
MONTHS = (8, 9, 10)
DEC = pa.decimal128(18, 8)
SCHEMA = pa.schema(
    [
        pa.field("agg_trade_id", pa.int64(), nullable=False),
        pa.field("price", DEC, nullable=False),
        pa.field("quantity", DEC, nullable=False),
        pa.field("transact_time", pa.int64(), nullable=False),
    ]
)


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_landing(root: Path, prices: dict[int, str] | None = None) -> None:
    """Tres meses con los 4 735 ticks reales; `prices` pisa el precio de una fila de un mes."""
    rows = [
        line.split(",")
        for line in (FIXTURES / "ticks.csv").read_text().splitlines()[1:]
    ]
    for month in MONTHS:
        price = [Decimal(p) for _, p, _ in rows]
        if prices and month in prices:
            price[10] = Decimal(prices[month])
        table = pa.table(
            {
                "agg_trade_id": pa.array([int(i) for i, _, _ in rows], pa.int64()),
                "price": pa.array(price, DEC),
                "quantity": pa.array([Decimal(1)] * len(rows), DEC),
                "transact_time": pa.array([int(t) for _, _, t in rows], pa.int64()),
            },
            schema=SCHEMA,
        )
        path = (
            root
            / f"provider=binance/market=spot/asset=BTCUSDT/year=2017/month={month:02d}"
            / "consolidated.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=1_000)


def run_l2(tmp_path: Path, *extra: str) -> dict:
    env = {
        "L2_LANDING_ROOT": str(tmp_path / "landing"),
        "L2_EVENTS_ROOT": str(tmp_path / "events"),
        "L2_DQ_ROOT": str(tmp_path / "dq"),
        "IMAGE_VERSION": "0.1.0+test",
    }
    argv = ["--series-start", "2017-08", "--mode", "backfill", *extra]
    assert l2_main(argv, env) == 0
    return env


@pytest.fixture
def lake(tmp_path):
    write_landing(tmp_path / "landing")
    run_l2(tmp_path, "--from", "2017-08", "--to", "2017-10")
    return tmp_path


def last_line(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def theta_partition(lake: Path, theta: str = "0.00010000", month: int = 9) -> Path:
    return (
        lake
        / "events/provider=binance/market=spot/asset=BTCUSDT"
        / f"theta={theta}/year=2017/month={month:02d}"
    )


def test_seam_l2_sobre_lo_que_escribe_l2(lake, capsys):
    seam = load("seam_l2")
    assert seam.main(["--events-root", str(lake / "events")]) == 0
    # 50 θ x 2 bordes (08 -> 09 y 09 -> 10).
    assert last_line(capsys) == "θ: 50; bordes: 100 (ok: 100; fallos: 0)"


def rewrite(path: Path, **changes) -> None:
    table = pq.read_table(path)
    for name, value in changes.items():
        index = table.schema.get_field_index(name)
        column = pa.array([value] * table.num_rows, table.schema.field(name).type)
        table = table.set_column(index, table.schema.field(name), column)
    pq.write_table(table, path)


def test_seam_l2_falta_un_carry_over(lake, capsys):
    (theta_partition(lake) / "carry_over.parquet").unlink()
    seam = load("seam_l2")
    assert seam.main(["--events-root", str(lake / "events")]) == 1
    out = capsys.readouterr().out
    assert "FALLO θ=0.00010000 2017-08 -> 2017-09: falta_carry_over" in out
    assert "FALLO θ=0.00010000 2017-09 -> 2017-10: falta_carry_over" in out
    assert out.strip().endswith("(ok: 98; fallos: 2)")


def test_seam_l2_pendiente_que_no_coincide(lake, capsys):
    # El carry-over de agosto promete otro evento pendiente que el que escribió septiembre.
    rewrite(
        theta_partition(lake, month=8) / "carry_over.parquet",
        pending_reference_agg_trade_id=1,
    )
    seam = load("seam_l2")
    assert seam.main(["--events-root", str(lake / "events")]) == 1
    out = capsys.readouterr().out
    reasons = {
        line.rsplit(": ", 1)[1] for line in out.splitlines() if line.startswith("FALLO")
    }
    assert reasons <= {"cadena_rota", "pendiente_no_coincide", "pendiente_perdido"}
    assert "2017-08 -> 2017-09" in out


def test_seam_l2_sin_eventos_es_fallo(tmp_path, capsys):
    seam = load("seam_l2")
    assert seam.main(["--events-root", str(tmp_path / "vacio")]) == 1
    assert "sin eventos" in capsys.readouterr().out


def test_pies_parquet_minimo_de_price_por_mes(lake, capsys):
    pies = load("pies_parquet")
    assert pies.main(["--landing-root", str(lake / "landing")]) == 0
    line = last_line(capsys)
    assert line.startswith("meses: 3; min(price): ")
    assert line.endswith("con min <= 0: 0")


def test_pies_parquet_caza_el_price_cero(tmp_path, capsys):
    write_landing(tmp_path / "landing", prices={9: "0"})
    pies = load("pies_parquet")
    assert pies.main(["--landing-root", str(tmp_path / "landing")]) == 1
    out = capsys.readouterr().out
    assert "FALLO 2017-09: filas=4735" in out
    assert out.strip().endswith("(2017-09); con min <= 0: 1")


def test_hashes_dos_corridas_del_mismo_mes_coinciden(lake, capsys):
    # Segunda corrida del mes 08 (otro run_id): L2 es determinista.
    run_l2(lake, "--from", "2017-08", "--to", "2017-08", "--force")
    hashes = load("hashes_events_summary")
    assert hashes.main(["--dq-root", str(lake / "dq"), "--month", "2017-08"]) == 0
    assert (
        last_line(capsys)
        == "θ: 50; unidades: 50 (con >= 2 corridas: 50; discrepancias: 0)"
    )


def test_hashes_detecta_una_discrepancia(lake, capsys):
    run_l2(lake, "--from", "2017-08", "--to", "2017-08", "--force")
    # Altera el hash de eventos de un θ en la corrida más reciente.
    tables = {path: pq.read_table(path) for path in (lake / "dq").rglob("*.parquet")}
    latest = max(
        (t["detected_at"][i].as_py(), t["run_id"][i].as_py())
        for t in tables.values()
        for i in range(t.num_rows)
    )[1]
    for path, table in tables.items():
        details = table["details"].to_pylist()
        for i in range(table.num_rows):
            if (
                table["check_type"][i].as_py() == "events_summary"
                and table["run_id"][i].as_py() == latest
            ):
                body = json.loads(details[i])
                body["events_content_hash"] = "0" * 64
                details[i] = json.dumps(body)
                column = table.schema.get_field_index("details")
                table = table.set_column(column, "details", pa.array(details))
                pq.write_table(table, path)
                break
        else:
            continue
        break
    hashes = load("hashes_events_summary")
    assert hashes.main(["--dq-root", str(lake / "dq"), "--month", "2017-08"]) == 1
    out = capsys.readouterr().out
    assert "DISCREPANCIA" in out
    assert "0" * 64 in out
    assert out.strip().endswith("discrepancias: 1)")


def test_hashes_con_una_sola_corrida_no_comprobo_nada(lake, capsys):
    hashes = load("hashes_events_summary")
    assert hashes.main(["--dq-root", str(lake / "dq")]) == 1
    assert "con >= 2 corridas: 0" in last_line(capsys)


def test_seam_l2_pendiente_perdido_cuando_el_mes_siguiente_no_tiene_eventos():
    seam = load("seam_l2")
    point = (Decimal(1), 1, 1)

    def carry(pending):
        row = {"has_pending_event": pending is not None}
        for name in ("reference", "confirm"):
            for column, value in zip(
                ("price", "time", "agg_trade_id"), point, strict=True
            ):
                row[f"pending_{name}_{column}"] = value if pending else None
        return row

    def month(held, first=None, last=None):
        return {"first": first, "last": last, "carry": carry(held)}

    # M+1 no confirmó nada: el pendiente debe pasar tal cual a su carry-over.
    assert seam.check_border(month(point), month(point)) is None
    assert seam.check_border(month(point), month(None)) == "pendiente_perdido"
    # Sin pendiente al cierre no hay costura que comprobar.
    assert seam.check_border(month(None), month(None)) is None
    assert seam.check_border(None, month(None)) == "falta_carry_over"


class FakeGcs:
    """Un GCS de juguete con los permisos de ops-script (o con los que se le den)."""

    def __init__(self, writable: set[str]):
        self.writable = writable
        self.objects: set[tuple[str, str]] = set()

    def bucket(self, name):
        gcs = self

        class Blob:
            def __init__(self, path):
                self.path = path

            def upload_from_string(self, data, if_generation_match=None):
                from google.api_core import exceptions

                if (name, self.path) in gcs.objects:
                    raise exceptions.Forbidden("sin storage.objects.delete")
                if not any(f"{name}/{self.path}".startswith(w) for w in gcs.writable):
                    raise exceptions.Forbidden("sin storage.objects.create")
                gcs.objects.add((name, self.path))

        return type("Bucket", (), {"blob": lambda _, path: Blob(path)})()

    def list_blobs(self, bucket, prefix, max_results):
        return iter([])


def test_permisos_pasa_con_los_permisos_de_ops_script(monkeypatch, capsys):
    from google.cloud import storage

    permisos = load("permisos")
    fake = FakeGcs(writable={"proj-ops/results/"})
    monkeypatch.setattr(storage, "Client", lambda: fake)
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://proj-ops/results/e1/")
    assert permisos.main() == 0
    assert last_line(capsys) == "permisos: 9 de 9 como se esperaba"


def test_permisos_avisa_si_la_cuenta_puede_escribir_de_mas(monkeypatch, capsys):
    from google.cloud import storage

    permisos = load("permisos")
    fake = FakeGcs(writable={"proj-ops/results/", "proj-landing/"})
    monkeypatch.setattr(storage, "Client", lambda: fake)
    monkeypatch.setenv("OPS_RESULTS_URI", "gs://proj-ops/results/e1/")
    assert permisos.main() == 1
    out = capsys.readouterr().out
    assert "FALLO escribir en landing: esperado 403, obtenido ok" in out
    assert out.strip().endswith("permisos: 8 de 9 como se esperaba")
