"""Los scripts de viz de `layers/ops_tools/scripts/` contra un día que escribe viz de verdad.

El día sale de correr `viz_tiles` sobre el lago de fixtures de su capa (L1 y L2 reales de
2017-08, 5 θ), así los invariantes se prueban contra lo que el job escribe y no contra lo
que se cree que escribe. Los casos que deben fallar alteran el día ya escrito.
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "layers/ops_tools/scripts"
sys.path.insert(0, str(ROOT / "layers/viz_tiles/tests"))

from lake_fixture import DAY, build_lake
from viz_tiles import cli

SERIES = "provider=binance/market=spot/asset=BTCUSDT"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def day(tmp_path):
    roots, _ticks, _pending = build_lake(tmp_path)
    assert cli.main(["--mode", "tiles", "--day", DAY], roots.env()) == 0
    return Path(roots.tiles)


def day_dir(tiles: Path) -> Path:
    return tiles / SERIES / f"day={DAY}"


def last_line(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def edit_events(
    tiles: Path, section: int, index: int, value: int, fmt: str = "<i"
) -> None:
    """Pisa un valor de `events.bin`: `section` 0 a 2 son tiempos, 3 a 5 posiciones."""
    path = day_dir(tiles) / "events.bin"
    raw = bytearray(path.read_bytes())
    n = len(raw) // 25
    struct.pack_into(fmt, raw, 4 * n * section + 4 * index, value)
    path.write_bytes(raw)


def test_check_day_pasa_con_lo_que_escribe_viz(day, capsys):
    check = load("viz_check_day")
    assert check.main(["--tiles-root", str(day)]) == 0
    index = json.loads((day_dir(day) / "index.json").read_text())
    events = sum(t["events"] for t in index["thetas"])
    assert last_line(capsys) == (
        f"día 2017-08-18: ticks=4735; θ=5; eventos={events}; fallos=0"
    )


def test_check_day_con_dir_y_sin_latest(day, capsys):
    check = load("viz_check_day")
    assert check.main(["--dir", str(day_dir(day))]) == 0
    assert "latest" not in capsys.readouterr().out


def test_check_day_latest_coincide_con_la_pagina(day, capsys):
    (day / "latest.html").write_bytes((day_dir(day) / "index.html").read_bytes())
    check = load("viz_check_day")
    assert check.main(["--tiles-root", str(day), "--day", DAY]) == 0
    assert "latest.html = index.html del 2017-08-18" in capsys.readouterr().out


def test_check_day_latest_distinto(day, capsys):
    (day / "latest.html").write_bytes(b"otra cosa")
    check = load("viz_check_day")
    assert check.main(["--tiles-root", str(day)]) == 1
    assert "FALLO latest_distinto" in capsys.readouterr().out


def test_check_day_ticks_truncados(day, capsys):
    path = day_dir(day) / "ticks.bin"
    path.write_bytes(path.read_bytes()[:-3])
    check = load("viz_check_day")
    assert check.main(["--tiles-root", str(day)]) == 1
    assert "FALLO ticks_dañado" in capsys.readouterr().out


def test_check_day_tamano_de_events(day, capsys):
    path = day_dir(day) / "events.bin"
    path.write_bytes(path.read_bytes()[:-1])
    check = load("viz_check_day")
    assert check.main(["--tiles-root", str(day)]) == 1
    assert "FALLO events_tamano" in capsys.readouterr().out


def test_check_day_un_tick_que_no_es_el_del_evento(day, capsys):
    # La confirmación del primer evento apunta a otro tick: su tiempo ya no coincide.
    path = day_dir(day) / "events.bin"
    raw = path.read_bytes()
    n = len(raw) // 25
    first = struct.unpack_from("<I", raw, 4 * n * 4)[0]
    edit_events(day, 4, 0, first + 1_000, "<I")
    check = load("viz_check_day")
    assert check.main(["--tiles-root", str(day)]) == 1
    out = capsys.readouterr().out
    assert "FALLO tick_no_coincide" in out or "FALLO ticks_desordenados" in out


def test_check_day_un_tiempo_fuera_del_dia(day, capsys):
    edit_events(day, 1, 0, 86_400_001)
    check = load("viz_check_day")
    assert check.main(["--tiles-root", str(day)]) == 1
    assert "FALLO tiempo_fuera_del_dia" in capsys.readouterr().out


def test_check_day_cadena_rota(day, capsys):
    # El extremo del primer evento deja de ser la referencia del segundo.
    index = json.loads((day_dir(day) / "index.json").read_text())
    theta = next(t for t in index["thetas"] if t["events"] >= 2)
    path = day_dir(day) / "events.bin"
    raw = path.read_bytes()
    n = len(raw) // 25
    at = theta["events_offset"]
    ext = struct.unpack_from("<i", raw, 4 * n * 2 + 4 * at)[0]
    edit_events(day, 2, at, ext + 1)
    check = load("viz_check_day")
    assert check.main(["--tiles-root", str(day)]) == 1
    assert "FALLO cadena_rota" in capsys.readouterr().out


def test_probe_ticks_mide_el_dia(day, capsys):
    probe = load("viz_probe_ticks")
    assert probe.main(["--tiles-root", str(day)]) == 0
    out = capsys.readouterr().out.splitlines()
    raw = (day_dir(day) / "ticks.bin").read_bytes()
    assert out[0].startswith(f"día {DAY}: ticks=4735 tramos=1 bytes={len(raw)} ")
    assert [line.split(":")[0] for line in out[1:4]] == [
        "sección dt_ms",
        "sección dprice_zigzag",
        "sección quantity_1e8",
    ]
    assert out[-1].startswith(
        f"sonda ticks: día={DAY} ticks=4735 bytes={len(raw)} gzip="
    )
    assert any(line.startswith("events.bin: ") for line in out)
    assert " events_gzip=" in out[-1]
    assert out[-1].endswith("presupuesto=ok")


def test_probe_ticks_presupuesto_excedido(day, capsys):
    probe = load("viz_probe_ticks")
    assert probe.main(["--dir", str(day_dir(day)), "--page-budget-mb", "0.001"]) == 1
    assert last_line(capsys).endswith("presupuesto=excedido")


def test_probe_ticks_dañado(day, capsys):
    path = day_dir(day) / "ticks.bin"
    path.write_bytes(path.read_bytes()[:-3])
    probe = load("viz_probe_ticks")
    assert probe.main(["--dir", str(day_dir(day))]) == 1
    assert "FALLO ticks_dañado" in capsys.readouterr().out
