"""Las sondas de L3 (`l3_probe_reader.py`, `l3_probe_ties.py`) contra el lago de fixtures de dc_frames.

Con roots locales, que es como corren en un equipo; en la nube solo cambian los roots por `gs://`.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "layers/ops_tools/scripts"
sys.path.insert(0, str(ROOT / "shared/dc_frames/tests"))

from dc_frames import lake as lake_module
from dc_frames import read_frames
from frames_lake import (
    MONTH,
    build_lake,
    read_events,
    read_ticks,
    theta_text,
    write_l1,
    write_l2,
)

MONTH_TEXT = f"{MONTH[0]:04d}-{MONTH[1]:02d}"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(l1, l2, *extra: str) -> list[str]:
    return ["--from", MONTH_TEXT, "--l1-root", str(l1), "--l2-root", str(l2), *extra]


@pytest.fixture
def lake(tmp_path):
    l1, l2, _ticks, events = build_lake(tmp_path, l1_row_group=40, l2_row_group=3)
    return l1, l2, events


def last_line(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def oracle(l1, l2, thetas):
    """Eventos y ticks por fase según el lector, sin la sonda de por medio."""
    events = confirmation = overshoot = 0
    for frames in read_frames(thetas, l1, l2, MONTH, MONTH):
        events += 1
        confirmation += len(frames.confirmation.price)
        overshoot += len(frames.overshoot.price)
    return events, confirmation, overshoot


def test_un_theta_cuenta_lo_mismo_que_el_lector(lake, capsys):
    l1, l2, events = lake
    theta = theta_text(min(events))
    probe = load("l3_probe_reader")
    assert probe.main(["--theta", theta, *args(l1, l2)]) == 0
    out = capsys.readouterr().out
    n, conf, over = oracle(l1, l2, theta)
    row = next(
        line.split() for line in out.splitlines() if line.lstrip().startswith(theta)
    )
    assert row[1:4] == [str(n), str(conf), str(over)]
    assert "veredicto=ok" in out
    assert "ok pared" in out and "ok RSS pico" in out


def test_todos_los_theta_decodifican_cada_row_group_una_vez(lake, capsys):
    l1, l2, events = lake
    probe = load("l3_probe_reader")
    # El archivo de fixtures pesa 300 KB y su pie se lee más de una vez: el 1,1 × no aplica.
    assert probe.main(["--theta", "all", *args(l1, l2, "--max-bytes-ratio", "3")]) == 0
    out = capsys.readouterr().out
    thetas = [theta_text(t) for t in sorted(events)]
    n, conf, over = oracle(l1, l2, thetas)
    final = out.strip().splitlines()[-1]
    assert f"thetas={len(thetas)}" in final and f"eventos={n}" in final
    assert "veredicto=ok" in final
    row_groups = -(-len(read_ticks()) // 40)
    # Ningún row group se decodifica dos veces; el último, que ningún evento cerrado toca, se salta.
    assert (
        f"decodificados del mes={row_groups - 1} de {row_groups} (saltados por estadísticas=1)"
        in out
    )
    assert "repetidos=0" in out
    assert "ok row groups de L1 decodificados más de una vez: 0" in out
    assert "ok bytes de L1" in out
    assert conf + over > 0


def test_los_bytes_medidos_cubren_lo_que_dicen_los_metadatos(lake, capsys):
    l1, l2, events = lake
    probe = load("l3_probe_reader")
    probe.main(["--theta", theta_text(min(events)), *args(l1, l2)])
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith("bytes leídos"))
    measured = int(line.split("L1=")[1].split()[0])
    expected = int(line.split("por metadatos L1=")[1])
    assert measured >= expected > 0


def test_los_bytes_de_meses_anteriores_no_cuentan_contra_el_umbral(tmp_path, capsys):
    ticks, events = read_ticks(), read_events()
    # La referencia del primer evento cae en 2017-07: el lector abre ese mes para leerla.
    reference = min(int(rows[0]["reference_agg_trade_id"]) for rows in events.values())
    l1, l2 = tmp_path / "l1", tmp_path / "l2"
    write_l1(l1, [t for t in ticks if t["id"] <= reference], 40, (2017, 7))
    write_l1(l1, [t for t in ticks if t["id"] > reference], 40)
    for theta, rows in events.items():
        write_l2(l2, theta, rows, 3)
    probe = load("l3_probe_reader")
    assert probe.main(["--theta", "all", *args(l1, l2, "--max-bytes-ratio", "3")]) == 0
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith("bytes leídos"))
    earlier = int(line.split("de meses anteriores ")[1].split(")")[0])
    total = int(re.search(r"bytes_l1=(\d+)", out.strip().splitlines()[-1])[1])
    in_range = int(re.search(r"bytes de L1 del rango (\d+)", out)[1])
    assert earlier > 0
    assert in_range == total - earlier


def test_la_sonda_restaura_el_lector_al_terminar(lake, tmp_path, capsys):
    l1, l2, events = lake
    original = lake_module.open_parquet
    probe = load("l3_probe_reader")
    probe.main(["--theta", theta_text(min(events)), *args(l1, l2)])
    assert lake_module.open_parquet is original
    # También cuando el lector falla.
    ticks = [t for t in read_ticks() if t["id"] != 3101]
    l1, l2, _, events = build_lake(tmp_path, l1_row_group=40, ticks=ticks)
    assert probe.main(["--theta", theta_text(min(events)), *args(l1, l2)]) == 1
    assert lake_module.open_parquet is original


def test_sin_contar_io_no_reporta_bytes_medidos(lake, capsys):
    l1, l2, events = lake
    probe = load("l3_probe_reader")
    assert (
        probe.main(
            ["--theta", theta_text(min(events)), *args(l1, l2, "--sin-contar-io")]
        )
        == 0
    )
    final = last_line(capsys)
    assert "lectura_s=n/d" in final and "bytes_l1=n/d" in final


def test_pared_excedida_sale_con_1(lake, capsys):
    l1, l2, events = lake
    probe = load("l3_probe_reader")
    code = probe.main(
        ["--theta", theta_text(min(events)), *args(l1, l2, "--max-wall-s", "0")]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "FALLO pared" in out and out.strip().endswith("veredicto=excedido")


def test_l1_y_l2_que_no_cuadran_es_un_fallo(tmp_path, capsys):
    ticks = [
        t for t in read_ticks() if t["id"] != 3101
    ]  # el tick de confirmación del primer evento
    l1, l2, _, events = build_lake(tmp_path, l1_row_group=40, ticks=ticks)
    probe = load("l3_probe_reader")
    assert probe.main(["--theta", theta_text(min(events)), *args(l1, l2)]) == 1
    out = capsys.readouterr().out
    assert "FALLO lector: FrameBoundaryError" in out
    assert out.strip().endswith("veredicto=fallo")


def test_la_suma_de_eventos_abiertos_ignora_los_que_se_tocan_en_la_frontera():
    probe = load("l3_probe_reader")
    # (referencia, extremo, ticks): los dos primeros se tocan en 20; el tercero se superpone a ambos: el máximo es 7 + 4, no 5 + 7.
    assert probe.open_sweep([(10, 20, 5), (20, 30, 7), (15, 25, 4)]) == 11
    assert probe.open_sweep([(10, 20, 5), (20, 30, 7)]) == 7
    assert probe.open_sweep([]) == 0


def test_theta_invalido_en_la_lista(lake):
    l1, l2, _ = lake
    probe = load("l3_probe_reader")
    with pytest.raises(SystemExit, match="no cabe en 8 decimales"):
        probe.main(["--theta", "0.123456789", *args(l1, l2)])


def with_ties(tmp_path):
    """Los ticks 3100..3102 comparten un µs (el de la confirmación de 3101) y 3103 es del mismo ms."""
    ticks = read_ticks()
    by_id = {t["id"]: t for t in ticks}
    shared = by_id[3101]["time"]
    for tick_id in (3100, 3102):
        by_id[tick_id]["time"] = shared
    by_id[3103]["time"] = shared + 500
    return build_lake(tmp_path, l1_row_group=40, ticks=ticks), shared


def iso(microseconds: int) -> str:
    probe = load("l3_probe_ties")
    return probe.render(microseconds - microseconds % 1000)[:-3].replace(" ", "T")


def test_empates_cuenta_los_transact_time_del_ms_y_el_grupo_de_cada_theta(
    tmp_path, capsys
):
    (l1, l2, _, _events), shared = with_ties(tmp_path)
    probe = load("l3_probe_ties")
    assert (
        probe.main(["--at", iso(shared), "--l1-root", str(l1), "--l2-root", str(l2)])
        == 0
    )
    out = capsys.readouterr().out
    # Dos transact_time distintos: uno con 3 ticks (3100..3102) y otro con 1 (3103).
    assert "ticks=4 (la card dice 4090), transact_time distintos=2" in out
    assert "1→1, 3→1" in out
    theta = theta_text(100000)
    row = next(
        line.split() for line in out.splitlines() if line.lstrip().startswith(theta)
    )
    # C = 3101: hasta C están 3100 y 3101; después, 3102; su extremo es C, así que ninguno es overshoot.
    assert row[3] == "3101"
    assert row[-4:] == ["3", "2", "1", "0"]
    final = out.strip().splitlines()[-1]
    assert "distintos=2 max_por_tt=3" in final and "max_empate=3" in final


def test_empates_sin_ticks_en_el_ms_falla(lake, capsys):
    l1, l2, _ = lake
    probe = load("l3_probe_ties")
    assert (
        probe.main(
            [
                "--at",
                "2030-01-01T00:00:00.000",
                "--l1-root",
                str(l1),
                "--l2-root",
                str(l2),
            ]
        )
        == 1
    )
    assert "FALLO" in capsys.readouterr().out


def test_empates_sin_l1_del_mes_falla(tmp_path, capsys):
    probe = load("l3_probe_ties")
    code = probe.main(
        [
            "--at",
            "2017-08-18T00:00:00.000",
            "--l1-root",
            str(tmp_path),
            "--l2-root",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert "FALLO l1_faltante" in capsys.readouterr().out


def test_empates_convierte_a_utc_una_zona_explicita(tmp_path, capsys):
    (l1, l2, _, _events), shared = with_ties(tmp_path)
    probe = load("l3_probe_ties")
    # El mismo instante, con el reloj de UTC-5 (cinco horas antes).
    local = iso(shared - 5 * 3_600 * 1_000_000) + "-05:00"
    assert probe.main(["--at", local, "--l1-root", str(l1), "--l2-root", str(l2)]) == 0
    assert "transact_time distintos=2" in capsys.readouterr().out


def test_empates_sin_directorios_theta_falla(tmp_path, capsys):
    (l1, _l2, _, _events), shared = with_ties(tmp_path)
    probe = load("l3_probe_ties")
    code = probe.main(
        ["--at", iso(shared), "--l1-root", str(l1), "--l2-root", str(tmp_path / "otra")]
    )
    assert code == 1
    assert "FALLO sin_thetas" in capsys.readouterr().out


def test_empates_con_un_events_faltante_falla(tmp_path, capsys):
    (l1, l2, _, events), shared = with_ties(tmp_path)
    theta = theta_text(min(events))
    for path in l2.rglob(f"theta={theta}/**/events.parquet"):
        path.unlink()
    probe = load("l3_probe_ties")
    code = probe.main(["--at", iso(shared), "--l1-root", str(l1), "--l2-root", str(l2)])
    assert code == 1
    out = capsys.readouterr().out
    assert (
        f"FALLO events_faltante: θ sin events.parquet del mes (no se midieron): {theta}"
        in out
    )
    # La línea de resumen sigue siendo la última.
    assert out.strip().splitlines()[-1].startswith("sonda empates:")
