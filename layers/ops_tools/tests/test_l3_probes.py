"""Las sondas de L3 (`l3_probe_reader.py`, `l3_probe_ties.py`) contra el lago de fixtures de dc_frames.

Con roots locales, que es como corren en un equipo; en la nube solo cambian los roots por `gs://`.
"""

import importlib.util
import io
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.fs as pafs
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


def run_probe(*argv: str) -> tuple[int, str]:
    """La sonda en un proceso aparte. Su RSS pico es el suyo (`VmHWM`, no `ru_maxrss`, que en Linux hereda el pico de pytest)."""
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "l3_probe_reader.py"), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, done.stdout


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


def test_un_theta_cuenta_lo_mismo_que_el_lector(lake):
    l1, l2, events = lake
    theta = theta_text(min(events))
    code, out = run_probe("--theta", theta, *args(l1, l2))
    assert code == 0, out
    n, conf, over = oracle(l1, l2, theta)
    row = next(
        line.split() for line in out.splitlines() if line.lstrip().startswith(theta)
    )
    assert row[1:4] == [str(n), str(conf), str(over)]
    assert "veredicto=ok" in out
    assert "ok pared" in out and "ok RSS pico" in out
    # Por defecto consume el lector en modo por chunks, con el umbral de 1 GiB.
    assert "modo=chunks over_budget=0" in out
    assert "≤ 1 GiB (modo chunks) (1024 MiB)" in out


def test_todos_los_theta_decodifican_cada_row_group_una_vez(lake, capsys):
    l1, l2, events = lake
    probe = load("l3_probe_reader")
    # El archivo de fixtures pesa 300 KB y su pie se lee más de una vez: el 1,1 × no aplica.
    assert (
        probe.main(
            [
                "--theta",
                "all",
                *args(l1, l2, "--modo", "evento", "--max-bytes-ratio", "3"),
            ]
        )
        == 0
    )
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


def chunk_oracle(l1, l2, thetas):
    """Eventos y ticks por fase según el lector en modo por chunks, sumando los trozos."""
    events = confirmation = overshoot = 0
    for frames in read_frames(thetas, l1, l2, MONTH, MONTH, chunks=True):
        events += 1
        confirmation += sum(len(frame) for frame in frames.confirmation)
        overshoot += sum(len(frame) for frame in frames.overshoot)
    return events, confirmation, overshoot


def test_el_modo_chunks_mide_el_pico_en_un_subproceso_y_cuenta_como_el_lector(lake):
    l1, l2, events = lake
    thetas = ",".join(theta_text(t) for t in sorted(events))
    # Un subproceso: el RSS pico es el de la sonda y no el de pytest.
    code, out = run_probe(
        "--theta",
        thetas,
        *args(l1, l2, "--modo", "chunks", "--max-bytes-ratio", "1000"),
    )
    assert code == 0, out
    final = out.strip().splitlines()[-1]
    assert "modo=chunks over_budget=0 veredicto=ok" in final
    # El RSS depende de la máquina: no se afirma un valor absoluto, solo que se midió y que
    # no pasa de 4 × el de un solo θ sobre el mismo fixture, medido en otro subproceso igual.
    rss = int(re.search(r"rss_pico_mib=(\d+)", final)[1])
    assert rss > 0
    assert re.search(r"arrow_pool=\w+ arrow_max_mib=\d+\.\d ", final)
    _, base_out = run_probe(
        "--theta",
        thetas.split(",")[0],
        *args(l1, l2, "--modo", "chunks", "--max-bytes-ratio", "1000"),
    )
    base = int(re.search(r"rss_pico_mib=(\d+)", base_out.strip().splitlines()[-1])[1])
    assert rss <= 4 * base, (rss, base)
    n, conf, over = chunk_oracle(l1, l2, thetas.split(","))
    assert f"eventos={n}" in final
    assert (n, conf, over) == oracle(l1, l2, thetas.split(","))
    totals = [
        line.split()
        for line in out.splitlines()
        if line.lstrip().startswith("0.") and len(line.split()) == 7
    ]
    assert sum(int(row[1]) for row in totals) == n
    assert sum(int(row[2]) for row in totals) == conf
    assert sum(int(row[3]) for row in totals) == over


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="VmHWM es de Linux")
def test_el_rss_de_la_sonda_no_hereda_el_pico_de_quien_la_lanza(lake):
    l1, l2, events = lake
    held = bytearray(512 << 20)
    for offset in range(0, len(held), 4096):
        held[offset] = 1
    # Con `ru_maxrss` el hijo mediría al menos los 512 MiB del padre tras el exec.
    _, out = run_probe("--theta", theta_text(min(events)), *args(l1, l2))
    del held
    rss = int(re.search(r"rss_pico_mib=(\d+)", out.strip().splitlines()[-1])[1])
    assert 0 < rss < 512, out


def test_el_modo_chunks_admite_las_relecturas_de_eventos_largos(lake, capsys):
    l1, l2, _ = lake
    probe = load("l3_probe_reader")
    # row group de 40 ticks: hay eventos de más de dos row groups, que el lector relee.
    code = probe.main(["--theta", "all", *args(l1, l2, "--max-bytes-ratio", "1000")])
    out = capsys.readouterr().out
    assert code == 0, out
    match = re.search(
        r"decodificados más de una vez: (\d+) \(relecturas admitidas[^:]*: (\d+)\)", out
    )
    assert 0 < int(match[1]) <= int(match[2])


def test_el_modo_evento_pasa_el_presupuesto_y_cuenta_los_over_budget(lake):
    l1, l2, events = lake
    theta = theta_text(max(events))
    budget = 300
    expected = [
        f
        for f in read_frames(theta, l1, l2, MONTH, MONTH, max_event_ticks=budget)
        if f.over_budget
    ]
    total = sum(1 for _ in read_frames(theta, l1, l2, MONTH, MONTH))
    assert 0 < len(expected) < total
    code, out = run_probe(
        "--theta",
        theta,
        *args(l1, l2, "--modo", "evento", "--max-event-ticks", str(budget)),
    )
    assert code == 0, out
    final = out.strip().splitlines()[-1]
    assert f"modo=evento over_budget={len(expected)} " in final
    assert f"over_budget)={len(expected)}" in out
    assert f"≤ 1 GiB + presupuesto de {budget} ticks" in out
    row = next(x.split() for x in out.splitlines() if x.lstrip().startswith(theta))
    assert row[-1] == str(len(expected))
    # Los eventos sobre el presupuesto no suman ticks.
    delivered = [
        f
        for f in read_frames(theta, l1, l2, MONTH, MONTH, max_event_ticks=budget)
        if not f.over_budget
    ]
    assert int(row[2]) == sum(len(f.confirmation.price) for f in delivered)


def test_el_umbral_de_rss_del_modo_evento_suma_el_presupuesto(lake, capsys):
    l1, l2, events = lake
    probe = load("l3_probe_reader")
    probe.main(
        [
            "--theta",
            theta_text(min(events)),
            *args(l1, l2, "--modo", "evento", "--max-event-ticks", "20000000"),
        ]
    )
    limit_mib = (1 << 30) / (1 << 20) + 20_000_000 * probe.BYTES_PER_TICK / (1 << 20)
    assert f"({limit_mib:.0f} MiB)" in capsys.readouterr().out


def test_modo_chunks_no_pasa_el_presupuesto_al_lector(lake):
    l1, l2, events = lake
    # `read_frames` rechaza chunks con presupuesto: la sonda no debe combinarlos.
    code, out = run_probe(
        "--theta", theta_text(min(events)), *args(l1, l2, "--max-event-ticks", "5")
    )
    assert code == 0, out
    assert "modo=chunks over_budget=0" in out


class RangedFile(io.BytesIO):
    """Un archivo como el de GCS: `seek(0, 2)` lee desde el final y el rango no existe."""

    def read(self, size=-1):
        if self.tell() >= len(self.getvalue()) and size != 0:
            raise OSError("Request range not satisfiable")
        return super().read(size)

    def seek(self, offset, whence=0):
        if whence == 2:
            raise OSError("Request range not satisfiable")
        return super().seek(offset, whence)


def test_el_envoltorio_resuelve_el_seek_al_final_con_el_tamano_conocido():
    probe = load("l3_probe_reader")
    data = bytes(range(100))
    wrapped = probe._CountedFile(RangedFile(data), "l1", "ruta", len(data))
    assert wrapped.seek(0, 2) == 100
    assert wrapped.tell() == 100
    assert wrapped.seek(-10, 2) == 90
    assert wrapped.read(10) == data[90:]
    assert wrapped.seek(5) == 5
    assert wrapped.read(3) == data[5:8]
    # El archivo falso sí falla si se le pide el final directamente.
    with pytest.raises(OSError, match="Request range"):
        RangedFile(data).seek(0, 2)


class FlatFS:
    """Un bucket: los directorios `theta=` no existen como entradas, solo los objetos."""

    def __init__(self, inner):
        self._inner = inner

    def get_file_info(self, selector):
        if isinstance(selector, pafs.FileSelector) and not selector.recursive:
            return []
        if isinstance(selector, pafs.FileSelector):
            return [
                info
                for info in self._inner.get_file_info(selector)
                if info.type == pafs.FileType.File
            ]
        return self._inner.get_file_info(selector)


def test_la_sonda_lista_los_theta_de_forma_recursiva(lake, monkeypatch):
    _l1, l2, events = lake
    probe = load("l3_probe_reader")
    real = probe.resolve_fs
    monkeypatch.setattr(
        probe, "resolve_fs", lambda root: (FlatFS(real(root)[0]), real(root)[1])
    )
    found = probe.lake_thetas(str(l2), ("binance", "spot", "BTCUSDT"), MONTH, MONTH)
    assert found == [theta_text(t) for t in sorted(events)]


def test_la_sonda_de_theta_all_omite_los_que_no_tienen_todo_el_rango(lake, capsys):
    _l1, l2, events = lake
    probe = load("l3_probe_reader")
    # Un θ con events.parquet solo de otro mes no tiene L2 en el rango.
    theta = theta_text(min(events))
    source = next(l2.rglob(f"theta={theta}/**/events.parquet"))
    other = source.parents[1] / "month=09" / "events.parquet"
    other.parent.mkdir(parents=True)
    other.write_bytes(source.read_bytes())
    source.unlink()
    found = probe.lake_thetas(str(l2), ("binance", "spot", "BTCUSDT"), MONTH, MONTH)
    assert theta not in found and len(found) == len(events) - 1
    assert f"omitidos: {theta}" in capsys.readouterr().out


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
    assert (
        probe.main(
            [
                "--theta",
                "all",
                *args(l1, l2, "--modo", "evento", "--max-bytes-ratio", "3"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith("bytes leídos"))
    earlier = int(line.split("de meses anteriores ")[1].split(")")[0])
    total = int(re.search(r"bytes_l1=(\d+)", out.strip().splitlines()[-1])[1])
    in_range = int(re.search(r"bytes de L1 del rango (\d+)", out)[1])
    assert earlier > 0
    assert in_range == total - earlier


def test_en_modo_chunks_solo_se_restan_las_relecturas_de_archivos_del_rango(
    tmp_path, capsys, monkeypatch
):
    ticks, events = read_ticks(), read_events()
    reference = min(int(rows[0]["reference_agg_trade_id"]) for rows in events.values())
    l1, l2 = tmp_path / "l1", tmp_path / "l2"
    earlier_path = str(
        write_l1(l1, [t for t in ticks if t["id"] <= reference], 5, (2017, 7))
    )
    write_l1(l1, [t for t in ticks if t["id"] > reference], 40)
    for theta, rows in events.items():
        write_l2(l2, theta, rows, 3)
    probe = load("l3_probe_reader")
    read_row_group = probe._ProbeFile.read_row_group

    def read_twice_in_earlier_month(self, i, columns=None, **kwargs):
        # El fixture solo hace releer al lector en el mes del rango: aquí se fuerza una
        # relectura en el mes previo, que el lector abre hacia atrás.
        table = read_row_group(self, i, columns, **kwargs)
        if self._probe_path == earlier_path:
            read_row_group(self, i, columns, **kwargs)
        return table

    monkeypatch.setattr(probe._ProbeFile, "read_row_group", read_twice_in_earlier_month)
    probe.main(
        [
            "--theta",
            "all",
            *args(l1, l2, "--modo", "chunks", "--max-bytes-ratio", "1000"),
        ]
    )
    out = capsys.readouterr().out
    repeated = probe.STATS.l1_repeated_bytes
    in_range_repeated = sum(n for path, n in repeated.items() if path != earlier_path)
    month_read = int(re.search(r"del rango (\d+) =", out)[1])
    counted = int(re.search(r"bytes de L1 del rango (\d+)", out)[1])
    assert repeated[earlier_path] > 0
    assert in_range_repeated > 0
    assert counted == month_read - in_range_repeated


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


def test_sin_contar_io_no_reporta_bytes_medidos(lake):
    l1, l2, events = lake
    code, out = run_probe(
        "--theta", theta_text(min(events)), *args(l1, l2, "--sin-contar-io")
    )
    assert code == 0, out
    final = out.strip().splitlines()[-1]
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


def test_empates_lista_los_theta_de_forma_recursiva(tmp_path):
    (_l1, l2, _, events), _shared = with_ties(tmp_path)
    probe = load("l3_probe_ties")
    root = f"{l2}/provider=binance/market=spot/asset=BTCUSDT"
    fs = FlatFS(pafs.LocalFileSystem())
    assert probe.lake_thetas(fs, root) == [theta_text(t) for t in sorted(events)]
    assert probe.lake_thetas(fs, f"{l2}/otra") == []


def test_empates_el_ms_por_defecto_es_el_de_los_4090_ticks(tmp_path, capsys):
    probe = load("l3_probe_ties")
    code = probe.main(["--l1-root", str(tmp_path), "--l2-root", str(tmp_path)])
    assert code == 1
    # Sin L1 no hay ms que medir, pero el mes pedido es el de 2026-09.
    assert "year=2026/month=09/consolidated.parquet" in capsys.readouterr().out
    assert "2026-09-30T12:40:26.942" in (SCRIPTS / "l3_probe_ties.py").read_text()


def test_empates_sin_ticks_informa_el_ms_mas_poblado_del_segundo(tmp_path, capsys):
    (l1, l2, _, _events), shared = with_ties(tmp_path)
    probe = load("l3_probe_ties")
    # El ms siguiente al de los empates no tiene ticks; el de los empates tiene 4.
    empty = iso(shared - shared % 1000 + 1000)
    code = probe.main(["--at", empty, "--l1-root", str(l1), "--l2-root", str(l2)])
    assert code == 1
    out = capsys.readouterr().out
    assert "FALLO sin_ticks" in out
    assert "el ms más poblado del segundo" in out
    assert f"es {iso(shared).replace('T', ' ')} con 4 ticks" in out


def test_empates_sin_ticks_en_el_segundo_lo_dice(lake, capsys):
    l1, l2, _ = lake
    probe = load("l3_probe_ties")
    probe.main(
        ["--at", "2017-08-18T00:00:00.000", "--l1-root", str(l1), "--l2-root", str(l2)]
    )
    out = capsys.readouterr().out
    assert "FALLO sin_ticks" in out and "ningún tick en el segundo" in out


def test_empates_con_un_events_faltante_falla(tmp_path, capsys):
    (l1, l2, _, events), shared = with_ties(tmp_path)
    theta = theta_text(min(events))
    for path in l2.rglob(f"theta={theta}/**/events.parquet"):
        # El θ sigue en el lago con otro mes, pero sin el del milisegundo.
        other = path.parents[1] / "month=09" / path.name
        other.parent.mkdir(parents=True)
        other.write_bytes(path.read_bytes())
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
