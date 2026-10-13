"""El catálogo de θ (TRD-L2 §7.3, TRD-L3 §11) y el límite de CPU que mide `unit_timing`."""

import pytest
from cli_support import write_catalog
from l3_dc_descriptors.cpu import CpuLimit, cpu_limit, throttled_s
from l3_dc_descriptors.thetas import (
    THETA_MAX,
    THETA_MIN,
    ThetasError,
    load_thetas,
    parse_thetas,
)


def test_the_catalog_is_sorted_and_read_from_a_file(tmp_path):
    path = write_catalog(tmp_path / "t.yaml", [300_000, 100_000])
    assert load_thetas(path) == [100_000, 300_000]


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("- 1\n- 2\n", "debe ser un mapa"),
        ("scale: 100\nthetas: [20000]\n", "`scale` debe ser 100000000"),
        ("scale: 100000000\nthetas: []\n", "lista no vacía"),
        ("scale: 100000000\nthetas: [0.001]\n", "debe ser un entero"),
        ("scale: 100000000\nthetas: [5]\n", "fuera de"),
        (f"scale: 100000000\nthetas: [{THETA_MAX + 1}]\n", "fuera de"),
        ("scale: 100000000\nthetas: [20000, 20000]\n", "repetidos"),
        ("scale: [\n", "no es YAML válido"),
    ],
)
def test_an_invalid_catalog_names_every_problem(text, fragment):
    with pytest.raises(ThetasError) as error:
        parse_thetas(text, "cat.yaml")
    assert error.value.source == "cat.yaml"
    assert any(fragment in problem for problem in error.value.problems)


def test_the_range_bounds_are_valid():
    text = f"scale: 100000000\nthetas: [{THETA_MIN}, {THETA_MAX}]\n"
    assert parse_thetas(text) == [THETA_MIN, THETA_MAX]


def test_a_missing_file_is_a_catalog_error(tmp_path):
    with pytest.raises(ThetasError, match="no se pudo leer"):
        load_thetas(str(tmp_path / "no-existe.yaml"))


def _write(directory, name, text):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text)


def _proc(tmp_path, text):
    proc = tmp_path / "proc_cgroup"
    proc.write_text(text)
    return proc


def test_cgroup_v2_quota_below_the_visible_cores_is_the_limit(tmp_path):
    _write(tmp_path, "cpu.max", "400000 100000\n")
    proc = _proc(tmp_path, "0::/\n")
    assert cpu_limit(tmp_path, proc, visible=6) == CpuLimit(4.0, 6, "cgroup-v2")


def test_no_quota_falls_back_to_the_visible_cores(tmp_path):
    _write(tmp_path, "cpu.max", "max 100000\n")
    proc = _proc(tmp_path, "0::/\n")
    assert cpu_limit(tmp_path, proc, visible=6) == CpuLimit(6.0, 6, "affinity")


def test_the_tightest_quota_up_the_hierarchy_wins(tmp_path):
    _write(tmp_path / "job" / "task", "cpu.max", "max 100000\n")
    _write(tmp_path / "job", "cpu.max", "300000 100000\n")
    proc = _proc(tmp_path, "0::/job/task\n")
    assert cpu_limit(tmp_path, proc, visible=8) == CpuLimit(3.0, 8, "cgroup-v2")


def test_cgroup_v1_quota(tmp_path):
    cpu = tmp_path / "cpu"
    _write(cpu, "cpu.cfs_quota_us", "200000\n")
    _write(cpu, "cpu.cfs_period_us", "100000\n")
    proc = _proc(tmp_path, "3:cpu,cpuacct:/\n")
    assert cpu_limit(tmp_path, proc, visible=6) == CpuLimit(2.0, 6, "cgroup-v1")


def test_throttled_time_comes_from_cpu_stat(tmp_path):
    _write(tmp_path, "cpu.stat", "usage_usec 5\nthrottled_usec 2500000\n")
    proc = _proc(tmp_path, "0::/\n")
    assert throttled_s(tmp_path, proc) == 2.5


def test_throttled_time_is_unknown_without_cpu_stat(tmp_path):
    assert throttled_s(tmp_path, tmp_path / "no-existe") is None
