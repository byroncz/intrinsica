from l2_dc_events.cpu import CpuLimit, cpu_limit, throttled_s


def _write(directory, name, text):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text)


def _proc(tmp_path, text):
    proc = tmp_path / "proc_cgroup"
    proc.write_text(text)
    return proc


def test_cgroup_v2_quota_below_the_visible_cores_is_the_limit(tmp_path):
    _write(tmp_path, "cpu.max", "200000 100000\n")
    proc = _proc(tmp_path, "0::/\n")
    assert cpu_limit(tmp_path, proc, visible=6) == CpuLimit(2.0, 6, "cgroup-v2")


def test_a_fractional_quota_is_kept(tmp_path):
    _write(tmp_path, "cpu.max", "50000 100000\n")
    proc = _proc(tmp_path, "0::/\n")
    assert cpu_limit(tmp_path, proc, visible=4).cores == 0.5


def test_no_quota_falls_back_to_the_visible_cores(tmp_path):
    _write(tmp_path, "cpu.max", "max 100000\n")
    proc = _proc(tmp_path, "0::/\n")
    assert cpu_limit(tmp_path, proc, visible=6) == CpuLimit(6.0, 6, "affinity")


def test_a_quota_above_the_visible_cores_does_not_limit(tmp_path):
    _write(tmp_path, "cpu.max", "800000 100000\n")
    proc = _proc(tmp_path, "0::/\n")
    assert cpu_limit(tmp_path, proc, visible=4) == CpuLimit(4.0, 4, "affinity")


def test_without_cgroup_files_it_is_the_affinity(tmp_path):
    proc = tmp_path / "no-existe"
    assert cpu_limit(tmp_path, proc, visible=3) == CpuLimit(3.0, 3, "affinity")


def test_the_tightest_quota_up_the_hierarchy_wins(tmp_path):
    _write(tmp_path / "job" / "task", "cpu.max", "max 100000\n")
    _write(tmp_path / "job", "cpu.max", "300000 100000\n")
    _write(tmp_path, "cpu.max", "max 100000\n")
    proc = _proc(tmp_path, "0::/job/task\n")
    assert cpu_limit(tmp_path, proc, visible=8) == CpuLimit(3.0, 8, "cgroup-v2")


def test_cgroup_v1_quota(tmp_path):
    cpu = tmp_path / "cpu"
    _write(cpu, "cpu.cfs_quota_us", "200000\n")
    _write(cpu, "cpu.cfs_period_us", "100000\n")
    proc = tmp_path / "no-existe"
    assert cpu_limit(tmp_path, proc, visible=6) == CpuLimit(2.0, 6, "cgroup-v1")


def test_cgroup_v1_without_quota_is_unlimited(tmp_path):
    cpu = tmp_path / "cpu"
    _write(cpu, "cpu.cfs_quota_us", "-1\n")
    _write(cpu, "cpu.cfs_period_us", "100000\n")
    proc = tmp_path / "no-existe"
    assert cpu_limit(tmp_path, proc, visible=6).source == "affinity"


def test_throttled_seconds_come_from_cpu_stat(tmp_path):
    _write(
        tmp_path, "cpu.stat", "usage_usec 10\nnr_throttled 4\nthrottled_usec 2500000\n"
    )
    proc = _proc(tmp_path, "0::/\n")
    assert throttled_s(tmp_path, proc) == 2.5


def test_throttled_seconds_are_unknown_without_cpu_stat(tmp_path):
    assert throttled_s(tmp_path, tmp_path / "no-existe") is None
