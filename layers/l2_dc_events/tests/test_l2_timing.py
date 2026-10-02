from l2_dc_events.timing import Timing


def _timing(**over) -> Timing:
    base = {
        "wall_s": 10.0,
        "read_s": 1.0,
        "decode_s": 2.0,
        "detect_s": 3.0,
        "write_s": 40.0,
        "carry_s": 0.5,
        "wait_s": 3.0,
        "row_groups": 5,
        "bytes_in": 123,
        "cores": 2.0,
        "cores_visible": 6,
        "cores_source": "cgroup-v2",
        "write_workers": 2,
        "cpu_throttled_s": None,
        "detect_cpu_s": 2.5,
        "fanout_threads": 2,
    }
    return Timing(**{**base, **over})


def test_other_is_the_wall_that_the_serial_phases_do_not_explain():
    # decode_s, detect_cpu_s y write_s son acumulados entre hilos: no entran.
    assert _timing().other_s == 2.5


def test_other_never_goes_negative():
    assert _timing(wall_s=1.0).other_s == 0.0


def test_details_carry_every_phase_and_the_cpu_limit():
    details = _timing(cpu_throttled_s=1.23456).details()
    assert details == {
        "wall_s": 10.0,
        "read_s": 1.0,
        "decode_s": 2.0,
        "detect_s": 3.0,
        "detect_cpu_s": 2.5,
        "write_s": 40.0,
        "carry_s": 0.5,
        "wait_s": 3.0,
        "other_s": 2.5,
        "row_groups": 5,
        "bytes_in": 123,
        "cores": 2.0,
        "cores_visible": 6,
        "cores_source": "cgroup-v2",
        "fanout_threads": 2,
        "write_workers": 2,
        "cpu_throttled_s": 1.235,
    }
