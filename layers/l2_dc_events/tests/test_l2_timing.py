import pytest
from l2_dc_events.timing import ThetaTiming, Timing


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
        "open_s": 0.0,
        "backpressure_s": 0.0,
        "drain_s": 0.0,
        "publish_s": 0.0,
        "encode_s": 0.0,
        "close_s": 0.0,
        "move_s": 0.0,
        "carry_write_s": 0.0,
        "theta_timing": [],
    }


def _theta(theta: int, **over) -> ThetaTiming:
    base = {
        "theta": theta,
        "events": 10,
        "open_s": 0.01,
        "encode_s": 1.0,
        "close_s": 0.2,
        "move_s": 0.3,
        "carry_write_s": 0.5,
        "events_done_s": 5.0,
        "done_s": 6.0,
    }
    return ThetaTiming(**{**base, **over})


def test_opening_the_writers_is_main_thread_wall_and_leaves_other():
    assert _timing(open_s=1.0).other_s == 1.5


def test_wait_is_split_in_backpressure_drain_and_publish():
    details = _timing(wait_s=3.0, backpressure_s=1.0, drain_s=0.5, publish_s=1.5)
    assert details.details()["wait_s"] == 3.0
    assert (details.backpressure_s, details.drain_s, details.publish_s) == (1, 0.5, 1.5)


def test_the_write_of_a_theta_adds_its_encode_close_move_and_carry():
    assert _theta(1).write_s == 2.0


def test_the_heaviest_and_the_last_theta_are_found_by_time():
    thetas = (
        _theta(1, encode_s=9.0, events_done_s=7.0),
        _theta(2, events_done_s=9.5),
        _theta(3, events_done_s=8.0),
    )
    timing = _timing(thetas=thetas)
    assert timing.heaviest.theta == 1
    assert timing.last.theta == 2
    assert timing.encode_s == 11.0
    assert (timing.close_s, timing.move_s) == pytest.approx((0.6, 0.9))
    assert timing.carry_write_s == 1.5


def test_no_thetas_have_no_heaviest_nor_last():
    timing = _timing()
    assert (timing.heaviest, timing.last) == (None, None)


def test_theta_timing_details_round_to_milliseconds():
    details = _theta(7, encode_s=1.23456, done_s=6.00049).details()
    assert details["theta"] == 7
    assert (details["encode_s"], details["done_s"]) == (1.235, 6.0)
