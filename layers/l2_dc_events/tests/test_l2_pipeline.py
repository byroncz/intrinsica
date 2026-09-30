import csv

import dc_pyo3
import pyarrow as pa
import pytest
from conftest import FIXTURES, price_int
from l2_dc_events.landing import LandingError, open_consolidated, read_batches, ticks_of
from l2_dc_events.pipeline import FEED_TICKS, RunContext, Unit, process_unit

# Por θ (`theta` = θ × 10⁸), el `agg_trade_id` de la confirmación con que abre
# cada ventana de discrepancia contra la v0. Es `VENTANAS` de
# shared/dc_core/tests/equivalence_v0.rs: las divergencias declaradas de
# ADR-L2-04 (instante de confirmación atómico). Con 2 % no hay ninguna.
VENTANAS = {
    100_000: [
        3259,
        3400,
        4302,
        4858,
        5299,
        5634,
        5922,
        5973,
        6415,
        6564,
        6726,
        7100,
        7160,
        7582,
    ],
    250_000: [3472, 4302, 5299, 5634, 5922, 5973, 6415, 6564, 7100, 7582],
    500_000: [4302, 7100, 7149],
    1_000_000: [3259, 3334, 3397, 4626, 7100, 7149, 7445],
    2_000_000: [],
}
THETAS = list(VENTANAS)
UNIT = Unit(2017, 8)


def _point(row, name):
    return (
        price_int(row[f"{name}_price"]),
        int(row[f"{name}_time"]),
        int(row[f"{name}_agg_trade_id"]),
    )


def v0_events():
    """Eventos de la v0 por θ: `(direction, reference, confirm, extreme | None)`."""
    out = {theta: [] for theta in THETAS}
    with (FIXTURES / "events_v0.csv").open() as f:
        for row in csv.DictReader(f):
            extreme = _point(row, "extreme") if row["extreme_price"] else None
            out[int(row["theta"])].append(
                (
                    int(row["direction"]),
                    _point(row, "reference"),
                    _point(row, "confirm"),
                    extreme,
                )
            )
    return out


def ctx(write_month, tmp_path):
    return RunContext(
        mode="backfill",
        run_id="test",
        image_version="0.1.0+test",
        series_start=(2017, 8),
        landing_root=write_month.landing,
        events_root=tmp_path / "events",
        dq_root=tmp_path / "dq",
    )


def fan_out_month(path, fanout):
    """Lo que hace `process_unit`, pero devolviendo los eventos: la lectura por
    row group, la alimentación lote a lote y el cierre del fin de entrada."""
    events = [[] for _ in fanout.thetas]
    for batch in read_batches(open_consolidated(str(path))):
        ticks = ticks_of(batch)
        for acc, closed in zip(
            events,
            fanout.feed_batch(ticks.prices, ticks.times, ticks.agg_trade_ids),
            strict=True,
        ):
            acc.extend(closed)
        del batch, ticks
    for acc, last in zip(events, fanout.finish(), strict=True):
        if last is not None:
            acc.append(last)
    return events


def as_tuple(event):
    return (event.direction, event.reference, event.confirm, event.extreme)


def rust_events(closed, carry):
    """Los eventos cerrados de un θ más el pendiente, sin extremo (como la v0)."""
    reference, confirm = carry.pending
    return [as_tuple(e) for e in closed] + [(carry.direction, reference, confirm, None)]


def align(mine, v0):
    """Alinea los eventos de las dos implementaciones (`align` de equivalence_v0.rs).

    Devuelve `(tipo, evento)`: `same`, `fields` (mismo evento con campos
    distintos), `mine_only` o `v0_only`.
    """
    a = b = 0
    out = []
    while a < len(mine) or b < len(v0):
        r = mine[a] if a < len(mine) else None
        v = v0[b] if b < len(v0) else None
        if r and v and r[0] == v[0] and r[2][2] == v[2][2]:
            out.append(("same" if r == v else "fields", r))
            a += 1
            b += 1
        elif r and (v is None or r[2][2] < v[2][2]):
            out.append(("mine_only", r))
            a += 1
        else:
            out.append(("v0_only", v))
            b += 1
    return out


def windows(items):
    """`agg_trade_id` de la confirmación con que abre cada racha de no `same`."""
    ids, is_open = [], False
    for kind, event in items:
        if kind == "same":
            is_open = False
        elif not is_open:
            ids.append(event[2][2])
            is_open = True
    return ids


def longest_run(items):
    run = longest = 0
    for kind, _ in items:
        run = 0 if kind == "same" else run + 1
        longest = max(longest, run)
    return longest


def test_events_through_the_reader_equal_the_v0_fixture(fixture_ticks, write_month):
    """Landing → lector por row group → fan-out da los eventos del crate.

    Igual que `eventos_iguales_salvo_las_divergencias_declaradas` de
    `equivalence_v0.rs`: para cada θ, la serie completa coincide con la v0 salvo
    en las ventanas declaradas de ADR-L2-04, que abren en los mismos eventos y
    no pasan de 3. Con θ = 2 % no hay ventanas: es idéntica evento a evento.
    """
    path = write_month(fixture_ticks, row_group_size=1_000)  # 5 row groups
    assert open_consolidated(str(path)).num_row_groups == 5
    fanout = dc_pyo3.FanOut(THETAS)
    events = fan_out_month(path, fanout)
    expected = v0_events()

    carries = fanout.carry_overs()
    for theta, closed, carry in zip(THETAS, events, carries, strict=True):
        items = align(rust_events(closed, carry), expected[theta])
        assert windows(items) == VENTANAS[theta], f"θ={theta}: ventanas"
        assert longest_run(items) <= 3, f"θ={theta}: ventana de más de 3 eventos"

    assert fanout.discarded() == [0] * len(THETAS)


def test_the_pending_event_of_the_carry_over_is_the_v0_one(fixture_ticks, write_month):
    path = write_month(fixture_ticks, row_group_size=1_000)
    fanout = dc_pyo3.FanOut([2_000_000])
    fan_out_month(path, fanout)
    (carry,) = fanout.carry_overs()
    (pending,) = [e for e in v0_events()[2_000_000] if e[3] is None]
    direction, reference, confirm, _ = pending
    assert (carry.direction, carry.pending) == (direction, (reference, confirm))


def test_row_group_size_does_not_change_the_events(fixture_ticks, write_month):
    by_size = []
    for size in (4_735, 1_000, 333):
        path = write_month(fixture_ticks, row_group_size=size)
        by_size.append(fan_out_month(path, dc_pyo3.FanOut(THETAS)))
    assert by_size[0] == by_size[1] == by_size[2]


def test_process_unit_feeds_the_50_thetas_batch_by_batch(
    fixture_ticks, write_month, tmp_path
):
    write_month(fixture_ticks, row_group_size=1_000)
    result = process_unit(UNIT, ctx(write_month, tmp_path))
    assert (result.n_ticks, result.n_row_groups) == (4_735, 5)
    assert len(result.events_per_theta) == 50
    # Más θ, menos eventos: la cadena de 0,01 % ve muchos más que la de 5 %.
    assert result.events_per_theta[0] > result.events_per_theta[-1] > 0


def test_process_unit_counts_the_closed_events_of_a_given_fanout(
    fixture_ticks, write_month, tmp_path
):
    path = write_month(fixture_ticks, row_group_size=1_000)
    expected = [len(e) for e in fan_out_month(path, dc_pyo3.FanOut(THETAS))]
    result = process_unit(
        UNIT, ctx(write_month, tmp_path), fanout=dc_pyo3.FanOut(THETAS)
    )
    assert result.events_per_theta == expected


# Un `EventColumns` vacío: el binding no tiene constructor, pero un fan-out sin
# ticks cierra 0 eventos.
NO_EVENTS = dc_pyo3.FanOut([100_000]).finish_columns()[0]


class SpyFanOut:
    """Un fan-out que anota qué recibe y cuánta memoria de Arrow hay en cada lote."""

    def __init__(self, thetas):
        self.thetas = thetas
        self.calls = []

    def __len__(self):
        return len(self.thetas)

    def feed_batch_columns(self, prices, times, ids):
        self.calls.append((len(times), pa.total_allocated_bytes()))
        return [NO_EVENTS] * len(self.thetas)

    def finish_columns(self):
        return [NO_EVENTS] * len(self.thetas)

    def discarded(self):
        return [0] * len(self.thetas)

    def carry_overs(self):
        point = (100_000_000, 1_000, 1)
        return [
            dc_pyo3.CarryOver(theta, dc_pyo3.STATE_VERSION, 0, point, point)
            for theta in self.thetas
        ]


def test_process_unit_never_holds_more_than_one_row_group(write_month, tmp_path):
    groups, rows = 10, 50_000
    ticks = [(100_000_000 * (1 + i), 1_000 + i, 1 + i) for i in range(groups * rows)]
    write_month(ticks, row_group_size=rows)
    spy = SpyFanOut([100_000, 200_000])
    baseline = pa.total_allocated_bytes()
    result = process_unit(UNIT, ctx(write_month, tmp_path), fanout=spy)

    assert result.n_ticks == groups * rows
    assert sum(n for n, _ in spy.calls) == groups * rows
    assert max(n for n, _ in spy.calls) <= FEED_TICKS
    one_group = rows * (16 + 8 + 8)
    # Igual que la prueba del lector: la memoria de Arrow es la misma en el
    # primer row group que en el último (retener el anterior la subiría un row
    # group por lote) y muy inferior al mes entero.
    used = [in_flight - baseline for _, in_flight in spy.calls]
    assert max(used) - min(used) < one_group / 2
    assert max(used) < groups * one_group / 3


def test_process_unit_fails_clearly_with_only_provisionals(write_month, tmp_path):
    ticks = [(100_000_000, 1_000, 1), (200_000_000, 1_001, 2)]
    write_month(ticks, row_group_size=2, name="provisional-day=05.parquet")
    with pytest.raises(LandingError, match="provisionales"):
        process_unit(UNIT, ctx(write_month, tmp_path))
