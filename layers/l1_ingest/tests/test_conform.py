from decimal import Decimal

import pyarrow as pa
import pytest
from l1_ingest.conform import TimestampUnitError, conform
from l1_ingest.schema import OUTPUT_SCHEMA, RAW_SCHEMA

MS = 1_700_000_000_000
US = 1_700_000_000_000_000


def make_table(
    times: list[int],
    prices: list[str] | None = None,
    quantities: list[str] | None = None,
) -> pa.Table:
    n = len(times)
    return pa.table(
        {
            "agg_trade_id": list(range(n)),
            "price": prices or ["42000.10"] * n,
            "quantity": quantities or ["0.5"] * n,
            "first_trade_id": list(range(n)),
            "last_trade_id": list(range(n)),
            "transact_time": times,
            "is_buyer_maker": [True] * n,
            "is_best_match": [False] * n,
        },
        schema=RAW_SCHEMA,
    )


def test_microsegundos_no_se_tocan():
    table, check = conform(make_table([US, US + 5]))

    assert table.schema.equals(OUTPUT_SCHEMA)
    assert table.column("transact_time").to_pylist() == [US, US + 5]
    assert (check.check_type, check.severity, check.status) == (
        "timestamp_unit_corrected",
        "info",
        "pass",
    )


def test_milisegundos_se_multiplican_por_mil():
    table, check = conform(make_table([MS, MS + 7]))

    assert table.column("transact_time").to_pylist() == [MS * 1000, (MS + 7) * 1000]
    assert (check.severity, check.status) == ("info", "corrected")
    assert check.details["unit"] == "ms"


@pytest.mark.parametrize("bad", [0, 999_999_999_999, 10**13, 10**15 - 1, 10**16])
def test_magnitud_invalida_lanza(bad):
    with pytest.raises(TimestampUnitError) as exc:
        conform(make_table([bad, bad]))

    check = exc.value.check
    assert (check.check_type, check.severity, check.status) == (
        "timestamp_unit_corrected",
        "error",
        "fail",
    )
    assert check.details == {"min": bad, "max": bad}


def test_bandas_mezcladas_lanza():
    with pytest.raises(TimestampUnitError) as exc:
        conform(make_table([MS, US]))

    assert exc.value.check.details == {"min": MS, "max": US}


def test_tabla_vacia_lanza():
    with pytest.raises(TimestampUnitError):
        conform(make_table([]))


def test_decimales_limite_sobreviven():
    prices = ["0.00000001", "1234567.12345678", "9999999999.99999999"]
    table, _ = conform(make_table([US] * 3, prices, prices))

    for name in ("price", "quantity"):
        assert table.column(name).to_pylist() == [Decimal(p) for p in prices]


@pytest.mark.parametrize("bad", ["0.123456789", "10000000000.0", "-10000000000"])
def test_decimal_que_no_cabe_lanza(bad):
    with pytest.raises(pa.ArrowInvalid):
        conform(make_table([US], [bad]))
    with pytest.raises(pa.ArrowInvalid):
        conform(make_table([US], quantities=[bad]))
