import pyarrow as pa
import pytest
from dq import Severity, Status
from l1_ingest.integrity import AggIdConflict, dedupe_agg_trade_id, ensure_order
from l1_ingest.schema import RAW_SCHEMA


def make_table(times: list[int], ids: list[int]) -> pa.Table:
    n = len(ids)
    return pa.table(
        {
            "agg_trade_id": ids,
            "price": ["1"] * n,
            "quantity": ["1"] * n,
            "first_trade_id": ids,
            "last_trade_id": ids,
            "transact_time": times,
            "is_buyer_maker": [True] * n,
            "is_best_match": [True] * n,
        },
        schema=RAW_SCHEMA,
    )


def test_tabla_ordenada_no_se_copia():
    table = make_table([1, 2, 2, 3], [1, 2, 3, 4])
    out, check = ensure_order(table)

    assert out is table
    assert check.check_type == "reorder_applied"
    assert (check.severity, check.status) == (Severity.INFO, Status.PASS)


def test_desorden_por_tiempo():
    table = make_table([1, 3, 2, 4], [1, 2, 3, 4])
    out, check = ensure_order(table)

    assert out.column("transact_time").to_pylist() == [1, 2, 3, 4]
    assert out.column("agg_trade_id").to_pylist() == [1, 3, 2, 4]
    assert (check.severity, check.status) == (Severity.WARNING, Status.CORRECTED)
    assert check.metric_value == 1


def test_empate_de_tiempo_con_ids_invertidos():
    table = make_table([5, 5, 6], [2, 1, 3])
    out, check = ensure_order(table)

    assert out.column("agg_trade_id").to_pylist() == [1, 2, 3]
    assert check.status == Status.CORRECTED
    assert check.metric_value == 1


def test_tabla_vacia_y_de_una_fila_estan_ordenadas():
    for table in (make_table([], []), make_table([1], [1])):
        out, check = ensure_order(table)
        assert out is table
        assert check.status == Status.PASS


def test_ids_sanos():
    table = make_table([1, 2, 3], [1, 2, 3])
    out, (gap, dup) = dedupe_agg_trade_id(table)

    assert out is table
    assert (gap.check_type, dup.check_type) == ("aggid_gap", "aggid_duplicate")
    for check in (gap, dup):
        assert (check.severity, check.status) == (Severity.INFO, Status.PASS)
        assert check.metric_value == 0


def test_un_hueco():
    _, (gap, dup) = dedupe_agg_trade_id(make_table([1, 2, 3], [1, 2, 6]))

    assert (gap.severity, gap.status) == (Severity.WARNING, Status.FAIL)
    assert gap.metric_value == 1
    assert gap.details["gaps"] == [[3, 5]]
    assert dup.status == Status.PASS


def test_un_duplicado_identico_se_elimina():
    table = make_table([1, 2, 2, 3], [1, 2, 2, 3])
    out, (gap, dup) = dedupe_agg_trade_id(table)

    assert out.column("agg_trade_id").to_pylist() == [1, 2, 3]
    assert out.column("transact_time").to_pylist() == [1, 2, 3]
    assert gap.status == Status.PASS
    assert (dup.severity, dup.status) == (Severity.WARNING, Status.CORRECTED)
    assert dup.metric_value == 1
    assert dup.details["ids"] == [2]


def test_bloque_duplicado_identico_cuenta_filas_eliminadas():
    # El bloque 3, 4, 5 viene dos veces seguidas: mismo id, tiempo y contenido.
    table = make_table(
        [10, 20, 30, 40, 50, 30, 40, 50, 60], [1, 2, 3, 4, 5, 3, 4, 5, 6]
    )
    out, (gap, dup) = dedupe_agg_trade_id(table)

    assert out.column("agg_trade_id").to_pylist() == [1, 2, 3, 4, 5, 6]
    assert out.num_rows == table.num_rows - 3
    assert dup.metric_value == 3
    assert dup.details["ids"] == [3, 4, 5]
    assert gap.status == Status.PASS


def test_duplicado_con_price_distinto_aborta():
    table = make_table([1, 2, 2, 3], [1, 2, 2, 3])
    prices = table.column("price").to_pylist()
    prices[2] = "2"
    table = table.set_column(1, "price", pa.array(prices, pa.string()))

    with pytest.raises(AggIdConflict) as exc:
        dedupe_agg_trade_id(table)

    check = exc.value.check
    assert check.check_type == "aggid_duplicate"
    assert (check.severity, check.status) == (Severity.ERROR, Status.FAIL)
    assert check.details["agg_trade_id"] == 2
    assert [row["price"] for row in check.details["rows"]] == ["1", "2"]


def test_combinacion_desordenada_no_modifica_la_entrada():
    table = make_table([1, 2, 3, 4, 5, 6], [7, 1, 2, 2, 2, 4])
    # Los tres ids 2 pasan a ser idénticos (mismo tiempo): quedan dos eliminadas.
    table = table.set_column(5, "transact_time", pa.array([1, 2, 3, 3, 3, 6]))
    before = table.to_pydict()
    out, (gap, dup) = dedupe_agg_trade_id(table)

    assert gap.metric_value == 2
    assert gap.details["gaps"] == [[3, 3], [5, 6]]
    assert dup.metric_value == 2
    assert dup.details["ids"] == [2]
    assert table.to_pydict() == before
    assert out.column("agg_trade_id").to_pylist() == [7, 1, 2, 4]


def test_details_limitado_a_10():
    ids = list(range(1, 60, 2))
    _, (gap, _) = dedupe_agg_trade_id(make_table(list(range(len(ids))), ids))

    assert gap.metric_value == len(ids) - 1
    assert len(gap.details["gaps"]) == 10
