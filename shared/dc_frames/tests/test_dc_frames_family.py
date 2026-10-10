"""El escritor común de archivos de familia: repite el esqueleto o aborta (TRD-L3 §7.3)."""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dc_frames import (
    FamilySkeletonMismatch,
    FamilyWriter,
    FramesInputError,
    family_path,
    l2_path,
)
from frames_lake import (
    FAMILY_SCHEMA,
    KEY,
    MONTH,
    build_lake,
    family_batches,
    theta_text,
    write_l2,
)
from pyutils import content_hash

THETA = 100000
VERSION = "1.0.0"


@pytest.fixture
def partition(tmp_path):
    """`events.parquet` de L2 (row groups de 3 filas) y las rutas de L3 de su partición."""
    _, l2_root, _, _ = build_lake(tmp_path)
    events = l2_path(l2_root, KEY, theta_text(THETA), MONTH)
    l3 = tmp_path / "l3"
    summaries = family_path(l3, KEY, theta_text(THETA), MONTH, "summaries")
    return (
        events,
        summaries,
        lambda name: family_path(l3, KEY, theta_text(THETA), MONTH, name),
    )


def write(path, skeleton, batches=None, schema=FAMILY_SCHEMA, version=VERSION):
    batches = family_batches(skeleton) if batches is None else batches
    with FamilyWriter(path, schema, version, skeleton) as writer:
        for batch in batches:
            writer.write_row_group(batch)
        return writer.commit()


def group_sizes(path) -> list[int]:
    meta = pq.read_metadata(path)
    return [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]


def keys(path) -> pa.Table:
    return pq.read_table(path, columns=["theta", "confirm_agg_trade_id"])


def test_summaries_repeats_the_skeleton_of_events(partition):
    events, summaries, _ = partition
    digest = write(summaries, events)
    assert group_sizes(summaries) == group_sizes(events)
    assert max(group_sizes(events)) > 1 and len(group_sizes(events)) > 10
    assert keys(summaries).equals(keys(events))
    assert digest == content_hash(pq.read_table(summaries))


def test_another_family_repeats_the_skeleton_of_summaries(partition):
    events, summaries, family = partition
    write(summaries, events)
    digest = write(family("other"), summaries)
    assert group_sizes(family("other")) == group_sizes(summaries)
    assert keys(family("other")).equals(keys(summaries))
    assert digest == content_hash(pq.read_table(family("other")))


def test_physical_properties(partition):
    events, summaries, _ = partition
    write(summaries, events, version="2.3.4")
    meta = pq.read_metadata(summaries)
    assert meta.metadata[b"state_version"] == b"2.3.4"
    for i in range(meta.num_row_groups):
        for j in range(meta.num_columns):
            column = meta.row_group(i).column(j)
            assert column.compression == "ZSTD"
            assert column.statistics is not None and column.statistics.has_min_max
            # Sin diccionario: ni página de diccionario ni codificación de diccionario.
            assert not column.has_dictionary_page
            assert not {"PLAIN_DICTIONARY", "RLE_DICTIONARY"} & set(column.encodings)
    sorting = meta.row_group(0).sorting_columns
    assert [(s.column_index, s.descending) for s in sorting] == [(1, False)]
    assert pq.read_schema(summaries).equals(FAMILY_SCHEMA)


def test_the_content_hash_is_stable_across_writes(partition):
    events, summaries, family = partition
    assert write(summaries, events) == write(family("again"), events)
    assert write(summaries, events) == write(summaries, events)


@pytest.fixture
def mismatches(partition):
    events, _, _ = partition
    batches = family_batches(events)

    def with_keys(index, **columns):
        batch = batches[index]
        data = {n: batch.column(n) for n in batch.schema.names} | columns
        return pa.RecordBatch.from_pydict(data, schema=FAMILY_SCHEMA)

    swapped = batches[1].take(pa.array([1, 0, 2]))
    ids = batches[2].column("confirm_agg_trade_id").to_pylist()
    first = batches[0]
    return {
        "una fila menos": batches[:-1]
        + [batches[-1].slice(0, batches[-1].num_rows - 1)],
        "row groups fundidos": [
            pa.Table.from_batches(batches[:2]).combine_chunks().to_batches()[0],
            *batches[2:],
        ],
        "un row group de más": [*batches, first],
        "un row group de menos": batches[:-1],
        "ningún row group": [],
        "otro id": batches[:2]
        + [with_keys(2, confirm_agg_trade_id=pa.array([ids[0], ids[1], ids[2] + 1]))]
        + batches[3:],
        "filas desordenadas": [batches[0], swapped, *batches[2:]],
    }


MISMATCH_CASES = [
    "una fila menos",
    "row groups fundidos",
    "un row group de más",
    "un row group de menos",
    "ningún row group",
    "otro id",
    "filas desordenadas",
]


@pytest.mark.parametrize("case", MISMATCH_CASES)
def test_a_broken_skeleton_aborts_and_publishes_nothing(partition, mismatches, case):
    events, summaries, _ = partition
    Path(summaries).parent.mkdir(parents=True)
    Path(summaries).write_bytes(b"anterior")
    with pytest.raises(FamilySkeletonMismatch, match="family_skeleton_mismatch"):
        write(summaries, events, mismatches[case])
    # El archivo anterior queda intacto y no sobra ningún temporal.
    assert Path(summaries).read_bytes() == b"anterior"
    assert os.listdir(Path(summaries).parent) == ["summaries.parquet"]


def test_a_different_theta_aborts(partition):
    events, summaries, _ = partition
    batches = family_batches(events)
    first = batches[0]
    wrong = pa.array(
        [first.column("theta")[0].as_py() * 2] * first.num_rows, pa.decimal128(9, 8)
    )
    bad = pa.RecordBatch.from_pydict(
        {n: first.column(n) for n in first.schema.names} | {"theta": wrong},
        schema=FAMILY_SCHEMA,
    )
    with pytest.raises(FamilySkeletonMismatch, match="theta"):
        write(summaries, events, [bad, *batches[1:]])
    assert not Path(summaries).exists()


def test_the_error_says_where_the_skeleton_breaks(partition, mismatches):
    events, summaries, _ = partition
    with pytest.raises(FamilySkeletonMismatch, match=r"row group 2, fila 8"):
        write(summaries, events, mismatches["otro id"])
    with pytest.raises(
        FamilySkeletonMismatch, match="tiene 1 filas y el del esqueleto 2"
    ):
        write(summaries, events, mismatches["una fila menos"])


def test_a_family_without_summaries_is_not_written(partition):
    _, summaries, family = partition
    with pytest.raises(FramesInputError, match="falta el esqueleto"):
        FamilyWriter(family("other"), FAMILY_SCHEMA, VERSION, summaries)
    assert not Path(family("other")).exists()


def test_a_partition_without_events_gives_an_empty_file(tmp_path):
    l2_root = tmp_path / "l2"
    events = Path(write_l2(l2_root, THETA, [], 3))
    summaries = family_path(tmp_path / "l3", KEY, theta_text(THETA), MONTH, "summaries")
    digest = write(summaries, events)
    table = pq.read_table(summaries)
    assert table.num_rows == 0 and table.schema.equals(FAMILY_SCHEMA)
    assert digest == content_hash(table)


def test_arguments_are_validated(partition):
    events, summaries, _ = partition
    with pytest.raises(ValueError, match="semver"):
        FamilyWriter(summaries, FAMILY_SCHEMA, "1.0", events)
    with pytest.raises(ValueError, match="clave 'theta'"):
        FamilyWriter(summaries, FAMILY_SCHEMA.remove(0), VERSION, events)
    nullable = FAMILY_SCHEMA.set(1, pa.field("confirm_agg_trade_id", pa.int64()))
    with pytest.raises(ValueError, match="nulable"):
        FamilyWriter(summaries, nullable, VERSION, events)
    wrong = FAMILY_SCHEMA.set(
        1, pa.field("confirm_agg_trade_id", pa.int32(), nullable=False)
    )
    with pytest.raises(ValueError, match="int64"):
        FamilyWriter(summaries, wrong, VERSION, events)
    # Un lote con otro esquema lo rechaza el escritor de Parquet.
    with (
        FamilyWriter(summaries, FAMILY_SCHEMA, VERSION, events) as writer,
        pytest.raises(ValueError, match="esquema"),
    ):
        writer.write_row_group(family_batches(events)[0].select(["theta"]))
