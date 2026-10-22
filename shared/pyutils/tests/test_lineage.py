import json
from datetime import UTC, datetime

import pytest
from pyutils import HASHER_VERSION, lineage

EVENTS = "a" * 64
CARRY = "b" * 64


def test_the_document_has_the_published_keys():
    moment = datetime(2026, 10, 21, 12, 30, 5, 999, tzinfo=UTC)
    assert lineage.document(EVENTS, CARRY, moment) == {
        "events_content_hash": EVENTS,
        "carry_over_content_hash": CARRY,
        "hasher_version": HASHER_VERSION,
        "written_at": "2026-10-21T12:30:05+00:00",
    }


def test_the_document_is_written_now_in_utc_by_default():
    written = datetime.fromisoformat(lineage.document(EVENTS, CARRY)["written_at"])
    assert written.utcoffset().total_seconds() == 0
    assert abs((datetime.now(UTC) - written).total_seconds()) < 60


def test_write_then_read_gives_the_document_back(tmp_path):
    path = str(tmp_path / "p" / "q" / lineage.LINEAGE)
    content = lineage.document(EVENTS, CARRY)
    assert lineage.write(path, content) == path
    assert lineage.read(path) == content
    # Sin temporales: el único archivo es el publicado.
    assert [p.name for p in (tmp_path / "p" / "q").iterdir()] == [lineage.LINEAGE]


def test_write_replaces_the_previous_document(tmp_path):
    path = str(tmp_path / lineage.LINEAGE)
    lineage.write(path, lineage.document(EVENTS, CARRY))
    lineage.write(path, lineage.document(CARRY, EVENTS))
    assert lineage.read(path)["events_content_hash"] == CARRY


def test_read_of_a_missing_object_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        lineage.read(str(tmp_path / lineage.LINEAGE))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"[]",
        b'{"events_content_hash": "a"}',
        json.dumps({**lineage.document(EVENTS, CARRY), "extra": "x"}).encode(),
        json.dumps({**lineage.document(EVENTS, CARRY), "written_at": ""}).encode(),
        json.dumps({**lineage.document(EVENTS, CARRY), "hasher_version": 1}).encode(),
        b"\xff\xfe",
    ],
)
def test_parse_refuses_what_is_not_the_published_format(data):
    with pytest.raises(ValueError):
        lineage.parse(data)


def test_retract_removes_the_object_and_tolerates_its_absence(tmp_path):
    path = str(tmp_path / lineage.LINEAGE)
    lineage.retract(path)
    lineage.write(path, lineage.document(EVENTS, CARRY))
    lineage.retract(path)
    assert not (tmp_path / lineage.LINEAGE).exists()
