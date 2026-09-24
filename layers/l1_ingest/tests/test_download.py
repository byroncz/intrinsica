import hashlib
import http.server
import threading

import pytest
from l1_ingest.download import (
    ChecksumError,
    DownloadError,
    L1DownloadError,
    fetch,
    source_url,
)

PAYLOAD = b"zip-bytes" * 100
GOOD = hashlib.sha256(PAYLOAD).hexdigest()


def publish(root, name="BTCUSDT-aggTrades-2024-01.zip", checksum=GOOD):
    (root / name).write_bytes(PAYLOAD)
    (root / f"{name}.CHECKSUM").write_text(f"{checksum}  {name}\n")
    return name


def test_source_url_monthly_and_daily():
    assert source_url("BTCUSDT", 2024, 1) == (
        "https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/"
        "BTCUSDT-aggTrades-2024-01.zip"
    )
    assert source_url("BTCUSDT", 2024, 1, 5, base_url="http://x") == (
        "http://x/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-05.zip"
    )


def test_fetch_valid(http_server):
    root, base = http_server
    name = publish(root)
    result = fetch(f"{base}/{name}", sleep=lambda s: None)
    assert result.data == PAYLOAD
    assert result.sha256 == GOOD
    assert result.file_bytes == len(PAYLOAD)
    assert result.source_url == f"{base}/{name}"
    assert result.downloaded_at.utcoffset().total_seconds() == 0


def test_fetch_recovers_on_second_attempt(http_server):
    root, base = http_server
    name = publish(root, checksum="0" * 64)
    waits = []

    def fix_then_record(seconds):
        waits.append(seconds)
        publish(root)

    result = fetch(f"{base}/{name}", sleep=fix_then_record)
    assert result.sha256 == GOOD
    assert waits == [2.0]


def test_fetch_persistent_checksum_failure(http_server):
    root, base = http_server
    name = publish(root, checksum="0" * 64)
    waits = []
    with pytest.raises(ChecksumError):
        fetch(f"{base}/{name}", sleep=waits.append)
    assert waits == [2.0, 4.0]  # 3 intentos, 2 esperas crecientes


def test_fetch_missing_file_raises_download_error(http_server):
    _, base = http_server
    with pytest.raises(DownloadError) as info:
        fetch(f"{base}/nope.zip", sleep=lambda s: None)
    assert isinstance(info.value, L1DownloadError)


def test_fetch_truncated_body_is_retried():
    class Truncating(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"corto")
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Truncating)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    waits = []
    try:
        with pytest.raises(DownloadError):
            fetch(f"http://127.0.0.1:{server.server_port}/x.zip", sleep=waits.append)
    finally:
        server.shutdown()
        server.server_close()
    assert waits == [2.0, 4.0]
