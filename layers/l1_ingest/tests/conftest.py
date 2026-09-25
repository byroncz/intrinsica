import functools
import hashlib
import http.server
import io
import threading
import zipfile

import pytest


@pytest.fixture
def http_server(tmp_path):
    """Sirve tmp_path por HTTP en un hilo; devuelve (directorio, base_url)."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(tmp_path)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


HEADER = (
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
    "transact_time,is_buyer_maker,is_best_match\n"
)
# Ids 1,2,2 (duplicado) y 5 (hueco 3-4); los dos últimos tiempos van desordenados.
ROWS = [
    "1,100.10000000,0.50000000,1,1,1709600000000,True,True",
    "2,100.20000000,1.00000000,2,2,1709600000100,False,True",
    "2,100.20000000,1.00000000,2,2,1709600000100,False,True",
    "5,100.30000000,2.00000000,5,5,1709600000300,True,True",
    "4,100.25000000,0.25000000,4,4,1709600000200,False,True",
]


@pytest.fixture
def publish_zip(http_server):
    """Publica un ZIP de aggTrades (header, tiempos en ms) y su .CHECKSUM."""
    root, base = http_server

    def publish(year=2024, month=3, day=None, checksum=None):
        kind, suffix = (
            ("monthly", f"{month:02d}")
            if day is None
            else (
                "daily",
                f"{month:02d}-{day:02d}",
            )
        )
        directory = root / "data/spot" / kind / "aggTrades/BTCUSDT"
        directory.mkdir(parents=True, exist_ok=True)
        name = f"BTCUSDT-aggTrades-{year}-{suffix}"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(f"{name}.csv", HEADER + "\n".join(ROWS) + "\n")
        data = buffer.getvalue()
        (directory / f"{name}.zip").write_bytes(data)
        checksum = checksum or hashlib.sha256(data).hexdigest()
        (directory / f"{name}.zip.CHECKSUM").write_text(f"{checksum}  {name}.zip\n")

    return publish, base
