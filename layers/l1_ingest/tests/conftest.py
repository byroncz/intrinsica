import functools
import http.server
import threading

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
