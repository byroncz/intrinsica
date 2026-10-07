"""GCS falso, en memoria y por HTTP, para medir qué pide el `GcsFileSystem` de Arrow.

Solo lo que usan las pruebas: subida resumible, lectura, metadatos, listado, borrado
y copia. Registra cada petición y niega con 403 la consulta a los objetos de `denied`
(el padre `tiles` de la raíz, fuera del prefijo en que la cuenta tiene permiso).

Como GCS real, transcodifica al leer: un objeto con `Content-Encoding: gzip` se entrega
descomprimido si la petición no trae `Accept-Encoding: gzip`, y el `GcsFileSystem` de
Arrow no lo trae (ITSC-320). El tamaño del metadato sigue siendo el almacenado.
"""

import gzip
import json
import re
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self
from urllib.parse import parse_qs, unquote, urlparse

import pyarrow.fs as pafs

BUCKET = "bucket"


class FakeGcs:
    def __init__(self, denied: tuple[str, ...] = ()) -> None:
        self.objects: dict[str, tuple[bytes, dict]] = {}
        self.requests: list[tuple[str, str]] = []
        self.denied = denied
        self._sessions: dict[str, dict] = {}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                fake._get(self)

            def do_POST(self) -> None:
                fake._post(self)

            def do_PUT(self) -> None:
                fake._put(self)

            def do_DELETE(self) -> None:
                fake._delete(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def filesystem(self) -> pafs.GcsFileSystem:
        return pafs.GcsFileSystem(
            anonymous=True,
            scheme="http",
            endpoint_override=f"127.0.0.1:{self._server.server_port}",
            retry_time_limit=timedelta(seconds=1),
        )

    def asked_for(self, name: str) -> bool:
        """Si alguien consultó el objeto `name` (su metadato o su contenido)."""
        return any(
            re.fullmatch(rf"/storage/v1/b/{BUCKET}/o/{re.escape(name)}", path)
            for _, path in self.requests
        )

    # --- Manejadores -------------------------------------------------------

    def _reply(self, h, code: int, body: bytes = b"{}", headers=()) -> None:
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        for key, value in headers:
            h.send_header(key, value)
        h.end_headers()
        h.wfile.write(body)

    def _body(self, h) -> bytes:
        return h.rfile.read(int(h.headers.get("Content-Length", 0)))

    def _resource(self, name: str) -> bytes:
        data, meta = self.objects[name]
        doc = {
            "name": name,
            "bucket": BUCKET,
            "size": str(len(data)),
            "generation": "1",
            "metageneration": "1",
            "contentType": meta.get("contentType", "application/octet-stream"),
            "contentEncoding": meta.get("contentEncoding", ""),
        }
        return json.dumps(doc).encode()

    def _get(self, h) -> None:
        url = urlparse(h.path)
        self.requests.append(("GET", h.path.split("&pageToken")[0]))
        query = parse_qs(url.query)
        if url.path == f"/storage/v1/b/{BUCKET}/o":
            prefix = query.get("prefix", [""])[0]
            items = [
                json.loads(self._resource(n))
                for n in sorted(self.objects)
                if n.startswith(prefix)
            ]
            self._reply(h, 200, json.dumps({"items": items}).encode())
            return
        name = unquote(url.path.removeprefix(f"/storage/v1/b/{BUCKET}/o/"))
        if name in self.denied:
            self._reply(h, 403, b'{"error":{"code":403,"message":"denegado"}}')
        elif name not in self.objects:
            self._reply(h, 404, b'{"error":{"code":404,"message":"no existe"}}')
        elif query.get("alt") == ["media"]:
            data, meta = self.objects[name]
            if meta.get("contentEncoding") == "gzip" and (
                "gzip" not in h.headers.get("Accept-Encoding", "")
            ):
                data = gzip.decompress(data)
            span = re.fullmatch(r"bytes=(\d+)-(\d*)", h.headers.get("Range", ""))
            if span:
                start = int(span[1])
                end = int(span[2]) + 1 if span[2] else len(data)
                self._reply(h, 206, data[start:end])
            else:
                self._reply(h, 200, data)
        else:
            self._reply(h, 200, self._resource(name))

    def _post(self, h) -> None:
        url = urlparse(h.path)
        self.requests.append(("POST", h.path))
        query = parse_qs(url.query)
        body = self._body(h)
        if url.path == f"/upload/storage/v1/b/{BUCKET}/o":
            session = str(len(self._sessions))
            meta = json.loads(body or b"{}")
            self._sessions[session] = {
                "name": query["name"][0] if "name" in query else meta["name"],
                "meta": meta,
                "data": b"",
            }
            location = f"http://{h.headers['Host']}/upload/session/{session}"
            self._reply(h, 200, headers=[("Location", location)])
            return
        copy = re.fullmatch(
            rf"/storage/v1/b/{BUCKET}/o/(.+)/rewriteTo/b/{BUCKET}/o/(.+)", url.path
        )
        if copy:
            src, dest = unquote(copy[1]), unquote(copy[2])
            if src not in self.objects:
                self._reply(h, 404, b'{"error":{"code":404,"message":"no existe"}}')
                return
            self.objects[dest] = self.objects[src]
            doc = {"done": True, "resource": json.loads(self._resource(dest))}
            self._reply(h, 200, json.dumps(doc).encode())
            return
        self._reply(h, 400, b'{"error":{"code":400,"message":"no soportado"}}')

    def _put(self, h) -> None:
        self.requests.append(("PUT", urlparse(h.path).path))
        session = self._sessions[urlparse(h.path).path.rsplit("/", 1)[1]]
        session["data"] += self._body(h)
        span = re.fullmatch(
            r"bytes (?:\d+-\d+|\*)/(\d+|\*)", h.headers["Content-Range"]
        )
        if span[1] == "*":
            end = len(session["data"]) - 1
            self._reply(h, 308, headers=[("Range", f"bytes=0-{end}")])
            return
        meta = {
            "contentType": session["meta"].get("contentType"),
            "contentEncoding": session["meta"].get("contentEncoding"),
        }
        self.objects[session["name"]] = (
            session["data"],
            {k: v for k, v in meta.items() if v},
        )
        self._reply(h, 200, self._resource(session["name"]))

    def _delete(self, h) -> None:
        self.requests.append(("DELETE", h.path))
        name = unquote(urlparse(h.path).path.removeprefix(f"/storage/v1/b/{BUCKET}/o/"))
        if self.objects.pop(name, None) is None:
            self._reply(h, 404, b'{"error":{"code":404,"message":"no existe"}}')
        else:
            self._reply(h, 204, b"")
