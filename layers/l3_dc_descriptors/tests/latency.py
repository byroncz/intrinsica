"""Un lago de pruebas con la latencia de GCS: un retardo fijo por apertura (ITSC-369).

En el lago de pruebas no hay red y la espera en serie de la compuerta, la apertura de los
`events.parquet` y la publicación no se ve sola: 100 aperturas cuestan microsegundos. Aquí el
sistema de archivos local se envuelve en uno que duerme `delay_s` antes de abrir un archivo
(para leer o para escribir), que es lo que GCS cobra por cada objeto: un viaje. `slow_commit`
hace lo mismo con la publicación, que en local es un `os.replace` y en GCS un `move` (copia
más borrado).

`install` cambia `resolve_fs` en los módulos que abren archivos con él; el retardo no alcanza a
`consolidated.parquet` de L1, cuyo pie y cuyas lecturas son otro tramo de la unidad.
"""

import threading
import time
from pathlib import Path

import pyarrow.fs as pafs
from dc_frames import lake as frames_lake
from l3_dc_descriptors import l2
from pyutils import lineage
from pyutils import parquet as pyutils_parquet


class Counter:
    """Cuántas aperturas pasaron por el retardo, y cuántas a la vez como máximo."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.opens = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.opens += 1
            self.peak = max(self.peak, self._active)

    def leave(self) -> None:
        with self._lock:
            self._active -= 1


class _Delayed(pafs.FileSystemHandler):
    """Delega en `LocalFileSystem` y duerme `delay_s` antes de cada apertura."""

    def __init__(self, delay_s: float, counter: Counter) -> None:
        self._fs = pafs.LocalFileSystem()
        self._delay = delay_s
        self._counter = counter

    def _wait(self, path: str) -> None:
        if path.endswith("consolidated.parquet"):
            return
        self._counter.enter()
        try:
            time.sleep(self._delay)
        finally:
            self._counter.leave()

    def __eq__(self, other) -> bool:
        return isinstance(other, _Delayed) and other._delay == self._delay

    def __ne__(self, other) -> bool:
        return not self == other

    def get_type_name(self) -> str:
        return "latency"

    def normalize_path(self, path):
        return path

    def get_file_info(self, paths):
        return self._fs.get_file_info(paths)

    def get_file_info_selector(self, selector):
        return self._fs.get_file_info(selector)

    def create_dir(self, path, recursive):
        self._fs.create_dir(path, recursive=recursive)

    def delete_dir(self, path):
        self._fs.delete_dir(path)

    def delete_dir_contents(self, path, missing_dir_ok):
        self._fs.delete_dir_contents(path, missing_dir_ok=missing_dir_ok)

    def delete_root_dir_contents(self):
        self._fs.delete_dir_contents("/", accept_root_dir=True)

    def delete_file(self, path):
        self._fs.delete_file(path)

    def move(self, src, dest):
        self._fs.move(src, dest)

    def copy_file(self, src, dest):
        self._fs.copy_file(src, dest)

    def open_input_stream(self, path):
        self._wait(path)
        return self._fs.open_input_stream(path)

    def open_input_file(self, path):
        self._wait(path)
        return self._fs.open_input_file(path)

    def open_output_stream(self, path, metadata):
        self._wait(path)
        return self._fs.open_output_stream(path, metadata=metadata)

    def open_append_stream(self, path, metadata):
        self._wait(path)
        return self._fs.open_append_stream(path, metadata=metadata)


def install(monkeypatch, delay_s: float) -> Counter:
    """Cada apertura de un archivo de L2 o de L3 tarda `delay_s` de más. Devuelve su cuenta."""
    counter = Counter()
    fs = pafs.PyFileSystem(_Delayed(delay_s, counter))

    def resolve_fs(root):
        return fs, str(Path(root).resolve())

    for module in (frames_lake, l2, lineage, pyutils_parquet):
        monkeypatch.setattr(module, "resolve_fs", resolve_fs)
    return counter


def slow_commit(monkeypatch, delay_s: float) -> Counter:
    """Publicar un archivo tarda `delay_s` de más, como el `move` de GCS."""
    counter = Counter()
    commit = pyutils_parquet.PartitionWriter.commit

    def slow(self):
        counter.enter()
        try:
            time.sleep(delay_s)
        finally:
            counter.leave()
        return commit(self)

    monkeypatch.setattr(pyutils_parquet.PartitionWriter, "commit", slow)
    return counter
