"""La cola de un objeto de GCS con el cliente de Google, para la verificación de la frontera (ITSC-424).

El acceso de pyarrow a GCS tiene un tope por proceso de ≈ 12 a 16 archivos por segundo (sonda de
ITSC-421): no es CPU ni red, y más lectores no lo mueven. El cliente de `google-cloud-storage`
con 8 lectores midió 201,7 pies por segundo. pyarrow sigue decodificando el pie; aquí solo se
trae.

Este módulo se importa solo cuando la raíz es `gs://` (`FooterReader`), así una corrida local no
necesita el cliente. Cada petición lleva un tiempo límite y a lo más `ATTEMPTS` intentos: sin
ellos, una petición colgada espera los 60 s por defecto del cliente y deja ocioso a su lector
(en la sonda, 64 lectores quedaron ociosos el 92 % del tiempo). Agotados los intentos el error
sale como `OSError`, que la verificación cuenta como «no se pudo leer»: nunca como válido.
"""

import threading

import requests.exceptions
from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions

# Segundos por petición (conexión y lectura) y peticiones por objeto antes de rendirse.
REQUEST_TIMEOUT_S = 10
ATTEMPTS = 3

# Lo que puede pasar una vez y no la siguiente: red caída o lenta, 5xx, 429 y tiempos agotados.
_TRANSIENT = (
    requests.exceptions.RequestException,
    api_exceptions.ServerError,
    api_exceptions.TooManyRequests,
    auth_exceptions.TransportError,
    ConnectionError,
    TimeoutError,
)
# Lo que el cliente dice de un objeto que no se puede leer y que repetir no arregla (403, 416...).
_PERMANENT = (api_exceptions.GoogleAPICallError, auth_exceptions.GoogleAuthError)


class GcsTail:
    """Un solo cliente de GCS por corrida, con un grupo de conexiones del tamaño de los lectores."""

    def __init__(self, connections: int, client=None) -> None:
        """`connections` son los lectores simultáneos; `client` solo lo pasan las pruebas."""
        self._connections = connections
        self._client = client
        self._lock = threading.Lock()

    def _storage(self):
        with self._lock:
            if self._client is None:
                from google.cloud import storage
                from requests.adapters import HTTPAdapter

                client = storage.Client()
                # El grupo por defecto de `requests` es de 10 conexiones: con más lectores,
                # los que sobran esperarían una conexión libre.
                adapter = HTTPAdapter(
                    pool_connections=self._connections, pool_maxsize=self._connections
                )
                client._http.mount("https://", adapter)
                self._client = client
        return self._client

    def read(self, path: str, size: int) -> bytes:
        """Los últimos `size` bytes de `gs://bucket/objeto` (el objeto entero si es menor).

        `FileNotFoundError` si no existe. `OSError` si tras los intentos no se pudo leer.
        """
        bucket, _, name = path.removeprefix("gs://").partition("/")
        blob = self._storage().bucket(bucket).blob(name)
        failure: Exception | None = None
        for _ in range(ATTEMPTS):
            try:
                # `retry=None`: los reintentos son los de este bucle, no los 120 s por defecto.
                # `checksum=None`: una petición de rango no trae el md5 del objeto para verificar.
                return blob.download_as_bytes(
                    start=-size,
                    timeout=REQUEST_TIMEOUT_S,
                    retry=None,
                    checksum=None,
                )
            except api_exceptions.NotFound:
                raise FileNotFoundError(path) from None
            except _TRANSIENT as exc:
                failure = exc
            except _PERMANENT as exc:
                raise OSError(f"{path}: {type(exc).__name__}: {exc}") from None
        # Sin `from`: el traceback anclaría los marcos y con ellos la respuesta parcial.
        raise OSError(
            f"{path}: {ATTEMPTS} intentos fallidos ({type(failure).__name__}: {failure})"
        )
