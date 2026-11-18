"""La cola de un objeto de GCS con el cliente de Google, para la verificación de la frontera (ITSC-424).

El acceso de pyarrow a GCS tiene un tope por proceso de ≈ 12 a 16 archivos por segundo (sonda de
ITSC-421): no es CPU ni red, y más lectores no lo mueven. El cliente de `google-cloud-storage`
con 8 lectores midió 201,7 pies por segundo. pyarrow sigue decodificando el pie; aquí solo se
trae.

Este módulo se importa solo cuando la raíz es `gs://` (`FooterReader`), así una corrida local no
necesita el cliente. Cada petición lleva un tiempo límite y a lo más `ATTEMPTS` intentos, con
espera exponencial entre ellos (`BACKOFF_S`, el doble cada vez): sin tiempo límite, una petición
colgada espera los 60 s por defecto del cliente y deja ocioso a su lector (en la sonda, 64
lectores quedaron ociosos el 92 % del tiempo), y sin espera, una ráfaga de 429 o 503 de un
segundo agota los intentos de golpe.

«No pude leer» no es «inválido»: agotados los intentos, o ante un error permanente (403...), el
error sale como `FooterUnreadable` y aborta la verificación; no se convierte en un mes inválido,
porque haría retroceder una frontera por una falla del transporte y no del archivo. Un 404 sale
como `FileNotFoundError` (el archivo falta) y un rango imposible (416: objeto vacío) como
`OSError` (el archivo no es un Parquet), igual que antes.
"""

import threading
import time
from collections.abc import Callable

import requests.exceptions
from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions

from l3_dc_descriptors.frontier import FooterUnreadable

# Segundos por petición (conexión y lectura) y peticiones por objeto antes de rendirse.
REQUEST_TIMEOUT_S = 10
ATTEMPTS = 5
# Espera antes del segundo intento; cada siguiente es el doble (0,5, 1, 2 y 4 s con 5 intentos).
BACKOFF_S = 0.5

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

    def __init__(
        self,
        connections: int,
        client=None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """`connections` son los lectores simultáneos.

        `client` y `sleep` (el reloj de la espera entre intentos) solo los pasan las pruebas.
        """
        self._connections = connections
        self._client = client
        self._sleep = sleep
        self._lock = threading.Lock()

    def _storage(self):
        with self._lock:
            if self._client is None:
                from google.cloud import storage
                from requests.adapters import HTTPAdapter

                # Sin mTLS: `mount` reemplaza el adaptador que el cliente monta con
                # `GOOGLE_API_USE_CLIENT_CERTIFICATE=true` y se perdería el certificado de
                # cliente. Cloud Run no lo usa (la identidad es la cuenta de servicio); si algún
                # día se usa, montar el adaptador solo si `not client._http.is_mtls`.
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

        `FileNotFoundError` si no existe; `OSError` si el rango no existe (objeto vacío);
        `FooterUnreadable` si el transporte falló (intentos agotados o error permanente).
        """
        bucket, _, name = path.removeprefix("gs://").partition("/")
        blob = self._storage().bucket(bucket).blob(name)
        failure: Exception | None = None
        for attempt in range(ATTEMPTS):
            if attempt:
                self._sleep(BACKOFF_S * 2 ** (attempt - 1))
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
            except api_exceptions.RequestRangeNotSatisfiable as exc:
                # Un objeto vacío no tiene cola que pedir: es el archivo, no el transporte.
                raise OSError(f"{path}: {type(exc).__name__}: {exc}") from None
            except _TRANSIENT as exc:
                failure = exc
            except _PERMANENT as exc:
                raise FooterUnreadable(path, f"{type(exc).__name__}: {exc}") from None
        # Sin `from`: el traceback anclaría los marcos y con ellos la respuesta parcial.
        raise FooterUnreadable(
            path,
            f"{ATTEMPTS} intentos fallidos ({type(failure).__name__}: {failure})",
        )
