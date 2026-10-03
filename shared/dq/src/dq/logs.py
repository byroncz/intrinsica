"""Logging de los jobs: líneas JSON estructuradas en Cloud Run, texto en local."""

import json
import logging
import os
import sys
from collections.abc import Mapping

# Atributo del LogRecord con los campos del hallazgo (lo pone `dq.emit`).
FINDING_ATTR = "finding"


class CloudRunFormatter(logging.Formatter):
    """Una línea JSON por registro, en el formato que Cloud Logging estructura.

    Cloud Run guarda la línea como `jsonPayload` y promueve `severity` a la
    severidad de la entrada. `message` es el texto del registro; en un hallazgo
    es su línea JSON completa, como antes. Los campos del hallazgo van además en
    el primer nivel (incluido `finding_id`) para que una alerta filtre por
    campos y no por texto.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        entry: dict[str, object] = {}
        finding = getattr(record, FINDING_ATTR, None)
        if finding:
            # `severity` del hallazgo ("error") es minúscula y el de Logging no.
            entry.update({k: v for k, v in finding.items() if k != "severity"})
        entry["severity"] = record.levelname
        entry["message"] = message
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(env: Mapping[str, str] | None = None) -> None:
    """Configura el logging raíz de un job a nivel INFO.

    Dentro de Cloud Run (el Job define `CLOUD_RUN_JOB`) escribe JSON a stdout;
    en cualquier otro lado, `<fecha> <mensaje>` como siempre.
    """
    env = os.environ if env is None else env
    if env.get("CLOUD_RUN_JOB"):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CloudRunFormatter())
        logging.basicConfig(level=logging.INFO, handlers=[handler])
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
