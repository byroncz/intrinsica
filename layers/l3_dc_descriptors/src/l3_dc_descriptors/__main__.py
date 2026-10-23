"""Punto de entrada de la imagen: `python -m l3_dc_descriptors --mode ...`."""

import sys

from l3_dc_descriptors.cli import main

if __name__ == "__main__":  # el hijo de una unidad (`spawn`) importa este módulo
    sys.exit(main())
