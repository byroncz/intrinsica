"""Punto de entrada de la imagen: `python -m ops_tools --script gs://... -- args`."""

import sys

from ops_tools.cli import main

sys.exit(main())
