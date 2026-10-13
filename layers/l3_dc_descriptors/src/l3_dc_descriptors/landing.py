"""La landing de L1 vista por la CLI: hasta qué mes hay y por qué falta uno (TRD-L3 §7.1).

L3 solo lee el canónico, `consolidated.parquet` (ADR-L3-08). Los row groups los lee el
lector de tramas; aquí solo se listan y se comprueba que el mes exista.
"""

import re

import pyarrow.fs as pafs
from dc_frames.lake import CONSOLIDATED, Month, l1_path
from pyutils import resolve_fs

_CONSOLIDATED = re.compile(
    rf"/year=(\d{{4}})/month=(\d{{2}})/{re.escape(CONSOLIDATED)}$"
)


def last_consolidated(root: str, key: tuple[str, str, str]) -> Month | None:
    """El último mes (año, mes) con `consolidated.parquet` en la landing, o `None`.

    Un solo listado recursivo del activo. Los provisionales no cuentan (ADR-L3-08): un
    mes con solo `provisional-day=DD.parquet` no está cerrado.
    """
    provider, market, asset = key
    prefix = f"{root.rstrip('/')}/provider={provider}/market={market}/asset={asset}"
    fs, resolved = resolve_fs(prefix)
    selector = pafs.FileSelector(resolved, recursive=True, allow_not_found=True)
    months = (
        (int(match[1]), int(match[2]))
        for info in fs.get_file_info(selector)
        if (match := _CONSOLIDATED.search(info.path))
    )
    return max(months, default=None)


def missing_input(
    root: str, key: tuple[str, str, str], month: Month
) -> tuple[str, dict] | None:
    """Por qué falta el consolidado del mes: `(check_type, details)`, o `None` si existe.

    `input_provisional_only` si el mes solo tiene provisionales y `input_missing` si no
    tiene nada (TRD-L3 §9.3).
    """
    path = l1_path(root, key, month)
    fs, resolved = resolve_fs(path)
    if fs.get_file_info(resolved).type != pafs.FileType.NotFound:
        return None
    parent = resolved.rpartition("/")[0]
    selector = pafs.FileSelector(parent, allow_not_found=True)
    provisional = [
        info.base_name
        for info in fs.get_file_info(selector)
        if info.base_name.startswith("provisional-day=")
    ]
    if provisional:
        return "input_provisional_only", {
            "expected_path": path,
            "provisionals": len(provisional),
        }
    return "input_missing", {"expected_path": path}
