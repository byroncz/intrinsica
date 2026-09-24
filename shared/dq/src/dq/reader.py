"""Lectura del lago de hallazgos: estado actual por finding_id (ADR-L1-08).

El lago es append-only: un hallazgo cambia de estado escribiendo un evento
nuevo con el mismo finding_id. El estado actual es el último evento por
detected_at. Requiere el extra `dq[reader]` (DuckDB).
"""

from pathlib import Path

import duckdb
import pyarrow as pa

CURRENT_FINDINGS_SQL = """
select * exclude (rn) from (
    select
        *,
        row_number() over (
            partition by finding_id order by detected_at desc
        ) as rn
    from read_parquet($pattern, hive_partitioning = true)
)
where rn = 1
"""


def current_findings(root: str | Path) -> pa.Table:
    """Devuelve una fila por finding_id: su evento más reciente.

    `root` es una raíz local con particiones `detected_date=YYYY-MM-DD/`.
    """
    pattern = f"{Path(root).resolve()}/detected_date=*/*.parquet"
    with duckdb.connect() as con:
        return con.execute(CURRENT_FINDINGS_SQL, {"pattern": pattern}).to_arrow_table()
