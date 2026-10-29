# dc_views

Vistas SQL versionadas de L3 sobre `summaries` unido a `events`, y el cargador
que las crea en una conexión de DuckDB. Hoy hay una: `derived`, la familia D
(`src/dc_views/derived.sql`).

## Qué es la vista `derived`

Una fila por evento, sin datos propios: las claves `(theta,
confirm_agg_trade_id)` y las 19 columnas de la familia D, calculadas por fila
con aritmética sobre `summaries.parquet` unido a `events.parquet` de L2. Fórmula
y regla de nulos de cada columna en
[`docs/data-contracts.md`](../../docs/data-contracts.md#vista-derived).

`DERIVED_VERSION` (semver, `dc_views.DERIVED_VERSION`) es la versión del
`.sql`; coincide con su primera línea (`-- derived_version: X.Y.Z`) y una
prueba lo exige. Cita esta versión al reportar cualquier número que salga de la
vista.

## `load_derived`

```python
import duckdb
from dc_views import DERIVED_VERSION, load_derived

con = duckdb.connect()
version = load_derived(con, summaries, events)  # == DERIVED_VERSION
con.execute("SELECT * FROM derived").fetchall()
```

`summaries` y `events` aceptan cada uno una de tres formas, que se pueden
mezclar:

| Entrada | Se registra como |
|---|---|
| `pyarrow.Table` | La tabla en memoria. |
| Ruta (`str` o `Path`, local o `gs://`) a un archivo `.parquet` | Un dataset de PyArrow de ese solo archivo. La vista no trae `year` ni `month`. |
| Ruta a un directorio | Un dataset de PyArrow con particionado hive descubierto bajo él: solo los archivos `summaries.parquet` o `events.parquet`, según el argumento. La vista trae `year` y `month` como enteros (`INTEGER`) entre las claves y las 19 columnas. |

Las rutas se resuelven con `pyutils.resolve_fs`, el mismo acceso al lago que
L3 y viz. DuckDB lee el dataset por lotes y no recibe credenciales ni carga
`httpfs`. Los segmentos `provider=`, `market=`, `asset=` y `theta=` de la ruta
se ignoran: `theta` es la columna `DECIMAL(9,8)` del archivo.

`load_derived` se puede llamar otra vez sobre la misma conexión: reemplaza las
tablas registradas y la vista.

## Ejemplo: una partición

La vista se evalúa por partición; no cargues el histórico completo en RAM. Un θ
y un mes son dos rutas de archivo:

```python
partition = (
    "provider=binance/market=spot/asset=BTCUSDT/theta=0.00100000/year=2026/month=09"
)
load_derived(
    con,
    f"gs://<bucket-l3>/{partition}/summaries.parquet",
    f"gs://<bucket-l2>/{partition}/events.parquet",
)
```

Con la identidad de una cuenta de servicio con permiso por prefijo (por ejemplo
`ops-script`), pasa siempre rutas de archivo: listar un directorio de GCS da
403.

## Advertencia: una raíz hive une por las dos claves

Con una raíz, la unión de `summaries` y `events` es por `(theta,
confirm_agg_trade_id)`. Un filtro de la consulta (`WHERE month = 10`) recorta
`summaries`, pero **no poda `events`**: DuckDB sigue abriendo los
`events.parquet` de todos los meses y θ de la raíz. Para un mes, pasa la
partición.

## Regla de nulos

Una división por cero da nulo (`NULLIF`), no `±inf`; con overshoot vacío
(`os_ticks = 0`) las columnas `os_*` salen nulas. Las tres banderas
(`atomic_sweep`, `empty_overshoot`, `provider_gap`) nunca son nulas. El detalle
por columna está en la tabla de `data-contracts.md`.

## Pruebas

```bash
uv run pytest shared/dc_views
```

Corre sin red: el fixture sintético tiene los valores calculados a mano y el
lago de prueba se escribe en `tmp_path`.
