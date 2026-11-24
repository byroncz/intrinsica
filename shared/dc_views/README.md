# dc_views

Vistas SQL versionadas de L3 sobre `summaries` unido a `events`, y los cargadores
que las crean en una conexión de DuckDB. Hoy hay dos: `derived`, la familia D, M y N
(`src/dc_views/derived.sql`), y `compare`, generada por regla (`src/dc_views/compare.py`).

## Qué es la vista `derived`

Una fila por evento, sin datos propios: las claves `(theta,
confirm_agg_trade_id)`, las 19 columnas de la familia D y las 4 del evento (M: `tmv`,
`osv`, `duration`; N: `net_duration`), calculadas por fila
con aritmética sobre `summaries.parquet` unido a `events.parquet` de L2 (y a
`path.parquet` de L3, solo para `net_duration`). Fórmula
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
version = load_derived(con, summaries, events, path)  # con net_duration
con.execute("SELECT * FROM derived").fetchall()
```

`summaries`, `events` y `path` aceptan cada uno una de tres formas, que se pueden
mezclar:

| Entrada | Se registra como |
|---|---|
| `pyarrow.Table` | La tabla en memoria. |
| Ruta (`str` o `Path`, local o `gs://`) a un archivo `.parquet` | Un dataset de PyArrow de ese solo archivo. La vista no trae `year` ni `month`. |
| Ruta a un directorio | Un dataset de PyArrow con particionado hive descubierto bajo él: solo los archivos `summaries.parquet`, `events.parquet` o `path.parquet`, según el argumento. Si es `summaries`, la vista trae `year` y `month` como enteros (`INTEGER`) entre las claves y las 23 columnas, sea cual sea la forma de `events` y de `path`; si es `events` o `path`, no las trae. |

Las rutas se resuelven con `pyutils.resolve_fs`, el mismo acceso al lago que
L3 y viz. DuckDB lee el dataset por lotes y no recibe credenciales ni carga
`httpfs`. Los segmentos `provider=`, `market=`, `asset=` y `theta=` de la ruta
se ignoran: `theta` es la columna `DECIMAL(9,8)` del archivo.

`load_derived` se puede llamar otra vez sobre la misma conexión: reemplaza las
tablas registradas y la vista. Sin `path`, registra una tabla vacía y `net_duration`
sale nula; con `path` une por `(theta, confirm_agg_trade_id)` (un `LEFT JOIN`: un evento
sin fila de `path` también da `net_duration` nula).

## Ejemplo: una partición

La vista se evalúa por partición; no cargues el histórico completo en RAM. Un θ
y un mes son dos rutas de archivo:

```python
partition = (
    "provider=binance/market=spot/asset=BTCUSDT/theta=0.00098052/year=2026/month=09"
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
(`atomic_sweep`, `empty_overshoot`, `provider_gap`), `tmv`, `osv` y `duration` nunca son
nulas. `net_duration` es nula sin `path` o sin fila de `path` para el evento. El detalle
por columna está en la tabla de `data-contracts.md`.

## `SILENCE_US` y `net_duration`

`dc_views.SILENCE_US` (10 000 000 µs) es el umbral de silencio de T-09 y T-10: un hueco
cuenta si es estrictamente mayor. Es el mismo valor que `SILENCE_US` de
`l3_dc_descriptors.schema` (una prueba en L3 exige que sean iguales). `derived.sql` lo
repite como literal marcado con `/* SILENCE_US */`, porque la sonda `l3_derived_check.py`
ejecuta el `.sql` sin importar este paquete; una prueba exige que el literal sea el
valor de la constante. Cambiar el umbral exige cambiar los tres sitios y subir
`DERIVED_VERSION`.

## La vista `compare`

Una fila por evento de `summaries` con una columna `cmp_<medida>` `DOUBLE` por cada
medida que existe en las dos fases (`c_X` y `os_X`): `os_X / c_X` si la etiqueta es
`ratio` (nulo con `c_X = 0`) y `os_X - c_X` si es `diff`. No lista las medidas: las
descubre. Regla, etiquetas y errores en
[`docs/data-contracts.md`](../../docs/data-contracts.md#vista-compare).

```python
from dc_views import COMPARE_OPS, COMPARE_VERSION, load_compare, phase_pairs

con = duckdb.connect()
load_compare(con, summaries, events, path=path, fit=fit)  # == COMPARE_VERSION
con.execute("SELECT * FROM compare").fetchall()
```

- `phase_pairs(names)` recibe los nombres de columna de **una** relación y devuelve
  `(paired, single)`: las medidas con `c_X` y `os_X`, y las columnas con solo una.
- `COMPARE_OPS[relación][medida]` es `"ratio"` o `"diff"`. Relaciones: `summaries`,
  `derived`, `path`, `fit` y `spectrum`.
- `load_compare(con, summaries, events, path=None, fit=None, spectrum=None)` crea
  `derived` con `load_derived`, registra `fit` y `spectrum` si se entregan, lee las
  columnas de cada relación con `DESCRIBE` y arma el SQL. Una familia no entregada no
  aporta columnas. Para agregar un indicador con pareja basta etiquetarlo en
  `COMPARE_OPS`.
- Falla con `ValueError`, nombrando la medida, si una pareja no tiene etiqueta, una
  etiqueta no tiene pareja o una medida está en dos relaciones. Una familia entregada
  debe traer todas sus parejas.
- Las columnas salen en el orden de cada relación, no en el de `COMPARE_OPS`.

## Pruebas

```bash
uv run pytest shared/dc_views
```

Corre sin red: el fixture sintético tiene los valores calculados a mano y el
lago de prueba se escribe en `tmp_path`.
