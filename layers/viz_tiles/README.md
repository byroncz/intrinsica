# viz_tiles

Capa viz: reduce un día de ticks de L1 y de eventos de L2 a **tiles** (arreglos
binarios planos por nivel de zoom) que el tablero descarga y dibuja sin cómputo
en caliente. Diseño en [`docs/TRD/viz.md`](../../docs/TRD/viz.md); el contrato
de los archivos, en [`docs/data-contracts.md`](../../docs/data-contracts.md)
("Tiles de viz"). Una prueba rompe el CI si ese contrato se desvía del código.

La imagen corre `python -m viz_tiles --mode tiles`: lee un mes de L1 y los
eventos de L2 (en disco o en `gs://`), construye los tiles de cada día pedido y
los escribe. La página HTML por día se agrega sobre este modo en una card
posterior.

## Uso en local

```bash
export VIZ_LANDING_ROOT=$PWD/sandbox.local/landing   # raíz de L1
export VIZ_EVENTS_ROOT=$PWD/sandbox.local/events     # raíz de L2
export VIZ_TILES_ROOT=$PWD/sandbox.local/tiles       # salida
export VIZ_DQ_ROOT=$PWD/sandbox.local/dq             # hallazgos

uv run python -m viz_tiles --mode tiles --day 2017-08-18      # un día
uv run python -m viz_tiles --mode tiles --from 2017-08        # un mes
uv run python -m viz_tiles --mode tiles --from 2017-01 --to 2017-08
uv run python -m viz_tiles --mode tiles                       # mes anterior
uv run python -m viz_tiles --mode tiles --day 2017-08-18 --force
```

Las cuatro variables son obligatorias (raíz local o `gs://`); si falta una, o un
argumento es inválido, el proceso sale con código 2. `--mode` solo acepta
`tiles`.

| Argumento | Efecto |
|---|---|
| `--day YYYY-MM-DD` | Un solo día. No se combina con `--from` ni `--to`. |
| `--from YYYY-MM` / `--to YYYY-MM` | Todos los días de cada mes del rango; `--to` es `--from` por defecto. |
| (ninguno) | El mes anterior al actual (UTC) y la revisión hacia atrás de los meses con cola provisional (abajo). |
| `--force` | Regenera lo seleccionado aunque el `input_hash` no haya cambiado. |
| `--asset` | Activo; por defecto `BTCUSDT`. Debe tener `price_scale` en `contract.py`. |

Un mes completo pide los días que L1 cubre, del primer al último tick del
archivo (agosto de 2017 arranca el 17). Un día sin ticks dentro de ese rango es un
hueco: deja `input_missing` con `what = "ticks"`.

Códigos de salida: `0` éxito (incluye "todo al día"), `1` algún hallazgo `error`
o una entrada faltante, `2` error de uso. Un rango sigue con el mes siguiente si
uno falla, y termina con 1.

## Cómo decide qué rehacer

El `input_hash` de un día es el SHA-256 de un texto con `tiles_version`, el día y
una línea `ruta<TAB>tamaño<TAB>CRC32C` por archivo de entrada, ordenadas por
ruta (rutas relativas a `VIZ_LANDING_ROOT` o `VIZ_EVENTS_ROOT`). El tamaño y el
CRC32C salen de los metadatos del objeto en GCS, sin descargarlo; en disco se
calculan sobre el archivo. Si el `index.json` del día trae el mismo `input_hash` y
`tiles_version`, el día se salta y no se escribe nada; no se lee ningún tick.

Los archivos de entrada del día son el `consolidated.parquet` del mes y el
`events.parquet` y `carry_over.parquet` de cada θ. Los θ son los que L2 tiene en
el mes; uno sin `events.parquet` o sin `carry_over.parquet` va a
`missing_thetas`, deja `input_missing` y el día se escribe con los demás.

**Cola provisional.** Si un θ tiene un evento pendiente al cierre del mes, sus
ticks tras la confirmación se dibujan con el extremo vigente del carry-over
([TRD-viz §7.6](../../docs/TRD/viz.md#76-de-dónde-salen-los-eventos-de-un-día)).
Mientras la cadena de carry-overs de los meses siguientes siga abierta, ese
extremo es un candidato; cuando L2 cierra el evento, el definitivo sale del
`events.parquet` de ese mes. Desde el día del candidato del propio mes, el
`input_hash` suma los archivos de esa cadena, así que un mes nuevo de L2 rehace
esos días. Se mide con el candidato del carry-over del mes y no con el de la última
carry-over de la cadena: si la primera ya mueve el candidato a un mes posterior,
los tiles del día cambian (pasan a overshoot certero) y con el candidato último el
hash seguiría igual.

**Revisión hacia atrás.** Sin `--day` ni `--from`, tras el mes anterior se revisan
los meses previos: desde `N−1`, si el `index.json` del último día trae algún
`provisional_from_s`, se retrocede por los días del mes mientras los encuentre
provisionales y se pasa al mes anterior. Se detiene en el primer mes cuyo último día
es definitivo para todos los θ. La idempotencia salta lo que no cambió.

## Hallazgos

Con `layer = viz`, `mode = tiles`, `stage = canonical` y el día en `details.day`
([TRD-viz §9.3](../../docs/TRD/viz.md#93-tipos-de-chequeo-check_type)). Se emiten
en una llamada por mes procesado: una línea JSON por hallazgo en el log (la que
resume `run-job.yml`) y las filas en `VIZ_DQ_ROOT`.

| `check_type` | `severity` | Cuándo |
|---|---|---|
| `tiles_summary` | info | Uno por día, construido o saltado: `skipped`, `bytes`, `objects`, `levels`, `input_hash`, `content_hash`, `thetas`, `provisional_thetas`, `provisional_tail` y `missing_thetas`. |
| `input_missing` | error | Sin `consolidated.parquet` (`what = l1`), sin `events.parquet` en el mes (`events`), un θ sin `events.parquet` o `carry_over.parquet` o con la cadena rota (`events`, `carry_over`, con `theta`), o un día sin ticks (`ticks`). Los de mes llevan el primer día pedido en `details.day` y `days`. |
| `price_rounded` | warning | Ticks fuera del tick del activo: se redondean y el día se escribe. |
| `price_unrepresentable` | error | El precio máximo no cabe en `int32`: el día no se escribe. |

Cada día termina con una línea sonda: `sonda: unit=<día> ticks=… wall_s=…
rss_mib=…`.

## Memoria

Una sola pasada por el `consolidated.parquet` del mes, row group a row group,
saltando por las estadísticas de `transact_time` los que no tocan los días por
construir. Un día cierra cuando los ticks pasan al siguiente. Por θ se leen solo
los row groups de `events.parquet` que tocan el día. En RAM: un row group, los
acumuladores del día en curso y los eventos de un θ.

## Imagen

```bash
docker build -f layers/viz_tiles/Dockerfile -t viz_tiles .
layers/viz_tiles/smoke.sh viz_tiles     # el día real 2017-08-18 de los fixtures de dc_core
```

Sin Rust: `viz_tiles` no importa `dc_pyo3` ni las capas L1 y L2; lee sus Parquet por
nombre de columna.

## API del paquete

```python
from viz_tiles.contract import price_scale
from viz_tiles.direction import PendingEvent, direction_tiles
from viz_tiles.reduce import reduce_day
from viz_tiles.write import ThetaTiles, write_day

# `batches`: RecordBatch de L1 (agg_trade_id, price, quantity, transact_time),
# ordenados por transact_time; se consumen uno a uno.
reduction = reduce_day(batches, day, price_scale("BTCUSDT"))

# `events`: filas del θ con el esquema de events.parquet que tocan el día;
# `pending`: PendingEvent.from_carry_over(fila) o None.
directions = direction_tiles(reduction.last_ids, events, pending)

write_day(
    root,
    provider="binance",
    market="spot",
    asset="BTCUSDT",
    day=day,
    reduction=reduction,
    thetas=[ThetaTiles("0.00010000", n_events, None, directions)],
    input_hash=input_hash,
    image_version="0.1.0",
)
```

- `reduce_day` guarda solo acumuladores de 4 096 columnas: la RAM es O(lote),
  no O(ticks). Los niveles gruesos se derivan del más fino (M4 es componible).
- `direction_tile` resuelve el estado con `searchsorted` sobre los ids de los
  eventos, sin recorrer ticks. Compara por `agg_trade_id` (TRD-viz §7.5).
- `write_day` escribe los arreglos y, al final, `index.json` (marca de
  commit); `latest.json` solo avanza. Se lee con `numpy.fromfile`.

## Pruebas

```bash
uv run pytest layers/viz_tiles/tests
```
