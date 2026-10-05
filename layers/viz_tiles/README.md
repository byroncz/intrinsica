# viz_tiles

Capa viz: reduce un día de ticks de L1 y de eventos de L2 a **tiles** (arreglos
binarios planos por nivel de zoom) que el tablero descarga y dibuja sin cómputo
en caliente. Diseño en [`docs/TRD/viz.md`](../../docs/TRD/viz.md); el contrato
de los archivos, en [`docs/data-contracts.md`](../../docs/data-contracts.md)
("Tiles de viz"). Una prueba rompe el CI si ese contrato se desvía del código.

Este paquete es el núcleo, sin CLI, sin Docker y sin GCS: la imagen, los modos
`tiles` y `export` y la lectura de L1 y L2 en la nube vienen en una card
posterior. Por eso `viz_tiles` aún no está en `LAYERS` de `ci.yml`.

## Uso

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
