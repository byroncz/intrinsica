# l1_ingest

Capa L1: descarga los ZIP de aggTrades de Binance, los conforma y escribe un
Parquet por día (provisional) o por mes (`consolidated.parquet`). Diseño en
[`docs/TRD/l1.md`](../../docs/TRD/l1.md); operación en la nube en
[`docs/runbooks/operacion-l1.md`](../../docs/runbooks/operacion-l1.md). El
contrato de la salida está en
[`docs/data-contracts.md`](../../docs/data-contracts.md). En local corre con
`uv run python -m l1_ingest --mode backfill --from 2024-03`.

## Unicidad de `agg_trade_id` (ITSC-351)

L1 entrega cada agg trade una sola vez: L2 recorre el mes por posición y L3
decide la pertenencia por valor de id, así que un id repetido rompe a las dos.

- Dos filas con el mismo `agg_trade_id` e idéntico contenido en todas las
  columnas se reducen a una. `aggid_duplicate` queda en `warning` /
  `corrected`, con `metric_value` = filas eliminadas y `details.ids` = hasta 10
  ids.
- Si difieren en alguna columna, L1 no puede saber cuál es la verdadera:
  `aggid_duplicate` en `error` / `fail` con el id y las dos filas, y la unidad
  aborta sin escribir Parquet ni manifiesto (`AggIdConflict`).

### Memoria

- **Ruta por lotes** (`stream.py`, la normal): los repetidos son contiguos
  porque los ids no retroceden, así que cada fila se compara con la anterior
  dentro de la misma pasada que ya calculaba los huecos. Sin duplicados, la
  pasada extra es una suma sobre un arreglo booleano (O(lote), un byte por
  fila) y devuelve el mismo lote, sin copia. Con duplicados añade una copia
  filtrada del lote que vive hasta escribirlo. El borde entre lotes es una sola
  fila en un `dict`, sin buffers de Arrow.
- **Ruta materializada** (datos desordenados): ya es O(mes). Añade el orden
  estable por id, los índices de las filas repetidas y una máscara booleana de
  un byte por fila para filtrar (sin materializar un arreglo int64 de N
  índices); no duplica la tabla salvo por el filtro final.
- Medido sobre un lote de 1 000 000 de filas (61 MiB): `feed` pasa de 5 ms a
  11 ms sin duplicados; con 1 de cada 1 000 repetido son 47 ms y el pico
  transitorio sube unos 40 MiB (la copia filtrada); con 1 de cada 10, 26 ms.
- Un bloque de ids que se repite entero más adelante (a, b, c, …, a, b, c)
  hace retroceder los ids y cae a la ruta materializada, como cualquier otro
  desorden.
