# dc_frames

Lector de tramas y escritor común de archivos de familia de L3.

El lector entrega los ticks de cada evento DC, por fase, recortando L1 con las
fronteras de L2. No guarda nada: las tramas se leen, no se materializan.
Contrato en [TRD-L3 §7.4](../../docs/TRD/l3.md#74-contrato-del-lector-de-tramas)
y decisión en ADR-L3-06. Es también el oráculo de las pruebas de L3.

El escritor arma los archivos de familia de una partición `theta/year/month` de
L3 y exige que repitan el esqueleto del archivo que los precede
([TRD-L3 §7.3](../../docs/TRD/l3.md#73-directorio-de-la-capa-y-esqueleto-común)).
Vive aquí y no en `pyutils` porque reutiliza la lectura de Parquet y los
errores del lector, y porque `pyutils` no sabe nada de L3.

## Qué contiene

- `read_frames(thetas, l1_root, l2_root, start, end, provider="binance",
  market="spot", asset="BTCUSDT", chunks=False, max_event_ticks=None)`:
  iterador de `EventFrames`, uno por evento cerrado de los meses `start`..`end`
  (tuplas `(año, mes)`, inclusivas, de las particiones de L2). `thetas` es un θ
  (`Decimal` o `str`) o varios; con varios, cada row group de L1 se decodifica
  una sola vez. Dentro de un θ salen por `confirm_agg_trade_id`; entre θ no hay
  orden.
- `frames_of(event, l1_root, ..., month=None)`: las tramas de un evento cuya
  fila de L2 ya se tiene (un `RecordBatch` o `Table` de una fila, o un mapa con
  sus 11 columnas). Decodifica solo los row groups que el evento toca. Con un
  mapa Arrow infiere los tipos (`direction` sale `int64`), así que `event` ya no
  es la fila de L2 sin transformar. `month` es la partición de L2 que cierra el
  evento (para `ticks_before_month`); la fila no la trae y por defecto es el mes
  de `extreme_time`. Es siempre modo por evento y sin presupuesto.
- `EventFrames(theta, event, confirmation, overshoot, ticks_before_month,
  over_budget)`: `event` es la fila de L2 sin transformar; `confirmation` son
  los ticks de `(R, C]` y `overshoot` los de `(C, E]`, vacío si `E = C`.
- `Frame(transact_time, price, quantity, is_buyer_maker)`: cuatro arrays de
  Arrow con los tipos de L1. No trae `agg_trade_id` ni `is_best_match`.
- `FamilyWriter(path, schema, state_version, skeleton_path)`,
  `family_path(root, key, theta, month, family)` y `l2_path(root, key, theta,
  month)`: el escritor de familias y las rutas de la partición. Ver
  [Escritor de familias](#escritor-de-familias).
- Errores: `FrameBoundaryError` (una frontera de L2 no es un tick de L1, o el
  tick de confirmación no tiene `transact_time = confirm_time`),
  `FramesInputError` (falta un archivo, el esquema o el orden no cumplen el
  contrato) y `FamilySkeletonMismatch` (el archivo de familia no repite el
  esqueleto).

```python
from dc_frames import read_frames

for event in read_frames("0.00100000", l1_root, l2_root, (2023, 3), (2023, 3)):
    prices = event.confirmation.price  # pa.Array decimal128(18, 8)
```

## Modos de lectura

**Por evento** (por defecto): cada fase es un `Frame` con todos sus ticks. El
pico es un row group de L1, uno de eventos por θ y el evento en curso de cada
θ, es decir O(evento). Úsalo con un θ por pasada: con varios, el pico suma los
eventos abiertos de todos.

**Por chunks** (`chunks=True`): cada fase es un iterador de `Frame`, uno por
row group de L1 con ticks de esa fase, en orden de la serie. Un row group sin
ticks de la fase no entrega trozo y el overshoot vacío es un iterador sin
elementos.

```python
for event in read_frames(thetas, l1_root, l2_root, start, end, chunks=True):
    for chunk in event.confirmation:  # Frame de a lo más un row group
        ...
    for chunk in event.overshoot:
        ...
```

- Cada iterador se recorre una sola vez. Se puede consumir después de pedir el
  siguiente evento: no depende del estado del lector.
- El lector no retiene ningún evento. El último trozo de una fase es un slice
  del row group en curso; los anteriores se releen de L1 al avanzar el
  iterador, de uno en uno, solo para el evento que cruza row groups. Los
  eventos dentro de un row group no releen nada.
- El pico es O(row group) con cualquier número de θ y de eventos: el row group
  del pase, el que se relee y los buffers de decodificación (unos 5 row groups
  de 48 B por tick medidos en la prueba de memoria). En modo por evento el
  mismo evento de 10 row groups pica a más de 20.
- Conservar un trozo ancla el row group del que es slice. Si el consumidor
  guarda `EventFrames` sin recorrerlos, ancla el row group en curso de cada uno.
- No admite `max_event_ticks` (`ValueError`): no tiene presupuesto.

## Presupuesto de ticks (`max_event_ticks`)

Solo en modo por evento. Un evento con `extreme_agg_trade_id -
reference_agg_trade_id > max_event_ticks` se entrega con `confirmation` y
`overshoot` vacías (largo 0, sin anclar ningún row group) y `over_budget=True`.
La medida sale de la fila de L2, una cota superior del número de ticks (los
huecos del proveedor la reducen); un evento justo en el presupuesto se calcula.
El lector no copia ni retiene ninguno de sus ticks, pero sí recorre sus row
groups: valida las fronteras y cuenta `ticks_before_month`. Sin
`max_event_ticks` no hay presupuesto.

## Ticks anteriores al mes (`ticks_before_month`)

Cuántos ticks de `(R, E]` están en archivos de L1 de un mes anterior al de la
partición de L2 que cierra el evento; 0 si el evento nació en ese mes. Se cuenta
mientras se lee, con los ticks reales de L1 (un hueco de ids no suma) y sin
retener nada. Un evento cuyo extremo es anterior al primer tick del mes de su
partición cuenta todos sus ticks. Está disponible en los dos modos, también con
`over_budget`.

## Cómo funciona

- **Pertenencia por valor.** Cada frontera (`reference_`, `confirm_` y
  `extreme_agg_trade_id`) se busca con `np.searchsorted` sobre los ids del row
  group, para todos sus eventos a la vez. Nunca se resta ni se compara por
  tiempo, porque el proveedor deja huecos de ids. Una frontera que no es un tick
  de L1 (un hueco, o L1 y L2 que no cuadran) falla con `FrameBoundaryError`.
- **Slices sin copia.** Un evento que cabe en un row group sale como slices de
  sus buffers. Si el consumidor los conserva, ancla ese row group.
- **Eventos que cruzan (modo por evento).** Si un evento sigue en el row group
  siguiente (o en el mes siguiente), el lector copia solo los ticks que ya leyó
  y suelta el row group. Al entregarlo concatena una columna a la vez.
- **Saltos.** Los row groups de L1 que ningún θ necesita se saltan con las
  estadísticas min/max de `agg_trade_id`; el primer evento puede empezar en
  meses anteriores a `start`, que el lector retrocede a abrir.

## Escritor de familias

`FamilyWriter` escribe un archivo de familia (`<familia>.parquet`) de una
partición, un row group a la vez, y exige que repita el esqueleto del archivo que
lo precede: las mismas filas, en el mismo orden y con los mismos límites de row
group.

| Archivo que se escribe | `skeleton_path` |
|---|---|
| `summaries.parquet` | `events.parquet` de L2 de la misma partición |
| cualquier otro archivo de familia | `summaries.parquet` de la misma partición |

```python
from dc_frames import FamilyWriter, family_path, l2_path

key = ("binance", "spot", "BTCUSDT")
events = l2_path(l2_root, key, "0.00100000", (2023, 3))
summaries = family_path(l3_root, key, "0.00100000", (2023, 3), "summaries")

with FamilyWriter(summaries, SCHEMA, "1.0.0", events) as writer:
    print(writer.row_groups)  # filas de cada row group que se espera
    for batch in batches:  # un RecordBatch por row group, en orden
        writer.write_row_group(batch)
    content_hash = writer.commit()  # publica el archivo
```

- `schema` trae `theta` (decimal) y `confirm_agg_trade_id` (`int64`), ambas no
  nulables, más las columnas de la familia. No se copian otras columnas de L2.
  Un esquema que no cumple, o una `state_version` que no es semver `X.Y.Z`,
  lanza `ValueError`.
- Cada `write_row_group` comprueba, contra el esqueleto, el número de filas del
  row group y que `theta` y `confirm_agg_trade_id` sean los mismos fila a fila;
  además `confirm_agg_trade_id` debe crecer. `commit` comprueba que no falte ni
  sobre ningún row group. Cualquier diferencia lanza `FamilySkeletonMismatch`,
  cuyo `check_type` es `family_skeleton_mismatch`, y no se publica nada.
- Un lote vacío no cuenta como row group. Una partición sin eventos da un
  archivo válido de cero filas.
- Si `skeleton_path` no existe falla con `FramesInputError`: un archivo de
  familia que no sea `summaries.parquet` no se escribe sin el `summaries.parquet`
  de su mes.
- Propiedades físicas: ZSTD-3, estadísticas por columna, `sorting_columns` por
  `confirm_agg_trade_id`, `state_version` en los metadatos del archivo y ninguna
  columna con diccionario (`PartitionWriter(compact_encoding=True)`: enteros en
  `DELTA_BINARY_PACKED`, el resto en `PLAIN`).
- `commit` devuelve el `content_hash` del contenido lógico (`pyutils`); no se
  guarda en el archivo, lo registra quien llama.
- Escritura atómica: va a un temporal del mismo directorio y `commit` lo renombra
  sobre el destino. Si algo falla, o se sale del bloque `with` sin `commit`, el
  archivo anterior queda intacto y no sobra ningún temporal. Úsalo siempre con
  `with`: cierra además el archivo del esqueleto.
- Memoria: un row group de la familia más las dos columnas de claves del
  esqueleto de ese row group.

## Pruebas

```bash
uv run pytest shared/dc_frames
```

Usan `ticks.csv` y `events_v0.csv` de `shared/dc_core/tests/fixtures`, con la
cantidad de cada tick igual a su `agg_trade_id` para leer los ids de una trama
sin que el lector los entregue. `tests/frames_lake.py` arma el lago: uno de un
mes (`build_lake`), uno repartido en dos meses con eventos que abarcan el corte
y otros cuyo extremo es anterior al primer tick del segundo mes
(`build_two_month_lake`) y las filas de una familia mínima sobre cualquier
esqueleto (`family_batches`).

- `test_dc_frames_reader.py`: el modo por evento contra un oráculo en Python
  puro.
- `test_dc_frames_chunks.py`: modo por chunks, `max_event_ticks` y
  `ticks_before_month` contra un conteo independiente.
- `test_dc_frames_family.py`: el escritor, sus propiedades físicas y cada forma
  de romper el esqueleto.
- `test_dc_frames_memory.py`: el pico de memoria, medido en un proceso aparte
  con el pool de Arrow.
