# dc_frames

Lector de tramas (por evento, por chunks y por row group) y escritor común de archivos de familia de L3.

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
  market="spot", asset="BTCUSDT", chunks=False, max_event_ticks=None,
  columns=None)`:
  iterador de `EventFrames`, uno por evento cerrado de los meses `start`..`end`
  (tuplas `(año, mes)`, inclusivas, de las particiones de L2). `thetas` es un θ
  (`Decimal` o `str`) o varios; con varios, cada row group de L1 se decodifica
  una sola vez (en modo por chunks, salvo por los eventos que abarcan más de dos
  row groups, ver [Modos de lectura](#modos-de-lectura)). Dentro de un θ salen por `confirm_agg_trade_id`; entre θ no hay
  orden. `columns` proyecta las columnas de `events.parquet` que se decodifican
  (siempre se leen además las tres fronteras y `confirm_time`); sin él, las 11.
- `read_row_groups(thetas, l1_root, l2_root, start, end, provider="binance",
  market="spot", asset="BTCUSDT", columns=None, readers=1)`: la entrega por row
  group (ADR-L3-14). Iterador de `RowGroupFrames`, uno por row group de L1 del pase,
  con los eventos de todos los θ que lo tocan como arreglos de posiciones. `readers`
  es cuántos row groups se leen a la vez. Ver
  [Entrega por row group](#entrega-por-row-group) y
  [Lectura: `pre_buffer`, pool de E/S y `readers`](#lectura-pre_buffer-pool-de-es-y-readers).
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
- `RowGroupFrames(month, before_start, ids, frame, thetas)` y `ThetaEvents(theta,
  events, positions, starts_before, ends_after, ticks_before_month)`: lo que
  entrega `read_row_groups`. `ids` son los `agg_trade_id` de los ticks del row
  group (vista de NumPy sin copia, crecientes), que el motor nativo usa para
  ubicar las fronteras él mismo.
- `Frame(transact_time, price, quantity, is_buyer_maker)`: cuatro arrays de
  Arrow con los tipos de L1. No trae `agg_trade_id` ni `is_best_match`.
- `FamilyWriter(path, schema, state_version, skeleton_path)`,
  `family_path(root, key, theta, month, family)` y `l2_path(root, key, theta,
  month)`: el escritor de familias y las rutas de la partición. Ver
  [Escritor de familias](#escritor-de-familias).
- `dc_frames.lake.EventFile` y `ParquetSource`: archivos abiertos una vez y leídos por tramos
  (ver [Archivos abiertos una vez](#archivos-abiertos-una-vez-y-leídos-por-tramos-itsc-363)).
- Errores: `FrameBoundaryError` (con los atributos `theta`, `field`, `got` y `wanted`;
  una frontera de L2 no es un tick de L1, o el
  tick de confirmación no tiene `transact_time = confirm_time`),
  `FramesInputError` (falta un archivo, el esquema o el orden no cumplen el
  contrato) y `FamilySkeletonMismatch` (el archivo de familia no repite el
  esqueleto; el atributo `detail` dice en qué difiere).

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

- Cada iterador se recorre una sola vez. **Se consume antes de pedir el
  siguiente evento**: consumirlo después funciona, pero ancla los row groups de
  sus trozos hasta entonces, y ese costo es del consumidor.
- El lector no retiene ningún evento. Conserva **una sola copia del row group
  anterior**, compartida por todos los θ, mientras algún evento abierto la
  necesite. Un evento que cruza un borde toma de ahí sus ticks del row group
  anterior (un slice sin copia) y el último trozo de cada fase es un slice del
  row group en curso. Los eventos dentro de un row group no releen nada.
- **Contrato de decodificación.** Con varios θ, cada row group de L1 se
  decodifica una vez, salvo por los eventos que abarcan más de dos row groups:
  de esos se releen de L1, de uno en uno, solo los row groups que sobran (los
  anteriores al row group previo). Su costo lo mide la sonda de TRD-L3 §14.
- El pico es O(row group) con cualquier número de θ y de eventos: el row group
  del pase, el anterior y los buffers de decodificación (unos 5 row groups de
  48 B por tick medidos en la prueba de memoria). Un evento de más de dos row
  groups suma el que relee (unos 6 en la prueba, con un evento de 10). En modo
  por evento el mismo evento pica a más de 20.
- Conservar un trozo ancla el row group del que es slice. Si el consumidor
  guarda `EventFrames` sin recorrerlos, ancla el row group en curso y el
  anterior de cada uno.
- No admite `max_event_ticks` (`ValueError`): no tiene presupuesto.

## Entrega por row group

La usa el resumidor y las familias por chunks (ADR-L3-14). No arma un objeto por
evento: por cada row group de L1 del pase, **una vez para todos los θ**, entrega los
cuatro arrays del row group y, por θ, las posiciones de las fronteras de los eventos
que lo tocan.

```python
from dc_frames import read_row_groups

for delivery in read_row_groups(thetas, l1_root, l2_root, start, end):
    frame = delivery.frame  # cuatro pa.Array, vistas del row group
    for te in delivery.thetas:  # solo los θ con eventos en este row group
        r, c, e = te.positions.T  # arreglos de n posiciones
        # confirmación = ticks [r, c); overshoot = ticks [c, e)
        ...
    del delivery  # antes de pedir el siguiente
```

- **Convención de posiciones.** `positions[i] = (R, C, E)` del evento `i` son
  `searchsorted(agg_trade_id, id, "right")`: cuántos ticks del row group tienen id ≤
  la frontera. La confirmación es `[R, C)` y el overshoot `[C, E)`, medio abiertos, y
  las fases consecutivas de un θ comparten corte (`E_i = R_{i+1}`). Una frontera fuera
  del row group queda en 0 (anterior) o en su largo (posterior); un overshoot vacío es
  `E = C`.
- **Qué eventos salen.** Uno aparece en cada row group desde el que contiene su
  referencia hasta el que contiene su extremo, incluso sin ticks en él (la referencia
  es el último tick del row group). `starts_before` y `ends_after` (booleanos por
  evento) dicen si empieza antes o termina después; con `starts_before` falso abre
  aquí y con `ends_after` falso cierra aquí. El consumidor completa los eventos que
  cruzan con su acumulador parcial: **nada se relee**.
- **`ticks_before_month`** (arreglo por evento) vale lo mismo que en `read_frames` y
  viene en el row group donde el evento cierra; en los demás es 0. El lector lo
  acumula sin guardar ticks.
- **`events`** son las filas de L2 de esos eventos: las siete columnas de TRD-L3 §7.1
  (`reference_`, `confirm_` y `extreme_agg_trade_id`, `confirm_time`, `direction`,
  `confirm_price` y `extreme_price`) o las de `columns` más las cuatro que el lector
  siempre lee. Es un slice sin copia del row group de eventos de L2; si los eventos
  vienen de dos, la unión es una copia de solo esas filas.
- **`month` y `before_start`.** El pase retrocede al row group más antiguo que
  necesita algún evento de `start`; los row groups de meses anteriores salen con
  `before_start = True`.
- **Garantías de pertenencia.** Las de TRD-L3 §7.4 (1 a 4) se validan por evento con
  `FrameBoundaryError`: toda frontera dentro del row group existe como tick
  (`ids[pos - 1] == id`) y en `C`, además, `transact_time = confirm_time`; un hueco de
  ids no corre la pertenencia. Si L1 termina antes de cerrar un evento, también falla.
- **Memoria.** Con `readers=1`: un row group de L1 en vuelo y ninguno retenido, más un row group de
  eventos de L2 por θ. El `for` conserva el elemento anterior hasta recibir el nuevo:
  suéltalo con `del` antes de pedir el siguiente o conviven dos row groups. Medido: el
  pico es el de decodificar un row group (unos 2,4 de 48 B por tick) y no sube con 10
  veces más eventos ni con un evento de 10 row groups; el modo por chunks llega a más
  de 5. Con `readers` > 1 la cuenta cambia, ver la sección siguiente.
- Cada row group del pase se decodifica **exactamente una vez**, sea cual sea el
  número de θ y la longitud de los eventos. Un row group que ningún evento abierto
  necesita se salta con las estadísticas del pie.

## Lectura: `pre_buffer`, pool de E/S y `readers`

Los archivos viven en GCS y cada petición cuesta una espera, no ancho de banda. Un
`pq.ParquetFile` abre por defecto con `pre_buffer=False`: `read_row_group` pide cada
columna (5 en L1) y cada página por separado y en serie, y un row group de 48 MB tardaba
≈ 2,3 s (≈ 21 MB/s) cuando GCS entrega más de 100 MB/s por conexión.

- **`pre_buffer=True`**, solo en aperturas de vida corta (`lake.open_parquet(...,
  pre_buffer=True)`). Arrow calcula con el pie del archivo los rangos de bytes del row
  group, junta los vecinos y los pide a la vez con su pool de E/S: una espera por row group.
  Pero retiene ese caché **comprimido** (≈ 15 MB) hasta la siguiente lectura o hasta cerrar
  el archivo, y mientras tanto convive con el decodificado. Por eso el lector abre un
  `ParquetFile` por row group, con el pie ya leído, lo lee y lo cierra (L1 con cualquier
  `readers`, y los row groups de `events.parquet`); los archivos que quedan abiertos todo el
  mes (los de `_steps`, el esqueleto de la familia y los que abren L2 y L3) se abren sin
  `pre_buffer`.
- **Pool de E/S de Arrow**, `DC_FRAMES_IO_THREADS` (entero ≥ 1, por defecto 8). Se fija
  **al importar `dc_frames`** con `pa.set_io_thread_count`; un `set_io_thread_count`
  posterior lo reemplaza (las pruebas de memoria lo usan así). Un valor inválido lanza
  `ValueError` al importar. Es global del proceso: si otra parte pide otro tamaño, gana el
  último en llamar.
- **`readers=R`** (solo en `read_row_groups`; por defecto 1). Con R > 1, R row groups se
  leen y decodifican a la vez, cada uno en su hilo, y salen **en el mismo orden y con las
  mismas posiciones** que con 1 (una prueba compara row group a row group). Cada hilo abre
  su propio `ParquetFile`, con el pie ya leído, porque un `ParquetFile` no admite lecturas
  concurrentes, y corre en una copia del contexto de quien consume (los `ContextVar`, como el
  contador de row groups de L3, valen en los hilos). `needed` (qué row groups piden los eventos abiertos) se vuelve a consultar
  al entregar, así que se saltan los mismos row groups; lo único distinto es que al final
  se pueden decodificar de más, a lo sumo R, los que el cierre del último evento dejó sin
  pedir.
- **Por qué el pico es R + K.** Con `readers=1` el lector es secuencial: el row group que
  el consumidor tiene en la mano, y nada más (K = 1). Con R > 1, mientras el consumidor
  trabaja con uno ya viajan los R siguientes: R en camino (los que terminaron esperan su
  turno decodificados) más el de la mano, o sea **K = R + 1 decodificados en RAM**. A eso se
  suman los buffers de decodificación de cada lectura en curso, ≈ 1,1 row groups (un
  `read_row_group` pica en ≈ 2,1 de 48 B por tick con un hilo) y ≈ 0,3 de comprimido de
  `pre_buffer`. Medido en la prueba de memoria: 2,1 con `readers=1`, 4,1 con 2 y 5,1 con 3.
  El umbral de la prueba es `(K + 1.5) × row group`: 2,5, 4,5 y 5,5, con ≈ 0,4 de margen en
  los tres.

  **Eso es lo medido con el fixture, no una cota.** Hasta R `read_row_group` corren a la vez
  (Arrow suelta el GIL), así que el peor caso es el de la mano más R lecturas solapadas:
  `1 + R × 2,1` row groups (R = 2: ≈ 5,2; R = 3: ≈ 7,3), más el comprimido en vuelo. En el
  fixture las decodificaciones duran milisegundos y casi no se solapan; con row groups de
  48 MB sí pueden.
  Cada lector de más cuesta ≈ 48 MB de RSS.
- `pre_buffer` y el pool de E/S valen también para `read_frames` y `frames_of`; `readers`
  no: ahí el lector sigue siendo secuencial.

### Por qué el umbral de memoria no era una cota (ITSC-363)

El umbral de la prueba, `(K + 1,5) × row group`, salió de un fixture y no valía en GCS: la sonda
de ITSC-362 midió un pico del pool de Arrow de 228 MiB con un lector (previsto 95) y 304 con dos
(previsto 172). Faltaban dos términos, medidos aparte:

1. **Leer un row group pica a ≈ 1,8× su tamaño decodificado**, no a 1×. Arrow descomprime cada
   chunk de columna en un temporal del tamaño del chunk antes de decodificarlo, y las cinco columnas
   se decodifican en paralelo. Con un archivo de un row group de 1 M de ticks (46,1 MiB
   decodificados): 78 MiB sin `pre_buffer`, 82 con él (1,78×); por columna, de 1,96× a 2,15×. Con
   `pre_buffer` quedan además ≈ 10 MiB de caché comprimido hasta cerrar el lector.
2. **Los row groups de eventos de L2 de los 50 θ, en la mano a la vez** (`read_row_groups`):
   ≈ 2 MiB por θ con siete columnas, 80 a 105 MiB con 50 θ.

La cuenta cierra con la sonda local sobre 2020-01 (50 θ, row groups de 45 MiB): 82 + 80 = 162 MiB
con un lector (medido 162) y 162 + 82 = 244 con dos (medido 249). En GCS, con row groups de 48 MiB:
82 × 48/46 + 100 ≈ 185 a 230 con un lector (medido 228) y +76 por el segundo lector (304 − 228).
El umbral honesto es entonces `R × 1,8 + 1` row groups más lo que L2 tenga en la mano, no `K + 1,5`.
El resumidor de L3 ya no usa `read_row_groups` (ver la sección siguiente); su cuenta de memoria está
en TRD-L3 ADR-L3-16.

## Archivos abiertos una vez y leídos por tramos (ITSC-363)

`dc_frames.lake` trae dos piezas para leer muchos pedazos de un mismo archivo sin pagar el viaje
de apertura cada vez (H8 de ITSC-362: 1 335 aperturas de `events.parquet` en 2021-01):

- `ParquetSource(path, what)`: abre el archivo como `NativeFile` y lee el pie una vez. `reader()`
  devuelve un `ParquetFile` de vida corta con `pre_buffer` sobre ese mismo archivo, con el pie ya
  leído; varios lectores pueden leer a la vez (un lector, no). Lo usan el escritor de familias
  (las dos claves del esqueleto de cada row group) y la conciliación de L3 (`L1Month`).
- `EventFile(path, what, columns)`: `events.parquet` de un θ y un mes por tramos.
  `lots(lot_bytes)` agrupa row groups seguidos que suman al menos `lot_bytes` comprimidos;
  `batches(row_groups, batch_rows, request_bytes)` los entrega en lotes de a lo más `batch_rows`
  filas; `id_range(columna)` da el mínimo y el máximo de una columna con las estadísticas del pie,
  sin leer datos. **Pocas peticiones grandes:** con las opciones de caché por defecto de Arrow,
  `pre_buffer` pide ≈ 2 rangos de ≈ 200 KiB por row group y solo cuando el decodificador los
  necesita (medido con un contador de lecturas en un archivo real: 40 lecturas de 184 KiB de
  mediana para 20 row groups). `batches` usa la API de datasets de Arrow con
  `CacheOptions(hole_size_limit = request_bytes, range_size_limit = request_bytes, lazy = False)`:
  Arrow junta los rangos de todas las columnas de todos los row groups del tramo, incluidas las que
  no se piden (caben en los huecos), y los pide a la vez (el mismo archivo da 2 lecturas de ≈ 7 MiB).
  Cuesta leer esas columnas de más (+40 % de bytes en el archivo de prueba) y el caché del tramo
  vive mientras se recorre. Un lote nunca cruza row groups y es una vista de los buffers de su row
  group: el último de cada row group puede ser menor y soltarlo suelta el row group cuando ya no
  queda ningún lote suyo. Se lee a lo más un lote por delante del que se pide.


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
- `test_dc_frames_row_groups.py`: la entrega por row group contra un oráculo con
  `bisect`, los slices por posiciones contra `read_frames`, bordes de row group,
  huecos de ids, fallos de frontera y el conteo de decodificaciones.
- `test_dc_frames_family.py`: el escritor, sus propiedades físicas y cada forma
  de romper el esqueleto.
- `test_dc_frames_memory.py`: el pico de memoria de los modos por chunks y por
  row group, medido en un proceso aparte con el pool de Arrow.
