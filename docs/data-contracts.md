# Contratos de datos

Índice de los contratos de interfaz entre capas. El contenido de cada
sección lo escriben E1 y E2; por ahora cada una solo enlaza al TRD.

## Salida Parquet de L1

Contrato hacia L2: el Parquet conformado de L1, una partición por mes.
Fuente de diseño: [TRD-L1 §7.2](TRD/l1.md#72-salida--parquet-conformado-de-l1-contrato-hacia-l2).
Fuente en código: `OUTPUT_SCHEMA` en
[`layers/l1_ingest/src/l1_ingest/schema.py`](../layers/l1_ingest/src/l1_ingest/schema.py)
y `write_partition` en
[`layers/l1_ingest/src/l1_ingest/write.py`](../layers/l1_ingest/src/l1_ingest/write.py).
La sobrescritura atómica (`PartitionWriter`) y el hash de contenido
(`ContentHasher`, `content_hash`) viven en
[`shared/pyutils`](../shared/pyutils/src/pyutils/) y son los mismos para L1 y L2.
Una prueba (`layers/l1_ingest/tests/test_output_contract_doc.py`) rompe el CI si esta
tabla se desvía del código.

### Esquema

| Columna | Tipo Arrow/Parquet | Nulable | Descripción |
|---|---|---|---|
| `agg_trade_id` | int64 | no | Id del trade agregado; clave de orden secundaria |
| `price` | decimal128(18, 8) | no | Precio exacto, escala 8 (ADR-L1-03) |
| `quantity` | decimal128(18, 8) | no | Cantidad exacta, escala 8 (ADR-L1-03) |
| `first_trade_id` | int64 | no | Primer trade individual del agregado |
| `last_trade_id` | int64 | no | Último trade individual del agregado |
| `transact_time` | int64 | no | Momento del trade en microsegundos desde la época (UTC); clave de orden primaria |
| `is_buyer_maker` | bool | no | Si el comprador fue el maker |
| `is_best_match` | bool | no | Si el precio fue el mejor del libro; se conserva del proveedor |

### Disposición física

- **Raíz**: una ruta local o `gs://<bucket landing>/l1`. El stack `l1` solo
  tiene acceso bajo el prefijo `l1/`; la ruta hive va debajo.
- **Partición**:
  `<raíz>/provider=<p>/market=<m>/asset=<a>/year=YYYY/month=MM/`, con el mes
  a dos dígitos.
- **Archivo**: `consolidated.parquet` para un mes cerrado;
  `provisional-day=DD.parquet` para un día del mes en curso. El lector
  prefiere `consolidated.parquet` si existe.
- **Formato**: Parquet con compresión ZSTD nivel 3, estadísticas (min/max) por
  columna, un row group por bloque de CSV de 64 MiB (~800k filas) y `sorting_columns`
  (`transact_time`, `agg_trade_id`) en los metadatos. `write_partition`
  rechaza una tabla que no llegue ordenada por esas claves.
- **Sobrescritura atómica**: `pyutils.PartitionWriter` (código en
  [`shared/pyutils`](../shared/pyutils/src/pyutils/parquet.py)) escribe a un temporal
  `.<nombre>.<uuid>.tmp` del mismo directorio (o del mismo prefijo en GCS) y
  `commit` lo renombra sobre el destino: nunca queda un archivo a medias ni
  se toca el anterior si algo falla. En local es `os.replace`; en GCS,
  `fs.move` (copia más borrado). Con el bucket versionado, cada `.tmp`
  borrado por `fs.move` queda como versión no vigente hasta que la regla de
  lifecycle de `landing` lo elimina (`matches_suffix = [".tmp"]`,
  `days_since_noncurrent_time = 1`), en vez de esperar a
  `num_newer_versions`. Esa regla es la mitigación definitiva, no una
  provisional (decisión del arquitecto en ITSC-234): en GCS se acepta que
  cada `commit` deje un `.tmp` no vigente durante un día, más el retraso con
  que GCS ejecuta las acciones de lifecycle, porque la copia dentro del
  mismo bucket no cobra red y el duplicado de un día cuesta centavos.
  Escribir directo sobre el destino en GCS no es alternativa: abortar sin
  llamar a `close()` no basta, porque `ParquetWriter.__del__` y los
  destructores de `GcsOutputStream`/`ObjectWriteStream` finalizan la subida
  igual, y un aborto podía sustituir la partición vigente por un Parquet con
  solo las filas ya escritas. Por eso se mantiene el temporal en ambos
  filesystems. Si
  el proceso muere sin pasar por `__exit__` (SIGKILL por OOM), el temporal
  queda huérfano: los lectores lo ignoran porque su nombre empieza por `.` y
  no es un `.parquet` de la partición, pero sigue vigente (LIVE) y la regla
  de lifecycle nueva solo actúa sobre versiones no vigentes
  (`with_state = "ARCHIVED"`): nunca lo alcanza. Solo una limpieza a mano con
  `gsutil rm` lo retira.
- **Idempotencia**: es contenido idéntico, no bytes idénticos. Se mide con
  `pyutils.content_hash(table)` (o `pyutils.ContentHasher`, que lo calcula lote a lote;
  código en [`shared/pyutils`](../shared/pyutils/src/pyutils/hashing.py)): el
  SHA-256 del esquema más un SHA-256 por columna sobre los bytes de sus
  valores en orden, sin importar el chunking. Se hashea el contenido lógico y
  no el archivo porque el Parquet puede diferir en bytes entre versiones de
  la librería sin que cambien los datos.

### Escribir

```python
from l1_ingest.write import (
    CONSOLIDATED,
    day_filename,
    partition_path,
    write_partition,
)

path = partition_path(
    "gs://<bucket landing>/l1",
    "binance",
    "spot",
    "BTCUSDT",
    2024,
    3,
    CONSOLIDATED,  # o day_filename(15)
)
write_partition(table, path)  # table.schema debe ser OUTPUT_SCHEMA
```

En GCS usa Application Default Credentials; no hay credenciales en código.

## Salida Parquet de L2

Contrato hacia L3: los eventos Directional Change de cada θ, una partición por
θ y mes, y el carry-over que encadena un mes con el siguiente.
Fuente de diseño: [TRD-L2 §7.2](TRD/l2.md#72-salida--eventsparquet-contrato-hacia-l3)
y [§7.4](TRD/l2.md#74-carry-over--contrato-y-disposición-física).
Fuente en código: `EVENTS_SCHEMA` y `CARRY_OVER_SCHEMA` en
[`layers/l2_dc_events/src/l2_dc_events/schema.py`](../layers/l2_dc_events/src/l2_dc_events/schema.py)
y el escritor en
[`layers/l2_dc_events/src/l2_dc_events/write.py`](../layers/l2_dc_events/src/l2_dc_events/write.py).
Una prueba (`layers/l2_dc_events/tests/test_l2_output_contract_doc.py`) rompe el
CI si alguna de las dos tablas se desvía del código.

Todo precio es el entero sin escalar de un `DECIMAL(18,8)` (escala `10⁸`, la
misma que L1) y todo tiempo son microsegundos UTC. Un punto es el trío
precio, tiempo y `agg_trade_id`.

### Esquema de `events.parquet`

Una fila por evento DC confirmado, con sus tres puntos completos.

| Columna | Tipo Arrow/Parquet | Nulable | Descripción |
|---|---|---|---|
| `reference_price` | decimal128(18, 8) | no | Precio del extremo que define el arranque del evento (el extremo del evento anterior) |
| `reference_time` | int64 | no | Tiempo de la referencia |
| `reference_agg_trade_id` | int64 | no | `agg_trade_id` de la referencia |
| `confirm_price` | decimal128(18, 8) | no | Precio de confirmación (DCC) según la regla conservadora de empates (ADR-L2-03) |
| `confirm_time` | int64 | no | Tiempo del último tick del grupo de empate; clave de orden del archivo |
| `confirm_agg_trade_id` | int64 | no | `agg_trade_id` del último tick del grupo de empate |
| `extreme_price` | decimal128(18, 8) | no | Precio que cierra el Overshoot; es la `reference_price` del evento siguiente |
| `extreme_time` | int64 | no | Tiempo del extremo; nunca anterior a `confirm_time` |
| `extreme_agg_trade_id` | int64 | no | `agg_trade_id` del extremo; nunca anterior a `confirm_agg_trade_id` |
| `direction` | int8 | no | `1` upturn, `-1` downturn |
| `theta` | decimal128(9, 8) | no | θ del evento; redundante con la partición, para leer sin `hive_partitioning` |

Un Overshoot vacío tiene `extreme_*` igual a `confirm_*`: L3 debe tratar su
duración como cero, no como un dato inválido.

### Esquema de `carry_over.parquet`

Una fila por `(theta, year, month)`: el estado del detector de ese θ al cierre
del mes. El mes siguiente la lee para continuar sin reprocesar ticks.

| Columna | Tipo Arrow/Parquet | Nulable | Descripción |
|---|---|---|---|
| `provider` | string | no | Proveedor del dato, redundante con la partición |
| `market` | string | no | Mercado del dato, redundante con la partición |
| `asset` | string | no | Activo del dato, redundante con la partición |
| `theta` | decimal128(9, 8) | no | θ de esta cadena |
| `year` | int32 | no | Año del mes que produjo este estado (se consume desde el mes siguiente) |
| `month` | int32 | no | Mes que produjo este estado (1 a 12) |
| `state_version` | string | no | Versión semver del esquema del estado; si no coincide con la que la imagen sabe leer, el mes siguiente aborta (ADR-L2-08) |
| `direction` | int8 | no | `0` indefinida (aún sin ninguna reversión), `1` upturn, `-1` downturn |
| `ext_high_price` | decimal128(18, 8) | no | Precio del extremo alto vigente |
| `ext_high_time` | int64 | no | Tiempo del extremo alto vigente |
| `ext_high_agg_trade_id` | int64 | no | `agg_trade_id` del extremo alto vigente |
| `ext_low_price` | decimal128(18, 8) | no | Precio del extremo bajo vigente |
| `ext_low_time` | int64 | no | Tiempo del extremo bajo vigente |
| `ext_low_agg_trade_id` | int64 | no | `agg_trade_id` del extremo bajo vigente |
| `has_pending_event` | bool | no | `true` si hay un evento confirmado cuyo extremo aún no se conoce (el caso normal al cierre de un mes) |
| `pending_reference_price` | decimal128(18, 8) | sí | Precio de la referencia del evento pendiente; nulo sin evento pendiente |
| `pending_reference_time` | int64 | sí | Tiempo de esa referencia; nulo sin evento pendiente |
| `pending_reference_agg_trade_id` | int64 | sí | `agg_trade_id` de esa referencia; nulo sin evento pendiente |
| `pending_confirm_price` | decimal128(18, 8) | sí | Precio de la confirmación del evento pendiente; nulo sin evento pendiente |
| `pending_confirm_time` | int64 | sí | Tiempo de esa confirmación; nulo sin evento pendiente |
| `pending_confirm_agg_trade_id` | int64 | sí | `agg_trade_id` de esa confirmación; nulo sin evento pendiente |

### Disposición física

- **Raíz**: una ruta local o `gs://<bucket>/<prefijo>` de eventos.
- **Partición**:
  `<raíz>/provider=<p>/market=<m>/asset=<a>/theta=<t>/year=YYYY/month=MM/`.
  `theta=<t>` es de ancho fijo, `0.` más ocho decimales sin recortar ceros
  (`theta=0.00010000`, `theta=0.05000000`): ordena bien como texto y deja a la
  vista la escala que comparte con el precio.
- **Archivos**: `events.parquet` y `carry_over.parquet`, juntos en la
  partición del mes que los produce. Un evento se escribe en el mes que
  confirma el **evento siguiente**, el que conoce su extremo (ADR-L2-06), no en
  el de su referencia ni necesariamente en el de `extreme_time`. El evento
  que sigue abierto al cierre del mes va al carry-over como
  `has_pending_event` y lo completa el mes que lo resuelve.
- **Formato**: Parquet con compresión ZSTD nivel 3 y estadísticas (min/max) por
  columna. `events.parquet` va ordenado por `confirm_time`
  (`sorting_columns` en los metadatos), en row groups de hasta 32 768 filas:
  se escriben a medida que los eventos cierran, para que la RAM no dependa
  del tamaño del mes. Un θ sin eventos en el mes publica igual un
  `events.parquet` válido con cero filas.
- **Codificación física** (ITSC-290): sin diccionario; las columnas enteras en
  `DELTA_BINARY_PACKED` y los decimales y el texto en `PLAIN`
  (`PartitionWriter(compact_encoding=True)`). Es un detalle físico sin efecto
  en el contrato: ni los tipos, ni los valores, ni el `content_hash` cambian, y
  pyarrow, DuckDB y Polars leen el archivo igual. Solo cambian los bytes (los
  `events.parquet` de 2020-01 pasan de 442 a 258 MB) y lo que cuesta
  codificarlos y decodificarlos. L1 no lo usa.
- **Sobrescritura atómica**: L2 usa el mismo `pyutils.PartitionWriter` que L1 (temporal
  `.<nombre>.<uuid>.tmp` en el mismo directorio o prefijo, y `commit` lo
  renombra sobre el destino). Nunca queda un archivo a medias ni se toca el
  anterior si algo falla; los detalles de GCS están en "Salida Parquet de L1".
  Primero se publican los `events.parquet` y luego cada `carry_over.parquet`,
  así que un carry-over presente significa que el mes de ese θ está completo.
- **Idempotencia**: es contenido idéntico, no bytes idénticos. `pyutils.ContentHasher`
  calcula un SHA-256 del esquema más uno por columna sobre sus valores en orden,
  sin importar cómo se parta la serie en lotes ni en row groups. Re-ejecutar un
  mes con el mismo carry-over de entrada da el mismo `content_hash` en ambos
  archivos de cada θ; el de cada θ queda en el hallazgo `events_summary`.
- **Fail-closed**: para un mes que no es el primero de la serie, si falta el
  carry-over del mes anterior o su `state_version` no es la de la imagen, la
  unidad aborta sin escribir ningún evento y emite un hallazgo por θ
  ([ADR-L2-08](TRD/l2.md#68-adr-l2-08--carry-over-faltante-o-de-otra-versión-fail-closed-no-log-and-continue)).
  El primer mes de la serie se declara (`--series-start`), no se infiere.

### Leer

```python
import pyarrow.parquet as pq

events = pq.read_table(
    "<raíz>/provider=binance/market=spot/asset=BTCUSDT"
    "/theta=0.00010000/year=2020/month=01/events.parquet"
)
```

## Lago de hallazgos de calidad de datos

Contrato del lago donde toda capa deja sus hallazgos de calidad de datos (DQ).
Fuente de diseño: [TRD-L1 §7.3](TRD/l1.md#73-lago-de-hallazgos-de-calidad-de-datos).
Fuente en código: `FINDING_SCHEMA` en
[`shared/dq/src/dq/schema.py`](../shared/dq/src/dq/schema.py). Las tres
listas de columnas (TRD, código y esta tabla) van en el mismo orden; una
prueba (`shared/dq/tests/test_contract_doc.py`) rompe el CI si esta tabla se
desvía del código.

### Esquema

| Columna | Tipo Arrow/Parquet | Nulable | Descripción |
|---|---|---|---|
| `finding_id` | string (UUID) | no | Identidad del hallazgo; se repite en todos los eventos del mismo hallazgo |
| `detected_at` | int64 | no | Marca del evento, en microsegundos desde la época (UTC) |
| `layer` | string | no | Capa que emite el hallazgo, p. ej. `l1` |
| `mode` | string | no | Modo de ejecución: `backfill`, `daily`, `monthly-close`, `seam-check` o `monthly` (L2, [TRD-L2 §8.3](TRD/l2.md#83-modo-monthly-incremental)) |
| `check_type` | string | no | Chequeo que lo generó (ver TRD-L1 §9) |
| `severity` | string | no | `info`, `warning` o `error` |
| `stage` | string | no | `provisional` o `canonical` |
| `status` | string | no | `pass`, `fail` o `corrected` |
| `provider` | string | no | Proveedor del dato afectado, p. ej. `binance` |
| `market` | string | no | Mercado del dato afectado, p. ej. `spot` |
| `asset` | string | no | Activo del dato afectado, p. ej. `BTCUSDT` |
| `year` | int32 | no | Año del dato afectado |
| `month` | int32 | no | Mes del dato afectado (1 a 12) |
| `metric_value` | float64 | sí | Medida del chequeo, p. ej. número de huecos; nulo si no aplica |
| `details` | string (JSON) | no | Payload libre de cada chequeo, como texto JSON |
| `run_id` | string | no | Ejecución que emitió el hallazgo |
| `image_version` | string | no | semver + git SHA de la imagen que lo emitió |

Por qué estas decisiones, escritas una sola vez:

- **`year` y `month` son `int32`**: son números pequeños y `int32` alcanza de
  sobra; con `int64` cada valor ocupa el doble sin ganar nada.
- **La partición se llama `detected_date=`**, no `year=` ni `month=`: esas dos
  son columnas del esquema (el período del dato afectado). Si la partición
  usara los mismos nombres, DuckDB y Arrow verían la columna duplicada y
  mezclarían dos ideas distintas: cuándo se detectó el hallazgo y a qué mes
  del dato se refiere.

### Disposición física

- **Raíz**: una ruta local o `gs://<project_id>-dq-findings`.
- **Partición**: `<raíz>/detected_date=YYYY-MM-DD/`, con la fecha UTC de
  `detected_at`.
- **Archivo**: `<run_id>-<uuid>.parquet`. El nombre es único por llamada, así
  que una emisión nunca sobrescribe otra.
- **Formato**: Parquet con compresión ZSTD nivel 3 y estadísticas de columna.
- **Append-only**: un hallazgo nunca se actualiza. Un cambio de estado es un
  evento nuevo con el mismo `finding_id` y un `detected_at` posterior.

### Emitir

```python
from dq import Finding, emit_findings

finding = Finding(detected_at=..., layer="l1", ...)  # campos del esquema
emit_findings([finding], "gs://<project_id>-dq-findings")
```

En GCS usa Application Default Credentials; no hay credenciales en código.

### Catálogo de hallazgos

`check_type` de L1 definidos en el diseño original
([TRD-L1 §9.3](TRD/l1.md#93-tipos-de-chequeo-check_type)): `checksum_fail`,
`checksum_drift`, `header_detected`, `timestamp_unit_corrected`,
`reorder_applied`, `aggid_gap`, `aggid_duplicate`, `seam_discontinuity`,
`daily_monthly_drift`. Los que se agregan después de un bug real se
documentan aquí, con su porqué:

- **`seam_skipped`** (ITSC-233): `daily` evalúa la costura del día recién
  escrito contra su día previo apenas termina de procesar la unidad. Cloud
  Run corre varias tareas del mismo Job en paralelo y no garantiza el orden:
  si el día previo todavía no se escribió (terminó su tarea después), la
  costura no tiene con qué compararse. No es un error de datos -el día
  previo se escribe segundos más tarde, dentro de la misma corrida- así que
  no vale un `warning` en `seam_discontinuity` (ese `check_type` es para
  huecos o solapes reales de `agg_trade_id`). Antes de ITSC-233 la omisión
  solo quedaba en el log (`costura omitida unidad=... no existe
  provisional-day=NN.parquet`); ahora además se emite `severity = info`,
  `status = pass`, `metric_value = null` y `details = {"reason":
  "previous_missing", "expected_path": <ruta que faltó>}`, para que quede en
  el lago y sea visible en BigQuery o Looker. En estado estacionario (una
  tarea por corrida) no ocurre; `monthly-close` cubre el hueco al validar el
  consolidado completo (`aggid_gap`) y la costura con M-1.
- **`zip_extra_members`** (ITSC-231): un ZIP mensual o diario trae, además
  del CSV esperado, otros miembros. Caso real: Binance publicó el ZIP de
  2021-12 con el CSV oficial duplicado, una vez en la raíz y otra bajo una
  ruta interna de su colector (`fsx-data/collector_data/...`), ambas copias
  con el mismo checksum publicado. L1 elige el miembro cuyo nombre base es
  exactamente `<asset>-aggTrades-<YYYY-MM>.csv` (o `<YYYY-MM-DD>` en diario);
  con varios candidatos, prefiere el de la raíz y, si ninguno está en la
  raíz, el primero en el orden del ZIP (`_pick_csv` en
  [`layers/l1_ingest/src/l1_ingest/parse.py`](../layers/l1_ingest/src/l1_ingest/parse.py)).
  Si sobran miembros no aborta: emite `severity = warning`,
  `status = pass`, `metric_value` = número de miembros sobrantes y
  `details.members` con esa lista. Si ningún miembro coincide con el nombre
  esperado, sigue abortando con `ValueError`, como antes de ITSC-231.

`check_type` de L2 (`layer = l2`, `stage = canonical`, `mode` `backfill` o
`monthly`; el θ viaja en `details.theta`, no en una columna). Son los de
[TRD-L2 §9.3](TRD/l2.md#93-tipos-de-chequeo-check_type). Los dos de catálogo
(`theta_catalog_invalid` y `theta_behind_frontier`) son de ITSC-285 y
`theta_config_drift` cambió de sentido en esa misma card:

- **`carry_over_missing`** (`error`, `fail`): mes distinto del primero de la
  serie sin el carry-over del mes anterior. La unidad aborta y emite uno por θ
  afectado, con `details.expected_path`.
- **`carry_over_version_mismatch`** (`error`, `fail`): el `state_version` del
  carry-over no es el de la imagen (`details.found`, `details.expected`), o el
  archivo no tiene el esquema de esa versión. También lo emite un
  carry-over cuya columna `theta` no coincide con la partición `theta=<t>`
  donde está (`details.reason`): el estado no se puede usar y la unidad aborta.
- **`theta_config_drift`** (`info`, `pass`): el lago tiene particiones de un θ
  que ya no está en el catálogo (`details.theta`, `details.months`,
  `details.last_month`; `metric_value` = meses). Informa, nunca es error:
  quitar un θ del catálogo no borra nada. Desde ITSC-285 ya no significa una
  columna `theta` distinta de la partición (ahora es
  `carry_over_version_mismatch`).
- **`theta_catalog_invalid`** (`error`, `fail`): el catálogo de θ
  (`L2_THETAS_URI`, [TRD-L2 §7.3](TRD/l2.md#73-el-catálogo-de-θ)) no existe, no
  se puede leer o incumple su contrato (`scale` ≠ 10⁸, θ no entero, repetido
  o fuera de [10⁻⁴, 5·10⁻²]). `details.source` es el objeto y
  `details.problems`, la lista de incumplimientos. La corrida termina con
  código 2 sin escribir datos; `year` y `month` son los del primer mes que
  iba a procesar.
- **`theta_behind_frontier`** (`warning`, `fail`): `monthly` no procesó un θ
  porque su frontera no es el mes previo (se agregó al catálogo sin backfill,
  o su cadena tiene un hueco). `details.theta` y `details.frontier`
  (`YYYY-MM`, o `null` si el θ no tiene carry-over). No falla la unidad: el
  resto de los θ avanza. Se resuelve lanzando `l2-backfill` sin `--from`.
- **`dc_zero_tick_discarded`** (`error`, `fail`): la guarda de "un DC tiene al
  menos un tick" descartó eventos de un θ. Con el instante de confirmación
  atómico se espera cero: solo se emite con conteo mayor (`metric_value`) y
  señala un defecto del detector.
- **`input_missing`** (`error`, `fail`): el mes no tiene `consolidated.parquet`
  ni provisionales (`details.expected_path`). Es lo mismo que espera L1 en
  `monthly-close`; L2 no puede empezar.
- **`input_provisional_only`** (`error`, `fail`): el mes solo tiene
  `provisional-day=DD.parquet` (`details.provisionals`). Se separa del
  anterior porque la acción es distinta: no hay que re-ingestar, hay que
  esperar el `monthly-close` de L1 (ADR-L2-09).
- **`events_summary`** (`info`, `pass`): uno por θ al terminar el mes.
  `metric_value` es el número de eventos escritos; `details` lleva
  `events`, `has_pending_event` y los `content_hash` de `events.parquet` y
  `carry_over.parquet`. Es la huella de la corrida: dos ejecuciones del mismo
  mes deben dar los mismos hashes.
- **`unit_timing`** (`info`, `pass`): uno por unidad (mes) al terminar, no por
  θ. `metric_value` es la pared en segundos (`wall_s`); `details` lleva los
  segundos por fase (`read_s`, `decode_s`, `detect_s`, `detect_cpu_s`,
  `write_s`, `carry_s`, `wait_s`, `other_s`), `row_groups`, `bytes_in`, el
  límite efectivo de CPU (`cores`, `cores_visible`, `cores_source`),
  `fanout_threads`, `write_workers` y `cpu_throttled_s` (`null` si el cgroup
  no lo expone). `read_s`, `detect_s`, `carry_s` y `wait_s` son pared del hilo
  principal y, con `other_s`, suman `wall_s`; `decode_s` (CPU del hilo lector),
  `detect_cpu_s` (CPU del hilo principal durante el fan-out) y `write_s` se
  acumulan entre hilos y no entran en esa suma (`write_s` puede pasar la
  pared). `detect_cpu_s` frente a `detect_s` distingue un detector lento (CPU
  alta) de uno desalojado (CPU baja). Desde ITSC-290 `read_s` es la espera del
  hilo principal por el lector anticipado y `decode_s` ya no es pared. Va aparte de `events_summary` porque sus tiempos cambian en cada
  corrida y el de `events_summary` no puede.

### Estado actual de un hallazgo

El estado actual es el último evento por `finding_id`
([ADR-L1-08](TRD/l1.md#68-adr-l1-08--hallazgos-append-only-con-event-sourcing-sin-motor-transaccional)).
La consulta de referencia es `CURRENT_FINDINGS_SQL` en `dq.reader`
(requiere el extra `dq[reader]`):

```sql
select * exclude (rn) from (
    select
        *,
        row_number() over (
            partition by finding_id order by detected_at desc
        ) as rn
    from read_parquet($pattern, hive_partitioning = true)
)
where rn = 1
```

Sobre una raíz local: `dq.reader.current_findings(root)`.

Política provisional a canonical: `daily` emite el hallazgo con
`stage = provisional`; `monthly-close` lo revalida y emite un evento nuevo con
el mismo `finding_id` y `stage = canonical`. Como el canónico tiene un
`detected_at` mayor, la consulta lo devuelve y el provisional queda como
historia.

Sobre `gs://`, `current_findings` todavía no acepta esa raíz. Hay dos
caminos:

- **DuckDB**: instala y carga la extensión `httpfs` y crea un secreto HMAC de
  GCS (`CREATE SECRET (TYPE gcs, KEY_ID ..., SECRET ...)`). A diferencia de
  `emit_findings` y del camino pyarrow, que usan Application Default
  Credentials, `httpfs` no las acepta. Luego ejecuta `CURRENT_FINDINGS_SQL` con
  `$pattern = "gs://<project_id>-dq-findings/detected_date=*/*.parquet"`.
- **pyarrow**: lee con
  `pyarrow.dataset.dataset("<project_id>-dq-findings", filesystem=GcsFileSystem(),
  partitioning="hive")` y registra el resultado en DuckDB con
  `con.register("findings", ds)`. Con `filesystem` explícito la ruta va sin
  `gs://` (`GcsFileSystem` rechaza URIs); si omites `filesystem`, pasa la URI
  `gs://<project_id>-dq-findings` y pyarrow resuelve el sistema de archivos. La
  consulta es la misma, pero cambia `read_parquet($pattern, hive_partitioning =
  true)` por `findings`.

## Manifiesto de checksums

Registro de procedencia de L1: una fila por archivo ingerido, con su SHA-256.
Fuente de diseño: [TRD-L1 §7.4](TRD/l1.md#74-manifiesto-de-checksums).
Fuente en código: `MANIFEST_SCHEMA` en
[`layers/l1_ingest/src/l1_ingest/manifest.py`](../layers/l1_ingest/src/l1_ingest/manifest.py).

### Esquema

| Columna | Tipo Arrow/Parquet | Nulable | Descripción |
|---|---|---|---|
| `provider` | string | no | Proveedor del archivo, p. ej. `binance` |
| `market` | string | no | Mercado del archivo, p. ej. `spot` |
| `asset` | string | no | Activo del archivo, p. ej. `BTCUSDT` |
| `year` | int32 | no | Año del dato |
| `month` | int32 | no | Mes del dato (1 a 12) |
| `granularity` | string | no | `monthly` o `daily` |
| `source_url` | string | no | URL de la que se descargó el archivo |
| `sha256` | string | no | SHA-256 del archivo, en hexadecimal |
| `downloaded_at` | int64 | no | Momento de la descarga, en microsegundos desde la época (UTC) |
| `file_bytes` | int64 | no | Tamaño del archivo en bytes |
| `image_version` | string | no | semver + git SHA de la imagen que lo ingirió |

### Disposición física

- **Raíz**: una ruta local o `gs://<bucket>/<prefijo>`.
- **Partición**:
  `<raíz>/provider=<p>/market=<m>/asset=<a>/year=YYYY/month=MM/`, por el
  scope del archivo y no por la fecha de descarga. Así el cierre mensual
  detecta una republicación comparando el `sha256` nuevo con los anteriores
  leyendo un solo prefijo.
- **Archivo**: `<run_id>-<uuid4>.parquet`. El nombre es único por llamada y
  lleva el `run_id` para cruzarlo con los hallazgos del mismo run; el
  esquema no lo incluye.
- **Formato**: Parquet con compresión ZSTD nivel 3 y estadísticas de columna.
- **Append-only**: una fila nunca se actualiza ni se borra. Descargar de nuevo
  el mismo archivo agrega otra fila.

### Escribir

```python
from l1_ingest.manifest import ManifestEntry, write_manifest

entry = ManifestEntry(provider="binance", ...)  # campos del esquema
write_manifest([entry], "gs://<bucket>/<prefijo>", run_id)
```

En GCS usa Application Default Credentials; no hay credenciales en código.
