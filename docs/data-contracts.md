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

### Invariante de valores

Toda fila del Parquet cumple `price > 0`, `quantity > 0`,
`first_trade_id >= 0` y `last_trade_id >= first_trade_id`. Binance marcó como
inválidos los agg trades duplicados que detectó al auditar su histórico Spot
en abril de 2022, con `price = 0`, `quantity = 0`, `first_trade_id = -1` y
`last_trade_id = -1`, y conservó su `agg_trade_id` y `transact_time`
([changelog de la API Spot, entrada 2022-04-12](https://github.com/binance/binance-spot-api-docs/blob/master/CHANGELOG_CN.md)). No son
transacciones: L1 las descarta antes de escribir y emite
`provider_invalid_marker`. Cualquier otra fila con `price <= 0`,
`quantity <= 0`, `first_trade_id < 0` o `last_trade_id < first_trade_id` hace
fallar la unidad (`price_out_of_range`). Por eso el
Parquet puede tener huecos de `agg_trade_id` solo donde el proveedor los
tenga, nunca por las marcas. Detalle en el runbook
[Marcas de Binance](runbooks/operacion-l1.md#marcas-de-binance).

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
| `mode` | string | no | Modo de ejecución: `backfill`, `daily`, `monthly-close`, `seam-check` o `monthly` (L2, [TRD-L2 §8.3](TRD/l2.md#83-modo-monthly-incremental)); `tiles` o `render` (viz, [TRD-viz §8](TRD/viz.md#8-pipeline-interno-y-modos-de-ejecución)) |
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

- **`provider_invalid_marker`** (ITSC-294): filas que Binance marcó como
  inválidas (`price = 0`, `quantity = 0`, `first_trade_id = -1`,
  `last_trade_id = -1`) y L1 descartó antes de escribir. `metric_value` =
  número de filas descartadas y `details.ids` = hasta 10 `agg_trade_id`. Sin
  marcas: `severity = info`, `status = pass`, `metric_value = 0`. Con marcas:
  `severity = warning`, `status = corrected`. Se emite siempre, en cada
  unidad.
- **`price_out_of_range`** (ITSC-294): fila con `price <= 0`, `quantity <= 0`,
  `first_trade_id < 0` o `last_trade_id < first_trade_id` que no es marca
  completa, es decir, dato corrupto. El nombre quedó por el caso original
  (precio) pero cubre también los trade ids. `severity = error`,
  `status = fail`, `metric_value` = filas corruptas y `details.ids` = hasta 10
  `agg_trade_id`. La unidad falla y no escribe Parquet ni manifiesto. Solo
  existe cuando falla.
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
- **`source_not_published`** (ITSC-295): Binance respondió 404 al ZIP de la
  unidad y la fecha UTC de la corrida aún está dentro del calendario de
  publicación (diario de D: el día D+1; mensual de M: el primer lunes de
  M+1). No es un error del pipeline, es la naturaleza del proceso. La unidad
  no escribe Parquet ni manifiesto y la CLI sale con 0: queda pendiente y la
  frontera de L2 la ignora. `metric_value` = días que faltan para la
  publicación (0 el propio día), `details.expected_publication`,
  `details.unit`, `details.source_url` y `details.reason`. Antes de la
  publicación: `severity = info`, `status = pass`. El día de publicación
  (Binance no publica hora, así que el día completo cuenta como a tiempo):
  `severity = warning`, `status = fail`.
- **`source_delayed`** (ITSC-295): mismo 404, pero la fecha UTC de la corrida
  es posterior al día de publicación: la fuente se retrasó sobre su
  calendario. `severity = error`, `status = fail`, `metric_value` = días de
  retraso (`details.reason` lo dice con palabras). La CLI sale con 3, la tarea
  falla y la orquestación no ejecuta los `next_jobs` (`l2-monthly` y
  `viz-tiles`). No escribe Parquet ni manifiesto. Solo existe cuando ocurre.
  Los jobs de L1 tienen `max_retries = 1`: Cloud Run reintenta la tarea con
  salida 3, que repite el 404 y deja una segunda fila `source_delayed` (otro
  `finding_id`) en el lago. Se acepta: el resumen de `run-job.yml` las junta
  por contenido (el hallazgo completo menos `finding_id`, `run_id` y
  `detected_at`).

Ni `source_not_published` ni `source_delayed` se reintentan: un 404 no cambia
por esperar. El backoff queda solo para fallos de red y 5xx.

Cada hallazgo, además de la fila en el lago, deja una línea de log JSON
(`dq.emit_findings`) con `finding_id`, `layer`, `mode`, `check_type`,
`severity`, `stage`, `status`, `provider`, `market`, `asset`, `year`, `month`,
`metric_value`, `details` y `run_id`. Es lo que lee el resumen de
`run-job.yml`, sin acceder al bucket.

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
  (`YYYY-MM`, o `null` si el θ no tiene carry-over). No falla la unidad mientras
  otros θ avancen o ya tengan el mes; si ninguno está listo ni lo tiene, `monthly` termina con código 1
  (fail-closed) y el workflow L1→L2 lo ve como fallo. Se resuelve lanzando `l2-backfill` sin `--from`.
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

## Tiles de viz

Contrato hacia el tablero: por día, `ticks.bin` con **todos** los ticks sin
reducir (deltas en varint, por tramos de hasta 65 536 ticks), `events.bin` con los eventos exactos de cada θ, un
índice JSON y la página que los lleva dentro. El navegador deriva de ahí el
precio, el volumen y las confirmaciones de cada píxel al dibujar. Fuente de
diseño: [TRD-viz §7](TRD/viz.md#7-contrato-de-datos). Fuente en código:
[`contract.py`](../layers/viz_tiles/src/viz_tiles/contract.py) (constantes),
[`ticks.py`](../layers/viz_tiles/src/viz_tiles/ticks.py) (`ticks.bin`),
[`events.py`](../layers/viz_tiles/src/viz_tiles/events.py) (`events.bin`),
[`write.py`](../layers/viz_tiles/src/viz_tiles/write.py) (escritura) y
[`render.py`](../layers/viz_tiles/src/viz_tiles/render.py) (la página del día). Una
prueba (`layers/viz_tiles/tests/test_tiles_contract_doc.py`) rompe el CI si
las tablas de esta sección se desvían de esas constantes.

Los archivos de un día no son fuente de verdad: se regeneran desde L1 y L2. Todos
los binarios son planos: `events.bin` sin cabecera y `ticks.bin` con la cabecera de
cada tramo. `tiles_version` es `2.0.0`: la 1.x guardaba arreglos M4 por nivel de zoom (`price-<w>`, `volume-<w>`, `dir-<w>`, `count-<w>`,
`confirms-<w>`, `simul-<w>`) y esa noción de nivel desapareció (ADR-VZ-14).

### Disposición

```
<raíz>/provider=<p>/market=<m>/asset=<a>/day=YYYY-MM-DD/
├── ticks.bin               # todos los ticks del día: tramos de tres secciones de varint
├── events.bin              # los eventos exactos de todos los θ del día
├── index.html              # la página del día: plantilla, uPlot, ticks.bin y events.bin
└── index.json              # se escribe al final: marca de commit
<raíz>/latest.json          # último día con index.json
<raíz>/latest.html          # copia de la página de ese día
```

Un día son 4 objetos, con cualquier número de θ. El día es UTC. El job escribe
`ticks.bin` **tramo a tramo mientras lee L1** (borra antes el `index.json` previo y
suelta cada tramo en cuanto lo escribe), y `write_day` escribe `events.bin` y la
página (que vuelve a leer `ticks.bin` por bloques) y deja el índice al final: un día
sin `index.json` no existe para el lector. Ningún paso de escritura retiene un
archivo del día entero en RAM.
`latest.json` y `latest.html` solo avanzan; un día anterior regenerado no los
retrocede.

Metadatos de los objetos: los binarios, `application/octet-stream` con
`Cache-Control: private, max-age=31536000, immutable`; `index.json` y
`latest.json`, `application/json` con `no-cache`; las páginas, `text/html;
charset=utf-8` con `no-cache` y, en un bucket, `Content-Encoding: gzip` (en
disco local van sin comprimir, para abrir por `file://`).

### Página del día

`index.html` es un solo documento sin peticiones de red: lleva dentro la
plantilla (`layers/viz_tiles/site/`), uPlot y los dos archivos del día en base64
bajo su nombre, en `window.VIZ_DATA = {tiles_version, generated_at, files, index}`.
`files` mapea `ticks.bin` y `events.bin` a su base64; `index` es el `index.json`
del día. Un `<meta name="viz-render" content="tiles_version=…;template=…">` al
principio guarda la versión de los tiles y el SHA-256 de la plantilla con que
se armó: el modo `render` lo lee para saltar lo que ya está al día.

### Archivos

`ticks.bin` guarda todos los ticks del día, en el orden del consolidado de L1
(`transact_time` y, dentro de un mismo instante, `agg_trade_id`), como una
secuencia de **tramos**: cada uno trae hasta `ticks_chunk` ticks (65 536; campo del
índice) y todos salen llenos salvo el último, así que los bytes no dependen de cómo
se partan los lotes al leer L1. El archivo es la concatenación de los tramos, sin
separadores, y ocupa el archivo entero.

Cada tramo es una **cabecera** de cuatro `uint32` little-endian (16 bytes) seguida de
**tres secciones consecutivas** de enteros varint, una por columna:

| Cabecera | Contenido |
|---|---|
| Bytes 0–3 | Ticks del tramo (de 1 a `ticks_chunk`): los valores de cada sección |
| Bytes 4–7 | Bytes de la sección Δtiempo |
| Bytes 8–11 | Bytes de la sección Δprecio |
| Bytes 12–15 | Bytes de la sección cantidad |

Un varint es un entero sin signo en LEB128: siete bits por byte, del menos al más
significativo, y el bit alto de cada byte marca que sigue otro (de 1 a 10 bytes). El
lector decodifica, en cada tramo, los valores de la primera sección, los de la segunda
y los de la tercera, y cada sección debe ocupar exactamente los bytes que declara la
cabecera. **El Δtiempo y el Δprecio del primer tick de un tramo son relativos al
último tick del tramo anterior** (en el primer tramo del día, al inicio del día y a
0): el lector arrastra los dos acumuladores de un tramo al siguiente. La suma de los
ticks de los tramos es `ticks`.

| Sección | Nombre | Contenido |
|---|---|---|
| 1 | `dt_ms` | Δtiempo: milisegundos (`⌊µs / 1000⌋`) desde el tick anterior. El primero del día se mide desde `t0`. Sin signo: los ticks no retroceden. Ticks del mismo milisegundo llevan 0 |
| 2 | `dprice_zigzag` | Δprecio en unidades de `1 / price_scale`, desde el tick anterior (el primero del día, desde 0), en zigzag: `(d << 1) ^ (d >> 63)`, que lleva 0, −1, 1, −2… a 0, 1, 2, 3… |
| 3 | `quantity_1e8` | Cantidad del tick en unidades de 10⁻⁸ (el entero exacto del `DECIMAL(18, 8)` de L1), sin signo. Puede pasar de 2³²: se lee con aritmética de coma flotante, no con operadores de bits de 32 bits |

El tiempo absoluto de un tick es `t0 + Σ dt_ms` y su precio, `Σ dprice / price_scale`
(las sumas corren sobre todos los ticks anteriores del día, de un tramo a otro).
Con los 948 740 ticks del 2026-09-30, `ticks.bin` pesa 4,85 MB (≈ 5,1 B por tick; las
cabeceras de 15 tramos suman 240 B). Un lector entero decodifica el día a tres
arreglos: tiempo y precio en `Int32Array` (el tiempo no pasa de 86 400 000 y el precio
cabe en `int32` por la guarda de abajo) y cantidad en `Float64Array`; el día
decodificado ocupa ≈ 16 B por tick.

**Memoria del job (regla, sin excepción).** El job codifica un lote de L1 en el tramo en
curso y, cuando el tramo se llena, lo escribe al objeto y lo suelta: en RAM nunca hay
más de un tramo (≈ 330 KB) además del lote. La página se arma leyendo `ticks.bin` de
vuelta por bloques de 1 MB y se transmite con gzip en streaming a un temporal que se renombra al
terminar: si el render falla a medias, la página vigente queda intacta (ADR-VZ-10).

`events.bin` se describe en Eventos exactos.

### Escala de precio

`price_scale` es **fijo por activo** e igual al tick de la cotización; nunca se
elige por día. Se declara en el código (`PRICE_SCALE_BY_ASSET`) y se escribe en
`index.json`. El precio en la cotización es `p / price_scale`; con 100, el
máximo representable es 21 474 836,47.

| Activo | `price_scale` |
|---|---|
| `BTCUSDT` | 100 |

El precio sale del `DECIMAL(18, 8)` de L1 como entero exacto (×10⁸) y se redondea
al tick más cercano (mitad al par) antes de calcular el delta. Un día siempre se
escribe: los ticks fuera del tick solo dejan el hallazgo `price_rounded`
(`warning`/`pass`; `details`: `day`, `count`, `max_abs_delta_int`), donde `count`
son los ticks del día fuera del tick y `max_abs_delta_int` la mayor distancia de
uno de ellos a su tick más cercano, en enteros de L1 (×10⁻⁸).
`price_unrepresentable` (`error`/`fail`) queda como guarda para un precio mayor
que `INT32_MAX / price_scale`: `TicksAccumulator.finish` lanza
`PriceUnrepresentable` y el día no se escribe. Detalle en
[TRD-viz §7.3](TRD/viz.md#73-ticksbin-los-ticks-del-día-sin-reducir)
y [§9.3](TRD/viz.md#93-tipos-de-chequeo-check_type).

### Eventos exactos

`events.bin` lleva los eventos de **todos** los θ del día, para que el navegador
dibuje cada franja en su instante real y derive de ellos las confirmaciones de
cada píxel. Son los eventos de `events.parquet` que tocan el día (los mismos de
`thetas[].events`), más la cola pendiente del carry-over con su candidato. Sin
cabecera, little-endian, cuatro secciones consecutivas de `N` valores, con `N` el
total de eventos del día (la suma de `thetas[].events`); 13 bytes por evento:

| Sección | Tipo | Contenido |
|---|---|---|
| referencia | int32 | Milisegundos desde `t0` (`⌊µs / 1000⌋`) del tick de referencia; recortado a `[0, 86 400 000]` |
| confirmación | int32 | Lo mismo, del tick de confirmación (`confirm_time`) |
| extremo | int32 | Lo mismo, del tick extremo; en la cola pendiente, el candidato vigente |
| banderas | uint8 | Suma de bits de la tabla siguiente |

El θ en la posición `k` de `thetas` ocupa de `events_offset` a
`events_offset + events − 1` en **cada** sección, en orden de referencia. Los
tiempos de un θ no decrecen, y el extremo de un evento es la referencia del
siguiente. Un valor fuera del día se recorta al borde y se marca; el tick
extremo pertenece al evento que cierra, `(referencia, extremo]`, así que el evento
siguiente arranca en el tick que lo sigue. Un evento toca el día si su referencia
es anterior al último `agg_trade_id` del día y su extremo no es anterior al primero
(`first_agg_trade_id` y `last_agg_trade_id` del índice).

| Bit | Bandera |
|---|---|
| `1` | alza |
| `2` | provisional |
| `4` | referencia recortada |
| `8` | confirmación recortada |
| `16` | extremo recortado |

Sin el bit `1` el evento es una baja. `provisional` marca la cola pendiente cuyo
candidato todavía puede cambiar (`provisional_from_s` no es `null`); si el
pendiente ya cerró, o su candidato cae después del día, no lleva la bandera. Una
cola cuyo candidato queda antes del fin del día **no** lleva eventos después de
él: los ticks posteriores no tienen todavía un evento que los cierre.

### Campos de `index.json`

Todos son obligatorios y van en este orden. `write_day` los escribe y la
prueba del contrato compara esta tabla con `INDEX_FIELDS`.

| Campo | Tipo | Descripción |
|---|---|---|
| `tiles_version` | string | semver del formato de los archivos del día (no la de la imagen); cambia con cualquier modificación de esta sección |
| `provider` | string | Proveedor de los ticks |
| `market` | string | Mercado |
| `asset` | string | Activo |
| `day` | string | Día UTC, `YYYY-MM-DD` |
| `t0` | integer | Inicio del día UTC en µs desde la época: el origen del tiempo relativo de `ticks.bin` y `events.bin` |
| `price_scale` | integer | Unidades de precio de `ticks.bin` por unidad de la cotización (ver Escala de precio); el precio es `p / price_scale` |
| `ticks` | integer | Ticks del día: la suma de los ticks de todos los tramos de `ticks.bin` |
| `ticks_chunk` | integer | Tamaño máximo de un tramo de `ticks.bin`, en ticks (65 536): ningún tramo declara más |
| `first_agg_trade_id` | integer | `agg_trade_id` del primer tick del día |
| `last_agg_trade_id` | integer | `agg_trade_id` del último tick del día |
| `ticks_file` | string | Nombre del archivo de ticks, `ticks.bin` |
| `events` | string | Nombre del archivo de eventos exactos, `events.bin` |
| `page` | string | Nombre de la página autocontenida del día, `index.html` |
| `thetas` | array | Un objeto por θ con eventos en `events.bin`, en el orden de sus bloques: `theta` (texto de ancho fijo de la partición de L2, `0.00010000`, de menor a mayor), `events` (filas de eventos que tocan el día, con la cola), `events_offset` (índice de su primer evento en cada sección de `events.bin`) y `provisional_from_s` (segundos desde `t0` desde los que la cola es provisional, o `null`) |
| `missing_thetas` | array | θ del catálogo sin eventos completos: no tienen bloque en `events.bin` |
| `input_hash` | string | Huella de los archivos de entrada; la calcula quien llama a `write_day` |
| `content_hash` | string | SHA-256 de los archivos: por cada uno en orden de nombre (`events.bin`, `ticks.bin`), el nombre, un byte nulo y sus bytes. No depende de `generated_at` |
| `generated_at` | string | Momento de la escritura, UTC, `YYYY-MM-DDTHH:MM:SSZ`; lo único del índice que cambia entre dos corridas idénticas |
| `image_version` | string | Versión de la imagen que escribió el día |

`latest.json` lleva `tiles_version`, `provider`, `market`, `asset` y `day`: es
mono-activo porque vive en la raíz.
