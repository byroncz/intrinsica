# l2_dc_events

Capa L2: eventos Directional Change de 50 θ sobre la landing de L1. Diseño en
[`docs/TRD/l2.md`](../../docs/TRD/l2.md). Lee el `consolidated.parquet` de un
mes, alimenta el fan-out de [`dc_pyo3`](../../shared/dc_pyo3/README.md) lote a
lote y escribe, por cada θ, los eventos del mes (`events.parquet`) y el estado
con que sigue el mes siguiente (`carry_over.parquet`). El contrato de la salida
está en [`docs/data-contracts.md`](../../docs/data-contracts.md) ("Salida
Parquet de L2"). CI construye su imagen (`Dockerfile`) y le corre el humo
(`smoke.sh`). En local corre con `uv run`.

## Uso en local

```bash
export L2_LANDING_ROOT=... L2_EVENTS_ROOT=... L2_DQ_ROOT=...   # local o gs://
export L2_SERIES_START=2020-01                                   # o --series-start
uv run python -m l2_dc_events --mode backfill --from 2020-01 [--to 2020-03] [--force] [--asset BTCUSDT]
uv run python -m l2_dc_events --mode monthly [--from 2020-04]
```

- `L2_LANDING_ROOT` es la raíz que escribió L1 (la ruta hive va debajo);
  `L2_EVENTS_ROOT`, dónde salen los eventos y el carry-over; `L2_DQ_ROOT`, el
  lago de hallazgos. Las tres son obligatorias.
- **`--series-start`** (o `L2_SERIES_START` si falta el flag; el flag gana) es
  el primer mes de la serie: el único que arranca sin carry-over previo. Se
  declara y no se infiere, porque "falta el carry-over" y "es el primer mes" se
  ven igual en disco y confundirlos corrompe la serie (ADR-L2-08). Sin flag ni
  variable, el proceso termina con código 2, en los dos modos. Al reanudar
  desde un mes intermedio sigue siendo el de la serie, no el de `--from`. Un
  rango que arranca antes de la serie termina con código 2.
- **`--mode backfill`** exige `--from` y procesa `[--from, --to]` (un solo mes
  si falta `--to`) **en orden y en un solo proceso**: cada mes lee el
  carry-over que el anterior acaba de escribir (TRD-L2 §8.2). Por eso corre en
  **una sola tarea**: con `CLOUD_RUN_TASK_INDEX` distinto de 0 termina con
  código 2 (el job de Cloud Run corre en una sola tarea, nunca como task
  array).
- **Reanudación (RF-L2-09).** El rango arranca en el primer mes de
  `[--from, --to]` al que le falte a algún θ un `carry_over.parquet` válido
  (existe, de la `state_version` de la imagen, con el esquema y el θ de su
  partición); los anteriores se saltan con una línea de log. Desde ahí procesa
  **todos** los meses en orden aunque ya tengan salida: así un mes escrito en
  frío por la sonda se reescribe encadenado. Reprocesar es idempotente
  (RF-L2-08). **`--force`** arranca en `--from` y reprocesa todo el rango. Si
  todos los meses tienen carry-over, no hace nada y sale con 0.
- **Un fallo detiene el rango.** Si el mes M falla (entrada ausente, carry-over
  que falta o de otra versión), los meses siguientes no se tocan y el proceso
  termina con código 1 y su hallazgo en el lago de DQ. Corregida la causa,
  relanzar el mismo comando retoma desde M.
- **`--mode monthly`** procesa un solo mes: `--from`, y sin él el mes anterior
  al actual en UTC, igual que `monthly-close` de L1, así que corre sin
  argumentos con `L2_SERIES_START` fijada. No admite `--to`.
- Uso inválido (argumentos, raíces faltantes, `CLOUD_RUN_TASK_INDEX` ≠ 0,
  serie sin declarar) termina con código 2. Una entrada ausente o un
  carry-over que falta o es de otra versión, con código 1.

Para producir la landing de un mes en local con L1 (los ZIP viven en RAM: un
mes de 2020 cabe holgado en los 7 GiB del contenedor):

```bash
export L1_LANDING_ROOT=$PWD/sandbox.local/x/landing L1_DQ_ROOT=$PWD/sandbox.local/x/dq L1_MANIFEST_ROOT=$PWD/sandbox.local/x/manifest
uv run python -m l1_ingest --mode backfill --from 2020-01
export L2_LANDING_ROOT=$L1_LANDING_ROOT L2_EVENTS_ROOT=$PWD/sandbox.local/x/events L2_DQ_ROOT=$PWD/sandbox.local/x/dq_l2
uv run python -m l2_dc_events --mode backfill --from 2020-01 --series-start 2020-01
```

## Qué hace una unidad

1. Abre `consolidated.parquet` del mes. Solo ese archivo (ADR-L2-09): si falta
   o solo hay `provisional-day=DD.parquet`, emite `input_missing` o
   `input_provisional_only` y falla.
2. **Compuerta de carry-over.** Si el mes no es el primero de la serie, carga
   el `carry_over.parquet` del mes anterior de cada uno de los 50 θ. Si a
   alguno le falta, o es de otra `state_version`, emite un hallazgo por cada
   uno y aborta la unidad **sin escribir nada** (fail-closed, ADR-L2-08): los
   50 θ comparten la lectura del mes, así que uno atrasado los detiene a todos.
3. Lee el mes **row group por row group**, solo `price`, `transact_time` y
   `agg_trade_id` (`landing.py`). Cada lote va a los 50 θ como buffers de Arrow
   sin copia, en tramos de 65 536 ticks (`FEED_TICKS`), y se suelta antes de
   pedir el siguiente.
4. Los eventos que cada tramo cierra salen de `dc_pyo3` **ya en columnas**
   (`feed_batch_columns`, sin un objeto `Event` por evento) y van a 50
   escritores abiertos (`events.py`), que los envuelven sin copia y los
   escriben en row groups de hasta 32 768 filas. Los 50 escritores codifican
   **en paralelo** (`parallel.py`): un pool de hilos, un θ a la vez y en orden
   cada uno, y a lo más 2 tramos sin escribir (`MAX_CHUNKS_IN_FLIGHT`), así que
   nunca se acumula el mes de un θ ni la cola de escritura: la RAM es O(lote).
5. Al final cierra el grupo de empate abierto (`finish_columns`, RF-L2-12) y
   publica, en paralelo por θ: primero `events.parquet` y luego
   `carry_over.parquet`, cada uno con escritura atómica (temporal más
   `commit`). Un carry-over presente significa que el mes de ese θ quedó
   completo.
6. Emite los hallazgos de resumen y deja la línea `sonda:`.

**Re-ejecutar un mes** da los mismos archivos: la entrada es la misma y nada
de la salida depende de la ejecución (ni el `run_id` ni la hora). Si el
proceso muere a medias solo quedan temporales `.<nombre>.<uuid>.tmp`, que los
lectores ignoran; la corrida siguiente publica sobre los destinos.

Un evento se escribe en el mes que confirma el **evento siguiente**, el que
conoce su extremo (ADR-L2-06). Por eso cada mes empieza completando el evento
pendiente del carry-over anterior, y `events.parquet` de un mes cerrado nunca
trae un evento sin extremo: el que sigue abierto al cierre va al carry-over.

### Hallazgos de DQ

Se emiten con [`shared/dq`](../../shared/dq/README.md) a `L2_DQ_ROOT`, con
`layer = l2` y `stage = canonical`. Catálogo completo en
[`docs/data-contracts.md`](../../docs/data-contracts.md) ("Catálogo de
hallazgos"):

| `check_type` | Cuándo |
|---|---|
| `input_missing`, `input_provisional_only` | El mes no tiene consolidado |
| `carry_over_missing`, `carry_over_version_mismatch`, `theta_config_drift` | La compuerta aborta la unidad; uno por θ afectado |
| `dc_zero_tick_discarded` | La guarda de §9.1 descartó eventos (esperado: cero, nunca se emite) |
| `events_summary` | Uno por θ al terminar: eventos escritos y `content_hash` de ambos archivos |

## Los 50 θ

[`src/l2_dc_events/config/thetas.yaml`](src/l2_dc_events/config/thetas.yaml): los enteros `round(θ × 10⁸)` de la
regla log-espaciada de ADR-L2-10. Los valores, no la fórmula, son la fuente de
verdad, y `tests/test_l2_thetas.py` verifica que siguen la regla (con 60
dígitos de precisión, para que ningún redondeo de float decida un empate). El
archivo viaja dentro del paquete y se resuelve relativo al módulo, así que
sirve igual con `uv run` que en la imagen, donde queda horneado (TRD-L2 §11).

## Resultado: un mes real de punta a punta (ITSC-244)

Insumo para E4a (la sonda que cierra ADR-04). Contenedor de desarrollo de 10
cores y 7 GiB; landing producida con L1 en local (`sandbox.local/itsc244/`).

| | 2020-01 (corrida 1) | 2020-01 (corrida 2) | 2020-02 (`monthly`, con carry-over de enero) |
|---|---|---|---|
| Ticks (filas de la landing) | 14 051 798 | 14 051 798 | 16 955 884 |
| Row groups | 15 | 15 | 18 |
| Eventos, 50 θ | 12 154 706 | 12 154 706 | 12 042 944 |
| RSS pico | 263 MiB | 260 MiB | 272 MiB |
| Tiempo de pared | 24,4 s | 24,1 s | 24,6 s |
| Ticks/s por core | 57 589 | 58 306 | 68 926 |
| θ·ticks/s por core | 2,88 M | 2,92 M | 3,45 M |

Línea sonda de la corrida 1:

```
sonda: unit=binance/spot/BTCUSDT/2020-01 mode=backfill rss_peak_mib=263 wall_s=24.4 ticks=14051798 cores=10 ticks_s_core=57589 theta_ticks_s_core=2879467
```

- **Eventos por θ (enero):** θ = 0,01 % (`theta=0.00010000`) cierra 2 080 306;
  θ ≈ 0,85 % (`0.00846907`) cierra 407; θ = 5 % (`0.05000000`), 5. Mínimo 5,
  máximo 2 080 306.
- **`content_hash` estable:** los 50 θ dan el mismo hash de `events.parquet` y
  de `carry_over.parquet` en las dos corridas de enero (sobre raíces de salida
  separadas). Es el que queda en cada hallazgo `events_summary`.
- **Parquet válido:** los 12,15 M de filas se releyeron con pyarrow. En los 50
  archivos `confirm_time` crece estrictamente, `extreme_*` nunca queda antes de
  `confirm_*`, y el extremo de cada evento es la referencia del siguiente.
- **La cadena entre meses cierra:** en los 50 θ, el primer evento de febrero
  es exactamente el evento pendiente del carry-over de enero, y el extremo del
  último evento de enero es su referencia.
- **Tamaño:** 421 MiB de `events.parquet` por mes (los 50 θ) frente a 159 MiB
  del `consolidated.parquet` de entrada.
- **RAM:** el pico es ~260 MiB para un mes de 14 M ticks, de los que ~60 MiB son
  el intérprete con pyarrow y `dc_pyo3` importados. Para comparar, L1 pica en
  1,4 GiB para el mismo mes. El pico no crece de enero a febrero (272 MiB con
  un 20 % más de ticks), que es lo que pide O(lote).

### Dónde se va el tiempo (para E4a)

El detector no es el cuello. Perfilado con `cProfile` sobre 2020-01 en la
versión de ITSC-244 (los tiempos absolutos se inflan por el perfilador; la
proporción es la que sirve; se repitió en ITSC-275 con el mismo resultado):

| Paso | Tiempo aproximado |
|---|---|
| `write_table` (Parquet ZSTD-3 de 12 M de filas) | ~8 s |
| Volcar los objetos `Event` a columnas (`EventColumns.extend`) | ~12 s |
| `feed_batch`: los 50 θ, multihilo | ~1,6 s |
| Leer los row groups | ~0,6 s |
| Hash de contenido | ~0,6 s |

Los 50 θ sobre 14 M ticks cuestan ~1,6 s de cómputo. El resto era trabajo
serial de Python: materializar 12 M de eventos como objetos y codificarlos a
Parquet, con casi todos los demás cores ociosos. Por eso los ticks/s por core de
arriba son los de la unidad entera y no los 2,8 M del benchmark de
[`dc_core`](../../shared/dc_core/README.md), que mide solo el detector.

## Antes y después de sacar el trabajo serial de Python (ITSC-275)

Dos cambios: `dc_pyo3` entrega los eventos en columnas
([`dc_pyo3`](../../shared/dc_pyo3/README.md#eventos-en-columnas-feed_batch_columns-finish_columns))
y los 50 escritores codifican en paralelo. Misma máquina (10 cores, 7 GiB) y
misma landing de 2020-01, con la línea sonda de la corrida:

| 2020-01 | Antes (ITSC-244) | Después (ITSC-275) |
|---|---|---|
| Tiempo de pared | 24,9 s (ITSC-244: 24,4 s) | 7,7 s (5 corridas: 7,7 a 8,6 s) |
| Ticks/s por core | 56 433 (57 589) | 182 491 |
| θ·ticks/s por core | 2,82 M | 9,12 M |
| RSS pico | 261 MiB (263) | 247-250 MiB |

```
antes:   sonda: ... rss_peak_mib=261 wall_s=24.9 ticks=14051798 cores=10 ticks_s_core=56433 theta_ticks_s_core=2821646
después: sonda: ... rss_peak_mib=249 wall_s=7.7 ticks=14051798 cores=10 ticks_s_core=182491 theta_ticks_s_core=9124544
```

2020-02 (`monthly`, con el carry-over de enero): 9,2 s y 285 MiB, frente a
24,6 s y 272 MiB de ITSC-244. Es el único punto donde el pico sube (+13 MiB, un
mes 20 % más grande): sigue sin crecer con los ticks, que es lo que pide O(lote),
pero no queda por debajo de la base como enero.

- **Nada de la salida cambió.** Los 50 θ dan el mismo `content_hash` de
  `events.parquet` y de `carry_over.parquet` que la línea base de ITSC-244, en
  enero y en febrero, y los archivos de enero son idénticos byte a byte a los de
  la versión anterior: mismos row groups, mismas filas.
- **Dónde quedó el tiempo:** el hilo principal ya solo lee, calcula el fan-out
  (~1,5 s) y espera a los escritores. El Parquet se codifica en los 10 cores en
  vez de en uno.
- **El tope de tramos es un intercambio entre pared y RAM**, medido en enero
  (`MAX_CHUNKS_IN_FLIGHT`, ver `parallel.py`): 1 tramo, ~10 s y ~238 MiB; 2,
  ~8 s y ~250 MiB; 3, ~7,1 s y ~265 MiB; 4, ~6,7 s y ~273 MiB. Se dejó en 2, el
  mayor que no sube el pico de la base (261 MiB).
- **El allocator importa tanto como el paralelismo.** Con los hilos, el pico se
  fue a ~580 MiB con solo ~70 MiB vivos en Arrow: el pool `mimalloc` de Arrow y
  el umbral dinámico de `mmap` de glibc retenían lo liberado. `memory.py` y
  `ARROW_DEFAULT_MEMORY_POOL=system` (fijada en `l2_dc_events/__init__.py`)
  lo resuelven, y la imagen fija las dos variables como `ENV`
  (`ARROW_DEFAULT_MEMORY_POOL=system`, `MALLOC_MMAP_THRESHOLD_=16384`). El
  detalle y las medidas están en el docstring de `memory.py`.
- **Lo que queda para E4a:** el paralelismo de escritura es el número de
  cores. Con un solo hilo de escritura (10 cores para el fan-out) la unidad
  tardó 12,2 s: el volcado a columnas ya no pasa por Python, y eso solo bajó
  la unidad a la mitad. Con 1 vCPU no está medido; conviene que la sonda de
  E4a lo mida.

## Imagen

`Dockerfile` (contexto: la raíz del workspace) tiene dos etapas. La de
compilación instala la toolchain de `rust-toolchain.toml` y corre
`uv sync --locked --package l2_dc_events --no-dev --no-editable`, que compila
`dc_pyo3` con maturin; la final es `python:3.14-slim` con solo `/app/.venv`,
sin cargo ni rustc. El `thetas.yaml` va dentro del paquete instalado.

```bash
docker build -f layers/l2_dc_events/Dockerfile -t l2_dc_events:dev .
layers/l2_dc_events/smoke.sh l2_dc_events:dev
```

`smoke.sh` convierte los 4 735 ticks de 2017-08-18
(`shared/dc_core/tests/fixtures/ticks.csv`) en un `consolidated.parquet` con el
esquema de L1, usando el pyarrow de la imagen, y corre el backfill de 2017-08.
Verifica los 50 `events.parquet` y `carry_over.parquet`, un Parquet en el lago
de DQ y la línea con `content_hash`.

## Pruebas

`uv run pytest layers/l2_dc_events`:

- `test_l2_thetas.py`: los θ contra la regla del TRD.
- `test_l2_landing.py`: el lector (un lote por row group, sin materializar la
  tabla).
- `test_l2_pipeline.py`: los eventos a través del lector y el binding contra
  el fixture de la v0 de `dc_core` (idénticos con θ = 2 %; con los demás, hasta
  la primera divergencia declarada de ADR-L2-04), y la RAM de Arrow por lote.
- `test_l2_output.py`: la partición de eventos (ruta hive, esquema, ZSTD,
  estadísticas, orden), el `content_hash` estable entre corridas y entre
  tamaños de row group, que dos meses encadenados por Parquet den lo mismo que
  un mes entero, el carry-over con y sin evento pendiente, que una falla no
  deje archivos, y cada hallazgo de DQ.
- `test_l2_parallel.py`: que cada θ reciba sus bloques en orden aunque los
  demás corran en paralelo, que un error del escritor llegue al hilo principal y
  detenga las escrituras, el tope de tramos en vuelo y que `to_batch` envuelva
  los buffers del binding sin cambiar los valores.
- `test_l2_output_contract_doc.py`: que las tablas de `docs/data-contracts.md`
  coincidan con `EVENTS_SCHEMA` y `CARRY_OVER_SCHEMA`.
- `test_l2_cli.py`: el CLI: el rango encadenado en un proceso, la reanudación
  y `--force`, que un fallo detenga el rango, el índice de tarea distinto de 0,
  `monthly` por defecto, `L2_SERIES_START` y el fail-closed.

## Acoplamiento con L1

Ninguno por código (RNF-14 del maestro): L2 no importa `l1_ingest`. El único
contrato es el Parquet de la landing
([`docs/data-contracts.md`](../../docs/data-contracts.md)); por eso `cli.py`
repite el mes por defecto de `monthly-close` de L1. El escritor atómico y el hash de
contenido (`PartitionWriter`, `ContentHasher`) vienen de
[`shared/pyutils`](../../shared/pyutils), no de `l1_ingest`.
