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
uv run python -m l2_dc_events --mode backfill [--from 2020-01] [--to 2020-03] [--force] [--thetas 0.00010000,...] [--asset BTCUSDT]
uv run python -m l2_dc_events --mode monthly [--from 2020-04]
export L2_THETAS_URI=gs://<proyecto>-manifest/l2/thetas.yaml     # o --thetas-uri; sin él, la semilla
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
- **`--mode backfill`** recorre `[--from, --to]` **en orden y en un solo
  proceso**: cada mes lee el carry-over que el anterior acaba de escribir
  (TRD-L2 §8.2). Por eso corre en **una sola tarea**: con
  `CLOUD_RUN_TASK_INDEX` distinto de 0 termina con código 2 (el job de Cloud
  Run corre en una sola tarea, nunca como task array). **`--from`** es
  opcional y por defecto es `--series-start`: la frontera de cada θ decide
  desde dónde avanza. **`--to`** por defecto es el último mes con
  `consolidated.parquet` en L1 (nunca provisionales); si L1 no tiene ninguno,
  termina con código 1.
- **Frontera por θ (RF-L2-09).** Antes de procesar, la unidad calcula la
  frontera de cada θ del catálogo: el último mes de su cadena de carry-over,
  contigua desde `--series-start` y con un `carry_over.parquet` válido
  (existe, de la `state_version` de la imagen, con el esquema y el θ de su
  partición; si el último no lo es, retrocede al anterior). Sale de **un solo
  listado** de `L2_EVENTS_ROOT` más la lectura del carry-over de la frontera de
  cada θ, no de listar partición por partición. La corrida deja una línea
  `θ=<t> frontera=<YYYY-MM>` por θ. Un θ sin frontera arranca en
  `--series-start`.
- **Un mes se lee una sola vez para todos los θ que lo necesitan.** En cada
  mes el fan-out lleva solo los θ cuya frontera es anterior; un mes que ningún
  θ necesita se salta **sin leer L1** (línea de log `ningún θ la necesita`). Con
  el catálogo sin cambios y todo al día, la corrida termina en segundos y no
  escribe. Reprocesar es idempotente (RF-L2-08).
- **`--force`** ignora la frontera: reprocesa todo `[--from, --to]` para los θ
  elegidos. Exige `--from` (código 2 sin él): así un `--force` suelto no
  reprocesa toda la serie por accidente. **`--thetas`** (decimales como en la ruta de la partición, p. ej.
  `0.00010000,0.00031313`) acota la corrida a un subconjunto del catálogo; uno
  que no esté en el catálogo termina con código 2.
- **Un fallo detiene el rango.** Si el mes M falla (entrada ausente, carry-over
  que falta o de otra versión), los meses siguientes no se tocan y el proceso
  termina con código 1 y su hallazgo en el lago de DQ. Corregida la causa,
  relanzar el mismo comando retoma desde M.
- **`--mode monthly`** procesa un solo mes: `--from`, y sin él el mes anterior
  al actual en UTC, igual que `monthly-close` de L1, así que corre sin
  argumentos con `L2_SERIES_START` fijada. No admite `--to`. Lleva solo los θ
  cuya frontera es el mes previo. Un θ **rezagado** (agregado al catálogo sin
  backfill) no se procesa: deja el hallazgo `theta_behind_frontier` con su
  frontera y la unidad no falla mientras otros θ avancen. Si ningún θ está listo
  y hay rezagados, termina con código 1 sin leer L1 (por ejemplo, tras un
  `monthly` perdido); si todos ya tienen el mes, con código 0.
- Uso inválido (argumentos, raíces faltantes, `CLOUD_RUN_TASK_INDEX` ≠ 0,
  serie sin declarar, catálogo de θ inválido) termina con código 2. Una entrada ausente o un
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
   el `carry_over.parquet` del mes anterior de cada θ que el mes procesa (los
   que no han llegado a él, según su frontera). Si a alguno le falta, o es de
   otra `state_version`, emite un hallazgo por cada uno y aborta la unidad
   **sin escribir nada** (fail-closed, ADR-L2-08): los θ del mes comparten la
   lectura, así que uno atrasado los detiene a todos.
3. Lee el mes **row group por row group**, solo `price`, `transact_time` y
   `agg_trade_id` (`landing.py`). Cada lote va a los θ del mes como buffers de Arrow
   sin copia, en tramos de 65 536 ticks (`FEED_TICKS`), y se suelta antes de
   pedir el siguiente.
4. Los eventos que cada tramo cierra salen de `dc_pyo3` **ya en columnas**
   (`feed_batch_columns`, sin un objeto `Event` por evento) y van a un
   escritor abierto por θ (`events.py`), que los envuelven sin copia y los
   escriben en row groups de hasta 32 768 filas. Los escritores codifican
   **en paralelo** (`parallel.py`): un pool de hilos, un θ a la vez y en orden
   cada uno, y a lo más 2 tramos sin escribir (`MAX_CHUNKS_IN_FLIGHT`), así que
   nunca se acumula el mes de un θ ni la cola de escritura: la RAM es O(lote).
5. Al final cierra el grupo de empate abierto (`finish_columns`, RF-L2-12) y
   publica, en paralelo por θ: primero `events.parquet` y luego
   `carry_over.parquet`, cada uno con escritura atómica (temporal más
   `commit`). Un carry-over presente significa que el mes de ese θ quedó
   completo.
6. Emite los hallazgos de resumen y de tiempo por fase, y deja la línea `sonda:`.

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
| `carry_over_missing`, `carry_over_version_mismatch` | La compuerta aborta la unidad; uno por θ afectado |
| `theta_catalog_invalid` | El catálogo de θ no existe o incumple §7.3: código 2, sin escribir datos |
| `theta_behind_frontier` | `monthly` no procesó un θ rezagado (warning); solo falla la unidad (código 1) si ningún θ quedó listo |
| `theta_config_drift` | Informativo (`info`): el lago tiene un θ que el catálogo ya no incluye |
| `dc_zero_tick_discarded` | La guarda de §9.1 descartó eventos (esperado: cero, nunca se emite) |
| `events_summary` | Uno por θ al terminar: eventos escritos y `content_hash` de ambos archivos |
| `unit_timing` | Uno por mes al terminar: pared y tiempo por fase, límite efectivo de CPU y bytes leídos (ITSC-289) |

## El catálogo de θ

θ es un parámetro del experimento, no código (TRD-L2 §7.3). La fuente de verdad
en la nube es `gs://<proyecto>-manifest/l2/thetas.yaml` (`L2_THETAS_URI` o
`--thetas-uri`); sin ninguno de los dos, la CLI usa la **semilla**
[`src/l2_dc_events/config/thetas.yaml`](src/l2_dc_events/config/thetas.yaml)
(local y pruebas), con los enteros `round(θ × 10⁸)` de la regla log-espaciada de
ADR-L2-10. Los valores, no la fórmula, son la fuente de verdad, y
`tests/test_l2_thetas.py` verifica que la semilla sigue la regla (con 60 dígitos
de precisión, para que ningún redondeo de float decida un empate). La semilla
viaja dentro del paquete y se resuelve relativo al módulo, así que sirve igual
con `uv run` que en la imagen.

Antes de tocar nada, la CLI valida el catálogo: `scale: 100000000`, una lista de
enteros **únicos** y dentro de **[10⁻⁴, 5·10⁻²]** (`10000 ≤ θ ≤ 5000000`), en
cualquier orden. Si falla, termina con código 2 y deja el hallazgo
`theta_catalog_invalid` en `L2_DQ_ROOT`, sin escribir datos.

### Cómo agregar θ

Sin PR y sin reprocesar los θ que ya existen. La trazabilidad la dan la
partición `theta=<t>`, la columna `theta`, `image_version` y el versionado del
bucket.

1. **Solo la primera vez**, el objeto no existe: súbelo desde un clon del repo
   en Cloud Shell, a partir de la semilla:

   ```bash
   gcloud storage cp layers/l2_dc_events/src/l2_dc_events/config/thetas.yaml gs://<proyecto>-manifest/l2/thetas.yaml
   ```

   Desde entonces, baja el catálogo, agrega los θ como `round(θ × 10⁸)` (por
   ejemplo `θ = 0,0003` es `30000`) y súbelo de nuevo:

   ```bash
   gcloud storage cp gs://<proyecto>-manifest/l2/thetas.yaml thetas.yaml
   $EDITOR thetas.yaml
   gcloud storage cp thetas.yaml gs://<proyecto>-manifest/l2/thetas.yaml
   ```

2. Lanza el job `l2-backfill` **sin `--from` ni `--to`**. Los θ que ya llegaron
   al último mes cerrado de L1 se saltan; los nuevos recorren la serie desde
   `--series-start`, un mes a la vez y leyendo cada mes de L1 una sola vez.
3. Mientras el backfill no termina, `monthly` no avanza los θ nuevos: los deja
   en `theta_behind_frontier` (warning) y sigue con los demás.

**Quitar un θ no borra nada.** L2 deja de avanzarlo y cada corrida lo reporta
como `theta_config_drift` (`info`). Para forzar solo un subconjunto del
catálogo, `--thetas`; para reprocesarlo, `--force`.

**Escala:** los eventos los dominan los θ pequeños. Agregar θ grandes es casi
gratis; agregar θ diminutos multiplica el volumen de `dc-events`.

## Resultado: un mes real de punta a punta (ITSC-244)

Insumo de la sonda que cerró ADR-04 (ITSC-281; cierre en
[TRD-L2 §14.1](../../docs/TRD/l2.md#141-adr-04--cómputo-de-l2-cloud-run-jobs-cerrado):
Cloud Run Jobs con 2 vCPU y 4 GiB). Contenedor de desarrollo de 10 cores y 7 GiB; landing producida con L1 en local (`sandbox.local/itsc244/`).

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

### Dónde se va el tiempo

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
- **Lo que quedó después de la sonda de ITSC-281:** el paralelismo de escritura
  es el número de cores. Con un solo hilo de escritura (10 cores para el
  fan-out) la unidad tardó 12,2 s: el volcado a columnas ya no pasa por Python,
  y eso solo bajó la unidad a la mitad. Con 1 vCPU sigue sin medirse (TRD-L2
  §14.1, pendiente a). La sonda leyó `cores` de la máquina y no el límite del
  job, así que con 2 vCPU hubo 6 hilos de escritura sobre 2 vCPU; no invalida
  el veredicto, porque con 8 vCPU (9 hilos) tampoco aceleró.


## Tiempo por fase y límite de CPU (ITSC-289)

La sonda de ITSC-281 dio la pared del mes más pesado (2023-03) con 2, 4 y 8
vCPU (292, 262 y 301 s) pero no dónde se va. Desde la versión 0.5.1 la línea
`sonda:` y el hallazgo `unit_timing` (uno por mes, en `dq-findings`; contrato en
[`docs/data-contracts.md`](../../docs/data-contracts.md)) separan el tiempo por
fase. Solo instrumenta: no cambia cómo se lee, se detecta ni se escribe.

```
sonda: unit=binance/spot/BTCUSDT/2020-01 mode=backfill rss_peak_mib=254 wall_s=8.3 ticks=14051798 cores=10 cores_visible=10 cores_source=affinity ticks_s_core=169299 theta_ticks_s_core=8464939 read_s=0.0 decode_s=0.7 detect_s=1.8 write_s=38.2 carry_s=0.0 wait_s=5.7 other_s=0.1 row_groups=15 bytes_in=81066590 cpu_throttled_s=0.0
```

- **`cores` es el límite efectivo.** Cuota del cgroup (`cpu.max`, v2, o
  `cpu.cfs_quota_us`, v1; el menor de la jerarquía) o, sin cuota, los cores
  visibles (`sched_getaffinity`). `cores_visible` conserva el dato que la sonda
  reportaba antes, y `cores_source` dice quién fijó `cores`. `ticks_s_core` se
  calcula con el límite efectivo. Lo hace `cpu.py`.
- **Las fases del hilo principal son disjuntas y suman `wall_s`:** `carry_s`
  (cargar el carry-over), `read_s` + `decode_s` (leer la landing), `detect_s`
  (el fan-out) y `wait_s` (esperar a los escritores, publicación final
  incluida); lo que falta es `other_s`. **`write_s` no entra en esa suma:** es
  el tiempo que los θ pasan codificando y subiendo Parquet, sumado entre los
  hilos de escritura, y puede pasar de la pared (en el ejemplo, 38 s de
  escritura en 8 s de pared, sobre 10 hilos). Para saber si la escritura
  frena la unidad se mira `wait_s`. Lo define `timing.py`.
- **Cómo se separa `read_s` de `decode_s`.** No se envuelve el archivo:
  `ParquetFile` lee con `pre_buffer=True` (lecturas en paralelo en hilos de
  Arrow), y un archivo Python las serializaría y cambiaría lo que se mide. Con
  `use_threads=False` el hilo que lee decodifica, así que su CPU
  (`time.thread_time`) es `decode_s` y el resto de la pared de cada row group
  es `read_s`. Límite: si la cuota estrangula al proceso, esa espera también
  cae en `read_s`; `cpu_throttled_s` (`throttled_usec` de `cpu.stat`) permite
  descontarla.
- **`unit_timing` y no `events_summary`.** El resumen es por θ (50 filas por
  mes) y su `details` debe ser idéntico entre dos corridas del mes; agregarle
  tiempos lo rompería y repetiría 50 veces un dato de la unidad.

### Costo de la instrumentación

Medido en local sobre 2020-01 (10 cores, 14 051 798 ticks), 8 corridas
intercaladas de la versión anterior (A) y la nueva (B) para cancelar la deriva
de la máquina, que entre tandas seguidas llegó a 0,8 s:

| 2020-01 | Antes (0.5.0) | Después (0.5.1) |
|---|---|---|
| Pared, media de 8 | 8,20 s | 8,16 s |
| Pared, mínima | 8,0 s | 7,8 s |
| RSS pico, media | 247,4 MiB | 248,2 MiB |
| `content_hash` de `events.parquet` y `carry_over.parquet` (50 θ) | los de la línea base | idénticos |

La sobrecarga de pared queda por debajo del ruido entre corridas (la diferencia
es −0,5 %, con corridas de 7,8 a 8,6 s), así que cumple el tope de 1 %. El RSS
no crece más de lo que varía entre corridas (244 a 254 MiB).

### Corrida en la nube sobre 2023-03

Pendiente (la corre el humano, runbook
[`sonda-l2.md`](../../docs/runbooks/sonda-l2.md#corrida-por-fases-itsc-289)):
una corrida con `force=true` en la config vigente del stack `l2`. Su línea
`sonda:` y el párrafo con la fase dominante y la optimización que propone la
card siguiente van aquí. La corrida que tiene la tubería de ITSC-290 está en la
sección siguiente.

## Optimización de la tubería sin cambiar la salida (ITSC-290)

La corrida de 2023-03 con 0.5.1 en Cloud Run (2 vCPU) tardó 613 a 733 s, y la
misma imagen 0.5.0, 262 a 301 s. El laboratorio (cgroup v1, cuota 1,97, 4 cores
visibles, 100 M de ticks sintéticos) no reprodujo esa diferencia, pero encontró
cuatro cuellos que explican por qué más vCPU no aceleraba. Se atacaron en dos
bloques; ninguno cambia el contrato de datos.

**Bloque 1, sin cambiar ni un byte de salida:**

- **GIL retenido durante la detección.** `feed_batch_columns` no lo soltaba, así
  que los hilos de escritura de Parquet (que viven en Python) esperaban a que
  terminara cada tramo. Ahora `dc_pyo3` valida y detecta dentro de `py.detach`.
  Laboratorio: pared 103,3 a 84,5 s (−18 %), `write_s` 218 a 78 s, `wait_s`
  25 a 7 s.
- **Hilos del fan-out = `floor(cuota)`.** `thread::available_parallelism` divide
  cuota entre período con enteros: con cuota 1,97 daba 1 hilo (`detect_s` 41 s
  con cuota 2,00; 62 s con 1,97). Cloud Run entregó 1,61, 1,86 y 1,97 para
  2 vCPU, así que el detector corrió siempre con un hilo. Ahora `cpu.py` decide:
  `ceil(cuota)` acotado a los cores visibles. `FanOut` recibe `threads`
  explícito desde Python.
- **Escritores = cores visibles, no cuota.** Con 5 o 6 hilos de escritura sobre
  una cuota de 2, el cgroup se sobresuscribía y desalojaba al hilo principal,
  que alimenta el detector. Ahora son `max(1, round(cuota))`.
- **Lectura de GCS en serie.** 237 row groups a ~0,3 s de latencia son 70 s de
  `read_s` con la CPU ociosa. Ahora un hilo lector pide `L2_READ_AHEAD` row
  groups (2 por defecto) por delante del que se procesa. En RAM hay a lo más
  1 + k row groups, que es el crecimiento que permite RNF-L2-01: en 2020-01
  (~30 MiB por row group) el pico pasa de ~247 MiB a ~270 con k = 1 y ~305 con
  k = 2.
- `cpu.py` también lee, en cgroup v1, la cuota del cgroup del proceso (antes
  leía la raíz, sin cuota) y `throttled_time` de su `cpu.stat`.
- La sonda suma `detect_cpu_s` (CPU del hilo principal durante el fan-out,
  `thread_time`) y `fanout_threads`. Frente a `detect_s`, distingue un
  detector lento de uno desalojado. `decode_s` pasa a ser la CPU del hilo
  lector y `read_s`, la espera del principal: ya no se suman a `other_s`.

**Bloque 2, mismo contenido lógico, otros bytes:** `events.parquet` y
`carry_over.parquet` se escriben sin diccionario, con los enteros en
`DELTA_BINARY_PACKED` y los decimales en `PLAIN`
(`PartitionWriter(compact_encoding=True)`). El diccionario no comprime precios,
tiempos ni ids, que son de alta cardinalidad. Sobre 3,8 M de eventos reales y
un core, zstd 3: con diccionario 179,6 MiB y 6,49 s; sin él, 107,0 MiB y
1,28 s (−40 % de volumen, 5× para codificar, 3,5× para decodificar).
`content_hash` no cambia: hashea valores, no bytes. L1 no usa esta opción.

### Medido en local sobre 2020-01 (10 cores, 14 051 798 ticks)

| 2020-01 | 0.5.1 | 0.5.2 |
|---|---|---|
| Pared, 10 cores | 8,2 s | 3,8 s |
| Pared, 1 core (`taskset -c 0`) | n. d. | 12,5 s |
| `events.parquet` de los 50 θ | 442 MB | 258 MB |
| RSS pico (k = 2) | ~247 MiB | ~266-312 MiB |
| `content_hash` de `events.parquet` y `carry_over.parquet` (50 θ) | los de la línea base | idénticos, con 1, 4 y 10 cores y con `L2_READ_AHEAD` = 0, 1 y 2 |

El RSS sube lo que permite k: ~30 MiB por row group de lectura anticipada. La
cuota fraccionaria no se puede fijar en un contenedor sin cgroup propio, así que
la prueba de regresión (`test_l2_parallelism.py`) la simula con
`CpuLimit(1.97, ...)` y las pruebas de `cpu.py` cubren la lectura de la cuota.

### Corrida en la nube, antes y después de ITSC-290

Pendiente (la corre el humano, runbook
[`sonda-l2.md`](../../docs/runbooks/sonda-l2.md#corrida-antes-y-después-itsc-290)):
2023-03 con la config vigente del stack `l2`, antes (0.5.1: pared 613 a 733 s,
`detect_s` 390 a 457 s, `write_s` 896 a 1 556 s de hilo, `read_s` 67 a 81 s) y
después (0.5.2). Meta: pared ≤ 50 % de la línea base de 0.5.1 en el mismo host
(comparar `cores_visible`). Las dos líneas `sonda:` completas y el párrafo con
la fase dominante van aquí.

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

- `test_l2_thetas.py`: la semilla contra la regla del TRD y la validación del catálogo (único, entero, rango).
- `test_l2_catalog.py` (ITSC-285): catálogo inválido, un θ nuevo sobre un lago de 3 meses y 2 θ (solo el nuevo se procesa, los viejos no cambian), todo al día sin escribir, meses sin θ pendiente, hueco en la cadena, `--to` por defecto, `--thetas`, drift informativo y `monthly` con un θ rezagado.
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
  por frontera y `--force`, que un fallo detenga el rango, el índice de tarea distinto de 0,
  `monthly` por defecto, `L2_SERIES_START` y el fail-closed.
- `test_l2_cpu.py` y `test_l2_timing.py`: el límite efectivo de CPU (cuota del
  cgroup v1 y v2, jerarquía, sin cuota), los hilos y escritores que de él salen
  y la aritmética de `other_s`.
- `test_l2_parallelism.py` y `test_l2_encoding.py` (ITSC-290): los hashes de los
  50 θ no dependen de la cuota ni de la lectura anticipada, y la salida sin
  diccionario se lee igual con pyarrow, DuckDB y Polars.

## Acoplamiento con L1

Ninguno por código (RNF-14 del maestro): L2 no importa `l1_ingest`. El único
contrato es el Parquet de la landing
([`docs/data-contracts.md`](../../docs/data-contracts.md)); por eso `cli.py`
repite el mes por defecto de `monthly-close` de L1. El escritor atómico y el hash de
contenido (`PartitionWriter`, `ContentHasher`) vienen de
[`shared/pyutils`](../../shared/pyutils), no de `l1_ingest`.
