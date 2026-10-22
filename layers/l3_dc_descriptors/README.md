# l3_dc_descriptors

Capa L3: describe lo que pasó entre las fronteras de cada evento DC. Diseño en
[`docs/TRD/l3.md`](../../docs/TRD/l3.md); contrato de la salida en
[`docs/data-contracts.md`](../../docs/data-contracts.md) ("Salida Parquet de
L3"). El paquete tiene dos partes: el **núcleo del resumidor** (`summarize_month`: de
`consolidated.parquet` de L1 y `events.parquet` de L2 a `summaries.parquet`, un mes y una
lista de θ por llamada) y la **imagen** (`python -m l3_dc_descriptors`), que recorre la
cadena de meses por θ con ese núcleo, lleva la frontera de cada θ y convierte sus
errores en hallazgos de DQ.

## Imagen y CLI

```sh
python -m l3_dc_descriptors --mode backfill --series-start 2020-01      # lo que falte
python -m l3_dc_descriptors --mode backfill --from 2023-03 --to 2023-03 --thetas 0.00010000
python -m l3_dc_descriptors --mode backfill --force --from 2023-03     # reprocesa
python -m l3_dc_descriptors --mode monthly --series-start 2020-01      # el mes anterior
```

Un solo proceso, un mes a la vez, y en cada mes el fan-out de los θ que lo necesitan. Un mes
que ningún θ necesita se salta sin leer L1.

| Opción | Qué hace |
|---|---|
| `--mode backfill\|monthly` | `backfill` recorre `--from`..`--to`; `monthly` es el backfill de un mes ([TRD-L3 §8.3](../../docs/TRD/l3.md#83-modo-monthly-incremental)) |
| `--from`, `--to` (`YYYY-MM`) | `backfill`: sin `--from`, la frontera de cada θ decide desde dónde avanza (el piso es `--series-start`); sin `--to`, el último mes con `consolidated.parquet` en L1. `monthly`: `--from` o el mes anterior al actual (UTC); no admite `--to` |
| `--series-start` / `L3_SERIES_START` | Primer mes de la serie de L2, el único sin mes anterior. Se declara, no se infiere |
| `--thetas` | Acota el catálogo, en decimal como en la ruta (`0.00010000,0.00031313`) |
| `--force` | Reprocesa el rango ignorando la frontera. Exige `--from`; solo en `backfill` |
| `--family` | `summaries` (la predeterminada y única por ahora); otro valor es código 2 |
| `--thetas-uri` / `L3_THETAS_URI` | **Obligatorio.** El catálogo de θ: el mismo objeto `l2/thetas.yaml` que usa L2 (ruta local o `gs://`). L3 no lleva copia |
| `--asset` | Activo; por defecto `BTCUSDT` |

Entorno: `L3_LANDING_ROOT` (L1), `L3_EVENTS_ROOT` (L2), `L3_SUMMARIES_ROOT` (L3) y
`L3_DQ_ROOT` (lago de hallazgos), además de `L3_SERIES_START` y `L3_THETAS_URI`. Opcionales,
enteros ≥ 1 (otro valor es código 2): `L3_READERS` (lectores de L1, R; por defecto 2),
`L3_INFLIGHT` (row groups en el motor, K; 4), `L3_WRITERS` (escritores, W; 12),
`L3_ENGINE_THREADS` (hilos del motor; por defecto `round(vCPU efectivos)`) y `L3_THREADS` (pool de
E/S; por defecto `ceil(3 × vCPU)`, nunca más que θ), más `L3_PROFILE=1`. El stack `l3` no fija
ninguna, así que los valores siguen a los vCPU del job; para sobrescribirlas en una corrida, ver el
[runbook](../../docs/runbooks/operacion-l3.md#hilos-y-topes-de-la-unidad-l3_readers-l3_inflight-l3_writers-l3_engine_threads-y-l3_threads).
**Memoria de los escritores (ITSC-365):** el orquestador no empuja más row groups mientras haya
`PENDING_PER_WRITER × W` row groups de salida sellados sin escribir (3 × 12 = 36 por defecto);
cada uno pesa ≈ 8 MiB (256 B por evento × ≈ 34 000 eventos), así que el tope son ≈ 290 MiB, lo
mismo que 8 × 4 de antes. Con 8 y W = 12 serían 96 (≈ 770 MiB): es lo que T4b pagó de más
(RSS 2 331 MiB). Si subes W o bajas el tamaño del row group, rehaz la cuenta y compárala con el
margen del job: con el asignador de la imagen (ver «Memoria») el RSS medido es ≈ 3,1 GiB de 4.
`L3_PREFETCH` ya no existe (ITSC-363). `CLOUD_RUN_TASK_INDEX` distinto de 0 es código 2. `IMAGE_VERSION` la fija la imagen.

**Códigos de salida** ([§8.4](../../docs/TRD/l3.md#84-códigos-de-salida)): 0 éxito; 1 fallo
de la unidad (entrada de L1 o L2 ausente, L2 incoherente, esqueleto, conciliación) o `monthly`
sin ningún θ listo y con rezagados; 2 uso o configuración (no escribe nada).

**Frontera** ([§8.2](../../docs/TRD/l3.md#82-modo-backfill-histórico)): por θ, el último mes de
su serie contigua desde `--series-start` cuyo `summaries.parquet` es válido (esquema, `theta`,
`state_version` y filas por row group iguales a `events.parquet` de L2 del mes). Sale de un
listado de la raíz de L3 más el pie del archivo de cada θ; no se consulta cada partición. Un
mes sin `events.parquet` y `carry_over.parquet` de L2 detiene el rango de ese θ con
`l2_input_missing` (los demás siguen, la corrida termina con código 1). En `monthly`, un θ
que no llegó al mes previo (`l3_frontier`) o al que L2 no le ha llegado (`l2_missing`) deja
`theta_behind_frontier` y no falla la unidad mientras otro avance.

**Hallazgos**: el catálogo de `check_type` y sus `details` están en
[`docs/data-contracts.md`](../../docs/data-contracts.md). Cada excepción tipada del núcleo
(tabla de abajo) se traduce a su hallazgo con los `details` que ya trae. `unit_timing` lleva lo
que necesita la medición del umbral 3 de [§10.3](../../docs/TRD/l3.md#103-sonda-del-lector-de-tramas--umbrales-de-aceptación-se-escriben-antes-de-medir):
pared total y por fase (todas son pared: `gate_s`, abrir `events.parquet` de los θ y arrancar la lectura de sus fronteras; `setup_s`,
el motor, el pie de L1 y la compuerta de carry-over; `aggregate_s` del motor, `publish_s`, `reconcile_s`; `lineage_s`
ya no existe: el linaje sale de los bytes que se leen; `read_s` es lo que el hilo principal esperó a las lecturas y
`read_wall_s` y `write_s` la pared con alguna lectura o algún escritor ocupado), la configuración
(`threads`, `engine_threads`, `readers`, `inflight`, `writers`), lo que hizo el motor
(`inflight_max`, `engine_busy`, `items`), `read_bytes`, `boundaries_peak_mib`,
`rss_peak_mib` (`VmHWM`, reiniciado al empezar la unidad), `row_groups_month` y
`row_groups_previous` (decodificaciones de L1 del mes y de meses anteriores, contadas con un
gancho sobre `dc_frames.lake.open_parquet` como hace la sonda de `ops_tools`),
`row_groups_repeated` (debe ser 0), eventos y ticks por θ y en total, el límite efectivo de
CPU y la CPU que consumió la unidad (`cpu_user_s`, `cpu_system_s`, `cpu_s` y
`cpu_utilization`, de `getrusage`, con `minor_faults`, `major_faults`, `voluntary_switches` e
`involuntary_switches`, que explican la CPU de sistema). Con `L3_PROFILE=1` la unidad corre bajo `cProfile` y suma el
hallazgo `unit_profile` (las 40 funciones más caras por tiempo propio y acumulado); sin la
variable no cambia nada. Cómo lanzarlo: [runbook](../../docs/runbooks/operacion-l3.md#perfilar-una-unidad).

## Orquestador sobre el motor nativo (ITSC-363)

La CLI procesa los meses en serie y `summarize_month` es el **orquestador** de cada mes
([ADR-L3-16](../../docs/TRD/l3.md), [TRD-L3 §8.1](../../docs/TRD/l3.md#81-núcleo-del-resumidor-común-a-los-dos-modos)):
Python lee, escribe y mide; **no calcula**. La aritmética, las posiciones de las fronteras y los
invariantes de §7.2 viven en el motor de Rust (`shared/dc_descriptors`, `dc_pyo3.Engine`), que
tiene sus propios hilos. Python orquesta, Arrow transporta, Rust calcula.

```text
 L2 (50 events.parquet) ─ pool de E/S ─► BoundaryFeed ─┐ fronteras, por delante de cada row group
 L1 (consolidated) ─ L1.row_groups(readers = R) ─► push ┤
                                                        ▼
                                          motor nativo (hilos, ítems (row group, θ))
                                                        │ pop_sealed
                                      Collector ────────┴─► SealedWriters (W hilos, cola por θ)
                                                                  ▼
                                                      FamilyWriter de cada θ ─► commit
```

| Módulo | Qué hace |
|---|---|
| `summarize.py` | `summarize_month`: la compuerta, la apertura, el bucle de `push` con contrapresión, `finish`, la conciliación y la publicación |
| `boundaries.py` | `BatchStream`: los lotes de eventos de L2 de un θ, dos por delante, leídos en el pool de E/S |
| `outputs.py` | `Collector` (vacía `pop_sealed`) y `SealedWriters` (pool de W escritores con una cola por θ y su «siguiente esperado») |
| `native.py` | El adaptador del motor: `BoundaryFeed`, `push`, `to_batch`, `to_summarize_error` |
| `timing.py` | `Phases`, `BusyClock` (pared de una fase concurrente), `ReadMeter`, `Retained` |

Qué sostiene la corrida:

- **Cuatro topes, no cuatro colas sin fondo.** R lectores de L1 traen row groups por delante; el
  orquestador no pide el siguiente mientras el motor tenga K en vuelo (`engine.inflight()`) ni
  haya `8 × W` row groups sellados sin escribir. En RAM hay a lo más `K + R + 1` row groups de L1
  (≈ 48 MB cada uno), las fronteras de L2 de los eventos de esos row groups, los buffers de salida
  abiertos y los sellados por escribir.
- **Las fronteras de L2 van por delante, y no más.** Antes de cada `push`, `cover(high)` entrega
  lotes hasta que el último evento entregado empieza después del último id del row group (lo que
  `Engine.push` exige). Cada θ abre su `events.parquet` una vez (`dc_frames.lake.EventFile`) y lo lee
  por tramos de ≈ 8 MiB comprimidos en pocas peticiones, no una por row group.
- **Cada θ escribe en orden.** El motor ya entrega cada θ en orden de row group de salida; la cola
  de `SealedWriters` lo garantiza aunque llegaran al revés y nunca atiende a dos escritores del
  mismo θ a la vez, porque el escritor de familias no lo admite.
- **Sin estado global.** Ningún módulo guarda un contenedor mutable ni usa `global`
  (`tests/test_l3_parallel.py` lo comprueba). `pipeline.process_unit(unit, ctx, thetas)` recibe
  datos inmutables (`Unit`, `RunContext`) y devuelve un `Result`; `Phases`, `RowGroupCount` y
  `Timing` los crea la unidad. El único punto de proceso es el gancho de `open_parquet`: se
  instala una vez, delega en el original y cuenta en un `ContextVar`.
- **Errores.** Un error del motor (`DescriptorError`), de un escritor o de la lectura aborta la
  unidad sin publicar; los hilos terminan antes de que se cierren los escritores y se borren los
  temporales. El motor valida solo los ticks que alguna fase toca: una cantidad negativa que
  ninguna fase toca (p. ej. el primer tick del mes) lo acepta, la referencia de Python la
  rechazaba (TRD-L3, ADR-L3-16, costo asumido 3).
- **Pruebas.** `tests/test_l3_parallel.py`: las 24 combinaciones de hilos del motor (1, 2, 4),
  lectores (1, 2), row groups en vuelo (1, 4) y escritores (1, 4), con las fronteras en un lote o
  en lotes de 3 eventos, dan exactamente la tabla, el `content_hash` y la conciliación de la
  referencia de Python (`tests/pyref.py`); más un lago con row groups de L1 repartidos a mano, la
  cola por θ, la contrapresión y los errores. No mide velocidad.
- **Qué mide `unit_timing`.** Todas las fases son pared. `engine_busy` es la fracción del tiempo
  con ítems en la cola del motor; si `cpu_utilization` no llega a 0,95, se explica con ella, con
  `read_s` y con `write_s` ([runbook](../../docs/runbooks/operacion-l3.md)). El perfil
  (`L3_PROFILE=1`) ve las llamadas de todos los hilos de Python (≥ 3.12, `sys.monitoring`), no las
  del motor: sus tiempos son de hilo, no de pared.

## Motor nativo de agregación (ITSC-358)

`native.py` adapta el motor de Rust (`shared/dc_descriptors`, expuesto por `dc_pyo3.Engine`):
`push_boundaries` entrega por lotes las columnas de L2 de los eventos de cada θ (Arrow, sin copia;
el índice de un evento es su casilla) y `BoundaryFeed` los entrega por delante del row group que los
toca, y no más (el motor suelta cada lote al cerrarse sus eventos); `push` empuja los buffers de
Arrow de cada row group sin copia, con la columna `agg_trade_id`, y el motor calcula él mismo dónde
cae cada frontera (ADR-L3-16); `to_batch` envuelve en Arrow los row groups de salida que el motor
sella y `to_summarize_error` convierte sus errores en las excepciones tipadas de `errors`, con el
mismo `check_type` y los mismos `details` (un evento que queda sin cerrar al terminar L1 es el
`l2_state_inconsistent` de «L1 terminó antes de cerrar todos los eventos»). Está conectado a
`summarize_month` desde ITSC-363. La referencia con la que se compara es la agregación de Python por
row group, que vive en `tests/` (`rowgroup.py` y `pyref.py`, el `summarize_month` anterior en
serie): `tests/test_l3_native.py` comprueba que el motor, con 1 y 4 hilos, en cualquier orden de
`push` y con las fronteras en lotes de pocos eventos intercalados con los `push`, da las mismas
tablas, el mismo `content_hash` y la misma conciliación, y `tests/test_l3_native_memory.py` que la
RSS que el motor añade no crece con los ticks del mes. `test_l3_native.py` comprueba también que el
motor suelta un lote de fronteras al cerrarse sus eventos, y compara además las posiciones del
motor con las de `read_row_groups`, row group a row group. El diseño y por qué es nativo están en
[TRD-L3 ADR-L3-16](../../docs/TRD/l3.md).

## Uso en local del núcleo

```python
from l3_dc_descriptors import summarize_month

results = summarize_month(
    "<raíz de L1>",
    "<raíz de L2>",
    "<raíz de L3>",
    (2020, 3),  # el mes de la partición de L2
    ["0.00010000", "0.00031313"],  # θ, con la escala de la partición
    series_start=(2020, 1),  # primer mes de la serie de L2
)
for theta, result in results.items():
    print(theta, result.rows, result.content_hash, result.reconciliation)
```

- Las raíces son rutas locales o `gs://`. `provider`, `market` y `asset` son
  opcionales (`binance`, `spot`, `BTCUSDT`).
- `series_start` es el primer mes de la serie de L2: el único que no tiene mes
  anterior. Se declara y no se infiere, como en L2. Un mes anterior a él es un
  `ValueError`.
- Devuelve, por θ, un `ThetaResult`: la ruta del archivo, las filas, el
  `content_hash` y la conciliación de ticks (`Reconciliation`).
- Pruebas: `uv run pytest layers/l3_dc_descriptors`.

## Qué lee y qué escribe

| Entrada | Qué usa |
|---|---|
| L1 `consolidated.parquet` del mes, y de los meses anteriores que algún evento necesite | `agg_trade_id`, `transact_time`, `price`, `quantity`, `is_buyer_maker`, a través del lector de tramas (`shared/dc_frames`) |
| L2 `events.parquet` del mes y θ | Las siete columnas del contrato: `reference_agg_trade_id`, `confirm_agg_trade_id`, `confirm_time`, `extreme_agg_trade_id`, `direction`, `confirm_price`, `extreme_price` |
| L2 `carry_over.parquet` del mes y θ | `direction`, `has_pending_event`, `pending_reference_agg_trade_id`, `ext_high_agg_trade_id`, `ext_low_agg_trade_id`: solo para conciliar los ticks y comprobar la coherencia de L2 |
| L2 `carry_over.parquet` del mes anterior | Solo `direction` |

Escribe `summaries.parquet` en la partición de L2 de cada θ
(`<raíz de L3>/provider=…/theta=<t>/year=YYYY/month=MM/`). No guarda estado entre
meses: el evento que cruza meses lo resuelve el lector, que arranca la pasada en el row
group de L1 más antiguo que algún evento del mes necesita, y el acumulador parcial.

## Mecánica de una unidad

1. **Apertura** (`gate_s`). En el pool de E/S, una tarea por θ: abrir `events.parquet`
   (`EventFile`: archivo y pie una sola vez; si falta, `l2_missing`), comprobar sus tipos y
   arrancar la lectura anticipada de sus fronteras (las 11 columnas, de las que el motor recibe
   las siete de §7.1, `EVENT_COLUMNS`). Con las estadísticas de los pies se sabe la primera
   referencia y el último extremo del mes sin leer datos, y los tamaños de row group que pide
   el motor.
2. **Compuerta de entrada y escritores** (`setup_s`). En el mismo pool, mientras este hilo crea el
   motor y espera el pie de L1: el carry-over del mes (y el del mes anterior, salvo en
   `series_start`) de cada θ, y los 50 escritores (`FamilyWriter`, con el esqueleto en RAM).
   Antes del primer `push` se esperan los carry-overs, y un carry-over con `direction ≠ 0` debe
   tener evento pendiente; los escritores no se esperan: cada θ espera al suyo al escribir su
   primer row group sellado, y uno que no logra abrirse detiene la unidad en el siguiente `push`.
3. **Lectura y empuje.** `L1.row_groups(readers = R)` entrega los row groups de L1 en orden, desde el
   que contiene la primera referencia hasta el que contiene el último extremo (más los de meses
   anteriores que algún evento necesita, ADR-L3-12); cada uno se decodifica **una vez** para todos
   los θ. Antes de cada `push` se entregan al motor las fronteras de L2 que el row group toca. El
   motor reparte ítems `(row group, θ)` en sus hilos, ubica las fronteras, suma las fases en
   enteros exactos (`u128`), cierra cada evento en su casilla (con los siete invariantes de §7.2
   contra su fila de L2) y sella los row groups de salida.
4. **Escritura.** El recolector pasa lo sellado a `SealedWriters`; cada θ escribe en orden con
   `FamilyWriter` de `shared/dc_frames`, que comprueba filas, orden y límites de row group contra
   `events.parquet`. ZSTD-3, estadísticas, sin diccionario, `state_version` en los metadatos.
5. **Conciliación** por θ y mes, en el pool: `N_M = (Σ_cerrados − P_M) + Cola_M + U_M`. `N_M`,
   `Cola_M` y `U_M` salen del pie de `consolidated.parquet` y de los ids de L2; `Σ_cerrados` y
   `P_M` (ticks cerrados y ticks de meses anteriores), de las cuentas del motor.
6. **Publicación** (`publish_s`). Atómica por archivo: se publica solo cuando todos los θ del mes
   pasaron los pasos anteriores, y los `commit` corren en el pool. Un error de entrada, de resumen
   o de conciliación borra los temporales y no deja ningún `summaries.parquet`. Si falla la
   publicación de un θ (p. ej. el `move` a GCS), los ya publicados quedan; re-ejecutar el mes los
   reescribe con el mismo contenido.

## Errores

El núcleo no emite hallazgos: lanza excepciones tipadas que llevan el `check_type`
del hallazgo y sus `details` (valores JSON).

| Excepción | `check_type` | `details` |
|---|---|---|
| `L2InputMissing` | `l2_input_missing` | `theta`, `month`, `file` |
| `L2StateInconsistent.mismatch` | `l2_state_inconsistent` | `theta`, `month`, `field`, `got`, `wanted` (y `message` si viene de una frontera del lector) |
| `L2StateInconsistent.no_pending` | `l2_state_inconsistent` | `theta`, `month`, `direction`, `has_pending_event` |
| `PhaseQuantityMismatch` | `phase_quantity_mismatch` | `theta`, `confirm_agg_trade_id`, `phase`, `expected`, `got` |
| `PhaseTicksMismatch` | `phase_ticks_mismatch` | `theta`, `month`, `N_M`, `sigma_closed`, `P_M`, `tail_M`, `U_M` |
| `SkeletonMismatch` | `family_skeleton_mismatch` | `theta`, `month`, `family`, `detail` (qué difiere del esqueleto de L2) |
| `PhaseValueError` | ninguno | `theta`, `confirm_agg_trade_id`: un valor no cabe en `DECIMAL(18,8)` |

Los valores de `got` y `wanted` de precios son enteros sin escalar (10⁻⁸). Además
propaga `FramesInputError` del lector: falta o no cumple el contrato un archivo de L1
(p. ej. el del mes, o un row group sin estadísticas de `agg_trade_id`). Es un L1 fuera
de contrato, no un fallo de L2.

Un id de frontera de L2 que no existe como tick en L1, o un tick de confirmación
cuyo `transact_time` no es el `confirm_time` de L2, llegan del lector como
`FrameBoundaryError` (con los atributos `theta`, `field`, `got` y `wanted`); el
resumidor los convierte en `L2StateInconsistent` con esos cuatro valores.

## Memoria

El pico es O(lote), no O(mes) ni O(eventos) ni O(ticks del evento más grande): `K + R + 1` row
groups de L1 (siete con los valores por defecto, ≈ 48 MB cada uno anclado o ≈ 1,8 × mientras se
decodifica), las fronteras de L2 de los eventos de esos row groups (`boundaries_peak_mib`; el motor
suelta cada lote al cerrarse sus eventos) más un tramo comprimido en curso por θ, los buffers de
salida abiertos y los sellados sin escribir (acotados por `8 × W`). Un evento que abarca muchos row
groups solo deja su acumulador parcial. La cuenta completa, y por qué la sonda de ITSC-362 medía
228 MiB y no los 95 previstos, está en TRD-L3 ADR-L3-16 («Memoria del orquestador»).
`tests/test_l3_memory.py` lo mide en un proceso aparte (pool de Arrow más `tracemalloc`): el pico
no crece con los eventos ni cuando el evento más grande pasa de 10 a 40 row groups.
`rss_peak_mib` de `unit_timing` es la medida de la corrida.

**Asignador.** La imagen corre con `MALLOC_MMAP_THRESHOLD_=33554432` (32 MiB) y
`MALLOC_TRIM_THRESHOLD_=1073741824` (sin recorte del heap), no con los 16 KiB de L2: en 2021-01 y a
igual cuota de CPU gasta −31 % de CPU y −25 % de pared, a cambio de un RSS de ≈ 3,1 GiB en vez de
2,0 (900 MiB bajo los 4 GiB del job). glibc retiene lo liberado para reutilizarlo; el pico sigue
acotado por K, R, W y los sellados pendientes, no por el mes. Sin la variable (en local y en pruebas)
`memory.tune_allocators()` fija el mismo umbral de `mmap`. Las cifras, la regla de retroceso (un mes
sobre 3 500 MiB quita el `trim`) y cómo comparar otra configuración están en TRD-L3 ADR-L3-16 y en
el runbook, «El asignador». `unit_timing` registra `mmap_threshold` y `trim_threshold`.

## Pruebas

`uv run pytest layers/l3_dc_descriptors`. `tests/rowgroup.py` y `tests/pyref.py` son la agregación
de Python por row group y el `summarize_month` en serie de antes de ITSC-363: la referencia con la
que `test_l3_parallel.py` y `test_l3_native.py` comparan el motor. El camino por evento anterior (Arrow compute
sobre `read_frames(chunks=True)`) vive en `tests/by_event.py` como **oráculo**: debe dar la
misma tabla y el mismo `content_hash` que el resumidor, también con eventos que cruzan row
groups, `R`, `C` y `E` en el primer y el último tick de un row group, overshoot vacío, un
evento de más de dos row groups y un instante de confirmación partido entre dos
(`tests/test_l3_row_groups.py`). `tests/oracle.py` es la tercera implementación, ingenua y
sin Arrow.

## Versiones

`VERSION` es la de la capa (la imagen y su tag se derivan de ella). La
`state_version` de `summaries.parquet` (`STATE_VERSION`) es la del archivo y viaja
en sus metadatos: sube con el esquema de la familia, no con el código.
