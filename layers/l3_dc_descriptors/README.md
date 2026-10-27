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

Un mes a la vez, y en cada mes el fan-out de los θ que lo necesitan. En `backfill` cada mes corre en
un proceso hijo (`L3_UNIT_PROCESS`, ITSC-372): el padre recorre el rango y emite los hallazgos, y la
memoria del mes muere con su proceso. Un mes que ningún θ necesita se salta sin leer L1.

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
enteros ≥ 1 (otro valor es código 2; `L3_LOT_EVENTS` admite 0): `L3_READERS` (lectores de L1, R; por defecto 2),
`L3_INFLIGHT` (row groups en el motor, K; 4), `L3_WRITERS` (escritores, W; 12),
`L3_ENGINE_THREADS` (hilos del motor; por defecto `round(vCPU efectivos)`), `L3_THREADS` (pool de
E/S; por defecto `ceil(3 × vCPU)`, nunca más que θ), `L3_META_WORKERS` (pool de metadatos: compuerta,
apertura de `events.parquet` y de los escritores, pie de L1 y publicación; entre 1 y 64, por defecto 16) y `L3_INFLIGHT_BYTES` (tope de bytes de eventos
abiertos; 805 306 368 = 768 MiB), `L3_LOT_EVENTS` (eventos de un tramo; 1 000 000; 0 desactiva el
corte), `L3_MEMORY_SAMPLE_S` (segundos entre ticks del muestreador de memoria; número > 0, por
defecto 0,25), `L3_TRIM_RSS_MIB` (marca de agua del trim periódico, en MiB; entero ≥ 0, por
defecto 1 024; 0 lo desactiva) y `L3_TRIM_MIN_S` (segundos mínimos entre dos trims; número ≥ 0, por
defecto 0,5), más `L3_PROFILE=1` y `L3_UNIT_PROCESS` (`0` o `1`; por defecto 1:
un proceso hijo por unidad en `backfill`; `monthly` no lo usa). El stack `l3` no fija
ninguna, así que los valores siguen a los vCPU del job; para sobrescribirlas en una corrida, ver el
[runbook](../../docs/runbooks/operacion-l3.md#hilos-y-topes-de-la-unidad-l3_readers-l3_inflight-l3_writers-l3_engine_threads-y-l3_threads).
**Memoria de los escritores (ITSC-365):** el orquestador no empuja más row groups mientras haya
`PENDING_PER_WRITER × W` row groups de salida sellados sin escribir (3 × 12 = 36 por defecto);
cada uno pesa ≈ 8 MiB (256 B por evento × ≈ 34 000 eventos), así que el tope son ≈ 290 MiB, lo
mismo que 8 × 4 de antes. Con 8 y W = 12 serían 96 (≈ 770 MiB): es lo que T4b pagó de más
(RSS 2 331 MiB). Si subes W o bajas el tamaño del row group, rehaz la cuenta y compárala con el
margen del job: con el asignador de la imagen (ver «Memoria») el RSS medido es ≈ 3,1 GiB de 4.
**Memoria acotada por los eventos en vuelo (ITSC-372):** el orquestador tampoco empuja un row group
nuevo mientras haya algún row group en el motor y los bytes de eventos abiertos pasen de
`L3_INFLIGHT_BYTES`. Los eventos abiertos son los que `BoundaryFeed` entregó al motor menos los que
ya cerraron (`Engine.needed`), por θ. Un evento cuesta 305 B: 240 B de la fila de salida (las 18
columnas de `ENGINE_SCHEMA`) y 65 B de las fronteras de L2 que el motor retiene hasta cerrarlo. Los
que `cover` trajo por delante del último tramo empujado (`BoundaryFeed.ahead`, eventos cuya
referencia pasa del último id cubierto) aún no tienen casilla y cuestan solo los 65 B; desde ITSC-376
así se cuentan, y antes se contaban a 305 B. 768 MiB son ≈ 2,6 M de eventos de 305 B. Un row group de un mes normal trae ≈ 0,6 M de eventos y 2017-12 trae
4,5 M: sin este tope, con K = 4 caben los tres row groups del mes y la unidad picó a 3 545 MiB. Un
tramo que por sí solo pasa del tope avanza igual (avisa una vez; `inflight_alone_bytes_max` >
`inflight_bytes` en `unit_timing`); el aviso dice qué tramo trajo cuántos eventos y que la unidad
avanza. Con el lote por defecto (1 M ≈ 305 MB) y el tope por defecto no debe salir en ningún mes: si sale,
un tramo solo pasa de 768 MiB, y se arregla con un lote más chico (`L3_LOT_EVENTS`: el tramo sale de
la densidad media del mes y uno localmente denso puede pasar del tope) o con más memoria en el job. El tope se mira
antes de empujar, así que `inflight_bytes_max` puede pasar de `inflight_bytes` hasta en un tramo sin
que sea esa señal.
**Lote acotado en eventos (ITSC-373):** el tope de bytes no puede acotar lo que entra de a uno, y un
row group de L1 de 2017-12 (≈ 0,94 M de ticks) trae 5,76 M de eventos abiertos: 1,67 GiB de casillas,
y el mes picó a 3 002 MiB aunque nunca convivieran dos row groups. Por eso `_push_all` parte cada
row group en **tramos** contiguos de a lo más `N` ticks y empuja cada uno al motor como si fuera un
row group (con su ordinal, su `cover(high)` y la bandera `before_month`). `N = floor(L3_LOT_EVENTS /
densidad)`, con la densidad del mes = eventos de los θ (pies de L2) ÷ ticks (pie de L1), así que un
tramo trae ≈ `L3_LOT_EVENTS` eventos con cualquier densidad. **La cuenta:** 1 M de eventos × 305 B
(`EVENT_BYTES`) ≈ 305 MB de casillas y fronteras abiertas, bajo el tope de 768 MiB con holgura. En
2017-12 (6,1 eventos por tick) son ≈ 164 000 ticks y ≈ 6 tramos por row group. **Lo que se probó
y qué dio:** ITSC-373 fijó 1 M y midió +38 % de CPU en los meses densos (sin aislar la causa);
ITSC-375 subió el lote a 2,5 M para ahorrarla y midió lo contrario: 2017-12 subió de 2 681 a
2 802 MiB y 2018-01 de 2 987 a 3 272 MiB con la misma CPU (55,9 contra 56,5 s; 103,6 contra
104,3 s) y la misma pared (± 2 %), porque con 2,5 M conviven dos tramos en el motor
(`inflight_max` 2) y los eventos abiertos suben de 3,3 M a 4,4 M. ITSC-376 vuelve a 1 M. El corte no copia: `ids[a:b]` es una vista
de NumPy y `column.slice(a, b - a)` un array de Arrow sobre el mismo buffer, y `Engine.push` honra el
desfase (en bits para `is_buyer_maker`). Un evento que cruza tramos se fusiona como ya se fusiona
entre row groups, y las tablas y los `content_hash` no cambian. Las fronteras de L2 que `cover` trae
por delante (un lote de 32 768 eventos por θ) también cuentan como abiertas.
`L3_PREFETCH` ya no existe (ITSC-363). `CLOUD_RUN_TASK_INDEX` distinto de 0 es código 2. `IMAGE_VERSION` la fija la imagen.

**Códigos de salida** ([§8.4](../../docs/TRD/l3.md#84-códigos-de-salida)): 0 éxito; 1 fallo
de la unidad (entrada de L1 o L2 ausente, L2 incoherente, esqueleto, conciliación) o `monthly`
sin ningún θ listo y con rezagados; 2 uso o configuración (no escribe nada).

**Frontera** ([§8.2](../../docs/TRD/l3.md#82-modo-backfill-histórico)): por θ, el último mes de
su serie contigua desde `--series-start` cuyo `summaries.parquet` es válido (esquema, `theta`,
`state_version` y filas por row group iguales a `events.parquet` de L2 del mes). Sale de un
listado de la raíz de L3 más el pie del archivo de cada θ; no se consulta cada partición. Un
mes sin `events.parquet` y `carry_over.parquet` de L2 detiene el rango de ese θ con
`l2_input_missing` (los demás siguen, la corrida termina con código 1). Una partición de L2
con sus dos Parquet pero sin `content_hash.json` (el linaje que L2 publica, ITSC-368) falla la
unidad con `l2_lineage_missing`: se cubre con el backfill del linaje antes del apply de la
imagen ([runbook](../../docs/runbooks/operacion-l3.md#cambios-requeridos)). En `monthly`, un θ
que no llegó al mes previo (`l3_frontier`) o al que L2 no le ha llegado (`l2_missing`) deja
`theta_behind_frontier` y no falla la unidad mientras otro avance.

**Hallazgos**: el catálogo de `check_type` y sus `details` están en
[`docs/data-contracts.md`](../../docs/data-contracts.md). Cada excepción tipada del núcleo
(tabla de abajo) se traduce a su hallazgo con los `details` que ya trae. `unit_timing` lleva lo
que necesita la medición del umbral 3 de [§10.3](../../docs/TRD/l3.md#103-sonda-del-lector-de-tramas--umbrales-de-aceptación-se-escriben-antes-de-medir):
pared total y por fase (todas son pared: `gate_s`, abrir `events.parquet` de los θ y arrancar la lectura de sus fronteras; `setup_s`,
el motor, el pie de L1 y la compuerta de carry-over; `aggregate_s` del motor, `publish_s`, `reconcile_s`; `lineage_s`
ya no existe: el linaje lo publica L2 y se lee; `read_s` es lo que el hilo principal esperó a las lecturas y
`read_wall_s` y `write_s` la pared con alguna lectura o algún escritor ocupado), la configuración
(`threads`, `engine_threads`, `readers`, `inflight`, `writers`), lo que hizo el motor
(`inflight_max`, `engine_busy`, `items`, `inflight_bytes`, `inflight_events_max`,
`inflight_bytes_max`, `inflight_alone_bytes_max`), el lote (`lot_events`, `lot_ticks` y `lots`, los
tramos empujados), `spawn_s` (el arranque del proceso hijo de la unidad), `read_bytes`, `boundaries_peak_mib`,
`rss_peak_mib` (`VmHWM`, reiniciado al empezar la unidad), la composición de ese RSS
(`mem_peak`, `mem_close_before_trim`, `mem_close_after_trim`, `mem_samples` y `mem_peak_coverage`:
ver «Composición del RSS» abajo), `row_groups_month` y
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
                                          motor nativo (hilos, ítems (tramo, θ))
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
| `unit_process.py` | `run_unit`: la unidad en un proceso hijo `spawn`; si el hijo muere sin resultado, deja `unit_process_died` (ITSC-372) |

Qué sostiene la corrida:

- **Cuatro topes, no cuatro colas sin fondo.** R lectores de L1 traen row groups por delante; el
  orquestador no empuja el siguiente tramo mientras el motor tenga K en vuelo (`engine.inflight()`),
  haya `3 × W` row groups sellados sin escribir ni, con alguno en el motor, más de
  `L3_INFLIGHT_BYTES` de eventos abiertos (el tramo es el tope por lote: ver arriba). En RAM hay a
  lo más `K + R + 1` row groups de L1
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
   arrancar la lectura anticipada de sus fronteras (las siete columnas de §7.1, `EVENT_COLUMNS`,
   más `theta` para el esqueleto: `EVENT_FILE_COLUMNS`; las otras cuatro no se decodifican). Con las estadísticas de los pies se sabe la primera
   referencia y el último extremo del mes sin leer datos, y los tamaños de row group que pide
   el motor.
2. **Compuerta de entrada y escritores** (`setup_s`). En el mismo pool, mientras este hilo crea el
   motor y espera el pie de L1: el carry-over del mes (y el del mes anterior, salvo en
   `series_start`) y la huella que L2 publicó del mes (`content_hash.json`, el linaje; no se
   recalcula) de cada θ, y los 50 escritores (`FamilyWriter`, con el esqueleto en RAM).
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
| `L2LineageMissing` | `l2_lineage_missing` | `theta`, `month`, `file` (`content_hash.json`), `reason` |
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
salida abiertos y los sellados sin escribir (acotados por `3 × W`). Un evento que abarca muchos row
groups solo deja su acumulador parcial. La cuenta completa, y por qué la sonda de ITSC-362 medía
228 MiB y no los 95 previstos, está en TRD-L3 ADR-L3-16 («Memoria del orquestador»).
`tests/test_l3_memory.py` lo mide en un proceso aparte (pool de Arrow más `tracemalloc`): el pico
no crece con los eventos ni cuando el evento más grande pasa de 10 a 40 row groups.

**La variable que faltaba (ITSC-372): la densidad de eventos por tick.** Los buffers de salida
abiertos cuestan ≈ 240 B por evento y los eventos de un row group dependen del mes: ≈ 0,6 M en los
meses medidos, 4,5 M en 2017-12 (4,8 eventos por tick con los 50 θ). Por eso el orquestador acota
también los eventos abiertos (`L3_INFLIGHT_BYTES`, arriba) y, en `backfill`, cada unidad corre en un
proceso hijo: el RSS que una unidad deja en el proceso (fragmentación de arenas, que `malloc_trim`
no recupera) sube de mes en mes y en 110 meses mata el job. `test_l3_memory.py` mide las dos
cosas: el `VmRSS` del padre entre seis unidades en hijos y un lago denso con y sin el tope.
`rss_peak_mib` de `unit_timing` es la medida de la corrida.

**Asignador.** La imagen corre con `MALLOC_MMAP_THRESHOLD_=33554432` (32 MiB) y
`MALLOC_TRIM_THRESHOLD_=1073741824` (sin recorte del heap), no con los 16 KiB de L2: en 2021-01 y a
igual cuota de CPU gasta −31 % de CPU y −25 % de pared, a cambio de un RSS de ≈ 3,1 GiB en vez de
2,0 (900 MiB bajo los 4 GiB del job). glibc retiene lo liberado para reutilizarlo; el pico sigue
acotado por K, R, W y los sellados pendientes, no por el mes (dentro de una unidad: entre unidades
el RSS crecía, y por eso la devolución al cerrar de más abajo, ITSC-371). Sin la variable (en local y en pruebas)
`memory.tune_allocators()` fija el mismo umbral de `mmap`. Las cifras, la regla de retroceso (un mes
sobre 3 500 MiB quita el `trim`) y cómo comparar otra configuración están en TRD-L3 ADR-L3-16 y en
el runbook, «El asignador». `unit_timing` registra `mmap_threshold` y `trim_threshold`. Al cerrar cada unidad `process_unit` devuelve
la memoria libre al kernel (`gc.collect()` y `malloc_trim(0)`), porque un `backfill` corre muchos meses en
un proceso y con el asignador de la imagen glibc no devuelve al kernel lo liberado, así que el RSS sube
unidad a unidad hasta el OOM (ITSC-371; la causa exacta es una hipótesis, ver ADR-L3-16); `unit_timing` registra
`rss_before_trim_mib` y `rss_after_trim_mib`.

**Trim por marca de agua (ITSC-376).** El trim del cierre no baja el pico: llega cuando la unidad ya
terminó. La medición de ITSC-375 (imagen 0.18.0) mostró que más de la mitad del pico de los meses
densos era memoria libre que glibc retenía (`fordblks`). Por eso el mismo hilo `l3-memory-sampler`
llama a `malloc_trim(0)` durante la unidad: en cada tick, si `VmRSS` ≥ `L3_TRIM_RSS_MIB` (1 024 MiB)
y pasaron al menos `L3_TRIM_MIN_S` (0,5 s) desde el último trim. Marca de agua y no «tras cada
sellado» ni «cada N segundos»: `malloc_trim` recorre todas las arenas con sus candados y con ≈ 2 GiB
libres fragmentados cada llamada puede costar decenas de ms; con marca de agua el costo es cero bajo
la marca (los meses chicos no pagan nada) y el pico queda acotado por la marca más lo que crece en
`L3_TRIM_MIN_S`. `L3_TRIM_RSS_MIB=0` lo desactiva (queda el del cierre). `unit_timing` suma `trims`
(cuántos), `trim_s` (cuánto tardaron en total) y `trim_rss_mib` (la marca con que corrió). Si el
trim estorba al motor (`trim_s` o la CPU suben), el ajuste es subir `L3_TRIM_MIN_S`, no quitarlo.
Lo que no baja es la memoria viva: casillas abiertas del motor y la salida sellada de Arrow que
espera a los escritores.

**Composición del RSS (ITSC-375).** `rss_peak_mib` dice cuánto pesó la unidad, no de qué. El hilo
`l3-memory-sampler` (uno por unidad, período `L3_MEMORY_SAMPLE_S`, 0,25 s) lee `VmRSS` en cada tick y,
solo cuando supera su máximo, toma una muestra con `mallinfo2` de glibc (`arena`, `hblkhd`,
`uordblks`, `fordblks`, `keepcost`), los bytes vivos de Arrow (`pa.total_allocated_bytes()`) y los
tramos anclados en el motor (`engine.inflight()`). `unit_timing` publica la del RSS más alto
(`mem_peak`) y dos del cierre, justo antes y después del trim (`mem_close_before_trim`,
`mem_close_after_trim`). Cómo leerlas: `uordblks` es memoria en uso del heap y `fordblks`, memoria
que el programa ya liberó y glibc retiene (`arena = uordblks + fordblks`; `hblkhd` son las regiones
`mmap`, aparte). Con `trims` = 0, si `fordblks_mib` es la mayor parte de `rss_mib` el pico es
retención del asignador y no datos vivos. Con `trims` > 0 esa lectura engaña (`fordblks` cuenta
páginas que el trim ya devolvió y puede pasar de `rss_mib`): se mira lo residente sin uso,
`rss_mib − (uordblks_mib + hblkhd_mib)`, y la memoria viva en `uordblks_mib` y `arrow_mib`. El trim
recorre cada arena y hace dos cosas: devuelve al kernel las
páginas interiores de los chunks libres de todas las arenas (baja el RSS y no toca `fordblks`) y
recorta solo el top de la arena principal (baja `arena` y `fordblks` en ≈ `keepcost`, que `mallinfo2`
también reporta solo para ella); los tops de las arenas secundarias siguen residentes. Entre las dos
muestras del cierre corre además `gc.collect()`, que puede subir `fordblks`. Por eso lo que glibc
retenía al cierre queda acotado: `mem_close_before_trim.rss_mib − mem_close_after_trim.rss_mib` es
una **cota inferior** (incluye lo que soltó `gc.collect()`), `fordblks_mib` antes del trim es la
**cota superior**, y `before.fordblks_mib − after.fordblks_mib` es la cima devuelta, no retención
que «desapareció». `mem_peak_coverage` = RSS de `mem_peak` ÷ `rss_peak_mib`: bajo 0,9 el
muestreador se perdió el pico y se baja `L3_MEMORY_SAMPLE_S`. El muestreo no cambia el pico; el trim
que hace el mismo hilo sí (arriba). La muestra del pico se toma antes del trim del tick.
`mallinfo2` toma el candado de cada arena y recorre sus listas libres, por eso no corre en cada tick.

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

### Pruebas de memoria (`tests/test_l3_memory.py`)

Miden el RSS de procesos hijos reales, así que dependen de la máquina: cuántos vCPU hay, cuánto
retiene glibc de lo liberado y quién llega primero, el lector o el motor. Por eso **ninguna
aserción es un valor en MiB**: cada una afirma una relación entre dos cifras de la misma corrida,
con la cuenta en el comentario del código (ITSC-377; tres relanzamientos del PR 158 fallaron cada
vez por un test distinto contra umbrales fijos).

| Prueba | Qué relación afirma | Por qué es esa |
|---|---|---|
| `test_peak_is_a_row_group_…` | El pico con 200 eventos por θ no pasa del de 10 en más de dos row groups fijos (`2 * ROW_GROUP_BYTES`, 9,6 MB). | Corre con `readers=1`, `inflight=1` y un hilo del motor: sin carrera lector-motor, en 100 pares many − few fue de -4,3 a +4,3 MB. Con los valores por defecto la variación llegaba a 11,9 MB; esa carrera la cubre `test_l3_parallel.py`. |
| `test_peak_does_not_grow_with_the_ticks_…` | El pico no crece con los ticks del evento más grande: sube a lo más lo que anclaron de más los row groups en vuelo (`inflight_max` por row group) más dos row groups. | Con `readers=1`, `inflight=1` la `inflight_max` es siempre 1 y esa parte vale 0: queda la holgura de dos row groups, medida en 50 corridas (a lo más 3,6 MB). |
| `test_rss_after_closing_…`, `test_without_the_trim_…` | El RSS al cerrar cada unidad sube, sobre el de la primera, al menos `UNITS_NO_TRIM_MIN_RISE` veces lo que la unidad tuvo en vuelo sin trim, y con trim se recupera al menos `UNITS_TRIM_MIN_SHARE` de esa subida; con trim la deriva no pasa de la de la serie sin trim más dos row groups ni de `UNITS_IN_FLIGHT_BYTES`. | La primera unidad trae el nivel base de la máquina (intérprete, librerías) y restarla lo cancela. Antes eran 57,6 MB fijos. |
| `test_the_parent_rss_is_flat_…` | El `VmRSS` del padre se mueve a lo más un row group de las unidades (`PARENT_FLAT_BYTES`; medido 0,02 a 0,14 MiB en 60 series) y los picos de los hijos difieren a lo más en lo que una unidad tiene en vuelo (`UNITS_IN_FLIGHT_BYTES`). | Antes `max ≤ 1,3 × min`: con 2 vCPU y otro proceso en paralelo cayó 3 de 6 veces, porque sin CPU de sobra el pico depende de cuántos row groups coinciden en memoria. |
| `test_the_byte_cap_…`, `test_the_lot_lowers_…` | El pico baja al menos `DENSE_MIN_GAIN` de lo que cuestan las casillas de más que la corrida midió (`inflight_events_max` × `SLOT_BYTES`). El motor sin tope puede no acumular nada (es una carrera, en 38 de 100 corridas): entonces el tope no debe subir el pico, y un tope que no hiciera nada también pasaría esa corrida (límite conocido; la cota `capped ≤ 1,2 × un row group` no depende de la carrera). | El pico es la marca más alta del heap y glibc retiene lo liberado, así que solo una parte del ahorro llega al RSS (0,28 a 0,70 en 100 pares). |
| `test_the_periodic_trim_lowers_the_peak_…` | Lo residente sin uso en el pico con trim periódico no pasa de `TRIM_MAX_RESIDUAL` del de la corrida sin él, y el pico no sube. | La baja del pico es lo libre que había en él, y eso varía (razón de 0,19 a 0,32); lo residente sin uso es lo que el trim puede quitar. |
| `test_the_trim_returns_pages_…` | El trim de cierre nunca empeora el RSS; lo baja de verdad solo si no corrió ningún trim periódico (`trims == 0`). | Con trims por marca de agua el último pudo correr justo antes del cierre y no queda nada que devolver. |
| `test_the_memory_sampler_sees_the_peak_…` | La muestra del pico trae `engine_inflight` si cayó dentro de la etapa de agregación (a lo más los K configurados); fuera de ella es `None` y no es un fallo. | El motor solo existe para el muestreador entre `new_engine` y su cierre; el RSS puede marcar su máximo en la publicación. |

**Si una vuelve a fallar:** no subas el margen. Corre el archivo 10 veces sin carga y 10 con otras
pruebas de subproceso a la vez, y anota por corrida las cifras que la aserción compara y las que
explican la causa (`inflight_max`, `trims`, `mem_peak`, `mem_close_*`). Si la relación sigue siendo
cierta y la cifra cae fuera, el margen estaba mal medido: corrígelo con la distribución nueva y el
comentario con la cuenta. Si la relación dejó de ser cierta, es un cambio real del orquestador.

## Versiones

`VERSION` es la de la capa (la imagen y su tag se derivan de ella). La
`state_version` de `summaries.parquet` (`STATE_VERSION`) es la del archivo y viaja
en sus metadatos: sube con el esquema de la familia, no con el código.
