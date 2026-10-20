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
`L3_DQ_ROOT` (lago de hallazgos), además de `L3_SERIES_START` y `L3_THETAS_URI`. Opcionales:
`L3_THREADS` (hilos del pool de θ; por defecto `ceil(3 × vCPU efectivos)`, nunca más que θ: 12 a
4 vCPU y 24 a 8; entero ≥ 1), `L3_PREFETCH` (row groups que el lector trae de ventaja; por defecto 2,
0 lo apaga; entero ≥ 0) y `L3_PROFILE=1`. El stack `l3` no fija ninguna de las dos, así que el valor
sigue a los vCPU del job; para sobrescribirlas en una corrida, ver el
[runbook](../../docs/runbooks/operacion-l3.md#hilos-de-la-unidad-l3_threads-y-l3_prefetch). Un valor que no sea un entero válido es código 2.
`CLOUD_RUN_TASK_INDEX` distinto de 0 es código 2. `IMAGE_VERSION` la fija la imagen.

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
pared total y por fase (`read_s` lectura y decodificación, `aggregate_s`, `write_s`),
`rss_peak_mib` (`VmHWM`, reiniciado al empezar la unidad), `row_groups_month` y
`row_groups_previous` (decodificaciones de L1 del mes y de meses anteriores, contadas con un
gancho sobre `dc_frames.lake.open_parquet` como hace la sonda de `ops_tools`),
`row_groups_repeated` (debe ser 0), eventos y ticks por θ y en total, el límite efectivo de
CPU y la CPU que consumió la unidad (`cpu_user_s`, `cpu_system_s`, `cpu_s` y
`cpu_utilization`, de `getrusage`). Con `L3_PROFILE=1` la unidad corre bajo `cProfile` y suma el
hallazgo `unit_profile` (las 40 funciones más caras por tiempo propio y acumulado); sin la
variable no cambia nada. Cómo lanzarlo: [runbook](../../docs/runbooks/operacion-l3.md#perfilar-una-unidad).

## Hilos dentro de la unidad

La CLI procesa los meses en serie, pero cada mes usa hilos (ADR-L3-15, [TRD-L3 §8.1](../../docs/TRD/l3.md#81-núcleo-del-resumidor-común-a-los-dos-modos)):
un hilo lector trae y decodifica el siguiente row group mientras se procesa el actual
(`L3_PREFETCH`) y un pool de `L3_THREADS` hilos reparte los θ de cada row group, cada uno
escribiendo su archivo. El mismo pool hace la compuerta de entrada de L2 antes de abrir ningún escritor y,
después del último row group, un pool del mismo tamaño hashea los archivos de L2 del linaje; ambos
conservan el orden de los θ y el primer error. El hilo principal espera los θ del row group antes de tomar el siguiente,
así que cada θ corre en un solo hilo a la vez y en orden. Con `L3_THREADS=1` y `L3_PREFETCH=0` el
flujo es el secuencial de antes y el resultado es idéntico en cualquier combinación
(`tests/test_l3_parallel.py`). Qué sostiene esto:

- **Sin estado global.** Ningún módulo guarda un contenedor mutable ni usa `global`
  (`tests/test_l3_parallel.py` lo comprueba). El trabajo por θ sobre un row group,
  `_Lane.add(Prefixes, ThetaEvents)`, solo lee sus argumentos y escribe en su propio `_Lane`
  (su agregador, su escritor, su buffer de salida y sus cuentas); `Prefixes` se calcula una vez
  por row group y es de solo lectura.
- **La unidad por mes es una función sin estado entre meses.** `pipeline.process_unit(unit, ctx,
  thetas)` recibe datos inmutables (`Unit`, `RunContext`) y devuelve un `Result` con sus
  hallazgos y tiempos; la CLI los emite al volver. Se puede invocar desde otra tarea o hilo.
- **Hallazgos y tiempos por unidad.** `Phases`, `RowGroupCount` y `Timing` los crea la unidad.
  El único punto de proceso es el gancho de `open_parquet`: se instala una vez, delega en el
  original y cuenta en un `ContextVar`, así que dos unidades en hilos distintos no se mezclan.
- **Prueba.** `summarize_month` con 1, 2 y 4 hilos y 0, 1 y 2 de prefetch da exactamente la misma
  tabla y el mismo `content_hash` que la referencia secuencial sobre el fixture. No mide velocidad.
- **Qué mide `unit_timing` con hilos.** `threads` y `prefetch` dicen lo usado. `read_s` es lo que el
  hilo principal esperó al lector; `aggregate_s` y `write_s` son **suma sobre los hilos** y pueden
  pasar de `wall_s`. El perfil (`L3_PROFILE=1`) ve las llamadas de todos los hilos (Python ≥ 3.12,
  `sys.monitoring`): sus tiempos son de hilo, no de pared.

**Límite conocido para los hilos.** Los tramos pesados están en NumPy y Arrow, que sueltan
el GIL: los prefijos y las diferencias de `Segments`, `reduceat`, la decodificación de Parquet, la
codificación ZSTD, la subida y el hash del escritor. Lo que queda en Python puro y lo retiene: la
**recomposición de `Σ price·quantity`** en entero de Python por segmento (`_vwaps` y
`Segments.words.tolist()`), el **VWAP** (`divide_half_even`), el control de flujo y, solo si una
comprobación del lote falla, el camino por evento de `build_row`. Ese tramo crece con los eventos
(≈ 5,7 µs de los 24,9 µs por evento) y los hilos no lo aceleran: por Amdahl, con ≈ 23 % serial el
techo es 4,3× y a 4 vCPU se espera 1,5× a 2,5×. Lo que los hilos sí esconden es la espera: T1
(ITSC-356, 2021-01) midió que la unidad esperaba el 62 % de su pared (`cpu_utilization` 0,092), en
lecturas de PyArrow y escritura que sueltan el GIL. Lo medido con hilos (corrida T3) está en
[TRD-L3 §10.2](../../docs/TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición). Lo ataca el
motor nativo (ITSC-358, ADR-L3-16; ver abajo) y no los procesos: la data no se replica. Dos cosas más que no son por unidad en un proceso con meses en
paralelo: `rss_peak_mib` y `cpu_throttled_s` son del proceso.

## Motor nativo de agregación (ITSC-358)

`native.py` adapta el motor de Rust (`shared/dc_descriptors`, expuesto por `dc_pyo3.Engine`) a
lo que entrega el lector: `set_boundaries` entrega una vez por θ las columnas de L2 de sus
eventos (Arrow, sin copia; el índice de un evento es su casilla), `push` empuja los buffers de
Arrow de cada row group sin copia, con la columna `agg_trade_id`, y el motor calcula él mismo
dónde cae cada frontera (ADR-L3-16), `to_batch` envuelve en Arrow los row groups de salida que el motor sella y
`to_summarize_error` convierte sus errores en las excepciones tipadas de `errors`, con el
mismo `check_type` y los mismos `details`. **Todavía no está en el flujo de `summarize_month`**:
lo conecta el orquestador (hija 3 de E6). Mientras tanto, `rowgroup.py` es la referencia contra
la que `tests/test_l3_native.py` comprueba que el motor, con 1 y 4 hilos y en cualquier orden
de `push`, da las mismas tablas, el mismo `content_hash` y la misma conciliación, y
`tests/test_l3_native_memory.py` que la RSS que el motor añade no crece con los ticks del mes
(las fronteras de L2 de los eventos sí son O(eventos) y quedan en la línea base; ver el costo en
ADR-L3-16). `test_l3_native.py` compara además las posiciones del motor con las de
`read_row_groups`, row group a row group. El diseño y por qué
es nativo están en [TRD-L3 ADR-L3-16](../../docs/TRD/l3.md).

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

1. **Compuerta de entrada.** Antes de abrir nada se comprueba, por θ, que existan
   `events.parquet` y `carry_over.parquet` del mes (y el carry-over del mes
   anterior, salvo en `series_start`), y que un carry-over con `direction ≠ 0`
   tenga evento pendiente.
2. **Lectura.** `read_row_groups` con todos los θ del mes: una pasada sobre L1 en la
   que cada row group se decodifica **una vez** para todos los θ, sin relecturas, y se
   entrega con las posiciones `R`, `C` y `E` de los eventos de cada θ que lo tocan
   (ADR-L3-14). De `events.parquet` solo se decodifican las siete columnas de §7.1
   (`EVENT_COLUMNS`). El row group se suelta antes de pedir el siguiente.
3. **Agregación por row group** (`rowgroup.py`), por θ y en llamadas vectorizadas, sin
   bucle por evento ni por tick:
   - `Prefixes`, una vez por row group: prefijos (sumas acumuladas) de `quantity` (todos
     los ticks, `is_buyer_maker = true` y `false`, en `int64`) y de `price·quantity`
     descompuesto en palabras de 21 bits (3 de precio × 3 de cantidad, 9 prefijos
     `uint64`). Cada producto parcial es menor que 2⁴² y un row group admite hasta 2²⁰
     ticks, así que toda diferencia de prefijos cabe en 63 bits. Nunca se multiplica
     `price` por `quantity` en `int64` ni se reduce antes de multiplicar.
   - `Segments`, por θ: el arreglo de cortes sale de las posiciones; los conteos son
     diferencias de cortes; las sumas, diferencias de prefijos; el máximo y el mínimo de
     `price`, `np.maximum.reduceat` y `np.minimum.reduceat` (solo sobre las fases con
     ticks, porque `reduceat` no admite segmentos vacíos); el primer y el último
     `transact_time`, indexación en los cortes; y el instante de confirmación, la máscara
     `transact_time == confirm_time` restringida a la fase de confirmación. `Σ price·
     quantity` se recompone en entero de Python por segmento.
   - `Open`, el acumulador parcial: el evento abierto al cierre del row group deja sus
     conteos, sumas (con las 5 palabras de `Σ price·quantity`), extremos, tiempos y los
     del instante de confirmación. Es estado O(1) por θ, nunca ticks; el siguiente row
     group lo fusiona con el tramo que le toca.
4. **Cierre de los eventos** (`rowgroup._close_batch`, `event.build_row`). `buy + sell` de
   cada fase debe igualar la suma de `quantity` de sus ticks. Los invariantes de fila del
   contrato (conteos, precios del instante, extremo del overshoot, duración y VWAP) se
   comparan con la fila de L2, sobre el lote de eventos que cerraron en el row group. El
   VWAP se divide en entero y se redondea half-even a 10⁻⁸. Si algo no cuadra, el lote se
   repite evento por evento con `build_row`, que es quien lanza el error de siempre. El
   evento que viene de un row group anterior se cierra con `build_row` sobre su acumulador.
5. **Escritura** (`FamilyWriter` de `shared/dc_frames`). Cada θ arma en
   un buffer de NumPy por columna un row group con el mismo número de filas que el row group
   de `events.parquet` y lo entrega al escritor, que comprueba filas, orden y
   límites de row group contra `events.parquet`. ZSTD-3, estadísticas, sin
   diccionario, `state_version` en los metadatos.
6. **Conciliación** por θ y mes: `N_M = (Σ_cerrados − P_M) + Cola_M + U_M`.
   `N_M`, `Cola_M` y `U_M` salen del pie de `consolidated.parquet` y de los ids de
   L2; `Σ_cerrados` y `P_M` (`ticks_before_month` del lector), del resumidor.
7. **Publicación.** Atómica por archivo y secuencial por θ: se publica solo cuando todos
   los θ del mes pasaron los pasos anteriores. Un error de entrada, de resumen o de
   conciliación borra los temporales y no deja ningún `summaries.parquet`. Si falla la
   publicación de un θ (p. ej. el `move` a GCS), los ya publicados quedan; re-ejecutar
   el mes los reescribe con el mismo contenido.

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

El pico es O(row group), no O(mes) ni O(eventos) ni O(ticks del evento más grande): un
row group de L1 en vuelo (el lector no retiene el anterior), los 12 prefijos de la
agregación (≈ 2 veces el row group decodificado) y sus temporales, un row group de
eventos de L2 por θ, los arreglos de las fases de los eventos del row group y el row group
de salida de cada θ. Un evento que abarca muchos row groups solo deja su acumulador
parcial. `tests/test_l3_memory.py` lo mide en un proceso aparte (pool de Arrow más
`tracemalloc`, que ve los arreglos de NumPy): el pico no crece con los eventos ni cuando el
evento más grande pasa de 10 a 40 row groups.

## Pruebas

`uv run pytest layers/l3_dc_descriptors`. El camino por evento anterior (Arrow compute
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
