# l3_dc_descriptors

Capa L3: describe lo que pasó entre las fronteras de cada evento DC. Diseño en
[`docs/TRD/l3.md`](../../docs/TRD/l3.md); contrato de la salida en
[`docs/data-contracts.md`](../../docs/data-contracts.md) ("Salida Parquet de
L3"). Este paquete es el **núcleo del resumidor**: de `consolidated.parquet` de L1
y `events.parquet` de L2 a `summaries.parquet`, un mes y una lista de θ por
llamada. Todavía no tiene CLI, imagen ni frontera por θ: eso lo trae la capa
ejecutable, que llama a `summarize_month` y convierte sus errores en hallazgos de
DQ.

## Uso en local

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
