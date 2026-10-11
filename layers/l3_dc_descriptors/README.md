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
meses: el evento que cruza meses lo resuelve el lector, que relee los row groups
de L1 anteriores que ese evento necesita.

## Mecánica de una unidad

1. **Compuerta de entrada.** Antes de abrir nada se comprueba, por θ, que existan
   `events.parquet` y `carry_over.parquet` del mes (y el carry-over del mes
   anterior, salvo en `series_start`), y que un carry-over con `direction ≠ 0`
   tenga evento pendiente.
2. **Lectura.** `read_frames(chunks=True)` con todos los θ del mes: una pasada
   sobre L1, un row group en vuelo más el anterior (compartido) y ningún evento
   retenido. Cada fase llega como un iterador de trozos, uno por row group.
3. **Agregación por fase** (`phase.py`), con Arrow compute sobre cada trozo:
   conteo; cantidad con `is_buyer_maker = false` (`buy`) y `true` (`sell`) y la
   suma de todas; máximo y mínimo de `price`; primer y último `transact_time`;
   `Σ price·quantity`. El instante de confirmación (`transact_time = confirm_time`)
   se agrega aparte, dentro de la fase de confirmación. Entre trozos solo viven
   escalares de Python (enteros sin escalar, exactos); el row group se suelta en
   cuanto se agregó. Los dos iteradores de un evento se consumen antes de pedir el
   siguiente.
4. **Cierre del evento** (`event.py`). `buy + sell` de cada fase debe igualar la
   suma de `quantity` de sus ticks. Los invariantes de fila del contrato (conteos,
   precios del instante, extremo del overshoot, duración y VWAP) se comparan con la
   fila de L2. El VWAP se divide en entero al cerrar y se redondea half-even a
   10⁻⁸.
5. **Escritura** (`FamilyWriter` de `shared/dc_frames`). Cada θ arma en una
   matriz de enteros un row group con el mismo número de filas que el row group
   de `events.parquet` y lo entrega al escritor, que comprueba filas, orden y
   límites de row group contra `events.parquet`. ZSTD-3, estadísticas, sin
   diccionario, `state_version` en los metadatos.
6. **Conciliación** por θ y mes: `N_M = (Σ_cerrados − P_M) + Cola_M + U_M`.
   `N_M`, `Cola_M` y `U_M` salen del pie de `consolidated.parquet` y de los ids de
   L2; `Σ_cerrados` y `P_M` (`ticks_before_month` del lector), del resumidor.
7. **Publicación.** Todo o nada: los archivos se publican (escritura atómica) solo
   cuando todos los θ del mes pasaron los pasos anteriores. Cualquier error borra
   los temporales y no deja ningún `summaries.parquet`.

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
| `PhaseValueError` | ninguno | `theta`, `confirm_agg_trade_id`: un valor no cabe en `DECIMAL(18,8)` |

Los valores de `got` y `wanted` de precios son enteros sin escalar (10⁻⁸). Además
propaga los errores del lector y del escritor de familias: `FramesInputError`
(falta o no cumple el contrato un archivo de L1, p. ej. el del mes) y
`FamilySkeletonMismatch` (`family_skeleton_mismatch`).

Un id de frontera de L2 que no existe como tick en L1, o un tick de confirmación
cuyo `transact_time` no es el `confirm_time` de L2, llegan del lector como
`FrameBoundaryError`; el resumidor los convierte en `L2StateInconsistent` con el θ,
el campo y los dos valores.

## Memoria

El pico es O(row group), no O(mes) ni O(eventos): dos row groups de L1 (el del
pase y el anterior), los temporales de la agregación de un trozo, un row group de
eventos de L2 por θ y el row group de salida de cada θ. Un evento que abarca más
de dos row groups relee de L1, de uno en uno, solo los que sobran.
`tests/test_l3_memory.py` lo mide en un proceso aparte: el pico de Arrow no crece
con el número de eventos.

## Versiones

`VERSION` es la de la capa (la imagen y su tag se derivan de ella). La
`state_version` de `summaries.parquet` (`STATE_VERSION`) es la del archivo y viaja
en sus metadatos: sube con el esquema de la familia, no con el código.
