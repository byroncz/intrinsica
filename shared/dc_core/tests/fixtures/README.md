# Fixture de equivalencia contra la v0 (ITSC-240)

Ticks reales de **2017-08-18 UTC** (BTCUSDT spot, 4 735 ticks) y los eventos que
`segment_events_kernel` de la v0 (tag `v0.2.0-legacy`) calcula sobre ellos para
θ = 0.1 %, 0.25 %, 0.5 %, 1 % y 2 %. La prueba
[`../equivalence_v0.rs`](../equivalence_v0.rs) alimenta los ticks a `Detector` y
compara con estos archivos. La v0 es el oráculo, no código del proyecto.

| Archivo | Contenido |
|---|---|
| `ticks.csv` | `agg_trade_id,price,transact_time`: precio con sus 8 decimales, tiempo en µs UTC, orden `(transact_time, agg_trade_id)`. |
| `events_v0.csv` | Un evento por fila y θ (`theta` = θ × 10⁸): referencia, confirmación y extremo, cada uno con precio, tiempo y `agg_trade_id`. El último evento de cada θ no tiene extremo (columnas vacías): es el pendiente. |
| `final_state_v0.csv` | Estado final del kernel por θ: `n_events`, tendencia, extremo alto y bajo, `last_os_ref` y `orphan_start_idx`. |
| `generate_v0_events.py` | Genera los tres archivos. |

El día sale de la landing de L1 (`consolidated.parquet` de 2017-08, hash de
contenido `6c96e620…90db`). Se eligió por bajo volumen (menos de 5 000 ticks,
los tres archivos suman ~340 KB) y porque el rango del día (11 %) da eventos a
todos los θ.

## Cómo regenerarlo

1. Un mes de landing en local (`data.binance.vision` está en la lista blanca):

   ```bash
   export L1_LANDING_ROOT=$PWD/sandbox.local/itsc240/landing \
          L1_DQ_ROOT=$PWD/sandbox.local/itsc240/dq \
          L1_MANIFEST_ROOT=$PWD/sandbox.local/itsc240/manifest
   uv run python -m l1_ingest --mode backfill --from 2017-08 --to 2017-08
   ```

2. El script en un entorno aparte: Numba no soporta el Python 3.14 del repo.
   `--no-project` evita que `uv` aplique el `requires-python` del repo.

   ```bash
   uv run --no-project --python 3.12 --with numba==0.63.1 --with numpy --with pyarrow \
     python shared/dc_core/tests/fixtures/generate_v0_events.py \
     "$L1_LANDING_ROOT/provider=binance/market=spot/asset=BTCUSDT/year=2017/month=08/consolidated.parquet" \
     2017-08-18
   ```

   El script baja `kernel.py` con `git show v0.2.0-legacy:src/intrinseca/core/kernel.py`
   a un temporal y lo importa; los precios entran como `float64`, como los
   alimentaba la v0. Como `quantities` pasa el `agg_trade_id`: el kernel lo
   copia junto al precio a sus búferes DC/OS y así se recuperan los
   `agg_trade_id` de referencia y confirmación, que la v0 solo maneja como
   índice de array. La regeneración es determinista.

Si cambias el día o los θ, actualiza también la tabla `VENTANAS` de la prueba.

## Qué difiere de la v0 y por qué

Con θ = 2 % los 12 eventos coinciden campo a campo. Con los otros θ, la prueba
encuentra **discrepancias, todas de una sola clase**: la política de
[ADR-L2-04](../../../../docs/TRD/l2.md) declara intencional que L2 trate el
instante de confirmación como atómico. Tras confirmar, la v0 sigue procesando
los ticks del mismo `transact_time`; L2 no. Cada discrepancia abre en la
confirmación de un instante con más de un tick, y la prueba lo verifica contra
los ticks del fixture. Por θ, los `agg_trade_id` de confirmación donde abre
cada ventana están fijados en `VENTANAS`.

- **Bug del detector: ninguno.** Fuera de esas ventanas los eventos coinciden
  campo a campo, y el estado final coincide en los 5 θ.
- **Redondeo float en un tick de frontera: ninguno en este día.** Un caso así
  aparecería como una ventana en una confirmación de un solo tick, y la prueba
  falla.
