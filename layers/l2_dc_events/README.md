# l2_dc_events

Capa L2: eventos Directional Change de 50 θ sobre la landing de L1. Diseño en
[`docs/TRD/l2.md`](../../docs/TRD/l2.md). Esta versión (ITSC-243) lee la
landing y alimenta el fan-out de [`dc_pyo3`](../../shared/dc_pyo3/README.md)
lote a lote; **no escribe** `events.parquet` ni `carry_over.parquet` todavía
(hijas siguientes de la Épica), y tampoco lleva Dockerfile, smoke ni entrada
en `LAYERS` del CI (E4a). Corre con `uv run`.

## Uso

```bash
export L2_LANDING_ROOT=... L2_EVENTS_ROOT=... L2_DQ_ROOT=...   # local o gs://
uv run python -m l2_dc_events --mode backfill --from 2017-08 [--to 2017-09] [--asset BTCUSDT]
```

- `--mode`: solo `backfill` por ahora (`monthly` llega con la escritura).
- La unidad es un mes. Se resuelve como en L1: `CLOUD_RUN_TASK_INDEX`
  (por defecto 0) elige el mes `--from + índice` dentro de `[--from, --to]`.
- Las tres raíces son obligatorias. Uso inválido (argumentos, raíces
  faltantes, índice fuera de rango) termina con código 2; una landing sin el
  mes, con código 1.

## Qué hace hoy una unidad

1. Abre `consolidated.parquet` del mes. Solo ese archivo (ADR-L2-09): si hay
   solo `provisional-day=DD.parquet`, falla diciéndolo.
2. Lo lee **row group por row group**, solo las columnas `price`,
   `transact_time` y `agg_trade_id` (`landing.py`).
3. Cada lote va a los 50 θ como buffers de Arrow sin copia, en tramos de
   65 536 ticks (`FEED_TICKS`), y se suelta antes de pedir el siguiente.
4. Al final cierra el grupo de empate abierto (`finish`) y registra cuántos
   eventos cerró cada θ y la sonda de RAM y tiempo.

Arranca siempre en frío: cargar el carry-over del mes anterior y guardar los
eventos es de la hija siguiente. `process_unit` ya acepta un `FanOut`
construido (`FanOut.from_carry_over`) para ese paso.

## Los 50 θ

[`config/thetas.yaml`](config/thetas.yaml): los enteros `round(θ × 10⁸)` de la
regla log-espaciada de ADR-L2-10. Los valores, no la fórmula, son la fuente de
verdad, y `tests/test_l2_thetas.py` verifica que siguen la regla (con 60
dígitos de precisión, para que ningún redondeo de float decida un empate). El
archivo se resuelve relativo al paquete, así que sirve con `uv run` y una
instalación editable; hornearlo en la imagen es de E4a.

## Pruebas

`uv run pytest layers/l2_dc_events`: los θ contra la regla del TRD, el lector
(un lote por row group, sin materializar la tabla: la memoria de Arrow es la
misma en el primer row group que en el último), el CLI, y los eventos a
través del lector y el binding contra el fixture de la v0 de `dc_core`
(idénticos con θ = 2 %; con los demás, hasta la primera divergencia declarada
de ADR-L2-04).

## Acoplamiento con L1

Ninguno por código (RNF-14 del maestro): L2 no importa `l1_ingest`. El único
contrato es el Parquet de la landing
([`docs/data-contracts.md`](../../docs/data-contracts.md)); por eso `cli.py`
repite la resolución de unidad de L1 en vez de compartirla.
