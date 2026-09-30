# l2_dc_events

Capa L2: eventos Directional Change de 50 θ sobre la landing de L1. Diseño en
[`docs/TRD/l2.md`](../../docs/TRD/l2.md). Lee el `consolidated.parquet` de un
mes, alimenta el fan-out de [`dc_pyo3`](../../shared/dc_pyo3/README.md) lote a
lote y escribe, por cada θ, los eventos del mes (`events.parquet`) y el estado
con que sigue el mes siguiente (`carry_over.parquet`). El contrato de la salida
está en [`docs/data-contracts.md`](../../docs/data-contracts.md) ("Salida
Parquet de L2"). Todavía no lleva Dockerfile, smoke ni entrada en `LAYERS` del
CI (E4a). Corre con `uv run`.

## Uso en local

```bash
export L2_LANDING_ROOT=... L2_EVENTS_ROOT=... L2_DQ_ROOT=...   # local o gs://
uv run python -m l2_dc_events --mode backfill --from 2020-01 [--to 2020-03] [--asset BTCUSDT]
uv run python -m l2_dc_events --mode monthly --from 2020-04 --series-start 2020-01
```

- `L2_LANDING_ROOT` es la raíz que escribió L1 (la ruta hive va debajo);
  `L2_EVENTS_ROOT`, dónde salen los eventos y el carry-over; `L2_DQ_ROOT`, el
  lago de hallazgos. Las tres son obligatorias.
- `--mode backfill` procesa la unidad `--from + CLOUD_RUN_TASK_INDEX` (por
  defecto 0) dentro de `[--from, --to]`. `--mode monthly` procesa un mes y
  **exige `--series-start`**.
- **`--series-start`** es el primer mes de la serie: el único que arranca sin
  carry-over previo. Se declara y no se infiere, porque "falta el carry-over"
  y "es el primer mes" se ven igual en disco y confundirlos corrompe la serie
  (ADR-L2-08). En `backfill` es `--from` si no se da.
- Los meses de una serie se procesan **en orden**: cada uno lee el carry-over
  del anterior. Para un rango, corre `--from` con `CLOUD_RUN_TASK_INDEX=0`,
  luego `1`, y así.
- Uso inválido (argumentos, raíces faltantes, índice fuera de rango) termina
  con código 2. Una entrada ausente o un carry-over que falta o es de otra
  versión, con código 1 y un hallazgo en el lago de DQ.

Para producir la landing de un mes en local con L1 (los ZIP viven en RAM: un
mes de 2020 cabe holgado en los 7 GiB del contenedor):

```bash
export L1_LANDING_ROOT=$PWD/sandbox.local/x/landing L1_DQ_ROOT=$PWD/sandbox.local/x/dq L1_MANIFEST_ROOT=$PWD/sandbox.local/x/manifest
uv run python -m l1_ingest --mode backfill --from 2020-01
export L2_LANDING_ROOT=$L1_LANDING_ROOT L2_EVENTS_ROOT=$PWD/sandbox.local/x/events L2_DQ_ROOT=$PWD/sandbox.local/x/dq_l2
uv run python -m l2_dc_events --mode backfill --from 2020-01
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
4. Los eventos que cada tramo cierra van a 50 escritores abiertos
   (`events.py`), que los vuelcan a columnas y los escriben en row groups de
   hasta 32 768 filas. Nunca se acumula el mes de un θ: la RAM es O(lote).
5. Al final cierra el grupo de empate abierto (`finish`, RF-L2-12) y publica:
   por cada θ, primero `events.parquet` y luego `carry_over.parquet`, cada uno
   con escritura atómica (temporal más `commit`). Un carry-over presente
   significa que el mes de ese θ quedó completo.
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

[`config/thetas.yaml`](config/thetas.yaml): los enteros `round(θ × 10⁸)` de la
regla log-espaciada de ADR-L2-10. Los valores, no la fórmula, son la fuente de
verdad, y `tests/test_l2_thetas.py` verifica que siguen la regla (con 60
dígitos de precisión, para que ningún redondeo de float decida un empate). El
archivo se resuelve relativo al paquete, así que sirve con `uv run` y una
instalación editable; hornearlo en la imagen es de E4a.

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

El detector no es el cuello. Perfilado con `cProfile` sobre 2020-01 (los
tiempos absolutos se inflan por el perfilador; la proporción es la que sirve):

| Paso | Tiempo aproximado |
|---|---|
| `write_table` (Parquet ZSTD-3 de 12 M de filas) | ~8 s |
| Volcar los objetos `Event` a columnas (`EventColumns.extend`) | ~12 s |
| `feed_batch`: los 50 θ, multihilo | ~1,6 s |
| Leer los row groups | ~0,6 s |
| Hash de contenido | ~0,6 s |

Los 50 θ sobre 14 M ticks cuestan ~1,6 s de cómputo, unos 0,9 M ticks/s por
core incluyendo la creación de los objetos `Event`. El resto es trabajo serial
de Python: materializar 12 M de eventos como objetos y codificarlos a Parquet,
con casi todos los demás cores ociosos. La ganancia está ahí, no en el detector: que
`dc_pyo3` devuelva los eventos ya en columnas y que los θ se escriban en
paralelo (pyarrow libera el GIL al codificar). Es una decisión de E4a, y por
eso los ticks/s por core de arriba son los de la unidad entera y no los 2,8 M
del benchmark de [`dc_core`](../../shared/dc_core/README.md), que mide solo el
detector.

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
- `test_l2_output_contract_doc.py`: que las tablas de `docs/data-contracts.md`
  coincidan con `EVENTS_SCHEMA` y `CARRY_OVER_SCHEMA`.
- `test_l2_cli.py`: el CLI, incluida la cadena de dos meses y el fail-closed.

## Acoplamiento con L1

Ninguno por código (RNF-14 del maestro): L2 no importa `l1_ingest`. El único
contrato es el Parquet de la landing
([`docs/data-contracts.md`](../../docs/data-contracts.md)); por eso `cli.py`
repite la resolución de unidad de L1 y `write.py` copia el patrón de
`PartitionWriter` y `ContentHasher` en vez de importarlo. Extraerlo a
`shared/` es una card aparte.
