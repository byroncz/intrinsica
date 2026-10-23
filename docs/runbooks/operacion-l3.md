# Runbook: operación de L3 (backfill histórico, lectura de fallos, cascada, cadena mensual, volumen y costo, perfil de una unidad, hilos de la unidad, verificación cruzada, comparación de corridas)

Lo ejecuta el humano desde *Actions → Run job* y Cloud Shell; ningún agente
dispara `terraform.yml` ni `run-job.yml`. Dice cómo se opera L3 una vez construido
su histórico: cómo se lanza, cómo se lee un fallo, cómo se relanza un mes, qué
hacer cuando L1 reprocesa, cómo corre la cadena mensual y con qué scripts se
comprueba. Referencia: [TRD-L3 §8, §9.3, §10 y §11](../TRD/l3.md). Mismo patrón que
el [runbook de L2](operacion-l2.md); aquí solo cambia lo que L3 hace distinto. Las
cifras de dimensionamiento salieron de la sonda y del histórico de ITSC-335.

Fechas calculadas al **2026-10-15**. Si lo ejecutas otro día, recalcula el
rango con la sección siguiente.

## Cambios requeridos

**ITSC-368 (imagen de L3 ≥ 0.15.0): el backfill del linaje va antes del apply.**
Desde esta imagen L3 no recalcula los `content_hash` de las entradas de L2 (le
costaba ≈ 100 s de CPU por unidad): los lee de `content_hash.json`, que L2 deja
junto a `events.parquet` y `carry_over.parquet` desde su 0.11.0
([TRD-L2 §7.5](../TRD/l2.md#75-content_hashjson--la-huella-publicada-contrato-hacia-l3)).
Las particiones publicadas antes no lo tienen. Orden obligatorio, todo lo
dispara y aprueba el humano:

1. **Merge** del PR de ITSC-368 (CI publica `l2_dc_events`, `l3_dc_descriptors` y
   `ops_tools` con su `VERSION` nueva).
2. **Apply de `ops` y de `l2`** (*Actions → Terraform*). El de `ops` da al job
   `ops-script` el permiso de **crear** `content_hash.json` bajo `l2/` de
   `dc-events` (nada más: no pisa ni borra); el de `l2`, la imagen que lo escribe
   en los meses nuevos.
3. **Backfill del linaje** con `ops-script`: sube y lanza
   [`l2_backfill_lineage.py`](operacion-l2.md#linaje-publicado-y-su-backfill-itsc-368)
   (la guía paso a paso está en el runbook de L2). Antes, con `--dry-run`, mira que
   no haya discrepancias. La última línea dice cuántas particiones escribió y cuántas ya existían: su suma
   debe ser la de L2 (50 θ por cada mes publicado), sin discrepancias, errores ni
   particiones incompletas.
4. **Apply de `l3`** con la imagen nueva.
5. **T8** (2021-01, valores por defecto): `l3-backfill` con `from` = `to` =
   `2021-01` y `force`. Metas y dónde se anota el resultado: TRD-L3 §10.2, fila T8.

Si el paso 4 va antes que el 3, la unidad falla con `l2_lineage_missing`, sin
publicar nada y sin avanzar la frontera: es lo previsto. Corre el backfill y
relanza los mismos inputs.

## Rango

- **Serie**: arranca en `2017-08` (`L3_SERIES_START` del stack `l3`), igual que
  L2. L3 no tiene estado entre meses ([ADR-L3-12](../TRD/l3.md)): cada mes se
  resume solo, leyendo hacia atrás los ticks que necesite.
- **Fin**: `l3-backfill` sin `to` llega hasta el último mes con
  `consolidated.parquet` en L1 (nunca provisionales). A 2026-10-15 es
  **2026-09**: el consolidado de 2026-10 se publica el primer lunes de
  noviembre. Son **110 meses** (2017-08 a 2026-09) y 50 θ: **5 500
  `summaries.parquet`**.
- **Un mes de L3 solo existe si L2 ya lo tiene.** La frontera de L3 nunca
  supera la de L2 (TRD-L3 §8.2): si L2 se atrasa, L3 se atrasa.

## Prerrequisitos

1. **Stacks aplicados.** `data` (lo aplica el humano desde Cloud Shell: crea el
   bucket `dc-descriptors` y da a `deploy-github` su IAM), `l3` (*Actions →
   Terraform*, `stack` = `l3`, `action` = `apply`, aprobado en el environment
   `gcp`) y `ops` (para los scripts de abajo: lee `l3/` desde ITSC-336). Orden
   y detalle en [infra/README.md](../../infra/README.md). Verifica los jobs:

   ```bash
   gcloud run jobs list --region <región> | grep l3
   gcloud run jobs describe l3-backfill --region <región> \
     --format='yaml(spec.template.spec.template.spec)'
   ```

   Deben existir `l3-backfill` y `l3-monthly`; el primero con `timeoutSeconds:
   86400`, `maxRetries: 1`, `cpu: '4'` y `memory: 4Gi` en `limits`. Las raíces
   (`L3_LANDING_ROOT`, `L3_EVENTS_ROOT`, `L3_SUMMARIES_ROOT`, `L3_DQ_ROOT`),
   `L3_SERIES_START` y `L3_THETAS_URI` van en `env`; `L3_THREADS`, `L3_READERS`, `L3_INFLIGHT`,
   `L3_WRITERS`, `L3_ENGINE_THREADS`, `L3_INFLIGHT_BYTES`, `L3_LOT_EVENTS` y `L3_UNIT_PROCESS` son opcionales (ver "Hilos y topes de la unidad" y "Memoria acotada").
2. **Imagen `l3_dc_descriptors`** de `layers/l3_dc_descriptors/VERSION`. El stack
   `l3` deriva el tag de ese archivo: sin `apply` posterior al merge, el job
   sigue con la imagen vieja.
3. **El catálogo de θ es el de L2**: `gs://<proyecto>-manifest/l2/thetas.yaml`.
   L3 no lo siembra ni lo copia ([TRD-L2 §7.3](../TRD/l2.md#73-el-catálogo-de-θ)).
   Si falta o es inválido, la corrida termina con código 2 y el hallazgo
   `theta_catalog_invalid`. Cómo se edita está en el
   [runbook de L2](operacion-l2.md#el-catálogo-de-θ).
4. **L2 completo** (`events.parquet` y `carry_over.parquet` de cada θ y mes) y
   **L1 completa** (`consolidated.parquet` de cada mes). Un θ al que le falta un
   mes de L2 sale del rango (`l2_input_missing`); un mes sin consolidado detiene
   todo (`input_missing`).
5. **Linaje de L2 publicado** (`content_hash.json` en cada partición de L2,
   ITSC-368). L3 lo lee y falla con `l2_lineage_missing` si falta. Es lo primero
   que hay que cubrir antes del apply de esta imagen: ver "Cambios requeridos".
6. Variables del environment `gcp` cargadas (`GCP_PROJECT_ID`, `GCP_REGION`,
   `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SERVICE_ACCOUNT`).

## Cómo funciona `run-job.yml` con L3

`l3-backfill` va en **una sola tarea** y un solo hilo, que recorre los meses en
orden. `l3-monthly` resume un mes.

| Job | `from` | `to` | `force` |
| --- | --- | --- | --- |
| `l3-backfill` | Vacío: la frontera de cada θ decide desde dónde avanza | Vacío: hasta el último mes con consolidado en L1. Si lo pasas, el workflow lo manda siempre, aunque sea igual a `from` | Reprocesa desde `from` e ignora la frontera; **exige `from`** |
| `l3-monthly` | El mes; vacío = el mes anterior | Vacío o igual a `from`; **la CLI rechaza otro valor** (código 2) | **No lo admite**: el workflow falla antes de lanzar. Usa `l3-backfill` con `from` |

`series_start` vacío usa el del stack. El workflow no expone `--thetas` ni
`--family`: por *Run job* siempre corren los 50 θ del catálogo y la familia
`summaries`. No lances a mano el mismo mes y modo que Scheduler esté ejecutando,
ni dos `l3-backfill` a la vez.

Al final, el run vuelca los logs de la ejecución y un resumen (estado, tareas,
inicio y fin) en el *Summary* del run. La CLI termina con **0** (todo avanzó o ya
estaba al día), **1** (falló la unidad: entrada ausente, L2 incoherente,
conciliación, esqueleto; o `monthly` sin ningún θ listo con rezagados) o **2**
(uso o configuración; no escribe datos). Detalle en
[TRD-L3 §8.4](../TRD/l3.md#84-códigos-de-salida).

## Backfill histórico por rangos

**Con el núcleo nuevo cabe en una sola ejecución.** La suma de los 110 meses son 4 828 s
(1,34 h; ITSC-364), bajo las 6 h que espera el run de Actions: `from` = `2017-08`, `to`
vacío y `force`, una vez. Sin `force`, o si una ejecución muere, se relanza y retoma donde
quedó. Lo que sigue (rangos) es de **ITSC-335, con el núcleo original**, cuya suma eran 11,5 h
(41 256 s), más que las 6 h que espera el run de Actions: Cloud Run no cortaba la tarea (el
timeout es el tope del módulo, 86 400 s), pero el run fallaría. Por eso entonces el
histórico se lanzaba por rangos, **uno a la vez y cada uno con éxito antes del siguiente**.
Los de ITSC-335:

| # | `from` | `to` | Resultado |
| --- | --- | --- | --- |
| 1 | `2017-08` | `2020-12` | 4 h 57 min (41 meses) |
| 2 | `2021-01` | `2021-12` | 2 h 09 min |
| 3 | `2022-01` | `2023-02` | 2 h 07 min |
| 4 | vacío | `2024-12` | 1 h 07 min |
| 5 | vacío | `2026-08` | Falló en 2026-02 (`l2_state_inconsistent`: ids repetidos de L1; ver "Cascada") |
| 6 | vacío | vacío | 19 min, tras ITSC-351: 2026-02 a 2026-08 y reconstrucción de 2026-09 |
| 7 | vacío | vacío | Todo al día: salida 0, sin escrituras |

El detalle de cada run está en [infra/README.md](../../infra/README.md#stack-l3-descriptores-por-evento).

**Regla para un backfill futuro de la serie completa** (por ejemplo, al agregar
una familia): si la suma de los `wall_s` esperados pasa de **≈ 5 h (18 000 s)**, fija cada
rango por su **pared estimada**, no por su número de meses. Con el núcleo nuevo la serie
suma 4 828 s para `summaries.parquet`, así que basta una ejecución mientras la familia no
sea ≈ 4 veces más lenta. Estima cada mes con el ajuste de
[TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición)
(`wall_s` ≈ 68 s + 0,038 s por millón de ticks + 24,9 s por millón de eventos; es del
**núcleo original** y sobrestima la suma unas 8,5 veces: no se reajustó con el nuevo) o con los
`unit_timing` de `dq-findings/l3/` y el CSV de `l3_compare_runs.py --timing`. La pared sigue a los **eventos**,
más que a los ticks: 2021-01 (46,0 M de eventos) tomó 1 419 s y 2023-03, el mes
con más ticks, 873 s. Los 41 meses de 2017-08 a 2020-12 promediaron ≈ 7,2 min por
mes; 2021 promedió ≈ 10,75 min. Lo más simple es repetir los rangos de la tabla.

La frontera de cada θ es la cadena **contigua desde `series_start`**: un rango
sin `from` arranca en el mes siguiente al último construido, y un mes hecho fuera
de orden (como 2023-03 y 2026-09 en ITSC-335) se reconstruye al llegar a él, con el
mismo `content_hash`. Para no repetirlo, termina el rango un mes antes.
Relanzar un rango interrumpido es seguro: la frontera retoma donde quedó.

### Timeouts: la regla de TRD-L2 §10.2 sobre la pared del núcleo nuevo (ITSC-364)

La regla ([TRD-L2 §10.2](../TRD/l2.md)): `monthly` ≥ 6 × la pared del mes más pesado, y
`backfill` ≥ 1,5 × la suma de los meses y ≤ 86 400 s (el tope del módulo `layer`). Con la
medición de ITSC-364 (mes más pesado 141,4 s en 2022-11; suma 4 828 s):

| Job | Desplegado hoy (núcleo original) | Recalculado | Cuenta |
| --- | --- | --- | --- |
| `l3-monthly` | 10 800 s | **900 s** | 6 × 141,4 = 849 s, redondeado hacia arriba: 6,4 × |
| `l3-backfill` | 86 400 s | **10 800 s** | 1,5 × 4 828 = 7 242 s, redondeado a 3 h: 2,2 × |

Los valores nuevos **no están aplicados**: los cambia ITSC-352 en `infra/` junto con el tamaño
(vCPU). Hasta entonces rigen los desplegados, que sobran. Son una cota, porque la suma corrió
con el asignador de T5 y con el de la imagen cada mes denso es más rápido. Si un job termina
por timeout con estos valores, mira primero `unit_timing` del mes: una unidad que pasa de
3 × su pared habitual es un hallazgo, no una razón para subir el timeout.

## Leer un fallo

Un fallo detiene el rango (los meses siguientes no se tocan) y deja un hallazgo
en `gs://<proyecto>-dq-findings/l3/detected_date=<fecha>/`. Primero distingue la
causa en el log de la ejecución:

- **OOM** ("Memory limit exceeded" o `signal 9`): no hay línea `unidad <mes>:` de ese mes.
  El pico medido con el núcleo nuevo es 3 545 MiB de 4 GiB, en 2017-12 (ITSC-364), por la
  densidad de eventos y no por el asignador; es el caso que acota ITSC-372. Un OOM en un
  mes más denso, o en uno chico tras varios meses, es el hallazgo: mira
  `rss_before_trim_mib` y `rss_after_trim_mib` de los meses anteriores y la sección
  «El asignador».
- **Timeout**: la tarea se corta sin mensaje de memoria. El de `l3-monthly` desplegado es de
  10 800 s (76 veces el mes más pesado con el núcleo nuevo, 141,4 s; ver «Timeouts»).
- **Un hallazgo `error`**: lee su `check_type` en la tabla siguiente.

Los 16 `check_type` de [TRD-L3 §9.3](../TRD/l3.md#93-tipos-de-chequeo-check_type),
y qué hacer con cada uno. En todos los casos de `error`, **el mes no se publica y
la frontera no avanza**: relanzar los mismos inputs retoma donde quedó. Nunca
repares un archivo de `l3/` a mano.

| `check_type` | Severidad | Qué significa | Qué hacer |
| --- | --- | --- | --- |
| `phase_quantity_mismatch` | `error` | `buy + sell` de una fase no es la suma de `quantity` de sus ticks | No debería ocurrir: es un defecto del núcleo o de los datos. Abre una card con `task-create` con `details` (θ, `confirm_agg_trade_id`, fase, los dos valores). No relances: dará lo mismo |
| `family_skeleton_mismatch` | `error` | Un archivo de familia no repite filas, orden, claves o row groups del esqueleto | Compara `details` con `events.parquet` de L2 del mes. Si L2 cambió después, relanza el mes con `force` (ver "Relanzar un mes"). Si persiste, card |
| `event_over_budget` | `warning` | Un evento supera el presupuesto de ticks (solo familias de serie completa) | `summaries` no tiene presupuesto y no lo emite. Informa; no falla la unidad |
| `l2_input_missing` | `error` | Falta `events.parquet` o `carry_over.parquet` de `(θ, M)` | Completa L2 para ese θ y mes ([runbook de L2](operacion-l2.md)); luego relanza `l3-backfill` con los mismos inputs. Ese θ salió del rango y los demás siguieron (código 1) |
| `l2_lineage_missing` | `error` | La partición de L2 de `(θ, M)` no tiene `content_hash.json` o no es válido; `details` trae θ, mes, archivo y `reason` | Corre el backfill del linaje ([L2](operacion-l2.md#linaje-publicado-y-su-backfill-itsc-368)), que escribe el de las particiones que no lo tienen; si el mes es nuevo (publicado por una imagen de L2 anterior a la 0.11.0), vuelve a correr el backfill. Luego relanza `l3-backfill` con los mismos inputs. L3 no recalcula el linaje por su cuenta, a propósito |
| `l2_state_inconsistent` | `error` | L2 contradice a L1 o a un invariante de fila (TRD-L3 §9.2); `details` trae θ, mes, campo y los dos valores | Descarta primero L1: ids repetidos o marcas del proveedor (ver "Cascada"). Si L1 está sano, es un defecto de L2 o de L3: card con `details`. Es el hallazgo que detectó los duplicados de 2026-02 |
| `phase_ticks_mismatch` | `error` | No cuadra la conciliación de ticks de §9.1; `details` trae `N_M`, `Σ_cerrados`, `P_M`, `Cola_M` y `U_M` | Corre el seam-check de [L1](operacion-l1.md) y el de [L2](operacion-l2.md#paso-3-seam-check-de-continuidad) sobre el mes. Si ambos pasan, card con `details` |
| `input_missing` | `error` | El mes no tiene `consolidated.parquet` ni provisionales | Corre `l1-backfill` o `l1-monthly-close` del mes ([runbook de L1](operacion-l1.md)); luego relanza |
| `input_provisional_only` | `error` | El mes solo tiene `provisional-day=DD.parquet` | Espera el cierre de L1 (el consolidado sale el primer lunes del mes siguiente) o lanza `l1-monthly-close`; luego relanza |
| `theta_catalog_invalid` | `error` | El catálogo no existe, no se lee o incumple TRD-L2 §7.3. Código 2, sin escribir | Corrige `manifest/l2/thetas.yaml` ([runbook de L2](operacion-l2.md#el-catálogo-de-θ)); relanza |
| `theta_behind_frontier` | `warning` | `monthly` no procesó un θ; `details.reason` es `l3_frontier` (la serie de L3 tiene un hueco o no existe) o `l2_missing` (L2 no tiene el mes) | `l3_frontier`: lanza `l3-backfill` sin `from` ni `to`, que retoma desde la frontera de cada θ. `l2_missing`: completa L2 primero. No falla la unidad si otros θ avanzan |
| `theta_config_drift` | `info` | L3 tiene particiones de un θ que ya no está en el catálogo | Informa. Quitar un θ del catálogo no borra nada; si fue un error, restáuralo |
| `summaries_summary` | `info` | Uno por θ y mes: filas, `content_hash` del archivo y de las entradas de L2 que usó | Es el linaje contra L2: sirve para comparar corridas y para detectar un mes obsoleto |
| `unit_timing` | `info` | Uno por unidad (mes): pared, tiempo por fase, CPU efectiva y consumida (`cpu_s`, `cpu_utilization`) y bytes leídos | Es la fuente de las cifras de "Volumen y costo" |
| `unit_process_died` | `error` | El proceso hijo de la unidad terminó sin resultado (`exit_signal` 9 es el OOM del cgroup, `signal_name` `SIGKILL`) | La unidad no se publicó y el rango se detuvo. Ver "Memoria acotada: un proceso por unidad y `L3_INFLIGHT_BYTES`" |
| `unit_profile` | `info` | Solo con `L3_PROFILE=1`, uno por unidad: las 40 funciones más caras por tiempo propio y acumulado | Es para proporciones, no para absolutos. Ver "Perfilar una unidad" |

Para leer los hallazgos de una corrida desde Cloud Shell, **baja primero solo las
carpetas de los días de la corrida y léelas en local**. Nunca abras un
`pyarrow.dataset` sobre `gs://<proyecto>-dq-findings/l3/` entero: descubre todos los
archivos del lago y lee el pie de cada uno por la red, y no termina (comprobado el
2026-10-17). Tampoco uses `pyarrow.fs` contra `gs://`: pide el token al servidor de
metadatos y falla si `gcloud` no tiene cuenta activa. (Desde `ops-script` aplica además
"La trampa del 403".)

Las particiones son **`detected_date=AAAA-MM-DD`** (el día en que corrió el job), no
el mes de los datos: el mes está en las columnas `year` y `month` de cada archivo. Por
eso se baja por la fecha de la corrida y se filtra por mes dentro del archivo.

```bash
# 1. Solo los días de la corrida (acepta comodines: detected_date=2026-10-1*).
rm -rf ~/dq-l3 && mkdir -p ~/dq-l3
gcloud storage cp -r "gs://<proyecto>-dq-findings/l3/detected_date=2026-10-17" ~/dq-l3/

# 2. Lee en local. MONTH es el mes de los datos; CHECK, el check_type.
DQ_LOCAL=~/dq-l3 MONTH=2026-09 CHECK=summaries_summary python3 - <<'PY'
import json
import os

import pyarrow.dataset as ds

year, month = (int(x) for x in os.environ["MONTH"].split("-"))
flt = (
    (ds.field("check_type") == os.environ["CHECK"])
    & (ds.field("year") == year)
    & (ds.field("month") == month)
)
table = ds.dataset(os.environ["DQ_LOCAL"], partitioning="hive").to_table(
    filter=flt, columns=["run_id", "detected_at", "details"]
)
for row in table.to_pylist():
    print(row["run_id"], row["detected_at"], json.dumps(json.loads(row["details"]))[:300])
PY
```

Variante para una corrida con perfil: imprime las claves de `unit_timing` del mes y las
15 primeras filas de `unit_profile`, por tiempo propio y por tiempo acumulado.

```bash
DQ_LOCAL=~/dq-l3 MONTH=2021-01 python3 - <<'PY'
import json
import os

import pyarrow.dataset as ds

year, month = (int(x) for x in os.environ["MONTH"].split("-"))
dataset = ds.dataset(os.environ["DQ_LOCAL"], partitioning="hive")


def details(check):
    flt = (
        (ds.field("check_type") == check)
        & (ds.field("year") == year)
        & (ds.field("month") == month)
    )
    table = dataset.to_table(filter=flt, columns=["detected_at", "details"])
    rows = sorted(table.to_pylist(), key=lambda r: r["detected_at"])
    return [json.loads(r["details"]) for r in rows]


for timing in details("unit_timing"):
    print("unit_timing:", ", ".join(timing))
    print({k: timing[k] for k in ("wall_s", "cpu_s", "cpu_utilization", "cores")})
for profile in details("unit_profile"):
    print(f"unit_profile: wall_s={profile['wall_s']}")
    for key in ("top_tottime", "top_cumtime"):
        print(key)
        for r in profile[key][:15]:
            print(
                f"  {r['tottime_s']:9.3f} {r['cumtime_s']:9.3f} {r['ncalls']:>10} "
                f"{r['function']} ({r['file']}:{r['line']})"
            )
PY
```

## Relanzar un mes

Un mes ya construido se reescribe con `force`, para los 50 θ, con el mismo
contenido si las entradas no cambiaron (`content_hash` idéntico: RNF-L3-03).

| Input | Valor |
| --- | --- |
| `job` | `l3-backfill` |
| `from` | el mes `M` |
| `to` | **el mismo mes `M`** |
| `force` | **marcado** |
| `series_start` | vacío |

**`to` no se deja vacío**: con `force` y `to` vacío el backfill reprocesaría de
`M` hasta el último mes. Debe dejar una sola línea `unidad <mes>: θ=50 eventos=… wall_s=…`. Si el mes
nunca falló y solo quieres que avance, no uses `force`: relanza los mismos inputs
y la frontera hace el resto.

## Agregar un θ

Sin PR ni reproceso de los θ existentes. Desde Cloud Shell, edita el catálogo
único como describe el [runbook de L2](operacion-l2.md#agregar-θ-nuevos), y en este
orden:

1. `l2-backfill` con todo vacío: solo el θ nuevo recorre la serie.
2. `l3-backfill` con todo vacío: los θ que ya llegaron al último mes se saltan y
   **solo el nuevo** recorre la serie desde `2017-08`. Mientras L2 no termine,
   `l3-monthly` deja `theta_behind_frontier` para el nuevo.

Un θ pequeño domina el volumen (`dc-descriptors` crece ≈ 45 B por evento y θ).

## Cascada cuando L1 reprocesa un mes

Si L1 corrige un mes `M`, quedan obsoletos L2 (cuyo carry-over encadena los
meses siguientes), los tiles de viz y las filas de L3 de los eventos que leyeron
ticks de `M`. El orden y los comandos están en el
[runbook de L1, "Ids repetidos"](operacion-l1.md#ids-repetidos-itsc-351):
`l1-backfill` con `force` sobre `M`; seam-check de L1; `l2-backfill` con `force`,
`from = M` y `to` = el último mes; `viz-tiles` con `force` en el mismo rango; y,
al final, **`l3-backfill` con `force`, `from = M` y `to` vacío**, que debe
completar sin `l2_state_inconsistent`. Cada paso con éxito antes del siguiente.

Caso real (ITSC-351): L1 había dejado 2 000 `agg_trade_id` repetidos en 2026-02
(3 000 filas sobrantes). L2 los procesó como precios reales sin error y fue la
compuerta de L3 la que los detectó (`l2_state_inconsistent`, rango 5 del
histórico). L2 no cambió de contenido; sí cambió L3 desde 2026-02.

## La cadena mensual

`monthly-close` de L1 dispara el workflow `l1-run-job`, que encadena
**`l1-monthly-close` → `l2-monthly` → `viz-tiles` → `l3-monthly`**, cada uno solo
si el anterior terminó bien y sin overrides (cada job toma el mes anterior). L3
va al final porque depende del cierre de L2 y de que viz no quede detrás
(decisión del 2026-10-13, [TRD-L3 §11](../TRD/l3.md#11-operaciones)). El workflow
no espera a `l3-monthly`: sigue su ejecución en *Cloud Run → Jobs → l3-monthly*.

**Si `viz-tiles` falla un mes, `l3-monthly` no corre.** Costo asumido del diseño.
Qué hacer:

1. Resuelve `viz-tiles` con el [runbook de viz](operacion-viz.md) y relánzalo
   (*Run job*, `job` = `viz-tiles`, `from` = el mes).
2. Lanza **`l3-monthly` a mano** (*Run job*, `job` = `l3-monthly`, `from` = el
   mes, `to` y `force` vacíos). No vuelvas a lanzar `monthly-close`: relanzaría
   `l2-monthly`.
3. Verifica que no queden `theta_behind_frontier`: si los hay,
   `l3-backfill` sin `from` ni `to` los pone al día.

Un `l3-monthly` solo termina con código 1 si ningún θ estaba listo y había
rezagados (un mes perdido no pasa como éxito); con algún θ al día o nada que
procesar termina con 0. El scheduler de `monthly-close` está pausado en el repo;
la primera prueba real de la cadena completa la cubre la card que enciende los
schedulers, no ésta.

## Volumen y costo

Medidos sobre el histórico completo: el volumen y el núcleo original, en ITSC-335
(2026-10-15), y el núcleo nuevo, en ITSC-364 (2026-10-22). Detalle en
[TRD-L3 §10.2, §10.4 y §10.5](../TRD/l3.md#104-tamaño-de-l3--estimación-por-fila).

| Medida | Valor |
| --- | --- |
| `dc-descriptors/l3/` | 46 742 260 408 B (**43,5 GiB**) en 5 500 `summaries.parquet`; 2,06 veces L2; ≈ 45 B por evento (1 042 M de filas) |
| Suma de `wall_s` de los 110 meses | **Núcleo nuevo (ITSC-364): 4 828 s (1,34 h); 18 992 vCPU-s (19 312 con los 4 asignados).** Núcleo original (ITSC-335): 41 256 s (11,5 h); ≈ 165 000 vCPU-s y GiB-s |
| Costo de lista (0,000018 USD por vCPU-s y 0,000002 por GiB-s) | **≈ 0,4 USD** con el núcleo nuevo; ≈ 3,3 USD con el original. Un cálculo sobre `wall_s`, no una cifra de Billing |
| Mes más pesado | Núcleo nuevo: 2022-11, 141,4 s. Original: 2021-01, 1 419 s (46,0 M de eventos); 2023-03, el máximo por ticks, 873 s |
| `cpu_utilization` (núcleo nuevo) | p10 0,312, p50 0,510, mínimo 0,116 (2017-08); ningún mes llega a 0,80 |
| RSS pico máximo | Núcleo nuevo: 3 545 MiB (2017-12), 86 % de 4 GiB. Original: 1 267 MiB (2018-01), 31 % |
| Cupo gratuito | Es de la cuenta de facturación, no de la capa: L3 fue el 41 % del CPU de octubre y los créditos cubrieron el costo (facturado 0) |
| Estado estacionario (`l3-monthly`) | Como máximo lo que el mes más pesado: centavos |

Para repetir la medición:

```bash
gcloud storage du -s -h gs://<proyecto>-dc-descriptors/l3
```

Los `unit_timing` (pared y fases por mes) y los `summaries_summary` (filas y
`content_hash` por θ y mes) están en `gs://<proyecto>-dq-findings/l3/`; el
snippet de "Leer un fallo" (baja primero la carpeta `detected_date=` del día) con `CHECK=unit_timing` o `CHECK=summaries_summary` los lee, y
`details` de `unit_timing` trae la pared y el tiempo de cada fase. La pared
facturable de una tarea es la de la ejecución
(`gcloud run jobs executions describe <ejecución> --format='value(status.startTime,status.completionTime)'`),
que incluye el arranque. Billing tarda hasta 24 h en reflejar el consumo y no
separa por job.

## Perfilar una unidad

Para saber en qué funciones se gasta la pared de un mes (por ejemplo, cuánto de la
escritura es `_check_keys`, la compresión ZSTD o Python) se corre la unidad con
`L3_PROFILE=1`: corre bajo `cProfile` y deja, además de sus hallazgos de siempre, un
`unit_profile` con las 40 funciones más caras por tiempo propio (`top_tottime`) y por
tiempo acumulado (`top_cumtime`). Está en
[TRD-L3 §9.3](../TRD/l3.md#93-tipos-de-chequeo-check_type).

**Es para proporciones, no para absolutos.** `cProfile` instrumenta cada llamada y
añade entre 30 % y 100 % de pared, más a las funciones pequeñas y muy llamadas. No
compares el `wall_s` de una corrida perfilada con el de una sin perfil; el `unit_timing`
de la corrida perfilada también sale inflado.

`run-job.yml` no admite variables de entorno, así que el perfil se lanza **desde Cloud
Shell**. `--update-env-vars` en `execute` vale solo para esa ejecución: el job no
cambia. Los `--args` son los mismos que arma `run-job.yml` para `l3-backfill` con
`from = to = 2021-01` y `force` (los imprime
`.github/scripts/run-job-args.sh l3-backfill 2021-01 2021-01 true ""`):

```bash
gcloud run jobs execute l3-backfill \
  --project <proyecto> --region <región> \
  --update-env-vars L3_PROFILE=1 \
  --args="--mode,backfill,--from=2021-01,--to=2021-01,--force" \
  --wait
```

Un mes pesado como 2021-01 tarda unos 1 419 s sin perfil y hasta el doble con él
(≈ 5 700 vCPU-s sin perfil, ≈ 0,10 USD de lista). `force` reescribe el mes con el
mismo contenido: los `content_hash` de los 50 θ deben coincidir con los de la corrida
anterior. No lances otra ejecución de `l3-backfill` a la vez.

Para leer el resultado, baja la carpeta `detected_date=` del día y corre la variante con
perfil del snippet de "Leer un fallo": imprime las claves de `unit_timing` y las 15
primeras filas de `unit_profile`.

## Hilos y topes de la unidad: `L3_READERS`, `L3_INFLIGHT`, `L3_WRITERS`, `L3_ENGINE_THREADS` y `L3_THREADS`

Cada mes es un orquestador de E/S que alimenta al motor nativo
([ADR-L3-16](../TRD/l3.md#616-adr-l3-16--motor-de-agregación-nativo-ítems-por-row-group-y-θ-casillas-por-evento-fusión-asociativa)):
Python lee, escribe y mide; los hilos nativos del motor calculan. Son opcionales y se dejan sin fijar:

| Variable | Qué fija | Por defecto |
|---|---|---|
| `L3_READERS` | R: row groups de L1 que se leen y decodifican a la vez. Entero ≥ 1 | 2 |
| `L3_INFLIGHT` | K: row groups empujados al motor que aún anclan sus ticks; el orquestador no pide el K + 1 hasta que uno termina. Entero ≥ 1 | 4 |
| `L3_WRITERS` | W: escritores de row groups de salida (cada θ en orden, uno a la vez por θ). Entero ≥ 1. 12 es el techo medido: con 24 la pared empeoró 17 % (T4c) | 12 |
| `L3_ENGINE_THREADS` | Hilos nativos del motor. Entero ≥ 1 | los vCPU efectivos redondeados (`round(cores)` de `unit_timing`): 4 en `l3-backfill` |
| `L3_THREADS` | Pool de E/S: pasada de entrada de cada θ (compuerta y apertura), lectura de las fronteras de L2, conciliación y publicación. Entero ≥ 1; nunca se usan más que θ | `ceil(3 × cores)`, nunca más que θ: 12 a 4 vCPU y 24 a 8 |

El stack `l3` no fija ninguna: los valores siguen a los vCPU y el resultado (tablas y
`content_hash`) es el mismo en cualquier combinación. Un valor que no sea un entero ≥ 1 termina
la corrida con código 2, sin escribir nada. `L3_THREADS=1` quita el pool de E/S (la compuerta, la
apertura, la conciliación y la publicación van en serie y las fronteras de L2 se leen en el hilo
principal); sirve de línea base. `L3_PREFETCH` ya no existe.

**Qué cuesta cada uno en memoria.** Cada row group de L1 en vuelo suma ≈ 48 MB anclados (K) o,
mientras se decodifica, ≈ 1,8 × 48 MB (R, por los temporales de Arrow). **Los escritores** pesan por lo
sellado que esperan: el orquestador no empuja otro row group mientras haya
`PENDING_PER_WRITER × W` row groups de salida sin escribir. Cada uno ocupa ≈ 8 MiB (256 B por
evento × ≈ 34 000 eventos por row group de L2 en 2021-01). Con `PENDING_PER_WRITER = 3` y
W = 12 el tope son 36 row groups ≈ 290 MiB, igual que los 8 × 4 de antes; con 8 y W = 12 eran
96 ≈ 770 MiB y T4b midió `rss_peak_mib` 2 331 contra 1 864 de T4 (W = 4). Si cambias W,
rehaz la cuenta: `3 × W × 8 MiB`. La cuenta completa está en
[TRD-L3 ADR-L3-16, «Memoria del orquestador»](../TRD/l3.md#616-adr-l3-16--motor-de-agregación-nativo-ítems-por-row-group-y-θ-casillas-por-evento-fusión-asociativa).
Se acepta una combinación si `rss_peak_mib` queda < 3 500 a 4 vCPU con el asignador de la imagen (la
configuración por defecto mide 3 126 MiB); con el asignador de 16 KiB de L2 el criterio es ≤ 2 048.
El límite y su regla de retroceso están en [«Asignador por defecto de L3»](#el-asignador-malloc_mmap_threshold_-y-malloc_trim_threshold_-itsc-366-itsc-367).

**Cómo leer `unit_timing`** (todas las fases son pared; [TRD-L3 §9.3](../TRD/l3.md#93-tipos-de-chequeo-check_type)):

| Campo | Qué dice |
|---|---|
| `cpu_utilization` | `cpu_s / (wall_s × cores)`: la medida del objetivo (meta de T4: ≥ 0,90; objetivo de E6: 0,95) |
| `engine_busy` | Fracción del tiempo del bucle con ítems en la cola del motor. Si es baja, el motor espera entrada: mira `read_s` y `inflight_max` |
| `inflight_max` | Máximo de row groups en el motor. Si toca `inflight` y `engine_busy` es alto, el motor no es el que espera; si no lo toca, subir K no ayuda |
| `read_s` | Lo que el hilo principal esperó a las lecturas (L1 y fronteras de L2) |
| `read_wall_s`, `read_bytes` | Pared con alguna lectura en vuelo y bytes comprimidos que viajaron (L1 y L2). La regla del humano: leer a la velocidad del volumen (1,25 GB en 2021-01, ≤ 15 s). Con contrapresión crece sin que la lectura sea lenta: `read_bytes / read_wall_s` es la tasa mientras hay lecturas |
| `write_s` | Pared con algún escritor ocupado. Pegada a `aggregate_s` y con `engine_busy` bajo, los escritores son el cuello: sube `L3_WRITERS` |
| `gate_s`, `setup_s`, `aggregate_s`, `publish_s`, `reconcile_s` | Tramos seguidos del hilo principal; `other_s` es lo que falta hasta `wall_s`. `gate_s` es abrir `events.parquet` de los θ y arrancar la lectura de sus fronteras (una tarea por θ) y `setup_s`, el motor, el pie de L1 y la compuerta de carry-over (los escritores se abren en el pool en ese tramo, sin que nadie los espere); `lineage_s` ya no existe |
| `minor_faults`, `major_faults`, `voluntary_switches`, `involuntary_switches` | Contadores del kernel de la unidad. Con `cpu_system_s` alto: muchos fallos de página (millones) apuntan a memoria y asignador; muchos cambios de contexto voluntarios, a candados o red; ninguno de los dos, a otra causa |
| `mmap_threshold` | Umbral de `mmap` de glibc con que corrió la unidad: `MALLOC_MMAP_THRESHOLD_` o 33554432; `null` si quedó dinámico. Dice con qué corrió cada fila del cuadro de §10.2 |
| `trim_threshold` | Umbral de recorte del heap con que corrió la unidad: `MALLOC_TRIM_THRESHOLD_` (1073741824 en la imagen); `null` si no está fijada y glibc recorta con su umbral dinámico |
| `rss_before_trim_mib`, `rss_after_trim_mib` | `VmRSS` (no el pico) justo antes y después de devolver la memoria al kernel al cerrar la unidad. La diferencia es lo que el trim devolvió; `rss_after_trim_mib` de mes a mes dice si el proceso acumula entre unidades |
| `boundaries_peak_mib` | Máximo de fronteras de L2 que el motor tuvo en mano |

Para comparar dos corridas usa `wall_s`, `cpu_s` y `cpu_utilization`. `run-job.yml` no admite
variables de entorno: para fijarlas en una corrida suelta se lanza **desde Cloud Shell**
(`--update-env-vars` vale solo para esa ejecución; el job no cambia). Por ejemplo, un solo
lector y un solo escritor:

```bash
gcloud run jobs execute l3-backfill \
  --project <proyecto> --region <región> \
  --update-env-vars L3_READERS=1,L3_WRITERS=1 \
  --args="--mode,backfill,--from=2021-01,--to=2021-01,--force" \
  --wait
```

**La corrida T5 de ITSC-365** (los valores por defecto: R = 2, K = 4, W = 12, motor de 4 hilos a
4 vCPU, imagen 0.12.0) se lanza igual, y antes una corrida con `L3_PROFILE=1` para explicar
`cpu_system_s` (ver la tabla de arriba y la entrada de ITSC-365 en
[TRD-L3 ADR-L3-16](../TRD/l3.md#616-adr-l3-16--motor-de-agregación-nativo-ítems-por-row-group-y-θ-casillas-por-evento-fusión-asociativa)).
Metas de T5: `cpu_utilization` ≥ 0,90 (objetivo 0,95), `wall_s` ≤ 100 s (el piso de T4b es
73 s), `rss_peak_mib` ≤ 2 048, `cpu_throttled_s` ≤ 25 y los 50 `content_hash` y los 50
`unassigned_ticks` idénticos a los de ITSC-335. El resultado va como comentario en ITSC-365 y
como fila en el cuadro de §10.2.

**La corrida T4 de ITSC-363** (los valores por defecto de entonces: R = 2, K = 4, W = 4, motor de 4 hilos a
4 vCPU) se lanza con *Actions → Run job*, `l3-backfill`, `from = to = 2021-01` y `force`, tras el
merge y el `apply` del stack `l3`; el resultado se registra como comentario en ITSC-363 y en el
cuadro de
[TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición).
Metas: `cpu_utilization` ≥ 0,90, `wall_s` ≤ 120 s, `rss_peak_mib` ≤ 2 048, `read_wall_s` ≤ 15 s y
los 50 `content_hash` idénticos a los de ITSC-335. Si `read_wall_s` no se cumple, antes de tocar
nada se mide el ancho de banda de GCS al job con la descarga de un objeto grande (un
`consolidated.parquet` entero) y se compara con `read_bytes / read_wall_s`. Para ajustar R, K y W
se repite la corrida con `--update-env-vars` y se anota cada una como una fila más del cuadro.
Un perfil con hilos (`L3_PROFILE=1`) ve todos los hilos y sus tiempos son de hilo, no de pared.

## El asignador: `MALLOC_MMAP_THRESHOLD_` y `MALLOC_TRIM_THRESHOLD_` (ITSC-366, ITSC-367)

**Qué corre por defecto.** La imagen de L3 (desde la 0.14.0) fija `MALLOC_MMAP_THRESHOLD_=33554432`
(32 MiB, el máximo de glibc) y `MALLOC_TRIM_THRESHOLD_=1073741824` (sin recorte del heap). Con el
umbral de 16 KiB que L3 heredó de L2, cada buffer grande era un `mmap`/`munmap` que se volvía a tocar
página por página: 32,6 M de fallos de página y ≈ 100 s de `cpu_system_s` por unidad. Con la
configuración nueva, en 2021-01 y a igual cuota de CPU, la unidad gasta **−31 % de CPU y −25 % de
pared** (376 a 260 s y 126 a 94 s) a cambio de **+1,1 GiB de RSS** (2 011 a 3 126 MiB, 900 MiB bajo
los 4 GiB del job). La decisión, las cifras de T5, T6 y T6b y la regla de retroceso están en
[ADR-L3-16](../TRD/l3.md#616-adr-l3-16--motor-de-agregación-nativo-ítems-por-row-group-y-θ-casillas-por-evento-fusión-asociativa),
«Asignador por defecto de L3».

`memory.tune_allocators()` respeta `MALLOC_MMAP_THRESHOLD_` si es un entero entre 0 y 33 554 432; un
valor inválido deja un aviso en el log y rige 33 554 432. Sin la variable (en local y en pruebas)
rige el mismo valor. `MALLOC_TRIM_THRESHOLD_` la lee glibc al arrancar y no pasa por
`tune_allocators`. `unit_timing` dice con cuáles corrió la unidad (`mmap_threshold` y
`trim_threshold`).

**Qué mirar en una corrida.** `rss_peak_mib` ronda 3 100 y es lo esperado, no una fuga: glibc retiene
en el heap lo que la unidad libera, para reutilizarlo. Lo que sí es una alarma es que crezca con el
mes. Metas de la corrida T7 de ITSC-367 (valores por defecto, sin variables): `mmap_threshold`
33554432, `trim_threshold` 1073741824, `minor_faults` < 2 M, `cpu_system_s` < 20 s,
`rss_peak_mib` < 3 500 y los 50 `content_hash` idénticos a los de ITSC-335. Se lanza con *Actions →
Run job*, `l3-backfill`, `from = to = 2021-01` y `force`, tras el merge y el `apply` del stack
`l3`; el resultado va como comentario en ITSC-367 y como fila T7 del cuadro de
[§10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición).

**Regla de retroceso.** Si la validación de 110 meses (ITSC-364) muestra un mes sobre 3 500 MiB de
`rss_peak_mib`, se vuelve al umbral solo: se quita `MALLOC_TRIM_THRESHOLD_` del `ENV` del Dockerfile
(T6 midió 2 922 MiB con 4,2× menos fallos que el umbral de 16 KiB). No se midieron valores
intermedios de `MALLOC_TRIM_THRESHOLD_` y no se inventan.

**Entre unidades: se devuelve la memoria al cerrar (ITSC-371).** Un `backfill` corre decenas de
meses en un proceso y, sin recorte del heap, glibc no devuelve al kernel lo que cada mes libera (la
causa exacta es una hipótesis: TRD-L3 ADR-L3-16, «El RSS crece entre unidades»). Con la
imagen 0.15.0 el RSS crecía de 350 a 650 MiB por mes y el contenedor murió con signal 9 en el
quinto mes. Desde la 0.15.1 `process_unit` llama a `gc.collect()` y a `malloc_trim(0)` al terminar
cada unidad, con éxito o con error, y `unit_timing` registra el `VmRSS` de antes y de después
(`rss_before_trim_mib`, `rss_after_trim_mib`). **Qué mirar:** `rss_after_trim_mib` de mes a mes. Sin
alarma si queda casi plano (en local sube ≈ 2 MiB por unidad, por fragmentación); sí si crece
decenas de MiB por mes. `rss_peak_mib` es un pico y no lo ve.

**Si un backfill muere con signal 9 (OOM).** Desde la 0.16.0 no debería: cada unidad corre en un
proceso hijo y el tope de bytes acota los eventos abiertos (sección siguiente). Si pasa, el padre
sigue vivo y deja `unit_process_died` con `exit_signal` 9. Qué mirar, en orden: (1) el
`unit_timing` del mes que murió no existe (el hijo murió antes de emitirlo), así que la señal sale
de otros dos lados. En Cloud Logging, busca el aviso `L3_INFLIGHT_BYTES` del mes: el hijo lo emite
al empujar un tramo que por sí solo pasa del tope, antes de morir. Y mide la densidad del mes
con lo que sí existe: suma `metric_value` de los `events_summary` de L2 de ese mes (los eventos de
todos los θ, [`operacion-l2`](operacion-l2.md#paso-2-reproducibilidad)) y divídela por los ticks de
`consolidated.parquet` de L1 de ese mes: el tramo es `L3_LOT_EVENTS` ÷ esa densidad (ITSC-373), y si
trae más eventos que el tope, baja `L3_LOT_EVENTS` (con 0 el corte no existe y el lote es el row
group entero). (2) Si es un mes normal, lo que cambió es el job o el asignador: compara `rss_peak_mib` con el de
los meses vecinos. (3) Relanza **desde Cloud Shell** con el asignador de T5, que devuelve la
memoria al kernel en cada `free`, o con más memoria en el job. Repite los `--args` del run que
murió, tal como los imprime `run-job-args.sh` (con `--force` si lo llevaba) y empieza en el mes
que murió: los meses anteriores ya quedaron publicados. Sin `--force` un mes ya publicado se
salta, así que quita la bandera solo si el run original no la llevaba.

```bash
gcloud run jobs execute l3-backfill \
  --project <proyecto> --region <región> \
  --update-env-vars MALLOC_MMAP_THRESHOLD_=16384,MALLOC_TRIM_THRESHOLD_=131072 \
  --args="--mode,backfill,--from=<AAAA-MM del mes que murió>,--to=<AAAA-MM>,--force" \
  --wait
```

Cuesta ≈ 100 s de CPU de sistema en los meses pesados, y los `content_hash` no cambian: no dependen
del asignador. (4) Abre una card con los `unit_timing` del run. Toda medición de memoria del
asignador se hace con **varias unidades** (`from` y `to` distintos), nunca con una sola: una
unidad no muestra lo que se acumula entre meses.

**La corrida de ITSC-371** (la lanza el humano tras el merge y el `apply` de `l3`): *Actions → Run
job*, `l3-backfill`, `from = 2017-08`, `to = 2018-01` y `force`, con los valores por defecto de la
imagen. Meta: `rss_peak_mib` de la última unidad ≤ 1,3 × el de la más grande del rango y
`rss_after_trim_mib` sin tendencia. El resultado va como comentario en ITSC-371 y en
[§10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición).

**Comparar con otro asignador.** `run-job.yml` no admite variables de entorno: una corrida suelta
con otra configuración se lanza **desde Cloud Shell**, sobre 2021-01 y con los `--args` de siempre;
`--update-env-vars` vale solo para esa ejecución. No lances otra ejecución de `l3-backfill` a la vez.
Por ejemplo, el umbral de 16 KiB de antes, para reproducir T5:

```bash
gcloud run jobs execute l3-backfill \
  --project <proyecto> --region <región> \
  --update-env-vars MALLOC_MMAP_THRESHOLD_=16384,MALLOC_TRIM_THRESHOLD_=131072 \
  --args="--mode,backfill,--from=2021-01,--to=2021-01,--force" \
  --wait
```

Compara `minor_faults`, `cpu_system_s`, `cpu_user_s`, `wall_s`, `cpu_throttled_s` y `rss_peak_mib`
con la fila de T7 y compara solo corridas con la misma cuota de CPU (`cores` de `unit_timing`): T6
corrió con 3,23 cores y su pared no se compara con la de T5. Si los fallos de página no bajan, la
fuente se busca con `strace -c -f` (conteo de `mmap`, `munmap`, `madvise` y `brk`) y `/usr/bin/time
-v` sobre el mes local 2020-01 en el contenedor.

## Memoria acotada: un proceso por unidad y `L3_INFLIGHT_BYTES` (ITSC-372)

Dos mecanismos, los dos por construcción y no por ajuste del asignador
([ADR-L3-16](../TRD/l3.md#616-adr-l3-16--motor-de-agregación-nativo-ítems-por-row-group-y-θ-casillas-por-evento-fusión-asociativa),
«Memoria acotada por construcción»):

| Variable | Qué fija | Por defecto |
|---|---|---|
| `L3_UNIT_PROCESS` | `1`: cada unidad de `backfill` corre en un proceso hijo (`spawn`) y su memoria muere con ella; `0`: en el proceso de la CLI. `monthly` no la usa. Otro valor es código 2 | `1` |
| `L3_INFLIGHT_BYTES` | Tope de bytes de eventos entregados al motor que aún no cerraron (305 B por evento). Entero ≥ 1 | 805 306 368 (768 MiB, ≈ 2,6 M de eventos) |
| `L3_LOT_EVENTS` | Eventos de un tramo: cada row group de L1 se parte en tramos de a lo más `L3_LOT_EVENTS ÷ densidad del mes` ticks (ITSC-373). Entero ≥ 0; `0` desactiva el corte | 1 000 000 (≈ 305 MB de casillas y fronteras) |

**Qué mirar en `unit_timing`.** `spawn_s` es el arranque del hijo (en local, 0,1 a 0,2 s; la
estimación para Cloud Run es ≈ 1 s por mes); `null` si la unidad corrió en el proceso de la CLI.
`rss_peak_mib` es el pico del hijo, de esa unidad sola: ya no hay deriva entre meses que mirar.
`inflight_events_max` e `inflight_bytes_max` son el máximo de eventos abiertos y su costo; el tope
se mira antes de empujar, así que `inflight_bytes_max` puede pasar de `inflight_bytes` hasta en un
tramo: es lo normal y no es una señal. La señal es `inflight_alone_bytes_max`, el máximo de
bytes abiertos tras empujar un tramo a un motor vacío: si pasa de `inflight_bytes`, un solo tramo
trajo más eventos que el tope (la unidad avanza igual y avisa en el log): el pico de ese mes es ≈
ese tramo, y se arregla bajando `L3_LOT_EVENTS` o subiendo la memoria del job. Con el valor por
defecto (305 MB contra un tope de 768 MiB) solo pasa si `L3_LOT_EVENTS` es 0 o se subió. `lot_events`
es el lote con que corrió (0 si no hay corte), `lot_ticks` el tamaño de tramo en ticks (`null` sin
corte) y `lots` los tramos empujados: en un mes normal `lots` es `row_groups_month` (más los de
meses anteriores que el lector entrega) y en uno denso es mayor. Con `L3_PROFILE=1` el perfil es del
hijo.

**El padre.** Su `VmRSS` es ≈ 80 MiB y no sube de un mes a otro: no importa lo que la unidad toca.
La memoria del job es la del padre más la del hijo de la unidad.

**Cuando un hijo muere.** El padre queda vivo, emite `unit_process_died` (`exit_signal` 9, `signal_name`
`SIGKILL`: el OOM del cgroup; `exit_code` -9) y termina con código 1 como con cualquier unidad
fallida: el mes no se publicó y los siguientes no se tocan. Una excepción que la unidad no tipó deja
el mismo hallazgo con `exit_code` 1 y el traceback en `error`.

**La corrida de ITSC-372** (la lanza el humano tras el merge y el `apply` de `l3`): *Actions → Run
job*, `l3-backfill`, `from = 2017-08`, `to = 2018-01` y `force`, con los valores por defecto de la
imagen. Metas: termina; `rss_peak_mib` de 2017-12 ≤ 1 500; el `VmRSS` del padre, plano; los
`content_hash` de los seis meses idénticos a los de ITSC-335 (`l3_compare_runs.py`). **Resultado:** terminó los
seis meses, pero 2017-12 picó a 3 002 MiB y 2018-01 a 3 400 MiB: un solo row group de 5,76 M de
eventos pasa solo del tope (`inflight_alone_bytes_max`) y el tope de bytes no lo acota. Lo cierra
la corrida siguiente.

**La corrida de ITSC-373** (la lanza el humano tras el merge y el `apply` de `l3`): *Actions → Run
job*, `l3-backfill`, `from = 2017-08`, `to = 2018-01` y `force`, con los valores por defecto de la
imagen. Metas: termina; `rss_peak_mib` de 2017-12 y de 2018-01 ≤ 1 500; `lots` > `row_groups_month`
en 2017-12 (el mes se parte); `wall_s` de 2017-12 y 2018-01 no más de un 20 % peor que 26,0 y 38,8 s;
`l3_compare_runs.py` con `--from 2017-08 --to 2018-01` da 300 de 300. **Qué esperar de 2017-12:**
`lot_ticks` ≈ 164 000, ≈ 6 tramos por row group e `inflight_events_max` en pocos millones, no en
los 5,76 M de antes; sin aviso `L3_INFLIGHT_BYTES`. 2021-01 no se parte (`lots` = row groups).
Después, el humano decide si el backfill completo se relanza con los valores por defecto (ITSC-352
lo necesita para fijar el tamaño del job). El resultado va como comentario en ITSC-373 y en
[§10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición).

## Lector de tramas: `DC_FRAMES_IO_THREADS` y `readers`

`dc_frames` lee cada row group de L1 con `pre_buffer=True`, en un archivo que abre y cierra por
row group, y las fronteras de L2 (`events.parquet`) por tramos de varios row groups sobre un
archivo abierto una vez (`EventFile`, [TRD-L3 ADR-L3-16](../TRD/l3.md#616-adr-l3-16--motor-de-agregación-nativo-ítems-por-row-group-y-θ-casillas-por-evento-fusión-asociativa)):
Arrow junta los rangos de bytes y los pide a la vez, en vez de una petición por columna y por página. Cuántas peticiones viajan en paralelo lo fija el pool de E/S de Arrow:

| Variable | Qué fija | Por defecto |
|---|---|---|
| `DC_FRAMES_IO_THREADS` | Hilos del pool de E/S de Arrow. Entero ≥ 1; un valor inválido hace fallar el import de `dc_frames` con `ValueError`, antes de leer nada | 8 |

Se lee al importar `dc_frames`, que solo importan `l3_dc_descriptors` y `ops_tools`. Los `VERSION` de L1, L2 y viz suben igual, porque `shared/` entra en cada imagen.
Como `L3_THREADS`, solo se sobrescribe lanzando desde Cloud Shell con `--update-env-vars`.
Más hilos de E/S no aceleran un row group suelto (es un solo rango de ≈ 15 MB): ayudan
cuando varios row groups se leen a la vez: los `L3_READERS` lectores de L1 y los tramos de
L2 de los 50 θ comparten este pool. Si `read_wall_s` de T4 no cumple, subir este valor es la
primera prueba. Cada lector de más cuesta ≈ 48 MB de RSS (≈ 1,8 × mientras decodifica) ([README de dc_frames](../../shared/dc_frames/README.md#lectura-pre_buffer-pool-de-es-y-readers)).

### Sonda del lector con `readers` (ITSC-362)

Los agentes no lanzan jobs: las dos corridas las lanza el humano con `ops-script`, después de
subir el script (`gcloud storage cp layers/ops_tools/scripts/l3_probe_reader.py
gs://<proyecto>-ops/scripts/l3_probe_reader.py`). Se pide `--sin-contar-io` porque envolver el
archivo para contar bytes serializa las lecturas (cada lectura toma el candado del archivo de
Python) y taparía lo que se quiere medir:

| Corrida | `args` |
|---|---|
| `readers = 1` | `--theta all --from 2021-01 --modo row-groups --readers 1 --sin-contar-io` |
| `readers = 2` | `--theta all --from 2021-01 --modo row-groups --readers 2 --sin-contar-io` |
| Línea base, opcional | `--theta all --from 2021-01 --modo row-groups --readers 1 --sin-contar-io --sin-pre-buffer` |

De la última línea (`sonda lector: …`) se copian `pared_s`, `s_por_row_group`, `rss_pico_mib`,
`arrow_pico_mib`, `s_read_row_group_l1` y `readers=`/`pre_buffer=`; la línea `pico del pool de Arrow=…` dice si quedó bajo el
umbral del lector, `(K + 1,5)` row groups (informativo; no es una cota: con `readers` > 1 el peor caso es `1 + readers × 2,1`); de `row groups de L1: …`, los decodificados. Los bytes salen por
metadatos (`bytes por metadatos`), porque `--sin-contar-io` no los mide. Con `readers` > 1 las
fases por hilo no se reportan (los hilos se solapan): decide la pared. Metas
([TRD-L3 §10.3](../TRD/l3.md#103-sonda-del-lector-de-tramas--umbrales-de-aceptación-se-escriben-antes-de-medir)):
`s_read_row_group_l1` ≤ 1,0 con `readers = 1` (es el tiempo medio de un `read_row_group` de L1, la misma métrica que la línea base de ≈ 2,3 s; `s_por_row_group` es la pared total, con L2 y la selección, y no se compara con ella) y `pared_s` ≤ 60 con `readers = 2`. Con `readers = 2`
se decodifican a lo sumo 2 row groups más que con 1, al final.

## Scripts de operación

Tres scripts de `layers/ops_tools/scripts/`, para el job `ops-script`
([runbook de operación ad hoc](operacion-ops.md)). Se suben a
`gs://<proyecto>-ops/scripts/` y se lanzan con `job` = `ops-script`. Todos solo
leen, toman las raíces del proyecto de `OPS_RESULTS_URI` y también aceptan rutas
locales (`uv run`).

| Script | Para qué | `args` típicos |
| --- | --- | --- |
| `l3_probe_reader.py` | Mide el lector de tramas (pared por fase, RSS, bytes, row groups) contra los umbrales de TRD-L3 §10.3 | `--theta all --from 2023-03`; con lectores, `--theta all --from 2021-01 --modo row-groups --readers 2 --sin-contar-io` |
| `l3_check_month.py` | Verificación cruzada: agrega por su cuenta las tramas de cada evento del mes y compara fila a fila con `summaries.parquet` | `--month 2026-09` |
| `l3_compare_runs.py` | Compara dos corridas por sus hallazgos (`content_hash` y `rows` de cada θ y mes) y, con `--timing`, mide la nueva | `--base-dates 2026-10-14:2026-10-15 --new-dates 2026-10-22 --timing` |

```bash
gcloud storage cp layers/ops_tools/scripts/l3_check_month.py \
  gs://<proyecto>-ops/scripts/l3_check_month.py
```

### `l3_check_month.py`

Cierra el criterio 7 de [TRD-L3 §13](../TRD/l3.md#13-criterios-de-aceptación) sobre
un mes real. Para cada θ recorre con `dc_frames` las tramas de cada evento
cerrado del mes y las agrega con NumPy (conteos, cantidad por lado según
`is_buyer_maker`, precio máximo y mínimo, duración, `confirm_instant_*` y VWAP
con la suma en entero de Python y redondeo half-even a 10⁻⁸). Compara por
`(theta, confirm_agg_trade_id)` con el `summaries.parquet` de la misma partición:
mismas filas, mismo orden, mismos valores en las 17 columnas de datos. **No
importa `l3_dc_descriptors`**: la agregación es propia, porque importar la capa
cambiaría `uv.lock` y subiría los `VERSION` de todas.

| `args` | Efecto |
| --- | --- |
| `--month 2026-09` | Los θ del catálogo `manifest/l2/thetas.yaml` |
| `--month 2023-03 --thetas 0.00010000,0.00011352` | Solo esos θ (para partir una corrida) |
| `--month 2026-09 --thetas all` | Los θ que tienen `summaries.parquet` del mes |
| `--l1-root`, `--l2-root`, `--l3-root`, `--catalog` | Otras raíces, también locales |
| `--max-diffs 100` | Cuántas diferencias imprime por θ (las cuentas las cuentan todas) |

Salida: una línea `DIFERENCIA θ=… confirm_agg_trade_id=… columna=… oráculo=… L3=…`
por diferencia, la tabla por θ (filas comparadas, diferencias, segundos) y la
última línea `check mes: mes=… thetas=… filas=… diferencias=… pared_s=…
veredicto=ok|diferencias|fallo`. La tabla queda también en `results/` como
`l3_check_month_<mes>_<fecha>.csv`. Códigos: **0** sin diferencias, **1** con
diferencias (o si el lector falla), **2** por uso. Una diferencia **nunca se
tolera**: es un defecto del núcleo o del lector; abre una card de bug con la
línea, y no relances el backfill esperando que desaparezca.

**Cuánto tarda (medido, ITSC-336, 2026-10-16).** El job `ops-script` tiene
3 600 s y `max_retries = 0`: un timeout cuesta una corrida. El script lee L1 una
vez para todos los θ. Los segundos por θ de la tabla son su cómputo; la lectura
compartida no se reparte.

| Mes | θ | Filas | Pared | Run |
| --- | --- | --- | --- | --- |
| 2026-09 | los 50 | 774 210 | 302,1 s | [38857887813](https://github.com/byroncz/intrinsica/actions/runs/38857887813) |
| 2023-03, parte A | los 5 menores (`0.00010000` a `0.00016608`) | 14 454 943 | 1 688,4 s | [38858924852](https://github.com/byroncz/intrinsica/actions/runs/38858924852) |
| 2023-03, parte B | los otros 45 (`0.00018854` a `0.05000000`) | 5 116 218 | 2 051,4 s | [38858999447](https://github.com/byroncz/intrinsica/actions/runs/38858999447) |

Cero diferencias en los tres. Las dos partes de 2023-03 suman 19 571 161 filas
(todos los eventos del mes, [TRD-L3 §10.3](../TRD/l3.md#103-sonda-del-lector-de-tramas--umbrales-de-aceptación-se-escriben-antes-de-medir))
y 3 739,8 s de pared, pero cada parte paga la lectura completa de L1 y L2. Con el
cómputo por θ de sus tablas (1 082,5 s y 1 740,7 s) y una sola lectura (310,7 s a
605,9 s, según la corrida), una corrida única habría tardado **≈ 3 130 a 3 430 s**.
Es una estimación: no se intentó. Queda demasiado cerca de los 3 600 s del job para
arriesgarla con `max_retries = 0`. De ahí la regla:

- **2026-09 (y cualquier mes de ≈ 1 M de eventos o menos) va en una sola
  corrida.**
- **2023-03 y los meses de su tamaño se verifican en dos corridas por
  `--thetas`, con los 5 θ menores aparte.** Es prudencia, no un límite medido:
  una corrida única habría quedado a ≈ 170 s del timeout. Los 5 θ menores son el 74 % de las filas y el 45 %
  de la pared; los otros 45 pagan, además de sus eventos, por cada tick que leen.

Para partir, toma del catálogo (`manifest/l2/thetas.yaml`) los θ en orden
ascendente: los cinco primeros, de `0.00010000` a `0.00016608`, van en `--thetas`
de la parte A, separados por coma, y los otros 45 en la B. La tabla por θ de cada corrida queda en `results/<ejecución>/`.

**Dónde está el costo.** Los θ pequeños pagan por evento: ≈ 0,12 a 0,14 ms por
evento (308 s para 4,5 M de filas en 2023-03). Los θ grandes tienen decenas de
filas y pagan por tick: 6 a 10 s cada uno en 2026-09 y 60 a 75 s en 2023-03.
Para un mes nuevo, extrapola de 2026-09 con la relación de ticks (7,3 × a
2023-03) y de eventos (25 ×), y parte la corrida si la suma pasa de ≈ 3 000 s.

### `l3_compare_runs.py`

Prueba que dos corridas de L3 produjeron lo mismo **sin releer los Parquet**: compara, por
`(θ, mes)`, el `content_hash` y las `rows` del hallazgo `summaries_summary` de una corrida
**base** con los de una **nueva**. Mismo `content_hash` es mismo contenido lógico
([RNF-L3-03](../TRD/l3.md)), así que 5 500 pares iguales (110 meses × 50 θ) son la prueba de
que el núcleo nuevo no cambió ningún archivo. La referencia es la corrida de ITSC-335, que
sigue en `dq-findings/l3/` aunque `force` reescriba L3: comparar es leer dos conjuntos de
hallazgos.

| `args` | Efecto |
| --- | --- |
| `--base-dates A:B --new-dates C:` | Cada lado por su `detected_date=` (UTC): `A`, `A:B`, `A:` o `:B`; los dos extremos entran |
| `--base-run <run_id> --new-run <run_id>` | Cada lado por `run_id` (repetible); se combina con las fechas. Solo con `run_id` lee **todos** los días del lago: usa fechas |
| `--from 2017-08 --to 2026-09` | Solo esos meses, en los dos lados |
| `--timing` | Resume los `unit_timing` de la corrida nueva (abajo) |
| `--baseline-vcpu-s 165000` | Con `--timing`: contra cuántos vCPU-s se compara |
| `--dq-root`, `--provider`, `--market`, `--asset`, `--max-diffs 100` | Otra raíz (también local), otra serie, cuántas diferencias imprime |

Un backfill se lanza por rangos y cada ejecución tiene su `run_id`, por eso el lado se elige
por fechas. El día es el UTC de `detected_at`: una ejecución que cruza la medianoche UTC deja
hallazgos en dos carpetas, y el rango debe incluir ambas. Si un `(θ, mes)` aparece en varias
corridas de un lado, vale la última por `detected_at`; si esas corridas discrepan entre sí, el
par sale `INCONSISTENTE` (una diferencia más): acota el lado con fechas o `run_id` hasta que
solo quede la corrida que quieres comparar.

Salida: una línea por diferencia (`DIFERENCIA θ=… 2021-01: content_hash base=… nueva=…`,
`SOLO EN BASE`, `SOLO EN NUEVA`, `INCONSISTENTE`) y, al final, `compare runs: pares=… coinciden=…
difieren=… solo_base=… solo_nueva=… inconsistentes=… veredicto=ok|diferencias`. Códigos: **0**
si todo coincide, **1** si hay una sola diferencia, un lado sin hallazgos o el lago no se
lee, **2** por uso. Una diferencia **nunca se tolera**: bloquea la épica hasta explicarla, y
no se relanza el backfill esperando que desaparezca.

Con `--timing` imprime después, de la corrida nueva y con el `unit_timing` más reciente de
cada mes: la tabla por mes (`wall_s`, `cpu_utilization`, `rss_peak_mib`, `engine_busy`,
`cores`), `cpu_utilization` p10, p50 y mínimo con el mes del mínimo, los meses bajo 0,80, la
suma de `wall_s`, los vCPU-s (`Σ wall_s × cores`, contra `--baseline-vcpu-s`), el
`rss_peak_mib` máximo y el `engine_busy` p50. La tabla queda en `results/` como
`l3_compare_runs_timing_<fecha>.csv`. Es informativo: no cambia el código de salida. Los
vCPU-s son el cómputo de las unidades; lo facturable es la pared de la ejecución, que incluye
el arranque (`gcloud run jobs executions describe`).

**Validar el núcleo nuevo (ITSC-364).** Todo lo dispara el humano; los agentes no lanzan
jobs:

1. Con el stack `l3` aplicado con la imagen vigente, *Actions → Run job*: `job` =
   `l3-backfill`, `from` = `2017-08`, `to` vacío y `force` marcado. Rehace los 110 meses
   (`force` exige `from`, y con `to` vacío llega al último mes con consolidado). Con la pared
   nueva (≤ 120 s por mes esperados; medido: 4 828 s en total) cabe en una sola ejecución
   y bajo las 6 h de Actions. Anota la fecha UTC de inicio y de fin.
2. Sube el script y lanza `ops-script` (`gcloud storage cp
   layers/ops_tools/scripts/l3_compare_runs.py gs://<proyecto>-ops/scripts/l3_compare_runs.py`),
   con las fechas de ITSC-335 como base y las del paso 1 como nueva:

   ```
   --base-dates <primer día de ITSC-335>:<último día> --new-dates <día del paso 1>: --timing --baseline-vcpu-s 165000
   ```

   El `--new-dates` queda abierto por el final para cubrir la medianoche UTC; si ya hubo
   otras corridas después, ciérralo con el último día del backfill.
3. Pega la última línea de la comparación (debe ser `pares=5500 coinciden=5500 …
   veredicto=ok`) y las líneas de `--timing` como comentario de ITSC-364. Con eso se
   escriben [TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición)
   y la decisión «Núcleo de L3 al 95 %».

**Resultado de ITSC-364 (ejecución `ops-script-pc4wp`, run de Actions 39877759318,
2026-10-23 00:23 UTC; backfill `l3-backfill-jgfrj`, base ITSC-335):**
`--base-dates 2026-10-14:2026-10-15 --new-dates 2026-10-22 --timing --baseline-vcpu-s 165000`.

```
compare runs: pares=5500 coinciden=5500 difieren=0 solo_base=0 solo_nueva=0 inconsistentes=0 veredicto=ok
cpu_utilization: p10=0.312 p50=0.510 mínimo=0.116 (2017-08); meses bajo 0.80: 110
wall_s: suma=4828 p50=37.9 máximo=141.4 (2022-11)
vCPU-s (Σ wall_s × cores): 18992 contra 165000 (11.5 %)
rss_peak_mib: máximo=3545.3 (2017-12). engine_busy: p50=0.525
```

La equivalencia se cumple; las metas de utilización (p50 ≥ 0,90, ningún mes bajo 0,80) no, y
siguen en [TRD-L3 §14, ítem 10](../TRD/l3.md#14-ítems-abiertos-y-a-validar). Esa corrida
usó el asignador de T5 (`MALLOC_MMAP_THRESHOLD_=16384` y `MALLOC_TRIM_THRESHOLD_=131072`)
porque la imagen con sus valores por defecto murió por memoria en el quinto mes (ITSC-371,
ITSC-372).

### La trampa del 403 por prefijo

La service account de `ops-script` lee `l1/`, `l2/`, `l3/` y `viz/tiles/` con
condiciones IAM por prefijo. `pyarrow.dataset.dataset("gs://<bucket>/l3")` (o
`pq.read_table` sobre un prefijo) pide primero el **objeto** `l3` sin barra, y la
condición `startsWith("…/l3/")` lo niega: 403, aunque los archivos de abajo sean
legibles (run 38659287542 de `ops-script`). Los tres scripts evitan el problema:
**listan de forma recursiva** y **abren cada archivo por su ruta completa**. Un
script nuevo hace lo mismo; desde Cloud Shell, con tus credenciales, el 403 no
aparece.
