# Runbook: operación de L3 (backfill histórico y de la familia T, lectura de fallos, cascada, cadena mensual, volumen y costo, perfil de una unidad, hilos de la unidad, verificación cruzada, comparación de corridas, vista derived)

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

**ITSC-416 (solo `ops_tools` 0.6.15; la imagen de L3 no cambia): la corrida de la sonda va después del apply de `ops`.**
El apply del stack `ops` (lo lanza el humano) publica la sonda nueva en `gs://<proyecto>-ops/scripts/`; sin él,
`ops-script` corre la versión de ITSC-411 y la última línea no trae los campos de la ubicación. Corre
`l3_spectrum_probe.py` sobre 2023-03 con θ `0.00098052` (ver «`l3_spectrum_probe.py`»).

**ITSC-402 (solo infraestructura; la imagen de L3 no cambia): dos applies, en este orden.**
Primero `l3` (crea el job `l3-monthly-fit` con `timeoutSeconds: 7500` y sube `l3-backfill` a
`timeoutSeconds: 43200`) y después `l1` (el workflow encadena `l3-monthly-fit` y recibe
`run.invoker` sobre él y `run.viewer` sobre `l3-monthly-path`). Si `l1` va primero, falla: el IAM
se da sobre un job que debe existir. Ambos los lanza el humano (*Actions → Terraform*). Después,
prueba `l3-monthly-fit` solo, como `l3-monthly` con `family` = `fit` (o `l1-monthly-close` del mes en
curso): `fit.parquet` del mes para los 50 θ, y `summaries_summary` de la familia `fit` en los
hallazgos. Pega el enlace del run en la card.

**ITSC-400 (imagen de L3 ≥ 0.22.0): el backfill de la familia A va después del apply.**
Tampoco hay orden previo: la imagen nueva escribe `fit.parquet` solo con `--family fit` y no
cambia nada de `summaries` ni de `path`. Tras el merge y el apply del stack `l3` (los lanza el
humano), corre `l3-backfill` con `family` = `fit`, sin `from` ni `to` y con los valores por
defecto, después del de S (ver «Backfill de la familia A»).

**ITSC-394 (imagen de L3 ≥ 0.21.0): el backfill de la familia T va después del apply.**
No hay orden previo que cumplir: la imagen nueva escribe `path.parquet` solo con `--family path`
y no cambia nada de `summaries`. Tras el merge y el apply del stack `l3` (los lanza el humano),
corre `l3-backfill` con `family` = `path`, sin `from` ni `to` y con los valores por defecto
(ver «Backfill de la familia T»).

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
3. **Backfill del linaje** con `ops-script`: lanza
   [`l2_backfill_lineage.py`](operacion-l2.md#linaje-publicado-y-su-backfill-itsc-368)
   (el apply de `ops` ya lo publicó en `scripts/`; la guía paso a paso está en el runbook de L2). Antes, con `--dry-run`, mira que
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

   Deben existir `l3-backfill`, `l3-monthly`, `l3-monthly-path` y `l3-monthly-fit`.
   `l3-monthly-path` lleva `args` `--mode monthly --family path` y los mismos recursos que
   `l3-monthly` (ITSC-396; compruébalo con `gcloud run jobs describe l3-monthly-path`).
   `l3-monthly-fit` lleva `args` `--mode monthly --family fit`, los mismos recursos salvo el
   timeout, `timeoutSeconds: 7500` (ITSC-402; `gcloud run jobs describe l3-monthly-fit`).
   `l3-backfill` tiene `timeoutSeconds: 43200` (ITSC-402; antes 10800, y antes de ITSC-352,
   86400), `maxRetries: 1`, `cpu: '2'` y `memory: 4Gi` en `limits` (desde el apply de ITSC-352;
   antes `'4'`). Las raíces
   (`L3_LANDING_ROOT`, `L3_EVENTS_ROOT`, `L3_SUMMARIES_ROOT`, `L3_DQ_ROOT`),
   `L3_SERIES_START` y `L3_THETAS_URI` van en `env`; `L3_THREADS`, `L3_READERS`, `L3_INFLIGHT`,
   `L3_WRITERS`, `L3_ENGINE_THREADS`, `L3_META_WORKERS`, `L3_INFLIGHT_BYTES`, `L3_LOT_EVENTS`, `L3_TRIM_RSS_MIB`, `L3_TRIM_MIN_S` y `L3_UNIT_PROCESS` son opcionales (ver "Hilos y topes de la unidad" y "Memoria acotada").
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
| `l3-monthly-path` | **No está en la lista de *Run job***: la cadena mensual lo corre solo. A mano, se lanza como `l3-monthly` con `family` = `path` | Igual que `l3-monthly` | Igual que `l3-monthly`: no lo admite |
| `l3-monthly-fit` | **No está en la lista de *Run job***: la cadena mensual lo corre solo, después de `l3-monthly-path`. A mano, se lanza como `l3-monthly` con `family` = `fit` | Igual que `l3-monthly` | Igual que `l3-monthly`: no lo admite |

`series_start` vacío usa el del stack. `family` (solo `l3-backfill` y `l3-monthly`;
`summaries` por defecto, `path` o `fit`) agrega `--family=<f>`; los demás jobs rechazan `path` y `fit`.
El workflow no expone `--thetas`: por *Run job* siempre corren los 50 θ del catálogo.
Las familias `path` y `fit` necesitan `summaries.parquet` del mes: se lanzan después de la de
`summaries` (ver «Backfill de la familia T» y «Backfill de la familia A»). No lances a mano el mismo mes y modo que Scheduler esté ejecutando,
ni dos `l3-backfill` a la vez.

Al final, el run vuelca los logs de la ejecución y un resumen (estado, tareas,
inicio y fin) en el *Summary* del run. La CLI termina con **0** (todo avanzó o ya
estaba al día), **1** (falló la unidad: entrada ausente, L2 incoherente,
conciliación, esqueleto; o `monthly` sin ningún θ listo con rezagados) o **2**
(uso o configuración; no escribe datos). Detalle en
[TRD-L3 §8.4](../TRD/l3.md#84-códigos-de-salida).

## Backfill histórico por rangos

**Con el núcleo nuevo cabe en una sola ejecución.** El backfill completo de 110 meses a
2 vCPU con la imagen 0.19.0 y los valores por defecto (ITSC-378, 2026-10-25) tardó 6 638 s
(1,84 h; Σ `wall_s` 6 556 s), bajo las 6 h que espera el run de Actions: `from` = `2017-08`,
`to` vacío y `force`, una vez. Sin `force`, o si una ejecución muere, se relanza y retoma
donde quedó. Los 4 828 s de ITSC-364 (4 vCPU) se midieron con el asignador de T5, un rodeo
que ya no hace falta desde la imagen 0.16.0 (run 39886368319). Lo que sigue (rangos) es de
**ITSC-335, con el núcleo original**, cuya suma eran 11,5 h
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

**Aviso (ITSC-402).** `l3-backfill` sube a `timeoutSeconds: 43200` para que un backfill completo
de la familia A (27 556 s) quepa en una ejecución. La espera de *Run job* sigue muriendo a las 6 h
mientras la ejecución continúa en Cloud Run: cómo seguirla y leer sus `unit_timing` está en
«Backfill de la familia A». Los rangos de la tabla de arriba se fijaron con el timeout de
10 800 s y valen para S y T.

**Regla para un backfill futuro de la serie completa** (por ejemplo, al agregar
una familia): manda el timeout de `l3-backfill` (10 800 s hasta ITSC-402; 43 200 s ahora), no las 6 h del run de Actions.
Basta una ejecución mientras la pared esperada del job no pase de **28 800 s**, el umbral de
margen de «Timeouts» (43 200 ÷ 1,5). Si pasa, fija cada rango por su **pared estimada**, no
por su número de meses, con cada rango bajo 28 800 s. Con el núcleo nuevo a 2 vCPU la serie
suma 6 556 s de `wall_s` para `summaries.parquet` (job 6 638 s, ITSC-378): una familia
≈ 4,4 veces más lenta ya pasa los 28 800 s, y una ≈ 6,5 veces más lenta agota el timeout. La
familia A (27 556 s, ×4,15 sobre S) queda bajo el umbral: el timeout la cubre 1,57 veces
(decisión del arquitecto, 2026-11-06; ver «Timeouts»), con poco margen para otra familia
igual de lenta.
Estima cada mes con el ajuste de
[TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición)
(`wall_s` ≈ 68 s + 0,038 s por millón de ticks + 24,9 s por millón de eventos; es del
**núcleo original** y sobrestima la suma unas 8,5 veces: no se reajustó con el nuevo) o con los
`unit_timing` de `dq-findings/l3/` y el CSV de `l3_compare_runs.py --timing`. Con el
**núcleo nuevo** la pared sigue a los **ticks**: a 2 vCPU, 2022-11 (8,7 G de ticks) tardó
184 s y 2021-01 (46,0 M de eventos, 3,3 G de ticks) 161 s
([TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición)).
Con el **núcleo original** (ITSC-335) era al revés, la pared seguía a los eventos: 2021-01
tomó 1 419 s y 2023-03, el mes con más ticks, 873 s; los 41 meses de 2017-08 a 2020-12
promediaron ≈ 7,2 min por mes y 2021 ≈ 10,75 min. Lo más simple es repetir los rangos de la
tabla.

La frontera de cada θ es la cadena **contigua desde `series_start`**: un rango
sin `from` arranca en el mes siguiente al último construido, y un mes hecho fuera
de orden (como 2023-03 y 2026-09 en ITSC-335) se reconstruye al llegar a él, con el
mismo `content_hash`. Para no repetirlo, termina el rango un mes antes.
Relanzar un rango interrumpido es seguro: la frontera retoma donde quedó.

### Backfill de la familia T (`path`, ITSC-394)

La familia T escribe `path.parquet` junto a `summaries.parquet` de cada θ y mes y **repite su
esqueleto** ([TRD-L3 §7.3](../TRD/l3.md#73-directorio-de-la-capa-y-esqueleto-común), RL3-11): por eso
va **después** de S. Una pasada de T lee L1 y L2 completas otra vez (una pasada por familia,
ADR-L3-11); su pared la fija la lectura de ticks, así que se espera del orden de la de S
(6 638 s para los 110 meses), con los mismos timeouts y la misma regla de arriba: una sola
ejecución mientras la pared esperada no pase de 28 800 s.

1. `l3-backfill` con `family` = `summaries` y los valores por defecto, hasta que termine con
   código 0 (S al día).
2. `l3-backfill` con `family` = `path`, **sin `from` ni `to`** y con los valores por defecto:
   la frontera de cada θ en `path` decide desde dónde avanza.
3. Al terminar, el *Summary* y los `unit_timing` (ahora con `details.family`) traen la pared
   total, el mes más lento y el `rss_peak_mib` máximo.

**La frontera es por familia.** Para `path` es el último mes de la serie contigua de
`path.parquet` cuyo archivo es válido (esquema `PATH_SCHEMA`, `state_version` y filas por
row group iguales a `events.parquet`), y **nunca supera la de `summaries`**. Una corrida de
`path` no toca un byte de `summaries.parquet`, y la frontera de `summaries` no se entera de
`path.parquet`. Un `path.parquet` con otra `state_version` o esquema retrocede su frontera y
el θ se reescribe desde ese mes. Si a un θ le falta `summaries.parquet` de un mes, sale del
rango con `family_skeleton_missing` y la corrida termina con código 1: corre el backfill de
`summaries` y relanza los mismos inputs.

Tras el apply del stack `l3` con la imagen nueva, pega en la card el resultado del paso 2:
pared total, mes más lento, `rss_peak_mib` máximo y el enlace al run (los registra la hija 4
de ITSC-391 en TRD-L3 §10.2).

### Backfill de la familia A (`fit`, ITSC-400)

La familia A escribe `fit.parquet` junto a `summaries.parquet` de cada θ y mes y, como T, **repite
su esqueleto** ([TRD-L3 §7.3](../TRD/l3.md#73-directorio-de-la-capa-y-esqueleto-común), RL3-11): va
**después** de S. Es independiente de T: no lee `path.parquet` y puede ir antes o después. Una
pasada de A lee L1 y L2 completas otra vez (una pasada por familia, ADR-L3-11). Se esperaba del
orden de la de S (6 638 s para los 110 meses), pero ITSC-401 midió **27 556 s** (×4,15; la pasada de
A es limitada por CPU), y por eso `l3-backfill` sube a 43 200 s (ver el aviso de abajo).

1. `l3-backfill` con `family` = `summaries` y los valores por defecto, hasta que termine con
   código 0 (S al día).
2. `l3-backfill` con `family` = `fit`, **sin `from` ni `to`** y con los valores por defecto:
   la frontera de cada θ en `fit` decide desde dónde avanza.
3. Al terminar, el *Summary* y los `unit_timing` (`details.family` = `fit`) traen la pared total,
   el mes más lento y el `rss_peak_mib` máximo.

**La frontera es por familia.** Para `fit` es el último mes de la serie contigua de `fit.parquet`
cuyo archivo es válido (esquema `FIT_SCHEMA`, `state_version` y filas por row group iguales a
`events.parquet`), y **nunca supera la de `summaries`**. Una corrida de `fit` no toca un byte de
`summaries.parquet` ni de `path.parquet`, y las fronteras de S y de T no se enteran de
`fit.parquet`. Un `fit.parquet` con otra `state_version` o esquema retrocede su frontera y el θ se
reescribe desde ese mes. Si a un θ le falta `summaries.parquet` de un mes, sale del rango con
`family_skeleton_missing` y la corrida termina con código 1: corre el backfill de `summaries` y
relanza los mismos inputs.

**El backfill de A no cabe en las 6 h que espera Actions (ITSC-402).** La pared medida de A es
27 556 s (7,7 h; ITSC-401, [TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición)),
2,5 veces los 10 800 s con que se dimensionó `l3-backfill` para S. Desde el apply de ITSC-402 el
job tiene `timeoutSeconds: 43200` (12 h, 1,57 veces esa pared; antes, `l3-backfill-7f85j` murió a
los 10 800 s), así que **una sola ejecución recorre los 110 meses**. Pero la espera de *Run job*
se corta a las 21 600 s (6 h): el run de Actions falla por tiempo de espera **mientras la
ejecución sigue viva en Cloud Run**. Un run en rojo con ese motivo no es un fallo del backfill.
No lo relances: dos `l3-backfill` a la vez escriben las mismas particiones.

1. **Sigue la ejecución en Cloud Run**, no en Actions. Saca su nombre de *Cloud Run → Jobs →
   l3-backfill* o de `gcloud run jobs executions list --job l3-backfill --region us-east1`, y míralo:

   ```bash
   gcloud run jobs executions describe <ejecución> --region us-east1 \
     --format='yaml(status.conditions,status.runningCount,status.succeededCount,status.failedCount,status.startTime,status.completionTime)'
   ```

   `runningCount: 1` es que sigue. Termina con `succeededCount: 1` (código 0) o `failedCount: 1`
   (código 1 o timeout; el reintento de `max_retries = 1` cuenta aparte).
2. **Lee los `unit_timing` desde Cloud Logging**, porque el *Summary* de Actions no llega a
   escribirse. Cada unidad (un mes) deja una línea `unidad <mes>: familia=fit θ=… wall_s=…
   rss_peak_mib=…`; filtra por la ejecución y la familia:

   ```bash
   gcloud logging read 'resource.type="cloud_run_job"
     AND resource.labels.job_name="l3-backfill"
     AND labels."run.googleapis.com/execution_name"="<ejecución>"
     AND (textPayload:"familia=fit" OR jsonPayload.message:"familia=fit")' \
     --project <proyecto> --freshness=7d --limit=1000 \
     --format='value(textPayload,jsonPayload.message)' |
     awk '{ for (i = 1; i <= NF; i++) {
              if ($i ~ /^wall_s=/) { split($i, a, "="); s += a[2]; n++
                                     if (a[2] > m) { m = a[2]; mes = $2; sub(/:$/, "", mes) } }
              if ($i ~ /^rss_peak_mib=/) { split($i, a, "="); if (a[2] > r) r = a[2] } }
            }
          END { printf "%d meses, Σ wall_s = %.0f s, más lento %s (%.0f s), rss_peak_mib máx = %.0f\n", n, s, mes, m, r }'
   ```

   Los hallazgos `unit_timing` con `details.family` = `fit` siguen en `dq-findings/l3/`; el CSV de
   `l3_compare_runs.py --timing` los lee de ahí.
3. Cuando la ejecución termine con éxito, corre `l3-backfill` con `family` = `fit` otra vez, sin
   `from` ni `to`: debe salir con código 0, sin escrituras ("todo al día"). Si murió por el
   timeout, la frontera retoma donde quedó: relanza `l3-backfill` con `family` = `fit`, sin
   `from` ni `to` (el paso 2 de la primera lista de esta sección, no el 2 de esta).

La pasada mensual de A (`l3-monthly-fit`) tiene su propio timeout de 7 500 s, 6 veces el mes más
lento de A (2023-03, 1 240 s). Los 1 200 s de `l3-monthly` no alcanzarían para ese mes.

Tras el apply del stack `l3` con la imagen nueva, pega en la card el resultado del paso 2: pared
total, mes más lento, `rss_peak_mib` máximo y el enlace al run (los registra la hija 4 de ITSC-397
en TRD-L3 §10.2, con las cotas +10 % y +5 % sobre S, como T).

### Timeouts: la regla de TRD-L2 §10.2 con 2 vCPU (ITSC-352)

La regla ([TRD-L2 §10.2](../TRD/l2.md)): `monthly` ≥ 6 × la pared del mes más pesado, y
`backfill` ≥ 1,5 × la suma de los meses y ≤ 86 400 s (el tope del módulo `layer`). ITSC-364
midió a 4 vCPU el mes más pesado (141,4 s en 2022-11) y la suma (4 828 s); ITSC-352 midió que
2 vCPU alarga 2021-01 en 1,16 (111,5 s contra 95,9 s) y fijó el tamaño en 2 vCPU con paredes
estimadas (por 1,16). ITSC-378 las reemplazó por el backfill completo de 110 meses a 2 vCPU
(imagen 0.19.0, valores por defecto, 2026-10-25): 2022-11 midió 183,7 s y el job 6 638 s
(Σ `wall_s` 6 556 s).

| Job | Desplegado hasta el apply de ITSC-352 | Definitivo (ITSC-352) | Vigente con ITSC-402 | Cuenta (medida) |
| --- | --- | --- | --- | --- |
| `l3-monthly` | 10 800 s | 1 200 s | **1 200 s** | 6 × 183,7 = 1 102 s: 6,5 × |
| `l3-monthly-path` | (no existía) | (los de `l3-monthly`) | **1 200 s** | los de `l3-monthly` |
| `l3-monthly-fit` | (no existía) | (no existía) | **7 500 s** | 6 × 1 240 (2023-03, A) = 7 440 s: 6,05 × con 7 500 s |
| `l3-backfill` | 86 400 s | 10 800 s | **43 200 s** | S: 1,5 × 6 638 = 9 957 s (1,63 × con 10 800 s); A: 1,5 × 27 556 = 41 334 s (1,57 × con 43 200 s) |

Los valores de ITSC-402 **los aplica el humano** con el `apply` del stack `l3`; hasta entonces rigen
los desplegados. La regla de subida de `l3-monthly` no cambia: si 2022-11 pasa de 200 s, sube.
La de `l3-backfill` es la misma, con el timeout nuevo: si un backfill completo pasa de **28 800 s**
(43 200 ÷ 1,5, el margen de 1,5 ×), sube en el siguiente PR de infra. La familia A la
cumple (27 556 s, un margen de 1,57 ×): el arquitecto fijó los 43 200 s el 2026-11-06 para
cubrir el 1,5 × sobre esa pared (41 334 s), y quedan bajo el tope de 86 400 s. Para `l3-monthly-fit` el umbral
es 1 250 s en el mes más lento (7 500 ÷ 6); 2023-03 ya está en 1 240 s. Mientras ninguna se
cumpla, nadie los sube «por si acaso». Si un job termina por timeout con estos valores, mira primero `unit_timing` del mes:
una unidad que pasa de 3 × su pared habitual es un hallazgo, no una razón para subir el timeout.

**Línea base por mes.** La tabla de los 110 meses de ese backfill (`wall_s`, `cpu_utilization`,
`rss_peak_mib`, `engine_busy`, `cores`) está en
`gs://intrinsica-dc-ops/results/ops-script-h8hf7/l3_compare_runs_timing_20261026T025519.csv`
(`run_id` 06af3d3a-374b-43e8-ac60-6ded7bb3ae44). Compara contra ella cualquier corrida nueva:
un mes que se aleja de su pared de esa tabla es la señal para mirar `unit_timing`. Mira también
`cores`: la cuota que Cloud Run entrega por «2 vCPU» varía con el host (1,613 en ese backfill,
2,086 en T5), y una pared distinta con otro `cores` no es una regresión
([TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición)).

## Leer un fallo

Un fallo detiene el rango (los meses siguientes no se tocan) y deja un hallazgo
en `gs://<proyecto>-dq-findings/l3/detected_date=<fecha>/`. Primero distingue la
causa en el log de la ejecución:

- **OOM** ("Memory limit exceeded" o `signal 9`): no hay línea `unidad <mes>:` de ese mes.
  El pico vigente, con el núcleo nuevo y el tamaño desplegado (2 vCPU), es 2 054 MiB, el
  50 % de 4 GiB, en 2020-03 (ITSC-378). La medida anterior, 3 545 MiB en 2017-12 a 4 vCPU
  (ITSC-364), venía de la densidad de eventos y no del asignador; es el caso que acotó
  ITSC-372. Un OOM en un
  mes más denso, o en uno chico tras varios meses, es el hallazgo: mira
  `rss_before_trim_mib` y `rss_after_trim_mib` de los meses anteriores y la sección
  «El asignador».
- **Timeout**: la tarea se corta sin mensaje de memoria. El de `l3-monthly` es de
  1 200 s desde el apply de ITSC-352 (10 800 s antes; 6,5 veces la pared medida del mes más
  lento a 2 vCPU, 183,7 s en 2022-11, ITSC-378); el de `l3-monthly-fit`, de 7 500 s, y el de
  `l3-backfill`, de 43 200 s desde ITSC-402 (ver «Timeouts»).
- **Un hallazgo `error`**: lee su `check_type` en la tabla siguiente.

Los 17 `check_type` de [TRD-L3 §9.3](../TRD/l3.md#93-tipos-de-chequeo-check_type),
y qué hacer con cada uno. En todos los casos de `error`, **el mes no se publica y
la frontera no avanza**: relanzar los mismos inputs retoma donde quedó. Nunca
repares un archivo de `l3/` a mano.

| `check_type` | Severidad | Qué significa | Qué hacer |
| --- | --- | --- | --- |
| `phase_quantity_mismatch` | `error` | `buy + sell` de una fase no es la suma de `quantity` de sus ticks | No debería ocurrir: es un defecto del núcleo o de los datos. Abre una card con `task-create` con `details` (θ, `confirm_agg_trade_id`, fase, los dos valores). No relances: dará lo mismo |
| `family_skeleton_mismatch` | `error` | Un archivo de familia no repite filas, orden, claves o row groups del esqueleto | Compara `details` con `events.parquet` de L2 del mes. Si L2 cambió después, relanza el mes con `force` (ver "Relanzar un mes"). Si persiste, card |
| `family_skeleton_missing` | `error` | Una familia distinta de `summaries` (`path` o `fit`) no encontró `summaries.parquet` del mes de `(θ, M)`; `details` trae `theta`, `month`, `family` y `file` | Corre `l3-backfill` con `family` = `summaries` para ese rango y relanza los mismos inputs. Ese θ salió del rango y los demás siguieron (código 1) |
| `event_over_budget` | `warning` | Un evento supera el presupuesto de ticks (solo familias de serie completa) | `summaries` no tiene presupuesto y no lo emite. Informa; no falla la unidad |
| `l2_input_missing` | `error` | Falta `events.parquet` o `carry_over.parquet` de `(θ, M)` | Completa L2 para ese θ y mes ([runbook de L2](operacion-l2.md)); luego relanza `l3-backfill` con los mismos inputs. Ese θ salió del rango y los demás siguieron (código 1) |
| `l2_lineage_missing` | `error` | La partición de L2 de `(θ, M)` no tiene `content_hash.json` o no es válido; `details` trae θ, mes, archivo y `reason` | Corre el backfill del linaje ([L2](operacion-l2.md#linaje-publicado-y-su-backfill-itsc-368)), que escribe el de las particiones que no lo tienen; si el mes es nuevo (publicado por una imagen de L2 anterior a la 0.11.0), vuelve a correr el backfill. Luego relanza `l3-backfill` con los mismos inputs. L3 no recalcula el linaje por su cuenta, a propósito |
| `l2_state_inconsistent` | `error` | L2 contradice a L1 o a un invariante de fila (TRD-L3 §9.2); `details` trae θ, mes, campo y los dos valores | Descarta primero L1: ids repetidos o marcas del proveedor (ver "Cascada"). Si L1 está sano, es un defecto de L2 o de L3: card con `details`. Es el hallazgo que detectó los duplicados de 2026-02 |
| `phase_ticks_mismatch` | `error` | No cuadra la conciliación de ticks de §9.1; `details` trae `N_M`, `Σ_cerrados`, `P_M`, `Cola_M` y `U_M` | Corre el seam-check de [L1](operacion-l1.md) y el de [L2](operacion-l2.md#paso-3-seam-check-de-continuidad) sobre el mes. Si ambos pasan, card con `details` |
| `input_missing` | `error` | El mes no tiene `consolidated.parquet` ni provisionales | Corre `l1-backfill` o `l1-monthly-close` del mes ([runbook de L1](operacion-l1.md)); luego relanza |
| `input_provisional_only` | `error` | El mes solo tiene `provisional-day=DD.parquet` | Espera el cierre de L1 (el consolidado sale el primer lunes del mes siguiente) o lanza `l1-monthly-close`; luego relanza |
| `theta_catalog_invalid` | `error` | El catálogo no existe, no se lee o incumple TRD-L2 §7.3. Código 2, sin escribir | Corrige `manifest/l2/thetas.yaml` ([runbook de L2](operacion-l2.md#el-catálogo-de-θ)); relanza |
| `theta_behind_frontier` | `warning` | `monthly` no procesó un θ; `details.family` es la pasada (`summaries`, `path` o `fit`) y `details.reason` es `l3_frontier` (la serie de L3 de la familia tiene un hueco o no existe), `l2_missing` (L2 no tiene el mes) o `summaries_missing` (familia `path` o `fit` sin `summaries.parquet` del mes) | `l3_frontier`: lanza `l3-backfill` de esa familia sin `from` ni `to`, que retoma desde la frontera de cada θ. `l2_missing`: completa L2 primero. `summaries_missing`: corre el backfill de `summaries`. No falla la unidad si otros θ avanzan |
| `theta_config_drift` | `info` | L3 tiene particiones de un θ que ya no está en el catálogo | Informa. Quitar un θ del catálogo no borra nada; si fue un error, restáuralo |
| `summaries_summary` | `info` | Uno por θ, mes y familia: filas, `content_hash` del archivo de la familia y de las entradas de L2 que usó | Es el linaje contra L2: sirve para comparar corridas y para detectar un mes obsoleto |
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
**`l1-monthly-close` → `l2-monthly` → `viz-tiles` → `l3-monthly` → `l3-monthly-path` → `l3-monthly-fit`**,
cada uno solo si el anterior terminó bien y sin overrides (cada job toma el mes
anterior). L3 va al final porque depende del cierre de L2 y de que viz no quede
detrás (decisión del 2026-10-13, [TRD-L3 §11](../TRD/l3.md#11-operaciones)).
`l3-monthly` escribe `summaries.parquet` (familia S) y `l3-monthly-path`, la pasada
de la familia T (`--mode monthly --family path`, ITSC-396), lo necesita como
esqueleto: por eso el workflow espera a `l3-monthly` y T corre solo si S terminó
con código 0. Desde ITSC-402 el workflow también espera a `l3-monthly-path`, y
`l3-monthly-fit`, la pasada de la familia A (`--mode monthly --family fit`), corre solo si T
terminó con código 0. A solo necesita `summaries.parquet`, no `path.parquet`, pero la cadena
es lineal y la Épica ITSC-397 fija ese orden. El workflow no espera a `l3-monthly-fit`: sigue
su ejecución en *Cloud Run → Jobs → l3-monthly-fit*. Así `path.parquet` y `fit.parquet` del
mes nuevo existen sin intervención y las fronteras de T y de A siguen a la de S mes a mes.

**Si `viz-tiles` falla un mes, `l3-monthly` no corre.** Costo asumido del diseño.
Qué hacer:

1. Resuelve `viz-tiles` con el [runbook de viz](operacion-viz.md) y relánzalo
   (*Run job*, `job` = `viz-tiles`, `from` = el mes).
2. Lanza **`l3-monthly` a mano** (*Run job*, `job` = `l3-monthly`, `from` = el
   mes, `to` y `force` vacíos). No vuelvas a lanzar `monthly-close`: relanzaría
   `l2-monthly`.
3. Lanza **`l3-monthly` con `family` = `path`** (mismo mes, `to` y `force` vacíos):
   la cadena no llegó a `l3-monthly-path`, y sin este paso la frontera de T
   queda un mes detrás de la de S.
4. Lanza **`l3-monthly` con `family` = `fit`** (mismo mes, `to` y `force` vacíos):
   la cadena tampoco llegó a `l3-monthly-fit`, y sin este paso la frontera de A
   queda un mes detrás.
5. Verifica que no queden `theta_behind_frontier`, de S (`details.family` =
   `summaries`), de T (`path`) ni de A (`fit`): si los hay, `l3-backfill` sin `from`
   ni `to`, con la familia que corresponda, los pone al día.

**Si `l3-monthly-path` falla un mes, S no se ve afectada, pero A tampoco corre**: S
ya publicó, y el workflow termina con error en ese eslabón sin lanzar
`l3-monthly-fit` (la cadena es lineal). El fallo se ve en *Cloud Run → Jobs →
l3-monthly-path* y en los hallazgos de T. Relánzalo a mano como `l3-monthly` con
`family` = `path` y el mes, y después `l3-monthly` con `family` = `fit` y el mes. Si el
mes quedó rezagado (`theta_behind_frontier` con `family` = `path`), `l3-backfill`
con `family` = `path`, sin `from` ni `to`, lo pone al día. `l3-monthly-path` termina
con código 1 en las mismas condiciones que `l3-monthly`: ningún θ listo y rezagados.

**Si `l3-monthly-fit` falla un mes, S y T no se ven afectadas**: ya publicaron, y el
workflow ya había terminado bien porque no espera al último eslabón; el fallo solo
se ve en *Cloud Run → Jobs → l3-monthly-fit* y en los hallazgos de A. Relánzalo a
mano como `l3-monthly` con `family` = `fit` y el mes. Si el mes quedó rezagado
(`theta_behind_frontier` con `family` = `fit`), `l3-backfill` con `family` = `fit`,
sin `from` ni `to`, lo pone al día. Termina con código 1 en las mismas condiciones
que `l3-monthly`. Su timeout es de 7 500 s (el mes más lento de A, 2023-03, tomó
1 240 s).

Un `l3-monthly` solo termina con código 1 si ningún θ estaba listo y había
rezagados (un mes perdido no pasa como éxito); con algún θ al día o nada que
procesar termina con 0. El scheduler de `monthly-close` (día 8) está con
`paused = false` desde ITSC-353 (`infra/stacks/batch/l1/main.tf`); no lances a
mano `monthly-close` ni `l3-monthly` para el mismo mes y modo que Scheduler esté
ejecutando. El primer cierre programado es el 2026-11-08.

## Volumen y costo

Medidos sobre el histórico completo: el volumen y el núcleo original, en ITSC-335
(2026-10-15), el núcleo nuevo, en ITSC-364 (2026-10-22), y con el tamaño definitivo de
2 vCPU, en ITSC-378 (2026-10-25). Detalle en
[TRD-L3 §10.2, §10.4 y §10.5](../TRD/l3.md#104-tamaño-de-l3--estimación-por-fila).

| Medida | Valor |
| --- | --- |
| `dc-descriptors/l3/` | 46 742 260 408 B (**43,5 GiB**) en 5 500 `summaries.parquet`; 2,06 veces L2; ≈ 45 B por evento (1 042 M de filas) |
| Suma de `wall_s` de los 110 meses | **2 vCPU (ITSC-378): 6 556 s (job 6 638 s, 1,84 h); ≈ 13 100 vCPU-s nominales (10 575 con `cores` 1,613).** 4 vCPU, núcleo nuevo (ITSC-364): 4 828 s (1,34 h); 18 992 vCPU-s (19 312 con los 4 asignados). Núcleo original (ITSC-335): 41 256 s (11,5 h); ≈ 165 000 vCPU-s y GiB-s |
| Costo de lista (0,000018 USD por vCPU-s y 0,000002 por GiB-s) | **≈ 0,29 USD a 2 vCPU (ITSC-378)**; ≈ 0,4 USD a 4 vCPU; ≈ 3,3 USD con el original. Un cálculo sobre `wall_s`, no una cifra de Billing |
| Mes más pesado | **2 vCPU: 2022-11, 183,7 s**; 4 vCPU: 2022-11, 141,4 s. Original: 2021-01, 1 419 s (46,0 M de eventos); 2023-03, el máximo por ticks, 873 s |
| `cpu_utilization` | **2 vCPU (ITSC-378): p10 0,470, p50 0,637, mínimo 0,120 (2017-08); 99 meses bajo 0,80.** 4 vCPU (ITSC-364): p10 0,312, p50 0,510, mínimo 0,116 (2017-08); ningún mes llega a 0,80 |
| RSS pico máximo | **2 vCPU (ITSC-378): 2 054 MiB (2020-03), 50 % de 4 GiB.** 4 vCPU, núcleo nuevo: 3 545 MiB (2017-12), 86 % de 4 GiB. Original: 1 267 MiB (2018-01), 31 % |
| Cupo gratuito | Es de la cuenta de facturación, no de la capa: L3 fue el 41 % del CPU de octubre y los créditos cubrieron el costo (facturado 0) |
| Estado estacionario (`l3-monthly`) | Como máximo lo que el mes más pesado: menos de un centavo (183,7 s × 2 vCPU) |

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
| `L3_ENGINE_THREADS` | Hilos nativos del motor. Entero ≥ 1 | los vCPU efectivos redondeados (`round(cores)` de `unit_timing`): 2 en `l3-backfill` |
| `L3_THREADS` | Pool de E/S: lectura de las fronteras de L2 y conciliación. Entero ≥ 1; nunca se usan más que θ | `ceil(3 × cores)`, nunca más que θ: 7 a 2 vCPU, 12 a 4 y 26 a 8 (con los `cores` medidos en T5: 2,086, 3,945 y 8,342) |
| `L3_META_WORKERS` | Pool de metadatos (ITSC-369): pasada de entrada de cada θ (`events.parquet`, carry-over, huella), pie de L1, apertura de los escritores y publicación. Son esperas de red de un viaje, no CPU: no sigue a los vCPU. Entero entre 1 y 64; nunca se usan más que θ | 16 (nunca más que θ) |

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
| `gate_open_s`, `gate_frontier_s`, `setup_carry_s`, `setup_carry_wait_s`, `setup_l1_s`, `setup_l1_wait_s`, `setup_engine_s`, `setup_writers_s`, `publish_work_s`, `publish_serial`, `meta_workers` | El desglose del costo fijo de la unidad (ITSC-369; [ADR-L3-16, «Costo fijo por unidad»](../TRD/l3.md#616-adr-l3-16--motor-de-agregación-nativo-ítems-por-row-group-y-θ-casillas-por-evento-fusión-asociativa)). Cada paso por θ trae `*_s`, su pared, y `*_work_s`, la suma de lo que duró cada tarea: `work` ≫ pared es espera solapada; `work` ≈ pared es espera en serie, y lo que se busca es que no ocurra. `gate_open_*` es abrir `events.parquet`; `gate_frontier_*`, arrancar sus fronteras; `setup_carry_*`, la compuerta de carry-over (carry del mes y del anterior y huella: 3 viajes por θ); `setup_writers_*`, abrir los escritores (nadie los espera). Lo que el hilo principal esperó dentro de `setup_s` es `setup_carry_wait_s` + `setup_l1_wait_s` + `setup_engine_s`; el resto es armar el motor y el escritor. Con `L3_THREADS=1` (`meta_workers` = 0) no hay pool: la compuerta y los escritores corren en el hilo principal, `setup_carry_wait_s` es 0 y esa espera está en `setup_carry_s` y `setup_writers_s`. `publish_serial` son las publicaciones que empezaron sin otra en curso: ≈ 1 con el solape, igual a los θ si fue en serie. **Cómo leerlo:** un mes liviano con `gate_s + setup_s + publish_s` alto y `*_work_s` ≈ n_θ × el viaje: sube `L3_META_WORKERS` (tope 64); con `meta_workers` ya en 16 y `setup_carry_work_s ÷ setup_carry_s` ≈ 16, el pool es el límite y son los viajes por θ los que cuestan. Con `*_work_s` bajo y la pared alta, la espera está en otra parte (hilo principal ocupado o el pool de E/S). `meta_workers` = 0 es `L3_THREADS=1` |
| `minor_faults`, `major_faults`, `voluntary_switches`, `involuntary_switches` | Contadores del kernel de la unidad. Con `cpu_system_s` alto: muchos fallos de página (millones) apuntan a memoria y asignador; muchos cambios de contexto voluntarios, a candados o red; ninguno de los dos, a otra causa |
| `mmap_threshold` | Umbral de `mmap` de glibc con que corrió la unidad: `MALLOC_MMAP_THRESHOLD_` o 33554432; `null` si quedó dinámico. Dice con qué corrió cada fila del cuadro de §10.2 |
| `trim_threshold` | Umbral de recorte del heap con que corrió la unidad: `MALLOC_TRIM_THRESHOLD_` (1073741824 en la imagen); `null` si no está fijada y glibc recorta con su umbral dinámico |
| `rss_before_trim_mib`, `rss_after_trim_mib` | `VmRSS` (no el pico) justo antes y después de devolver la memoria al kernel al cerrar la unidad. La diferencia es lo que el trim devolvió; `rss_after_trim_mib` de mes a mes dice si el proceso acumula entre unidades |
| `mem_peak`, `mem_close_before_trim`, `mem_close_after_trim` | De qué está hecho el RSS (ITSC-375): la muestra del RSS más alto que vio el hilo `l3-memory-sampler` y las del cierre, antes y después del trim. Cada una trae `rss_mib`, `arena_mib` (lo que las arenas piden al sistema), `hblkhd_mib` (regiones `mmap`), `uordblks_mib` (en uso), `fordblks_mib` (libre que glibc retiene), `keepcost_mib`, `arrow_mib`, `engine_inflight` (tramos anclados; `null` al cerrar) y `elapsed_s`. Con `trims` = 0, si `fordblks_mib` es la mayor parte de `rss_mib` en `mem_peak` el pico es retención del asignador y no datos vivos. Con `trims` > 0 esa lectura engaña (`fordblks` cuenta páginas que el trim ya devolvió y puede pasar de `rss_mib`): se mira lo residente sin uso, `rss_mib − (uordblks_mib + hblkhd_mib)`, y la memoria viva en `uordblks_mib` y `arrow_mib`. Sin `mallinfo2` los campos de glibc son `null` |
| `mem_samples`, `mem_peak_coverage` | Ticks del muestreador y `rss_mib` de `mem_peak` ÷ `rss_peak_mib`. Bajo 0,9 el muestreador no vio el pico: baja `L3_MEMORY_SAMPLE_S` (ver «Variables») y repite; es un ajuste, no una falla |
| `trims`, `trim_s`, `trim_rss_mib` | El trim por marca de agua (ITSC-376): cuántos `malloc_trim(0)` hizo el muestreador durante la unidad, la suma de lo que tardaron (s) y la marca con que corrió (`L3_TRIM_RSS_MIB`; 0 si estaba desactivado). `trims` = 0 en los meses chicos es lo esperado: no llegan a la marca. En los densos, `trim_s` ÷ `wall_s` es lo que cuesta el trim; si estorba al motor (`trim_s` o `cpu_s` suben), sube `L3_TRIM_MIN_S`. `fordblks_mib` no mide lo que el trim devolvió: `malloc_trim(0)` entrega páginas con `madvise(MADV_DONTNEED)` sin sacar los chunks de las listas libres, así que con el trim funcionando `fordblks_mib` puede seguir alto. Lo que el trim sí mueve es lo residente sin uso, `mem_peak.rss_mib − (mem_peak.uordblks_mib + mem_peak.hblkhd_mib)`, y `rss_peak_mib` contra el de la misma unidad sin trim |
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
sigue vivo y deja `unit_process_died` con `exit_signal` 9. Qué mirar, en orden: (0) los `trims` y el
`mem_peak` de los meses vecinos que sí terminaron (ITSC-376): si `trims` es 0 con un `rss_peak_mib`
sobre `trim_rss_mib`, el trim está desactivado (`L3_TRIM_RSS_MIB=0`) o el muestreador no lo vio
(`mem_peak_coverage` bajo); si `rss_peak_mib` queda sobre 2 000 con `trims` > 0, el resto es memoria
viva: mira `uordblks_mib` y `arrow_mib` de `mem_peak` (casillas del motor y salida sellada de Arrow)
y se baja con el diseño, no con el asignador. (1) el
`unit_timing` del mes que murió no existe (el hijo murió antes de emitirlo), así que la señal sale
de otros dos lados. En Cloud Logging, busca el aviso `L3_INFLIGHT_BYTES` del mes: el hijo lo emite
al empujar un tramo que por sí solo pasa del tope, antes de morir. Y mide la densidad del mes
con lo que sí existe: suma `metric_value` de los `events_summary` de L2 de ese mes (los eventos de
todos los θ, [`operacion-l2`](operacion-l2.md#paso-2-reproducibilidad)) y divídela por los ticks de
`consolidated.parquet` de L1 de ese mes: el tramo es `L3_LOT_EVENTS` ÷ esa densidad (ITSC-373), y si
trae más eventos que el tope, ese tramo no cabe: baja `L3_LOT_EVENTS` (el tramo sale de la densidad
media del mes y uno localmente denso puede pasar del tope) o sube la memoria del job (con
`L3_LOT_EVENTS=0` el corte no existe y el lote es el row group entero, peor). (2) Si es un mes normal, lo que cambió es el job o el asignador: compara `rss_peak_mib` con el de
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
| `L3_INFLIGHT_BYTES` | Tope de bytes de eventos entregados al motor que aún no cerraron, a una cifra por familia (ITSC-407): 337 B por evento en `summaries`, 769 B en `fit` y 369 B en `path` (cota inferior: T guarda una corrida por tramo suelto de un evento que cruza row groups); antes 337 B para las tres. Entero ≥ 1 | 805 306 368 (768 MiB: ≈ 2,4 M de eventos de S, ≈ 1,0 M de A, ≈ 2,2 M de T) |
| `L3_LOT_EVENTS` | Eventos de un tramo: cada row group de L1 se parte en tramos de a lo más `L3_LOT_EVENTS ÷ densidad del mes` ticks (ITSC-373). Entero ≥ 0; `0` desactiva el corte | 1 000 000 (≈ 337 MB de casillas y fronteras en S, ≈ 769 MB en A y ≈ 369 MB en T; ITSC-375 probó 2 500 000 y no ganó nada, ITSC-376 lo revirtió) |
| `L3_MEMORY_SAMPLE_S` | Segundos entre ticks del hilo `l3-memory-sampler`, que lee `VmRSS` y, solo si supera su máximo, toma una muestra con `mallinfo2`. Número > 0; otro valor es código 2 (ITSC-375) | `0,25` |
| `L3_TRIM_RSS_MIB` | Marca de agua del trim periódico (ITSC-376): en cada tick, si `VmRSS` llega a estos MiB y pasó `L3_TRIM_MIN_S`, el hilo `l3-memory-sampler` llama a `malloc_trim(0)`. Entero ≥ 0; `0` lo desactiva (queda el del cierre); otro valor es código 2 | `1024` |
| `L3_TRIM_MIN_S` | Segundos mínimos entre dos trims periódicos. Número ≥ 0; otro valor es código 2 | `0,5` |

**Qué mirar en `unit_timing`.** `spawn_s` es el arranque del hijo (en local, 0,1 a 0,2 s; la
estimación para Cloud Run es ≈ 1 s por mes); `null` si la unidad corrió en el proceso de la CLI.
`rss_peak_mib` es el pico del hijo, de esa unidad sola: ya no hay deriva entre meses que mirar.
`inflight_events_max` e `inflight_bytes_max` son el máximo de eventos abiertos y su costo; el tope
se mira antes de empujar, así que `inflight_bytes_max` puede pasar de `inflight_bytes` hasta en un
tramo: es lo normal y no es una señal. La señal es `inflight_alone_bytes_max`, el máximo de
bytes abiertos tras empujar un tramo a un motor vacío: si pasa de `inflight_bytes`, un solo tramo
trajo más eventos que el tope (la unidad avanza igual y avisa en el log, con el tramo y los eventos
que trajo): el pico de ese mes es ≈ ese tramo, y se arregla con un lote más chico (`L3_LOT_EVENTS`) o
subiendo la memoria del job. Con el valor
por defecto (1 M de eventos ≈ 337 MB en S, ≈ 769 MB en A y ≈ 369 MB en T, contra un tope de 805 MB) un tramo cabe con holgura en S y T y casi justo en A (ITSC-407: el tope se mira antes de empujar, así que caben a lo más `⌊tope ÷ (lote × cifra)⌋ + 1` tramos: 3 en S, 2 en A y 3 en T; antes A contaba a 337 B y admitía 3. `inflight_bytes_max` de la unidad de A sube con la cifra correcta y el pico de RSS no debe cambiar), y las
fronteras de L2 que `cover` trae por delante (hasta ≈ 1,6 M de eventos con 50 θ) se cuentan a 97 B,
lo que valen (ITSC-376, hoy 97 B con ITSC-399; antes de ITSC-376 se contaban a la tarifa entera (305 B entonces) e inflaban `inflight_alone_bytes_max`; las cifras de ITSC-376 son de cuando valían 65 B y 305 B; las de ITSC-393 en adelante, 81 B y 321 B): el aviso
no debe salir en S ni en T, y ahí si sale es una señal real. **En A puede salir** en los meses cuyo row group se parte en tramos de 1 M (ITSC-407): el lote ya ocupa 769 de los 805 MB del tope, y el aviso suma lo que sigue abierto de tramos anteriores y las fronteras que `cover` trae por delante, con un margen de ≈ 36 MB (un tramo ≈ 5 % más denso que la media basta); ahí no es una señal de falla y no hay que actuar. Ajustar el lote por familia es trabajo de otra card. `lot_events`
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
eventos pasa solo del tope (`inflight_alone_bytes_max`) y el tope de bytes no lo acota. El aviso
`L3_INFLIGHT_BYTES` salió una vez por mes (uno en 2017-12 y uno en 2018-01), no en cada row group.
Lo cierra la corrida siguiente.

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

**La corrida de ITSC-375** (la lanza el humano tras el merge y el `apply` de `l3`): *Actions → Run
job*, `l3-backfill`, `from = 2017-08`, `to = 2018-01` y `force`, con los valores por defecto de la
imagen (lote de 2,5 M de eventos y `L3_MEMORY_SAMPLE_S` de 0,25). Metas: termina;
`l3_compare_runs.py` con `--from 2017-08 --to 2018-01` da 300 de 300; `lots` de 2017-11 vuelve a 1 y el
de 2017-12 sigue mayor que 3; `mem_peak_coverage` ≥ 0,9 en 2017-12 y 2018-01 (si no, baja
`L3_MEMORY_SAMPLE_S`, por ejemplo a 0,1, y repite: es un ajuste del muestreador); `wall_s` de 2017-12
no más de un 3 % peor que 29,9 s. **Qué esperar:** con tramos de 2,5 M el pico puede subir ≈ 100 MiB
sobre los 2 681 de 2017-12 (≈ 130 MiB por millón de eventos abiertos) y la CPU de los meses densos
debería recuperar parte del +38 %; si 2018-01 pasa de 3 072 MiB se anota, el job tiene 4 GiB y la
medición es el objetivo. El aviso `L3_INFLIGHT_BYTES` es esperable en 2017-12 y 2018-01: las
fronteras de L2 por delante del tramo (sobrestimadas a 321 B cuando valen 81 B; en esa corrida eran 305 B y 65 B) bastan para pasar el
tope. No es razón para bajar `L3_LOT_EVENTS`; se anota en la card. Un tramo que no cabe de verdad
sería `inflight_alone_bytes_max` mayor que el tope + 1,6 M × 321 B ≈ 1,32 GB (con 305 B, ≈ 1,29 GB). Si muere por memoria (`exit_signal` 9), relanza **desde Cloud Shell** con
`L3_LOT_EVENTS=1000000` (`--update-env-vars`) para obtener la medición igual; que 2,5 M no cabe entra
en la decisión. **Qué leer:** de `mem_peak` y `mem_close_before_trim` de 2017-12 y 2018-01, qué parte
del `rss_mib` es `fordblks_mib` (libre que glibc retiene), qué parte `uordblks_mib` + `hblkhd_mib` (en
uso) y qué parte `arrow_mib`. Lo que glibc retenía al cierre queda acotado:
`mem_close_before_trim.rss_mib − mem_close_after_trim.rss_mib` es una cota inferior (incluye lo que
soltó `gc.collect()`) y `fordblks_mib` antes del trim es la cota superior. Entre las dos muestras del
cierre la caída de `fordblks_mib` no pasa de la de `arena_mib`: es la cima devuelta de la arena
principal (el trim recorta solo ese top) más los heaps vacíos que `heap_trim` liberó de las
secundarias tras `gc.collect()`; las páginas interiores de los chunks libres de todas las arenas se
devuelven sin tocar `fordblks`, y `fordblks_mib` puede subir por lo que libera `gc.collect()`. Esa
diferencia es memoria devuelta, no retención que «desapareció» (README de la capa y TRD-L3 §6, fila
«Medición del pico»). Hasta TRD-L3 v2.11.0 se leía que `fordblks_mib` caía en ≈ `keepcost_mib`. El resultado va como comentario en ITSC-375 y en la entrada de decisión.

**Las corridas de ITSC-376** (las lanza el humano tras el merge y el `apply` de `l3`, **las dos
seguidas el mismo día** para que el host no cambie entre una y otra). (a) *Actions → Run job*,
`l3-backfill`, `from = 2017-08`, `to = 2018-01` y `force`, con los valores por defecto de la imagen
(lote de 1 M de eventos, trim en 1 024 MiB cada 0,5 s). (b) La misma desde Cloud Shell con el corte
en tramos apagado, y quitando la variable al terminar:

```bash
gcloud run jobs execute l3-backfill \
  --project <proyecto> --region <región> \
  --update-env-vars L3_LOT_EVENTS=0 \
  --args="--mode,backfill,--from=2017-08,--to=2018-01,--force" \
  --wait
```

`--update-env-vars` en `execute` vale solo para esa ejecución y el job no cambia (ver arriba), así que
no hay nada que quitar al terminar; si alguien la fijó con `gcloud run jobs update`, quítala con
`--remove-env-vars L3_LOT_EVENTS`.

Metas sobre (a): termina; `l3_compare_runs.py --from 2017-08 --to 2018-01` da 300 de 300;
`rss_peak_mib` de 2017-12 y 2018-01 ≤ 2 000; `cpu_s` de 2017-12 y 2018-01 no más de un 5 % sobre 55,9
y 103,6 s. Son las tres metas bloqueantes. Lo residente sin uso de `mem_peak`
(`rss_mib − (uordblks_mib + hblkhd_mib)`) es dato informativo, no meta: `mem_peak` se toma cuando el
RSS acaba de subir, justo antes del siguiente trim, así que lo retenido ahí está siempre en su máximo
(la card decía `fordblks_mib` < 30 %, que `malloc_trim(0)` no baja; ADR-L3-16, fila «Trim por marca
de agua»). Sobre (a) contra (b): si `cpu_s`
por mes difiere menos de 5 %, el corte en tramos no cuesta CPU y el +38 % de ITSC-373 se atribuye al
host o a otro cambio de 0.17.0 (se anota, sin card); si difiere más, se abre una card para encontrar
dónde paga el corte. Si 2018-01 queda sobre 2 000 MiB, la card siguiente (salida sellada de Arrow
acotada o asignador del motor) se crea con los `mem_*` de esta corrida. Compara solo corridas con la
misma cuota de CPU (`cores`). El resultado va como comentario en ITSC-376 y en la entrada de
decisión de ITSC-375.

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
que el apply del stack `ops` lo publique (verifica con `gcloud storage ls
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

Nueve scripts de `layers/ops_tools/scripts/`, para el job `ops-script`
([runbook de operación ad hoc](operacion-ops.md)). El apply del stack `ops` los publica en
`gs://<proyecto>-ops/scripts/` (ya no se suben a mano) y se lanzan con `job` = `ops-script`. Todos solo
leen, toman las raíces del proyecto de `OPS_RESULTS_URI` y también aceptan rutas
locales (`uv run`).

| Script | Para qué | `args` típicos |
| --- | --- | --- |
| `l3_probe_reader.py` | Mide el lector de tramas (pared por fase, RSS, bytes, row groups) contra los umbrales de TRD-L3 §10.3 | `--theta all --from 2023-03`; con lectores, `--theta all --from 2021-01 --modo row-groups --readers 2 --sin-contar-io` |
| `l3_check_month.py` | Verificación cruzada: agrega por su cuenta las tramas de cada evento del mes y compara fila a fila con `summaries.parquet` | `--month 2026-09` |
| `l3_compare_runs.py` | Compara dos corridas por sus hallazgos (`content_hash` y `rows` de cada θ y mes) y, con `--timing`, mide la nueva | `--base-dates 2026-10-14:2026-10-15 --new-dates 2026-10-22 --timing` |
| `l3_derived_check.py` | Prueba la vista `derived` (familia D) sobre un θ y un mes: resumen por columna, conteo de banderas e invariantes en un CSV | `--theta 0.00098052 --month 2026-09` |
| `l3_gap_probe.py` | Mide los huecos entre ticks consecutivos de L1 en un mes (cuantiles, máximo y conteos por umbral, con los huecos entre row groups) para fijar `silence_us` de la familia T | `--month 2018-02` |
| `l3_path_check.py` | Prueba `path.parquet` (familia T) contra `summaries.parquet` sobre un θ y un mes: resumen por columna, esqueleto, invariantes y regla de nulos en un CSV | `--theta 0.00098052 --month 2023-03` |
| `l3_fit_bridge_probe.py` | Sonda de la familia A (ITSC-397): compara `c_accel` y `c_jump_share` observados con 20 puentes brownianos por evento sobre un θ y un mes; da D de Kolmogorov-Smirnov y el veredicto `sigue\|retirada` | `--theta 0.00098052 --month 2023-03` |
| `l3_fit_check.py` | Prueba `fit.parquet` (familia A) contra `summaries.parquet` y `events.parquet` sobre un θ y un mes: resumen por columna, esqueleto, invariantes, regla de nulos y cifras informativas en un CSV | `--theta 0.00098052 --month 2023-03` |
| `l3_spectrum_probe.py` | Sonda doble de la familia F (ITSC-410, ITSC-416): energía por banda de retornos y de precios por fase contra sus nulos, Haar contra Fourier, ubicación en el tiempo de la energía por nivel de Haar y total, redundancia con A y T, fases bajo 16 ticks por θ y pico de ticks en vuelo desde L2; da D y ρ, la regla de la ubicación y el veredicto `sigue\|retirada` | `--theta 0.00098052 --month 2023-03` |

El apply del stack `ops` publica todos estos scripts en `scripts/`; verifica con
`gcloud storage ls gs://<proyecto>-ops/scripts/`.

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
   nueva (a 2 vCPU se midieron 183,7 s el mes más pesado, 2022-11, y 6 638 s el job; ITSC-378) cabe en una sola ejecución
   y bajo las 6 h de Actions. Anota la fecha UTC de inicio y de fin.

   **Con la imagen 0.16.0 (ITSC-372) basta con los valores por defecto.** La corrida de
   2017-08 a 2018-01 con ellos terminó (run 39886368319). El `--update-env-vars` con el asignador
   de T5 (`MALLOC_MMAP_THRESHOLD_=16384`, `MALLOC_TRIM_THRESHOLD_=131072`) fue el rodeo de ITSC-364
   cuando la imagen anterior moría por memoria, y ya no hace falta. El backfill completo con los
   valores por defecto y 2 vCPU ya se corrió (ITSC-378, 6 638 s; ver «Timeouts» arriba).
2. Lanza `ops-script` (el apply del stack `ops` publica el script; verifica con `gcloud storage ls
   gs://<proyecto>-ops/scripts/l3_compare_runs.py`),
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

### `l3_derived_check.py`

Prueba la vista `derived` (familia D, [`derived.sql`](../../shared/dc_views/src/dc_views/derived.sql);
contrato en [data-contracts](../data-contracts.md#vista-derived)) sobre **un θ y un mes** reales
(ITSC-386). Abre el `summaries.parquet` de L3 y el `events.parquet` de L2 de esa partición por su
ruta completa, los registra en DuckDB como datasets de PyArrow, ejecuta `derived.sql` y cuenta
todo con consultas de agregación: ninguna fila pasa por Python. **No importa `dc_views`**: la
imagen de `ops_tools` no lo trae e importarlo cambiaría `uv.lock`; por eso el `.sql` se sube
aparte y `derived_version` sale de su primera línea (`-- derived_version: X.Y.Z`).

Hacen falta **dos** archivos en `scripts/` (el `.sql` es de lectura, como el script). El apply del
stack `ops` publica los dos; verifica con `gcloud storage ls gs://<proyecto>-ops/scripts/`.
Si cambias `derived.sql`, el siguiente apply de `ops` lo actualiza (un cambio en esa ruta dispara el
Deploy): la sonda usa la copia de `scripts/`, no la del repo.

| `args` | Efecto |
| --- | --- |
| `--theta 0.00098052 --month 2026-09` | Obligatorios: un θ (8 decimales como máximo) y un mes `YYYY-MM` |
| `--summaries-root`, `--events-root` | Otras raíces, también locales (por defecto `<proyecto>-dc-descriptors/l3` y `<proyecto>-dc-events/l2`) |
| `--sql gs://<proyecto>-ops/scripts/derived.sql` | Otra copia de la vista, local o `gs://` (por defecto, la de `scripts/` bajo `ops-script` y la del repo en local) |
| `--provider`, `--market`, `--asset` | Otra serie (por defecto `binance`, `spot`, `BTCUSDT`) |
| `--out` | Otra ruta del CSV; una ruta que termina en `/` es un directorio |

**El CSV** (`l3_derived_check_<mes>_<θ>.csv`, en `results/<ejecución>/`) tiene el encabezado
`seccion,columna,filas,nulos,min,mediana,max,verdaderas,violaciones` y una fila por cosa medida:

- `doble`: las 16 columnas `DOUBLE` de D, con `filas`, `nulos`, `min`, `mediana` y `max`.
- `bandera`: `atomic_sweep`, `empty_overshoot` y `provider_gap`, con `filas` y `verdaderas`.
- `invariante`: una fila por invariante con sus `violaciones` (0 es lo esperado): filas de la
  vista = filas de `summaries`; unión 1:1 por `(theta, confirm_agg_trade_id)` (sin huérfanas ni
  claves repetidas); `|c_flow_imbalance| ≤ 1`; `|os_flow_imbalance| ≤ 1`; `c_range ≥ 0`;
  `c_instant_bias ≥ 0` (tolerancia 1e-9 por el redondeo de `DOUBLE`); la cota de confirmación;
  `atomic_sweep` ⇔ `c_duration = 0` ⇔ `c_tick_rate` nulo; `empty_overshoot` ⇔ todos los `os_*`
  de D nulos; banderas sin nulos; columnas «nunca nulas» sin nulos; sin `±inf` ni `NaN`.

La **cota de confirmación** se calcula sobre `summaries` unido a `events`, porque D no trae
`c_price_max` ni `c_price_min`: en un upturn `(c_price_max − reference_price) / reference_price ≥ θ`
y en un downturn `(reference_price − c_price_min) / reference_price ≥ θ`. **`c_range ≥ θ` no es
invariante**: el tick de referencia no pertenece a la fase
([TRD-L3 §6.1 (ADR-L3-01)](../TRD/l3.md#61-adr-l3-01--pertenencia-de-un-tick-a-una-fase-por-agg_trade_id-no-por-tiempo-ni-por-posición), consecuencia 3), así que
`c_price_min` puede quedar por encima de `p_R` en un upturn y `c_range` por debajo de θ en esa
distancia.

**Última línea:** `derived check: mes=… theta=… derived_version=… filas=… atomic_sweep=…
empty_overshoot=… provider_gap=… violaciones=… pared_s=… veredicto=ok|violaciones|fallo`. Los
tres conteos son las banderas verdaderas; `violaciones` suma todos los invariantes. Con
`veredicto=fallo` (falta un archivo, el `.sql` no se lee o DuckDB falla) los conteos salen `-`.

**Códigos:** **0** sin violaciones; **1** con violaciones, sin alguno de los dos archivos, si el
`.sql` no se lee o si DuckDB falla; **2** por uso (`--theta` o `--month` ausentes o inválidos, sin
raíces ni `OPS_RESULTS_URI`, `--out` `gs://` fuera de `results/`). Si el CSV no se puede
escribir, la sonda sale con **0** (si no hay violaciones) y lo avisa en el log con
`no se guardó la tabla en …`. Una violación **nunca se
tolera**: es un defecto de `derived.sql` o del dato de entrada; abre una card de bug con la línea.

**Cómo correrla.** *Actions → Run job*: `job` = `ops-script`, `script` = `l3_derived_check.py`,
`args` = `--theta <θ> --month <mes>`. Un θ y un mes por corrida: la memoria es O(partición) (las
medianas retienen los valores de las 16 columnas). Para el primer mes real elige un θ con
decenas de miles de filas, no el más fino.

**Corrida real (ITSC-386, 2026-10-29).**

| Mes | θ | Filas | `atomic_sweep` | `empty_overshoot` | `provider_gap` | Violaciones | Pared | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-03 | `0.00098052` | 36 239 | 1 872 | 2 706 | 0 | 0 | 2,2 s | [40804620149](https://github.com/byroncz/intrinsica/actions/runs/40804620149) |

### `l3_gap_probe.py`

Mide los **huecos entre ticks consecutivos de L1** de un mes (ITSC-392): la diferencia entre
`transact_time` contiguos del `consolidated.parquet`, con el hueco entre el último tick de un row
group y el primero del siguiente. Con los resúmenes de varios meses el arquitecto fija `silence_us`,
el umbral de silencio de la familia T: separa «mercado en silencio» de «proveedor caído». Mide
**ticks, no eventos**: el hueco del proveedor de 2018-02 sale como el máximo de ese mes. La corrida
real lo midió en 33,5 h y lo cierra un tick del 2018-02-09 10:00 UTC (tabla de «Corrida real»).
**No importa ninguna capa** (una importación cambiaría `uv.lock`) y abre el archivo por su
ruta completa, por la [trampa del 403](#la-trampa-del-403-por-prefijo).

El apply del stack `ops` lo publica en `scripts/`; verifica con
`gcloud storage ls gs://<proyecto>-ops/scripts/l3_gap_probe.py`.

| `args` | Efecto |
| --- | --- |
| `--month 2018-02` | Obligatorio: el mes `YYYY-MM` |
| `--landing-root` | Otra raíz de L1, también local con la misma disposición hive (por defecto `gs://<proyecto>-landing/l1`) |
| `--provider`, `--market`, `--asset` | Otra serie (por defecto `binance`, `spot`, `BTCUSDT`) |
| `--out` | Otra ruta del CSV; una ruta que termina en `/` es un directorio |

**Memoria.** Lee un row group a la vez y solo `transact_time`; el último tiempo de cada grupo se
conserva para medir el hueco con el primero del siguiente. Los huecos de hasta 1 s van a un
histograma exacto de una cuenta por µs (10⁶ + 1 enteros, 8 MB) y los mayores se guardan uno a uno.
El pico es O(row group): no crece con los ticks del mes. Al final lee `agg_trade_id` de **un** row
group, el del tick que cierra el hueco máximo. Sobre un mes sintético de 6 M ticks en 8 row groups
tarda 0,1 s y el RSS sube 54 MB.

**Cuantil.** Rango más cercano: con N huecos, el cuantil q es el que ocupa el lugar ⌈N·q⌉ al
ordenarlos. Es exacto y siempre es un hueco que ocurrió, nunca una interpolación.

**El CSV** (`l3_gap_probe_<mes>.csv`, en `results/<ejecución>/`) tiene el encabezado
`medida,valor,legible,agg_trade_id,transact_time` y una fila por medida, en este orden:

- `ticks` y `huecos` (`ticks − 1`).
- `p50_us`, `p90_us`, `p99_us`, `p999_us`, `p9999_us` y `max_us`: `valor` en µs y `legible` en
  segundos, minutos u horas. `max_us` trae además el `agg_trade_id` y el `transact_time` (µs) del
  tick que **cierra** el hueco máximo; si el máximo se repite, el primero.
- `ge_1s`, `ge_10s`, `ge_60s` y `ge_600s`: cuántos huecos son ≥ 1 s, ≥ 10 s, ≥ 60 s y ≥ 600 s.

**Última línea:** `gap probe: mes=… ticks=… p50_us=… p90_us=… p99_us=… p999_us=… p9999_us=…
max_us=… ge_1s=… ge_10s=… ge_60s=… ge_600s=… pared_s=… veredicto=ok|fallo`. Con
`veredicto=fallo` las medidas salen `-`; un mes de un solo tick sale `ok` con los cuantiles y el
máximo en `-`.

**Códigos:** **0** si midió; **1** sin `consolidated.parquet` del mes, si la lectura falla o si
`transact_time` decrece; **2** por uso (`--month` ausente o inválido, sin `--landing-root` ni
`OPS_RESULTS_URI`, `--out` `gs://` fuera de `results/`). Si el CSV no se puede escribir, sale con
**0** y lo avisa en el log con `no se guardó la tabla en …`. Solo lee; lo único que escribe es el CSV.

**Cómo correrla.** *Actions → Run job*: `job` = `ops-script`, `script` = `l3_gap_probe.py`,
`args` = `--month <mes>`. Un mes por corrida. Las corridas las lanza y aprueba el humano; pega las
tres últimas líneas en ITSC-392.

**Corrida real (ITSC-392), 2026-10-29.** Las cifras son las de la última línea de cada corrida;
las de 2026-08 vienen del CSV, porque su salida llegó a Cloud Logging después del paso de logs del
run. Antes de 2025, L1 viene de milisegundos convertidos a µs: los huecos de 2018-02 y 2023-03 son
múltiplos de 1 000 µs y el p50 de 2023-03 sale en 0 por los ticks del mismo ms. Con estas cifras
`silence_us` quedó en 10 000 000 (10 s), escrito en la Épica ITSC-391.

| Mes | Ticks | p50 | p99 | p99,99 | Máximo | ≥ 1 s | ≥ 10 s | ≥ 60 s | ≥ 600 s | Pared | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-02 | 9 811 197 | 36 ms | 2,482 s | 9,708 s | 33,5 h | 550 420 | 830 | 3 | 3 | 3,0 s | [40869457621](https://github.com/byroncz/intrinsica/actions/runs/40869457621) |
| 2023-03 | 190 227 841 | 0 | 234 ms | 1,593 s | 2,5 h | 89 008 | 1 | 1 | 1 | 38,1 s | [40869510256](https://github.com/byroncz/intrinsica/actions/runs/40869510256) |
| 2026-08 | 23 939 440 | 68 µs | 1,62 s | 5,489 s | 13,5 s | 631 803 | 22 | 0 | 0 | pendiente | [40869570000](https://github.com/byroncz/intrinsica/actions/runs/40869570000) |

### `l3_path_check.py`

Prueba `path.parquet` (familia T, [contrato](../data-contracts.md#esquema-de-pathparquet); [TRD-L3 §7.2.1](../TRD/l3.md#721-salida--pathparquet-familia-t))
sobre **un θ y un mes** reales (ITSC-395). Abre el `path.parquet` y el `summaries.parquet` de L3 de
esa partición por su ruta completa, los registra en DuckDB como datasets de PyArrow y cuenta todo
con consultas de agregación: ninguna fila pasa por Python. `state_version` y `silence_us` salen de
los metadatos de `path.parquet`, leídos con PyArrow. **No importa ninguna capa** (una importación
cambiaría `uv.lock`). A diferencia de `l3_derived_check.py`, no necesita un `.sql` ni `events.parquet`
de L2: todo lo que cruza sale de `summaries.parquet`.

El apply del stack `ops` lo publica en `scripts/`; verifica con
`gcloud storage ls gs://<proyecto>-ops/scripts/l3_path_check.py`.

| `args` | Efecto |
| --- | --- |
| `--theta 0.00098052 --month 2023-03` | Obligatorios: un θ (8 decimales como máximo) y un mes `YYYY-MM` |
| `--summaries-root` | Otra raíz de L3, también local, donde viven los dos archivos (por defecto `gs://<proyecto>-dc-descriptors/l3`) |
| `--provider`, `--market`, `--asset` | Otra serie (por defecto `binance`, `spot`, `BTCUSDT`) |
| `--out` | Otra ruta del CSV; una ruta que termina en `/` es un directorio |

**El CSV** (`l3_path_check_<mes>_<θ>.csv`, en `results/<ejecución>/`) tiene el encabezado
`seccion,columna,filas,nulos,min,mediana,max,violaciones` y una fila por cosa medida:

- `columna`: las 13 columnas de datos de T (T-01 a T-13), con `filas`, `nulos`, `min`, `mediana` y `max`.
- `invariante`: una fila por invariante con sus `violaciones` (0 es lo esperado). Los límites con
  tolerancia 1e-9 por el redondeo de `DOUBLE`:
  - **Esqueleto:** filas de `path` = filas de `summaries`; unión 1:1 por `(theta, confirm_agg_trade_id)`
    (sin huérfanas ni claves repetidas); mismos row groups y filas por row group según los pies de
    los dos archivos.
  - **Rangos:** `c_realized_var ≥ 0` y `os_realized_var ≥ 0`; `c_tortuosity ≥ 1` y `os_tortuosity ≥ 1`;
    `|c_sign_autocorr| ≤ 1` y `|os_sign_autocorr| ≤ 1`; `0 ≤ c_max_gap ≤ c_duration` y
    `0 ≤ os_max_gap ≤ os_duration` (las duraciones de `summaries`); `0 ≤ *_idle_fraction ≤ 1`;
    `c_idle_fraction = 0` donde `silence_us > c_duration`; `0 ≤ c_t25 ≤ c_t50 ≤ c_t75 ≤ 1`.
  - **Regla de nulos**, una invariante `nulos_<columna>` por cada una de las 13: cuenta las filas
    donde la columna es nula y la regla dice que no, o al revés. `os_realized_var` y `os_max_gap` ⇔
    `os_ticks = 0`; `c_sign_autocorr` ⇔ `c_ticks < 2` **o todos los ticks del mismo lado**
    (`c_quantity_buy = 0` o `c_quantity_sell = 0`), y `os_sign_autocorr` igual con `os_*`;
    `c_idle_fraction` y `c_t25` a `c_t75` ⇔ `c_duration = 0`; `os_idle_fraction` ⇔ `os_ticks = 0` o
    `os_duration = 0`; `c_realized_var`, `c_tortuosity` y `c_max_gap` nunca nulas.
  - **`doubles_finitos`:** ni `±inf` ni `NaN` en las once columnas `DOUBLE`.

Dos precisiones sobre la regla de nulos, ambas del contrato:

- **`*_sign_autocorr`.** La nulidad no es solo `ticks < 2`: con dos o más ticks todos del mismo lado la
  varianza de los lados es 0 y el contrato la deja nula ([TRD-L3 §7.2.1](../TRD/l3.md#721-salida--pathparquet-familia-t)).
  Un mismo lado se lee de `summaries` como una cantidad de compra o de venta igual a 0 (todo tick tiene
  cantidad positiva).
- **`os_tortuosity`.** Es nula con `os_ticks = 0` y también con `p_E = p_C`, pero `summaries` no trae
  `confirm_price` ni `extreme_price` y la sonda no lee L2. Comprueba lo que sí ve: una no nula con
  `os_ticks = 0` es violación, y una nula con `os_ticks > 0` solo es válida si el extremo del
  overshoot toca el de la confirmación (`os_price_max = c_price_max` u `os_price_min = c_price_min`,
  necesario para `p_E = p_C`). Es una comprobación laxa a propósito, sin falsos positivos: un `inf` o
  `NaN` donde `p_E = p_C` lo atrapa `doubles_finitos`.

**Última línea:** `path check: mes=… theta=… state_version=… silence_us=… filas=… violaciones=…
pared_s=… veredicto=ok|violaciones|fallo`. `violaciones` suma todas las invariantes. Con
`veredicto=fallo` (falta un archivo, no se lee un pie, `path.parquet` no trae `silence_us` o DuckDB
falla) lo que no se pudo medir sale `-`.

**Códigos:** **0** sin violaciones; **1** con violaciones, sin alguno de los dos archivos, si un pie
no se lee o si DuckDB falla; **2** por uso (`--theta` o `--month` ausentes o inválidos, sin
`--summaries-root` ni `OPS_RESULTS_URI`, `--out` `gs://` fuera de `results/`). Si el CSV no se puede
escribir, la sonda sale con **0** (si no hay violaciones) y lo avisa en el log con
`no se guardó la tabla en …`. Una violación **nunca se tolera**: es un defecto del motor de T o del
dato de entrada; abre una card de bug con la línea.

**Memoria.** Un θ y un mes por corrida: el pico es O(partición), porque las medianas retienen los
valores de las 13 columnas. Para el primer mes real elige un θ con decenas de miles de filas, no el
más fino.

**Cómo correrla.** Necesita `path.parquet` del θ y el mes, o sea el backfill de T de ITSC-394.
*Actions → Run job*: `job` = `ops-script`, `script` =
`gs://<proyecto>-ops/scripts/l3_path_check.py` (la URI completa: `run-job-tasks.sh` rechaza otra
cosa con código 2), `args` = `--theta <θ> --month <mes>`. Las corridas las lanza y aprueba el humano; pega la última
línea en ITSC-395.

**Corrida real (ITSC-395, 2026-10-30).** Las 29 invariantes dieron 0 violaciones; las 36 239 filas
coinciden con las de `summaries.parquet` del mismo θ y mes (`l3_derived_check.py`, ITSC-386). El CSV
quedó en `results/ops-script-pd47c/l3_path_check_2023-03_0.00098052.csv`.

| Mes | θ | `state_version` | `silence_us` | Filas | Violaciones | Pared | Run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-03 | `0.00098052` | 1.0.0 | 10 000 000 | 36 239 | 0 | 2,4 s | [41052835807](https://github.com/byroncz/intrinsica/actions/runs/41052835807) |

### `l3_fit_bridge_probe.py`

Decide si la familia A (Forma, Épica ITSC-397) vale la pena construirla (ITSC-398). Sobre **un θ y un
mes** reales calcula, de cada evento, `c_accel` y `c_jump_share` observados y los compara con los mismos
registros sobre 20 puentes brownianos simulados por evento. Si `c_accel` observado se distribuye igual
que el de un puente, la forma de la trayectoria no dice nada que `c_realized_var` y el tiempo no digan, y
la familia se retira. **No importa ninguna capa** (una importación cambiaría `uv.lock`).

El apply del stack `ops` lo publica en `scripts/`; verifica con
`gcloud storage ls gs://<proyecto>-ops/scripts/l3_fit_bridge_probe.py`.

| `args` | Efecto |
| --- | --- |
| `--theta 0.00098052 --month 2023-03` | Obligatorios: un θ (8 decimales como máximo) y un mes `YYYY-MM` |
| `--l1-root`, `--l2-root`, `--l3-root` | Otras raíces, también locales (por defecto `gs://<proyecto>-landing/l1`, `-dc-events/l2` y `-dc-descriptors/l3`) |
| `--provider`, `--market`, `--asset` | Otra serie (por defecto `binance`, `spot`, `BTCUSDT`) |
| `--bridges 20` | Puentes por evento |
| `--seed` | Semilla del generador de NumPy (por defecto la fija del script, `397398`) |
| `--out` | Otra ruta del CSV; una ruta que termina en `/` es un directorio |

**Qué mide.** En la fase de confirmación `(R, C]` de cada evento, con `τ = (t − t_R) / (t_C − t_R)` y
`y = (p − p_R) / (p_C − p_R)` (el ancla y el cierre salen de `events.parquet`):

- `c_accel`: coeficiente `a` del ajuste `y = a·τ² + b·τ` sin intercepto, por mínimos cuadrados. Positivo
  si el recorrido se concentra al final de la fase; cercano a 0 si es parejo.
- `c_jump_share`: `(p_C − p_prev) / (p_C − p_R)`, con `p_prev` el último tick con `t < confirm_time`.
  Cuánto del recorrido hace el último instante.
- **El puente**, por evento: `n = c_ticks` (`summaries.parquet`), `σ² = c_realized_var / n`
  (`path.parquet`) y `ℓ_k = ℓ_R + (ℓ_C − ℓ_R)·k/n + σ·(W_k − (k/n)·W_n)`, con `W` un paseo gaussiano
  estándar, en los **tiempos reales** de los ticks: así la comparación aísla la forma del precio de la
  del reloj. Mismo `--seed`, mismos archivos y mismos argumentos dan el mismo CSV, byte a byte.
- Un evento con `c_accel` nulo (menos de 2 ticks, todos en el mismo `t`, o `t_C = t_R`) queda fuera de
  las dos muestras y se cuenta en `excluidos`. También el que arranca en el mes anterior y no cabe
  entero en el L1 del mes (a lo sumo uno por mes y θ).

**D (cómo leerla).** Es la distancia de Kolmogorov-Smirnov: el mayor desfase vertical entre las
distribuciones acumuladas empíricas observada y simulada, `sup |F_obs − F_sim|`, de 0 (idénticas) a 1
(sin solapamiento). Es un tamaño del efecto, no un valor p: con decenas de miles de eventos, hasta una
diferencia irrelevante sería «significativa». El umbral se fijó **antes de medir** (decisión 1 de la
Épica): **`D_accel ≥ 0,10` → `veredicto=sigue`** (siguen las hijas 2 a 5); **`D_accel < 0,10` →
`retirada`** (la familia A no se construye). `D_jump` se reporta para interpretar, pero no decide.
Referencia de la prueba local: con trayectorias de forma `D_accel ≈ 0,92`; con puentes puros ≈ 0,03.

**El CSV** (`l3_fit_bridge_probe_<mes>_<θ>.csv`, en `results/<ejecución>/`) tiene el encabezado
`registro,fuente,medida,valor`:

- Por cada `registro` (`c_accel`, `c_jump_share`) y `fuente` (`observado`, `simulado`): las medidas
  `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95` y `p99` (rango más cercano: con N valores, el
  que ocupa el lugar ⌈N·q⌉ al ordenarlos) y `N` (observado: eventos medidos; simulado: puentes·N).
- Por cada `registro`, con `fuente=ambas`: `D`.
- Con `registro` y `fuente` vacíos: `excluidos`, `puentes` y `semilla`.

**Última línea:** `fit probe: mes=… theta=… eventos=… excluidos=… puentes=… semilla=… D_accel=…
D_jump=… pared_s=… veredicto=sigue|retirada|fallo`. `eventos` son los N de las muestras. Con
`veredicto=fallo` lo que no se pudo medir sale `-`.

**Códigos:** **0** si midió (con `sigue` o con `retirada`: ambos son resultados); **1** si falta alguno
de los cuatro archivos, la lectura falla, los archivos no cuadran entre sí (otro número de filas o de
ticks, otro `confirm_agg_trade_id`, eventos más allá de los ticks del mes) o no queda ningún evento
medible; **2** por uso (`--theta` o `--month` ausentes o inválidos, `--bridges` o `--seed` no enteros o
`--bridges` menor que 1, sin las tres raíces ni `OPS_RESULTS_URI`, `--out` `gs://` fuera de
`results/`). Si el CSV no se puede escribir, la sonda sale con **0** y lo avisa con
`no se guardó la tabla en …`. Solo lee; lo único que escribe es el CSV.

**Memoria.** Los ticks se leen un row group a la vez (`agg_trade_id`, `transact_time` y `price`) y los
tres archivos por evento en lotes de 32 768 filas; el estado del evento abierto son unos pocos KB. Lo
único que crece con el mes son los resultados, necesarios para que D y los cuantiles sean exactos:
≈ 336 B por evento con 20 puentes (≈ 12 MB para los 36 239 eventos de 2023-03).

**Cómo correrla.** Necesita `events.parquet` de L2 y `summaries.parquet` y `path.parquet` de L3 del θ y
el mes (backfill de T), más el `consolidated.parquet` del mes. *Actions → Run job*: `job` = `ops-script`,
`script` = `gs://<proyecto>-ops/scripts/l3_fit_bridge_probe.py` (la URI completa), `args` =
`--theta 0.00098052 --month 2023-03`. Las corridas las lanza y aprueba el humano; pega la última línea y
los cuantiles en ITSC-398. El veredicto lo escribe el arquitecto en la Épica ITSC-397, sección
«Veredicto de la sonda».

**Corrida real (ITSC-398, 2026-11-05).** `D_accel` = 0,278658, por encima del umbral de 0,10: la forma
de la trayectoria no es la de un puente, así que `veredicto=sigue`. El veredicto completo y su lectura
están en la Épica ITSC-397, sección «Veredicto de la sonda».

| Mes | θ | Eventos | Excluidos | `D_accel` | `D_jump` | Pared | Veredicto | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-03 | `0.00098052` | 34 207 | 2 032 | 0,278658 | 0,408690 | 120,9 s | sigue | [41951296143](https://github.com/byroncz/intrinsica/actions/runs/41951296143) |

### `l3_fit_check.py`

Prueba `fit.parquet` (familia A, [contrato](../data-contracts.md#esquema-de-fitparquet); [TRD-L3 §7.2.2](../TRD/l3.md#722-salida--fitparquet-familia-a))
sobre **un θ y un mes** reales (ITSC-401). Abre el `fit.parquet` y el `summaries.parquet` de L3 y el
`events.parquet` de L2 de esa partición por su ruta completa, los registra en DuckDB como datasets de
PyArrow y cuenta todo con consultas de agregación: ninguna fila pasa por Python. `state_version` sale de
los metadatos de `fit.parquet`, leídos con PyArrow. **No importa ninguna capa** (una importación
cambiaría `uv.lock`). Necesita `events.parquet` porque la regla de nulos de A depende del ancla y el
cierre de cada fase, que son datos de L2 (`reference_time`, `confirm_time`, `extreme_time` y los tres
precios).

El apply del stack `ops` lo publica en `scripts/`; verifica con
`gcloud storage ls gs://<proyecto>-ops/scripts/l3_fit_check.py`.

| `args` | Efecto |
| --- | --- |
| `--theta 0.00098052 --month 2023-03` | Obligatorios: un θ (8 decimales como máximo) y un mes `YYYY-MM` |
| `--summaries-root` | Otra raíz de L3, también local, donde viven `fit.parquet` y `summaries.parquet` (por defecto `gs://<proyecto>-dc-descriptors/l3`) |
| `--l2-root` | Otra raíz de L2, también local, para `events.parquet` (por defecto `gs://<proyecto>-dc-events/l2`) |
| `--provider`, `--market`, `--asset` | Otra serie (por defecto `binance`, `spot`, `BTCUSDT`) |
| `--out` | Otra ruta del CSV; una ruta que termina en `/` es un directorio |

**El CSV** (`l3_fit_check_<mes>_<θ>.csv`, en `results/<ejecución>/`) tiene el encabezado
`seccion,columna,filas,nulos,min,mediana,max,q01,q05,q95,q99,violaciones,conteo,fraccion` y una fila por
cosa medida:

- `columna`: las 9 columnas de datos de A (A-01 a A-09), con `filas`, `nulos`, `min`, `mediana` y `max`;
  las tres `*_accel` traen además los cuantiles 1, 5, 95 y 99 %.
- `invariante`: una fila por invariante con sus `violaciones` (0 es lo esperado), con tolerancia 1e-9 por
  el redondeo de `DOUBLE`:
  - **Esqueleto:** filas de `fit` = filas de `summaries`; unión 1:1 por `(theta, confirm_agg_trade_id)`
    con `summaries` (`union_1a1`) y con `events` (`union_events_1a1`), sin huérfanas ni claves
    repetidas; mismos row groups y filas por row group según los pies de `fit` y `summaries`.
  - **Leyes de una fila** (decisión 7 de la Épica, corregida el 2026-11-06): `0 < c_jump_share ≤ 1`,
    `os_jump_share ≥ 0` y cuatro leyes de `ev_jump_share`, una invariante por caso
    (`ev_jump_share_caso_1` a `_4`), que `summaries` y `events` permiten decidir
    ([TRD-L3 §7.2.2](../TRD/l3.md#722-salida--fitparquet-familia-a)): (1) `os_ticks = 0`:
    `ev_jump_share = c_jump_share`; (2) `os_ticks > 0` y `os_duration > 0`: `0 ≤ ev_jump_share ≤ 1 + θ`;
    (3) `os_ticks > 0`, `os_duration = 0` y `extreme_time = confirm_time`: `0 < ev_jump_share ≤ 1 + θ`
    (imposible por ADR-L3-01, que pone el instante de confirmación entero en la fase `c`; queda como guarda);
    (4) `os_ticks > 0`, `os_duration = 0` y `extreme_time > confirm_time`: `ev_jump_share` entre
    `(p_E − confirm_instant_price_best)/(p_E − p_R)` y `(p_E − confirm_instant_price_worst)/(p_E − p_R)`
    (puede ser negativo). La primera versión de la sonda exigía `ev_jump_share ≥ 0` siempre; en 2023-03
    dio 240 violaciones, todas de esa ley, porque estaba mal derivada y no por un error del motor.
  - **Regla de nulos**, una invariante `nulos_<columna>` por cada `*_slope`, `*_accel` y `*_jump_share`:
    - `*_jump_share` es nula **exactamente** cuando `p_fin = p_0` de su fase (y, en `os`, con
      `os_ticks = 0`); nula donde no toca o no nula donde toca es violación.
    - `*_slope` es nula si la fase tiene menos de dos ticks (`c_ticks`, `os_ticks` o su suma) o si el
      cierre cae en el instante del ancla (`confirm_time = reference_time`, `extreme_time =
      confirm_time` o `extreme_time = reference_time`). Una `*_slope` **no nula** donde la regla manda
      nula es violación.
    - `*_accel`, además, si `p_fin = p_0`. Misma lectura.
  - **`doubles_finitos`:** ni `±inf` ni `NaN` en las nueve columnas.
- `informativo`: cifras que **no** suman a `violaciones`, porque la sonda no ve los ticks:
  - `mismo_t_{c,os,ev}_slope`: una `*_slope` nula con ticks suficientes y tiempos distintos. Es el caso
    «todos los ticks en el mismo `transact_time`»: no se puede decidir sin los ticks, así que se cuenta
    aparte. Su suma es el `mismo_t` de la última línea.
  - `accel_sin_explicar_{c,os,ev}_accel`: una `*_accel` nula con `*_slope` no nula, ticks suficientes,
    tiempos distintos y `p_fin ≠ p_0`. Lo explican un único `τ` positivo distinto o la singularidad
    numérica del sistema normal tras cuantizar (`det ≤ 0`), que `summaries` no permite predecir.
    Esperado: cerca de 0.
  - `ev_caso_{1_sin_overshoot,2_overshoot_con_duracion,3_overshoot_en_el_instante,4_instante_de_confirmacion}`:
    cuántos eventos caen en cada caso de las leyes de `ev_jump_share`. Los cuatro casos parten todas las
    filas, así que suman `filas`.
  - `abs_{c,os,ev}_accel_mayor_2`: filas con `|*_accel| > 2` y su `fraccion` sobre las no nulas.
    `|a| ≤ 2` **no es una ley**: el ajuste de dos parámetros sobre pocos ticks apiñados da cualquier
    coeficiente. Si la fracción cambia mucho entre meses, es una señal para mirar, no un fallo.

**Última línea:** `fit check: mes=… theta=… state_version=… filas=… violaciones=… mismo_t=… pared_s=…
veredicto=ok|violaciones|fallo`. `violaciones` suma todas las invariantes y ninguna cifra informativa.
Con `veredicto=fallo` (falta un archivo, no se lee un pie o DuckDB falla) lo que no se pudo medir sale `-`.

**Códigos:** **0** sin violaciones; **1** con violaciones, sin alguno de los tres archivos, si un pie no
se lee o si DuckDB falla; **2** por uso (`--theta` o `--month` ausentes o inválidos, sin las dos raíces ni
`OPS_RESULTS_URI`, `--out` `gs://` fuera de `results/`). Si el CSV no se puede escribir, la sonda sale
con **0** (si no hay violaciones) y lo avisa en el log con `no se guardó la tabla en …`. Una violación
**nunca se tolera**: es un defecto del motor de A o del dato de entrada; abre una card de bug con la
línea.

**Memoria.** Un θ y un mes por corrida: el pico es O(partición), porque las medianas y los cuantiles
retienen los valores de las 9 columnas.

**Cómo correrla.** Necesita `fit.parquet` del θ y el mes, o sea el backfill de A de ITSC-400.
*Actions → Run job*: `job` = `ops-script`, `script` = `gs://<proyecto>-ops/scripts/l3_fit_check.py` (la URI
completa: `run-job-tasks.sh` rechaza otra cosa con código 2), `args` = `--theta <θ> --month <mes>`. Las
corridas las lanza y aprueba el humano; pega la última línea en ITSC-401.

**Corrida real (ITSC-401).** La primera corrida (ley `ev_jump_share ≥ 0` siempre) dio 240 violaciones, las
240 de esa ley; las otras 16 invariantes dieron 0. El arquitecto corrigió la ley (cuatro casos) el
2026-11-06 y la sonda cambió con ella. La segunda corrida dio `violaciones=0`, con las 20 invariantes en 0.
Eventos por caso de `ev_jump_share`: 2 706 (caso 1), 32 278 (caso 2), 0 (caso 3) y 1 255 (caso 4). El caso 3
queda en 0 porque no puede darse (ADR-L3-01 pone el instante de confirmación entero en la fase `c`): es una
guarda, y un valor mayor que 0 sería un defecto del motor o de L2. Informativos de la segunda corrida:
`mismo_t` 1 083 / 595 / 513 (`c`, `os`, `ev`), `accel_sin_explicar` 159 / 0 / 116 y `|a| > 2` en
1 566 / 8 749 / 1 365 filas. El CSV quedó en `results/ops-script-x8xqb/l3_fit_check_2023-03_0.00098052.csv`.

| Corrida | Mes | θ | `state_version` | Filas | Violaciones | `mismo_t` | Pared | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (ley antigua) | 2023-03 | `0.00098052` | no pegada | 36 239 | 240 (`ev_jump_share_no_negativo`) | 2 191 | 4,3 s | [42147884536](https://github.com/byroncz/intrinsica/actions/runs/42147884536) |
| 2 (cuatro leyes) | 2023-03 | `0.00098052` | 1.0.0 | 36 239 | 0 | 2 191 | 4,1 s | [42242276494](https://github.com/byroncz/intrinsica/actions/runs/42242276494) |

### `l3_spectrum_probe.py`

Decide qué registros de la familia F (Frecuencia, Épica ITSC-410) valen la pena construir (ITSC-411). Sobre
**un θ y un mes** reales calcula, de cada fase de cada evento (`c` = `(R, C]` y `os` = `(C, E]`), cuánta
energía de la serie de ticks cae en cada una de 7 bandas de período, con dos candidatos (retornos y precios)
y con la descomposición de Haar, y la compara con sus nulos. Si la distribución observada se parece a la del
nulo, o repite lo que `c_accel` (A) o `c_t50` (T) ya dicen, el registro no se construye. Además estima, solo
con L2 y L3, el pico de ticks en vuelo de los 50 θ a la vez y la fracción de fases bajo 16 ticks por θ.
**No importa ninguna capa** (una importación cambiaría `uv.lock`).

El apply del stack `ops` lo publica en `scripts/`; verifica con
`gcloud storage ls gs://<proyecto>-ops/scripts/l3_spectrum_probe.py`.

| `args` | Efecto |
| --- | --- |
| `--theta 0.00098052 --month 2023-03` | Obligatorios: un θ (8 decimales como máximo) y un mes `YYYY-MM` |
| `--l1-root`, `--l2-root`, `--l3-root` | Otras raíces, también locales (por defecto `gs://<proyecto>-landing/l1`, `-dc-events/l2` y `-dc-descriptors/l3`) |
| `--provider`, `--market`, `--asset` | Otra serie (por defecto `binance`, `spot`, `BTCUSDT`) |
| `--nulls 20` | Nulos por fase y evento (retornos barajados y puentes brownianos) |
| `--seed` | Semilla del generador de NumPy (por defecto la fija del script, `410411`) |
| `--inflight-months 2017-12,2021-01,2023-03` | Meses del pico de ticks en vuelo; vacío omite ese pase |
| `--out` | Otra ruta del CSV; una ruta que termina en `/` es un directorio |

**Qué mide.** Por fase, con `n` ticks y el ancla `p_0` (`p_R` en `c`, `p_C` en `os`, como T), en tiempo-tick
sin remuestrear:

- **Retornos:** `r_k = ln(p_k / p_{k−1})`, con el primer paso desde el ancla; los retornos cero se conservan.
  Nulo: los retornos de la fase barajados.
- **Precios:** `(p_k − recta_k) / p_0`, con la recta del ancla (índice 0) al último tick (índice `n`). Nulo:
  el puente browniano de ITSC-398 en los `n` ticks, con `σ² = c_realized_var / n` (`c`) o
  `os_realized_var / os_ticks` (`os`) de `path.parquet`, llevado a precios con la misma recta.
- **Espectro:** serie centrada, ventana de Hann periódica, `numpy.fft.rfft`, energía `|X_k|²` de los bins
  `k = 1..⌊n/2⌋` y fracción de la energía total por banda de período `n/k`: `[2, 4)`, `[4, 8)`, `[8, 16)`,
  `[16, 32)`, `[32, 64)`, `[64, 128)` y `128 o más` (bandas 1 a 7; suman 1). Ejemplo para entender la ventana:
  una senoide de período 8 con `n = 64` deja 5/6 en la banda 3 y 1/6 en la 2.
- **Haar** (decisión 2): sobre los últimos `2^⌊log₂ n⌋` retornos de la fase, centrados, energía por nivel
  `Σ d²` con la transformada ortonormal; los niveles 1 a 6 son las escalas 2 a 64 ticks y el séptimo junta de
  128 en adelante, alineados con las 7 bandas. Es la convención que la hija 2 escribe en el contrato.
- **Fases excluidas** (todas contadas): menos de 16 ticks; evento con `extreme_agg_trade_id −
  reference_agg_trade_id > 20 000 000` (sus dos fases); energía cero (todos los retornos iguales, o precios
  sobre su recta); y el evento que arranca en el mes anterior y no cabe en el L1 del mes. Una fase `os` con
  `os_ticks = 0` no existe y se cuenta aparte (`fases_vacias`).

**Cómo leer D y ρ.** `D` es la distancia de Kolmogorov-Smirnov, `sup |F_obs − F_sim|`, de 0 (idénticas) a 1
(sin solapamiento), de la fracción de energía en la banda más rápida (período 2 a 4, banda 1) y en la más
lenta (128 o más, banda 7). Es un tamaño del efecto, no un valor p. `ρ` es el coeficiente de Spearman entre
las fracciones, banda a banda: precios contra retornos, Haar contra Fourier de retornos y cada candidato
contra `c_accel` y `c_t50`; una columna constante da `nan`, que no cuenta como redundancia. Los umbrales se
fijaron **antes de medir** (decisión 1): **`veredicto=sigue`** si y solo si algún candidato tiene
`D_fast ≥ 0,10` o `D_slow ≥ 0,10` sobre `c` y su |ρ| máximo contra `c_accel` y `c_t50` no pasa de 0,9; si no,
`retirada`. Qué registros de F-01 a F-04 entran lo decide el arquitecto con las cifras, en la Épica ITSC-410
(«Veredicto de la sonda»). `f32_err` es el mayor |fracción en f32 − fracción en f64| sobre las fases del mes
(decisión 5: si cabe en `f32` el búfer de retornos es de 4 B por tick).

**Ubicación de la energía en el tiempo (ITSC-416).** F guarda cuánta energía hay por nivel de Haar y pierde
*dónde* ocurre. El mapa completo (un coeficiente por tick y nivel) pesa lo mismo que los ticks y no se guarda;
el candidato es un número por nivel y fase, el centro de masa de la energía. Sobre la misma serie de Haar (los
últimos `2^m` retornos, `m = ⌊log₂ n⌋`, centrados), el coeficiente de detalle `d_{j,i}` del nivel `j`
(`i = 0..N_j − 1`) está en el centro de su bloque dentro de la fase:

- `t_{j,i} = (n − 2^m + (i + 0,5)·2^j) / n`, en [0, 1] (0 es el comienzo de la fase y 1 el final);
- `loc_j = Σ t_{j,i}·d_{j,i}² / Σ d_{j,i}²`. El nivel 7 junta los niveles de escala 128 en adelante, cada
  coeficiente con su posición. Es nulo (y se cuenta en `nulos`) si el nivel tiene menos de 2 coeficientes o
  energía cero: con `n` entre 16 y 31 solo existen los niveles 1 a 3 con ubicación.
- `loc_total = Σ t_k·r_k² / Σ r_k²` con `t_k = (k − 0,5)/n` sobre los `n` retornos de la fase: dónde ocurrió la
  agitación sin separar por nivel. Es la línea base: si todos los niveles se mueven con ella, la ubicación por
  nivel no agrega nada.

Ejemplo para entender la posición: con `n = 10`, Haar mira los últimos 8 retornos y el nivel 1 tiene 4
coeficientes en `t = 0,3 / 0,5 / 0,7 / 0,9`: los dos primeros retornos no cuentan, y por eso la ubicación de
ruido iid sale cerca de `0,5 + (n − 2^m)/(2n)` y no de 0,5.

El nulo son los mismos retornos barajados de siempre (la sonda no pide números nuevos al generador: las cifras
del CSV de ITSC-411 no cambian con la misma semilla). `D` es la de Kolmogorov-Smirnov exacta entre observado y
simulado, sin los nulos, por nivel y fase y para `loc_total`; además la corrida calcula la `D` de las
**fracciones** de Haar por nivel, que la primera corrida no calculó.

**Redundancia nueva** (Spearman, fase c; `nan` no cuenta; el máximo de una matriz no cuenta la diagonal):
`loc_j` contra `loc_total`, contra `c_jump_share` y `c_accel` (`fit.parquet`) y contra `c_t50`; `loc_total`
contra esos tres; `loc_j` entre niveles (7 × 7). Y las que la primera corrida no midió (consecuencia 6 del
veredicto de ITSC-410): cada banda de Fourier de retornos y cada nivel de Haar contra `c_sign_autocorr` y
`c_realized_var` (`path.parquet`) y `c_ticks` (`summaries.parquet`), y las bandas de Fourier entre sí (7 × 7).

**Regla de la ubicación** (fijada antes de medir; no cambia el `veredicto`):

- la ubicación de un nivel **pasa** si su `D ≥ 0,10` en c y su |ρ| máximo contra `loc_total`, `c_jump_share`,
  `c_accel` y `c_t50` no pasa de 0,9;
- `loc_total` **pasa** si su `D ≥ 0,10` y su |ρ| máximo contra `c_jump_share`, `c_accel` y `c_t50` no pasa de
  0,9;
- un `nan` no pasa en `D` y no cuenta en ρ. `locs_pasan` es cuántos de los 7 niveles pasan.

Qué entra lo decide el arquitecto con las cifras, en la Épica ITSC-410 bajo el veredicto. Si la ubicación pasa,
entra al motor como columnas nuevas de `spectrum.parquet` en una card aparte, antes del backfill de F; si no,
queda documentado y F sigue con sus 28 columnas.

**El CSV** (`l3_spectrum_probe_<mes>_<θ>.csv`, en `results/<ejecución>/`) tiene el encabezado
`seccion,candidato,fase,banda,fuente,medida,valor`:

- `cuantiles`: por `candidato` (`retornos`, `precios`, `haar`), `fase` (`c`, `os`), `banda` (1 a 7) y `fuente`
  (`observado`, `simulado`): `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99` (rango más cercano:
  con N valores, el que ocupa el lugar ⌈N·q⌉) y `N` (simulado: nulos·N). Los simulados se guardan en `float32`.
- `ubicacion`: por `candidato` (`ubicacion` con `banda` 1 a 7, y `ubicacion_total` con `banda` vacía), `fase` y
  `fuente` (`observado`, `simulado`): `p1` … `p99` de los valores no nulos, `N` (no nulos) y `nulos`.
- `D`: por `candidato` (`retornos`, `precios`) y `fase`, `D_fast` y `D_slow`; y, por `banda` (nivel 1 a 7) y
  `fase`, `D` de `haar` (fracciones por nivel) y de `ubicacion`, más `D` de `ubicacion_total`. Sin muestras
  no nulas, `D` es `nan`.
- `rho`: por par (`precios_retornos`, `haar_fourier`, `retornos_accel`, `retornos_t50`, `precios_accel`,
  `precios_t50`), `fase` y `banda`, más `max_abs_rho` por par. Los pares con A y T son solo de `c`. Pares de
  ITSC-416, solo de `c`: por nivel o banda, `loc_loctotal`, `loc_jump`, `loc_accel`, `loc_t50`,
  `retornos_sign`, `retornos_rvar`, `retornos_ticks`, `haar_sign`, `haar_rvar` y `haar_ticks`; de una sola
  cifra (`banda` vacía), `loctotal_jump`, `loctotal_accel` y `loctotal_t50`; y las matrices `loc_loc` y
  `retornos_retornos` (49 filas, `banda` = fila y `fuente` = `con=<columna>`).
- `regla`: `pasa` (0 o 1) por nivel de `ubicacion` y para `ubicacion_total`, y `locs_pasan`.
- `f32`: `f32_err`.
- `fases_lt16`: por θ del mes (en `fuente`, `theta=<θ>`), `eventos` y, por fase, `lt16` y `fraccion_lt16`; en
  `os` además `vacias`. Una fase `os` vacía cuenta como bajo 16 (decisión 4: sus columnas serían nulas igual).
- `ticks_en_vuelo`: por mes de `--inflight-months` (en `fuente`, `mes=<mes>`), `thetas` (cuántos θ entraron
  al pico ese mes), `inflight_peak_ticks` y su equivalente en bytes a 8 B y a 4 B por tick (`bytes_8B`,
  `bytes_4B`).
- `excluidos`: `fases_cortas`, `fases_vacias` y `energia_cero` por fase (y por candidato), más
  `eventos_sobre_presupuesto` y `evento_mes_anterior`. `muestras`: las N por fase. `config`: `nulls` y `semilla`.

**Pico de ticks en vuelo** (decisión 6; no lee L1). Con `R`, `C` y `E` de `events.parquet` de todos los θ que
tienen L2 en el mes, la fase abierta de un θ en el tick `x` es `c` si `R < x ≤ C` (cuenta `x − R`) y `os` si
`C < x ≤ E` (cuenta `x − C`); un evento sobre el presupuesto cuenta 0 porque no guarda retornos. El pico de
la suma sobre θ está siempre en un fin de fase, así que se recorre una mezcla ordenada de los archivos por
lotes (estado O(θ), memoria O(lote × θ)). Es una cota superior: los ids del proveedor tienen huecos. Con el
pico el arquitecto fija el tamaño del job de F (≤ 1 GiB: job de 4 GiB; entre 1 y 4 GiB: 8 GiB; más: plan B).

**Última línea:** `spectrum probe: mes=… theta=… eventos=… D_ret_fast=… D_ret_slow=… D_price_fast=…
D_price_slow=… rho_price_ret=… rho_haar_fourier=… rho_ret_accel=… rho_ret_t50=… rho_price_accel=…
rho_price_t50=… f32_err=… n_lt16=… inflight_peak_ticks=… D_haar_fast=… D_haar_slow=… D_loc_fast=…
D_loc_slow=… D_loc_total=… rho_loc_total=… rho_loc_jump=… rho_ret_sign=… rho_ret_ticks=… locs_pasan=…
pared_s=… veredicto=sigue|retirada|fallo`. `eventos` son los N de `c`; cada `rho_*` es el máximo de |ρ| de las
7 bandas o niveles; `n_lt16` es el número de fases `c` del θ bajo 16 ticks; `inflight_peak_ticks` es el mayor
de los meses. Los campos de ITSC-416, todos de la fase `c`: `D_haar_*` y `D_loc_*` son `D` de las fracciones de
Haar y de `loc_j` en el nivel 1 (`fast`) y el 7 (`slow`); `D_loc_total` la de `loc_total`; `rho_loc_total` y
`rho_loc_jump`, el máximo de |ρ| de los 7 niveles de `loc_j` contra `loc_total` y `c_jump_share`;
`rho_ret_sign` y `rho_ret_ticks`, el de las 7 bandas de Fourier de retornos contra `c_sign_autocorr` y
`c_ticks`; `locs_pasan`, cuántos de los 7 niveles pasan la regla. Con `veredicto=fallo` lo que no se pudo
medir sale `-`.

**Códigos:** **0** si midió (con `sigue` o con `retirada`: ambos son resultados); **1** si falta alguno de los
archivos, un listado de GCS o la lectura falla (la API de GCS devuelve un error: sale como `veredicto=fallo`,
sin traceback), los archivos no cuadran entre sí (otro número de filas o de ticks, otro
`confirm_agg_trade_id`, eventos más allá de los ticks del mes), un mes de `--inflight-months` no tiene L2 o
no queda ningún evento medible. Los listados corren antes de medir, para fallar barato; **2** por uso (`--theta` o `--month` ausentes o inválidos, `--nulls` o
`--seed` no enteros o `--nulls` menor que 1, `--inflight-months` inválido, sin las tres raíces ni
`OPS_RESULTS_URI`, `--out` `gs://` fuera de `results/`). Si el CSV no se puede escribir, la sonda sale con
**0** y lo avisa con `no se guardó la tabla en …`. Solo lee; lo único que escribe es el CSV.

**Memoria.** Los ticks se leen un row group a la vez (`agg_trade_id` y `price`) y los cuatro archivos por
evento en lotes de 32 768 filas. El estado es la serie de la fase abierta (`n` precios en f64, a lo sumo el
presupuesto: 160 MB), que se libera al cerrar la fase; el cálculo suma varias copias transitorias (retornos,
serie centrada, espectro) y los nulos se generan en grupos de a lo sumo 2²⁰ valores. Medido en local: ≈ 2 ms
por fase típica; una fase de 2 M ticks sube el pico unos 130 MiB y una de 20 M ticks (el presupuesto) llega a
1,7 GiB (≈ 9 veces la serie), dentro de los 8 GiB del job `ops-script`. Lo único que crece con el mes son los resultados por
evento, necesarios para que D y los cuantiles sean exactos: ≈ 3,7 KB por evento con 20 nulos, más 8 números
`float32` por fase y por nulo de la ubicación (≈ 1,3 KB), bajo 10 KB por evento (≈ 180 MB para los 36 239
eventos de 2023-03). El estado sigue siendo la fase abierta. El pase del pico en vuelo usa 8 192 filas por θ.

**Cómo correrla.** Necesita `events.parquet` de L2, `summaries.parquet`, `path.parquet` y `fit.parquet` de L3
del θ y el mes, el `consolidated.parquet` del mes de L1, y L2 de los meses de `--inflight-months` (los θ que
tengan L2 en cada mes). *Actions → Run job*: `job` = `ops-script`, `script` =
`gs://<proyecto>-ops/scripts/l3_spectrum_probe.py` (la URI completa), `args` =
`--theta 0.00098052 --month 2023-03`. Las corridas las lanza y aprueba el humano; pega la última línea y los
cuantiles en ITSC-411. El veredicto lo escribe el arquitecto en la Épica ITSC-410, sección «Veredicto de la
sonda».

**Corrida real (ITSC-416):** la lanza el humano con `ops-script` sobre 2023-03, θ `0.00098052`, con los
mismos `args` (el apply de `ops` publica el script). Pega la última línea en ITSC-416; la decisión sobre la
ubicación la escribe el arquitecto en la Épica ITSC-410, bajo el veredicto.

**Corrida real (ITSC-411):** pendiente. La tabla se llena con la última línea cuando el humano la corra.

| Mes | θ | Eventos | `D_ret_fast` | `D_ret_slow` | `D_price_fast` | `D_price_slow` | `f32_err` | `n_lt16` | Pico en vuelo | Pared | Veredicto | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-03 | `0.00098052` | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |

### La trampa del 403 por prefijo

La service account de `ops-script` lee `l1/`, `l2/`, `l3/` y `viz/tiles/` con
condiciones IAM por prefijo. `pyarrow.dataset.dataset("gs://<bucket>/l3")` (o
`pq.read_table` sobre un prefijo) pide primero el **objeto** `l3` sin barra, y la
condición `startsWith("…/l3/")` lo niega: 403, aunque los archivos de abajo sean
legibles (run 38659287542 de `ops-script`). Los scripts evitan el problema:
**listan de forma recursiva** y **abren cada archivo por su ruta completa**. Un
script nuevo hace lo mismo; desde Cloud Shell, con tus credenciales, el 403 no
aparece.
