# Runbook: operación de L1 (backfill, seam-check y ciclo daily a monthly-close)

Lo ejecuta el humano desde *Actions → Run job*; ningún agente dispara
`terraform.yml` ni `run-job.yml`. Cierra la Épica de L1 con evidencia: carga el
histórico completo, valida las costuras y recorre el ciclo diario a mensual
sobre un mes real. Referencia: [TRD-L1 §8, §10.3 y §13](../TRD/l1.md). La
dimensión de las tareas salió de la [sonda](sonda-l1.md).

Fechas de este runbook calculadas al **2026-09-26**. Si lo ejecutas otro día,
recalcula los rangos con la regla de la sección siguiente.

## Rangos

- **Último mes cerrado**: el mes anterior al actual. A 2026-09-26 es
  **2026-08** (el consolidado se publica el primer lunes del mes siguiente).
- **Backfill**: de `2017-08` al **penúltimo** mes cerrado, `2026-07`: 108 meses.
- **Ciclo daily a monthly-close**: el último mes cerrado, `2026-08`. Se deja
  fuera del backfill a propósito, para probar con él el reemplazo de
  provisionales por el consolidado.

## Prerrequisitos

1. Stack `data` aplicado con [ITSC-222](https://github.com/byroncz/intrinsica/pulls?q=ITSC-222)
   (buckets landing, dq-findings y manifest). Lo aplica el humano desde
   Cloud Shell.
2. Stack `l1` aplicado (*Actions → Terraform*, `stack` = `l1`, `action` =
   `apply`, aprobado en el environment `gcp`) con la imagen que incluye
   ITSC-227 y, para evitar los problemas que este mismo runbook encontró,
   también ITSC-231 (ZIP con CSV duplicado), ITSC-232 (`from = to` en
   `run-job.yml`) e ITSC-233 (hallazgo `seam_skipped`):
   `layers/l1_ingest/VERSION` ≥ 0.5.2. Los cuatro jobs de ITSC-224:
   `l1-backfill`, `l1-daily`, `l1-monthly-close`, `l1-seam-check`. Verifica
   que existen: `gcloud run jobs list --region <región>`.
3. **Timeout por tarea.** Ya fijado por ITSC-219 en el módulo `layer`:
   **3600 s** para `backfill` y `monthly-close`, **900 s** para `daily` y
   `seam-check`, `max_retries = 1`. El mes más pesado medido hasta ahora
   (2026-02, ver "Resultados") tardó 537 s: el margen sigue siendo amplio.
   Si cambia la config de cómputo (vCPU/GiB) o aparece un mes claramente más
   pesado, vuelve a comparar la pared contra estos topes antes de lanzar.
4. **Cuota de Cloud Run.** Cada tarea pide 4 vCPU y 16 GiB. El módulo no fija
   `parallelism`, así que Cloud Run lanza todas las tareas que la cuota
   regional permita a la vez; el resto espera. La cuota de CPU y memoria
   la ves en *IAM y administración → Cuotas* (`Cloud Run Admin API`, región
   del stack). El paralelismo cambia el tiempo de pared, no el costo
   (§10.2). Si la cuota es baja, parte el backfill en rangos más chicos
   (ejemplo: un run por año).
5. Variables del environment `gcp` cargadas (`GCP_PROJECT_ID`, `GCP_REGION`,
   `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SERVICE_ACCOUNT`).

## Cómo funciona `run-job.yml`

El workflow cuenta las unidades del rango con `.github/scripts/run-job-tasks.sh`
y las pasa como `--tasks`. Cada tarea toma `from` + `CLOUD_RUN_TASK_INDEX`:
la tarea 0 es el primer mes (o día), la 1 el siguiente, etc. Por eso
**`--tasks` = número de meses** en `backfill` y `monthly-close`, y número de
días en `daily`. `seam-check` siempre es una sola tarea que recorre todo el
rango. Los jobs `l2-*` no usan task array (sus meses son secuenciales): el
script devuelve una sola unidad y el workflow lanza `--tasks 1`. Regla: no
lances a mano el mismo mes y modo que Scheduler esté ejecutando.

Al final, el run vuelca los logs de la ejecución y un resumen (estado,
tareas completadas y fallidas, inicio y fin) en el *Summary* del run. Bajo
esa tabla va la de **hallazgos de calidad de datos** de la ejecución, de
cualquier capa: check, severidad, unidad, valor y detalle, con los `error`
primero. Sale de las líneas de log JSON que deja `emit_findings`, filtradas
por la ejecución; si no hubo hallazgos dice "sin hallazgos".

## Paso 1: backfill 2017-08 a 2026-07

*Actions → Run job → Run workflow*, rama `main`:

| Input | Valor |
| --- | --- |
| `job` | `l1-backfill` |
| `from` | `2017-08` |
| `to` | `2026-07` |
| `force` | sin marcar |

El paso "Unidades" debe listar 108 meses y el job lanza `--tasks 108`. Cada
tarea escribe `consolidated.parquet` de su mes, registra el checksum en el
manifiesto y emite hallazgos con `stage = canonical`.

Anota por cada run: URL, nombre de la ejecución Cloud Run, tareas
completadas y fallidas, inicio y fin. Para el costo, necesitas la **pared de
cada tarea** (ver "Costo real").

### Reintentar solo lo fallido

Una ejecución con tareas fallidas termina `Fallida` y el resumen muestra
cuántas. Para saber cuáles:

```bash
gcloud logging read \
  'resource.type="cloud_run_job" AND resource.labels.job_name="l1-backfill" AND severity>=ERROR' \
  --project <proyecto> --freshness=7d --format='value(timestamp,textPayload)'
```

Distingue la causa como en la [sonda](sonda-l1.md#qué-leer-y-qué-anotar):
OOM ("Memory limit exceeded"), timeout (tarea cortada a los 3600 s de
`backfill`, ITSC-219) o error de datos. Luego relanza `l1-backfill` **con el mismo rango completo** o solo
con los meses fallidos (`from` = `to` = ese mes, un run por mes o tramo
contiguo), sin `force`. Los meses ya procesados se saltan solos: la tarea
compara el `.CHECKSUM` publicado con el del manifiesto (ITSC-221) y, si
coinciden, no descarga ni reescribe. Repetir el rango completo cuesta solo
una petición de checksum por mes ya hecho. Usa `force` únicamente si cambió el
código de L1 y quieres reprocesar todo.

## Paso 2: seam-check histórico

Cuando el backfill esté completo (sin tareas fallidas), lanza:

| Input | Valor |
| --- | --- |
| `job` | `l1-seam-check` |
| `from` | `2017-08` |
| `to` | `2026-07` |
| `force` | sin marcar |

Una tarea recorre los 107 bordes (M, M+1) leyendo solo los footers Parquet,
sin cargar los datos. Emite un hallazgo `seam_discontinuity` por borde, con
estado `pass` o `fail`. En el log verás `costura (a, m) -> (a, m): <estado>`.

Por cada borde `fail`, decide si es un **hueco del proveedor** (Binance no
publicó esos `agg_trade_id`, el caso habitual, RL1-04) o un problema nuestro.
Lee `details` del hallazgo (ver "Consultar los hallazgos") y confirma:
si el `agg_trade_id` mínimo del mes siguiente supera en más de 1 al máximo del
mes anterior, falta un tramo en el origen; contrástalo con el archivo diario
del proveedor de esa fecha. Un solapamiento o duplicado en la costura no es
un hueco del proveedor: abre una card (`task-create`).

## Paso 3: ciclo daily a monthly-close sobre 2026-08

### 3a. daily de todos sus días

| Input | Valor |
| --- | --- |
| `job` | `l1-daily` |
| `from` | `2026-08-01` |
| `to` | `2026-08-31` |
| `force` | sin marcar |

Son 31 tareas, una por día. Cada una escribe `provisional-day=DD.parquet` y
verifica la costura contra el día previo (hallazgos `stage = provisional`).
Confirma que el resumen diga 31 tareas completadas y 0 fallidas. Lista los
provisionales:

```bash
gcloud storage ls "gs://<bucket landing>/l1/provider=binance/market=spot/asset=BTCUSDT/year=2026/month=08/"
```

Deben aparecer `provisional-day=01.parquet` a `provisional-day=31.parquet` y
**ningún** `consolidated.parquet`.

### 3b. monthly-close

| Input | Valor |
| --- | --- |
| `job` | `l1-monthly-close` |
| `from` | `2026-08` |
| `to` | `2026-08` |
| `force` | sin marcar |

Descarga el consolidado, lo escribe, compara contra los provisionales
(`daily_monthly_drift`), valida la costura con 2026-07 y, solo entonces,
borra los provisionales. Es idempotente: si se corta, relanzar completa el
cierre.

### 3c. Verificar el cierre

1. **`consolidated.parquet` existe** en la partición
   `year=2026/month=08` (mismo `gcloud storage ls` de 3a).
2. **Sin provisionales:** el mismo listado no debe mostrar ningún
   `provisional-day=*.parquet`. El log del run trae
   `cierre completado unidad=...: 31 provisionales borrados`.
3. **`daily_monthly_drift`** y **hallazgos canonical**: ver la sección
   siguiente. Debe haber un `daily_monthly_drift` de 2026-08 y hallazgos
   `stage = canonical` del mes que superan a los `provisional`.

## Archivo aún no publicado

Binance publica el diario de D el día **D+1** y el consolidado mensual de M
el **primer lunes de M+1** (README de `binance-public-data`; TRD-L1 §1.1 y
§8.4). Antes de eso el ZIP da 404, y eso no es una falla: L1 no lo reintenta
y lo registra como hallazgo según la fecha **UTC** de la corrida. Ejemplo, mes
2026-09: el primer lunes de octubre de 2026 es el 5.

| Corrida (UTC) | Hallazgo | Severidad | Salida |
| --- | --- | --- | --- |
| Antes del primer lunes de M+1 (diario: antes de D+1) | `source_not_published`, "dentro del calendario de Binance; publicación esperada el YYYY-MM-DD" | info | 0 |
| El primer lunes de M+1 (diario: el día D+1) | `source_not_published`, "día de publicación, aún sin archivo" | warning | 0 |
| Después (diario: después de D+1) | `source_delayed`, "N días de retraso sobre el calendario publicado de Binance", `metric_value = N` | error | 3 |

Binance no publica hora ni SLA, por eso el día de publicación completo cuenta
como a tiempo y recién el día siguiente es retraso. Con salida 0 no se escribe
`consolidated.parquet` ni manifiesto: la unidad queda pendiente y la frontera
de L2 (fail-closed) la ignora. Con salida 3 la tarea falla y la orquestación
no ejecuta el `next_job` (`l2-monthly`), que es lo correcto. El scheduler de
`monthly-close` corre el día 8, así que en estado estacionario solo puede
darse `source_delayed`.

**Cómo se ve en el resumen de Actions:** una corrida con `source_not_published`
termina en verde y la tabla de hallazgos trae la fila con su `info` o
`warning`, la fecha esperada y la unidad. Un `source_delayed` deja la
ejecución en rojo y la fila `error` sale primera en la tabla. El caso real de
ITSC-295: `l1-backfill` de 2017-12 a 2026-09 lanzado el sábado 2026-10-03
falló en 2026-09 tras cuatro descargas y sin hallazgo; con la regla termina en
verde con `source_not_published` info.

**Qué hacer:**

- `info` o `warning`: nada, o relanzar `l1-backfill` con `from = to = <mes>`
  desde el día de publicación en adelante y comprobar que el mes se ingiere.
- `error`: Binance se retrasó. Confirma en
  [data.binance.vision](https://data.binance.vision) que el archivo sigue sin
  estar y relanza cuando aparezca. Si nunca aparece, escala: es un hueco de la
  fuente.

## Marcas de Binance

En abril de 2022 Binance auditó su histórico Spot, recuperó agg trades
faltantes y marcó como inválidos los duplicados, con `p = 0`, `q = 0`,
`f = -1`, `l = -1`, conservando su `agg_trade_id` y `transact_time` para no
romper la secuencia ([changelog de la API Spot, entrada 2022-04-12](https://github.com/binance/binance-spot-api-docs/blob/master/CHANGELOG_CN.md)).
No son transacciones.

**Regla de L1** (ITSC-294):

- Una fila es marca si y solo si cumple las cuatro igualdades: `price = 0`,
  `quantity = 0`, `first_trade_id = -1` y `last_trade_id = -1`.
- L1 la descarta antes de escribir `consolidated.parquet` y emite
  `provider_invalid_marker` con el conteo y hasta 10 `agg_trade_id`.
- Cualquier otra fila con `price <= 0`, `quantity <= 0`, `first_trade_id < 0`
  o `last_trade_id < first_trade_id` es dato corrupto: `price_out_of_range`
  (`error`) y la unidad falla sin escribir.
- `aggid_gap` y `aggid_duplicate` se calculan sobre los ids crudos, antes del
  descarte, así que las marcas no aparecen como huecos. Una marca nunca es la
  primera ni la última fila del mes, por eso la costura entre meses no cambia.

**Meses afectados en BTCUSDT:** 2017-12 y 2018-01. Son los únicos dos de los
360 ZIP mensuales regenerados que incluyen BTCUSDT en la
[lista oficial de archivos regenerados](https://github.com/binance/binance-public-data) (`updates/2022-04-21_aggregate_trade_updates.zip`).
Un escaneo de los 109 meses de L1 (mínimo de `price` por row group) confirma
que solo esos dos tenían precio 0.

**Si L1 ya había escrito esos meses sin la regla** (imagen anterior a 0.5.14),
reprocésalos: `l1-backfill` con `from = 2017-12`, `to = 2018-01` y `force`
marcado. Después, `l2-backfill` con todos los campos vacíos: la frontera
reanuda en 2017-12 (2017-11 no cambió, su carry-over sigue válido). Los
SHA-256 esperados de los ZIP corregidos son `2017-12`
`45261b647c70862edd60e788ce20f32d3b78d0753f9841ba3ca0e50afdbfbefe` y `2018-01`
`645f8f581a6e4828df12f16b73f9f7451f12bef2ab24d4fc4b0466805e793e7d`.

## Consultar los hallazgos

Los hallazgos son Parquet append-only bajo
`gs://<bucket dq-findings>/l1/detected_date=<fecha>/`. Desde Cloud Shell:

```bash
gcloud storage cp -r "gs://<bucket dq-findings>/l1" /tmp/dq
python3 - <<'PY'
import duckdb
q = """
select detected_date, unit, check_type, stage, status, severity, metric_value, details
from read_parquet('/tmp/dq/**/*.parquet', hive_partitioning=true)
where check_type in ('seam_discontinuity', 'daily_monthly_drift')
   or (unit like '%2026-08%' and stage = 'canonical')
order by detected_date, unit
"""
print(duckdb.sql(q).df().to_string())
PY
```

Ajusta los nombres de columna al contrato del lago
([§7.3 del TRD](../TRD/l1.md) y `docs/data-contracts.md`) si difieren.

## Costo real

Con la configuración de 4 vCPU y 16 GiB y `s` = pared de la tarea en
segundos:

- GiB-s = 16 × Σ`s`; vCPU-s = 4 × Σ`s`, sumando **todas** las tareas del
  backfill, no solo el mes más pesado. La pared de cada tarea sale de las
  líneas `sonda: unit=... wall_s=<s>` del log, o de la duración de cada tarea
  en la ejecución Cloud Run (*Cloud Run → Jobs → l1-backfill → ejecución →
  Tareas*). Los reintentos también facturan: suma cada intento.
- **Facturado según Billing:** *Facturación → Informes*, filtro por servicio
  `Cloud Run`, rango de las fechas de la ejecución, agrupado por SKU.
  Anota lo facturado neto de crédito y de cupo gratis. Billing tarda hasta
  24 h en reflejar el consumo.
- **Estado estacionario estimado:** un `daily` por día y un `monthly-close`
  por mes. Toma la pared de las 31 tareas de 3a (Σ) y la de 3b, y proyecta a
  un mes: GiB-s/mes ≈ 16 × (Σ paredes daily del mes + pared monthly-close).
  Compáralo con el cupo gratis mensual de 360.000 GiB-s y 180.000 vCPU-s.
- **Contra [§10.3](../TRD/l1.md):** backfill ≈ 0–1 USD; `daily` y
  `monthly-close` ≈ 0 (free tier). La sonda ya advirtió que el backfill
  completo, ×96 meses con 2023-03, da 886.272 GiB-s (2,46 veces el cupo):
  esta ejecución dice si el real, con meses más livianos, se acerca o no.
  Anota si el excedente factura y cuánto.

## Resultados

Ejecutados por el humano el 2026-09-26 (backfill) y el 2026-09-27 (el
resto), hora UTC-5. Imagen `l1_ingest`, stack `l1` con
4 vCPU / 16 GiB, timeout 3600/900 s y `max_retries = 1` (ITSC-219). El
backfill inicial corrió con `l1_ingest:0.5.0`, ya con ITSC-230 (workflow) en
`main`. Antes del reintento de 2021-12 se mergeó ITSC-231 (ZIP con CSV
duplicado, 0.5.1, 14:07): el reintento y el seam-check corrieron con
`0.5.1`. Antes del daily se mergeó también ITSC-232 (`from = to` en
`run-job.yml`, 14:35): el daily y el monthly-close corrieron con `0.5.1`
igual. ITSC-233 (hallazgo `seam_skipped`, 0.5.2, 15:07) se mergeó después
del daily, a partir de lo que este runbook encontró (ver "Ciclo daily a
monthly-close" más abajo): no cubre estas corridas, solo las futuras.

**Runs**

| Paso | URL del run | Rango | Tareas OK / fallidas | Reintentos (meses) | Ejecución Cloud Run | Inicio → fin (run de Actions, UTC-5) |
| --- | --- | --- | --- | --- | --- | --- |
| Backfill | [36283117940](https://github.com/byroncz/intrinsica/actions/runs/36283117940) | 2017-08 a 2026-07 | 106 OK, 1 saltado por checksum (2023-03), 1 fallido (2021-12) | 2021-12 → [36344381547](https://github.com/byroncz/intrinsica/actions/runs/36344381547), OK | `l1-backfill-zzpft` | 2026-09-26 19:38 → 21:35 |
| Seam-check | [36344788264](https://github.com/byroncz/intrinsica/actions/runs/36344788264) | 2017-08 a 2026-07 | 1 tarea, OK | - | no reportada | 2026-09-27 14:33 → 14:36 |
| daily | [36345285379](https://github.com/byroncz/intrinsica/actions/runs/36345285379) | 2026-08-01 a 2026-08-31 | 31 OK | - | no reportada | 2026-09-27 14:41 → 14:48 |
| monthly-close | [36346362202](https://github.com/byroncz/intrinsica/actions/runs/36346362202) | 2026-08 | 1 tarea, OK | - | no reportada | 2026-09-27 14:59 → 15:02 |

"Ejecución Cloud Run" queda como "no reportada" salvo en el backfill, cuyo
nombre reportó el humano junto con la Σ de paredes (ver "Costo"): el paso 1
pide anotarla, pero el log del workflow no la deja en claro para una
ejecución fallida. "Inicio → fin" es el del run de GitHub Actions (el
único dato disponible sin credenciales de GCP), no el de la ejecución de
Cloud Run Jobs en sí.

2021-12 falló por un ZIP con dos CSV (el bug de ITSC-231, corregido antes del
reintento); el reintento (2026-09-27 14:26 → 14:30) pasó el chequeo nuevo
`zip_extra_members`. 2023-03, el mes más pesado según la sonda original, se
saltó porque el `.CHECKSUM` ya coincidía con el manifiesto (ITSC-221): no
hubo que reprocesarlo.

**Timeout por tarea:** aplicado (ITSC-219), 3600/900 s. Ningún mes se acercó
al tope: la pared máxima medida (2026-02, backfill) fue 537 s, bien dentro
del margen.

**seam_discontinuity**

Los 107 bordes del rango 2017-08–2026-07 dieron `pass` en el `seam-check`
histórico: sin huecos del proveedor ni solapamientos que explicar. No hay
ningún borde `fail` que anotar para esta ejecución.

**Ciclo daily a monthly-close (2026-08)**

| Verificación | Resultado |
| --- | --- |
| 31 provisionales tras daily | OK, 31/31; 6 costuras diarias (2026-08-04, 08, 12, 22, 25 y 27) no se pudieron evaluar por la carrera entre tareas paralelas (el día previo aún no estaba escrito). El daily corrió con `l1_ingest:0.5.1`, antes de ITSC-233: cada omisión solo dejó un WARNING "costura omitida" en el log, sin hallazgo en el lago. ITSC-233 (0.5.2) agrega el hallazgo `seam_skipped` (`severity=info`, `status=pass`) para las corridas futuras |
| `consolidated.parquet` presente | OK, escrito por el `monthly-close` |
| Sin provisionales tras el cierre | OK, los 31 `provisional-day=*.parquet` se borraron |
| `daily_monthly_drift` | `pass` |
| Hallazgos canonical | Emitidos por el `monthly-close`; costura con 2026-07 también `pass` |

**Costo**

RSS y pared por paso (de lo reportado por el humano desde Cloud Run):

| Paso | Tareas | RSS pico máx. | RSS pico mediana | Pared máx. | Pared mediana |
| --- | --- | --- | --- | --- | --- |
| Backfill | 108 | 9.194 MiB (2026-02) | 2.568 MiB | 537 s (2026-02) | 80 s |
| Backfill, reintento 2021-12 | 1 | 3.590 MiB | - | 113,8 s | - |
| daily | 31 | 570 MiB | - | 11,1 s | - |
| monthly-close | 1 | 2.347 MiB | - | 81,2 s | - |

**Σ real de las paredes del backfill**, el dato primario que pide
[§10.3](../TRD/l1.md). La leyó el humano de la línea `sonda: ... wall_s=` de
cada tarea, en el log de la ejecución `l1-backfill-zzpft` (run
[36283117940](https://github.com/byroncz/intrinsica/actions/runs/36283117940))
y del reproceso de 2021-12 (run
[36344381547](https://github.com/byroncz/intrinsica/actions/runs/36344381547)),
y la dejó en el PR 54 (comentario del 2026-09-28). El log de GitHub Actions
del reproceso trae la línea (`wall_s=113.8`); el del backfill no, porque
Actions solo muestra el `gcloud run jobs execute --wait`, no los logs de las
tareas.

| Tramo | Pared Σ | GiB-s (×16) | vCPU-s (×4) |
| --- | --- | --- | --- |
| 107 tareas OK (106 procesadas + 2023-03 saltada por checksum) | 11.412,4 s | 182.598,4 | 45.649,6 |
| 2 intentos fallidos de 2021-12 (`max_retries = 1`) | 89,2 s | 1.427,2 | 356,8 |
| Ejecución `l1-backfill-zzpft`, total | 11.501,6 s | 184.025,6 | 46.006,4 |
| Reproceso de 2021-12 | 113,8 s | 1.820,8 | 455,2 |
| **Backfill completo** | **11.615,4 s (3,23 h)** | **185.846,4 (0,52× cupo)** | **46.461,6 (0,26× cupo)** |

Cupo gratis mensual: 360.000 GiB-s y 180.000 vCPU-s. Los intentos fallidos
entran en la suma porque también facturaron. Para repetir la medición sin
pasar por la consola:

```sh
gcloud logging read 'resource.type="cloud_run_job"
  AND resource.labels.job_name="l1-backfill"
  AND labels."run.googleapis.com/execution_name"="l1-backfill-zzpft"
  AND textPayload:"sonda: unit="' \
  --project intrinsica-dc --format='value(textPayload)'
```

y sumar cada `wall_s`. Filtrar por `execution_name` deja afuera el
reproceso manual de 2021-12 y la sonda de ITSC-218, que usan el mismo job y
el mismo formato de log.

**Cotas previas a la Σ.** Antes de tener la Σ, el runbook la acotaba con los
dos únicos datos reportados, la mediana (80 s) y la máxima (537 s, 2026-02).
Al menos 54 de las 108 tareas duran ≥ 80 s y de ellas la mayor dura 537 s,
así que Σ ≥ 53 × 80 s + 537 s = 4.777 s. Por arriba, Σ ≤ 108 × 537 s =
57.996 s. La Σ real de la ejecución, 11.501,6 s, cae dentro de ese rango y
queda un 33 % por encima de 108 × la mediana (8.640 s): la cola derecha
existe, pero es corta. Se dejan como registro de cómo se razonó sin el dato:

| Concepto | 108 × mediana (no es cota) | Cota inferior (53 × 80 s + 537 s) | Cota superior (108 × 537 s) | Σ real (ejecución) |
| --- | --- | --- | --- | --- |
| Backfill, GiB-s | 138.240 (0,38× cupo) | 76.432 (0,21× cupo) | 927.936 (2,58× cupo) | 184.025,6 (0,51× cupo) |
| Backfill, vCPU-s | 34.560 (0,19× cupo) | 19.108 (0,11× cupo) | 231.984 (1,29× cupo) | 46.006,4 (0,26× cupo) |

La sonda ([`docs/runbooks/sonda-l1.md`](sonda-l1.md), extrapolación ×96
meses) estimó 886.272 GiB-s asumiendo que el mes más pesado (2023-03, 577 s)
se repetía en los 96 meses. El real, 185.846,4 GiB-s, es 0,21× esa
extrapolación, 4,8 veces menos: la mayoría de los meses son mucho más
livianos que el más pesado. Por volumen, el backfill completo cabe en el cupo
gratis mensual.

**Contraste con Billing:** rango 2026-09-26 a 2026-09-27, proyecto
`intrinsica-dc`, cuenta en COP (≈4.000 COP/USD). Cubre el backfill (09-26)
junto con el seam-check, el daily y el monthly-close (09-27), así que el
monto es del rango completo, no solo del backfill: Cloud Run bruto 5.197 COP
(~1,30 USD), crédito de prueba -5.197 COP, neto 0. La Σ, a precio de lista
con 4 vCPU / 16 GiB, da ≈1,58 USD brutos solo para el backfill (cálculo del
humano). Los dos coinciden en el orden de magnitud. Billing da menos aunque
incluye tres runs más, así que la tarifa efectiva que aplicó es menor que la
de lista usada para estimar; eso no cambia el veredicto. Billing tampoco
muestra descontado el cupo gratis, aunque por volumen la Σ cabe en él: el
bruto completo lo absorbió el crédito de prueba.

**Veredicto frente a §10.3:** el backfill costó entre ~1,30 USD (Billing) y
≈1,58 USD (Σ a precio de lista) brutos, algo por encima del ≈0–1 USD que
prevé §10.3. El neto 0 no es mérito del diseño: lo absorbió un crédito de
prueba temporal, que no estará en una cuenta de producción.

**Cloud Storage:** sin cargo consolidado aún en el reporte de esos días
(Cloud Storage factura por día con retraso, más que Cloud Run). 38,4 GiB
vigentes en `landing`, estimado ~3.000 COP/mes en clase Standard antes de que
el lifecycle del bucket (ITSC-222) los pase a Nearline.

**Estado estacionario estimado**, con la máxima de `daily` (no se reportó
mediana, así que esta es ya una cota conservadora) y el valor único de
`monthly-close`:

GiB-s/mes ≈ 16 × (31 × 11,1 s + 81,2 s) = 16 × 425,3 s ≈ 6.805 GiB-s/mes
vCPU-s/mes ≈ 4 × 425,3 s ≈ 1.701 vCPU-s/mes

Ambos muy por debajo del cupo gratis mensual: confirma §10.3 ("`daily` /
`monthly-close` ≈ 0, free tier").

**Estado del mes en curso:** sin provisionales hasta que el humano encienda
los schedulers (ITSC-226) o lance `l1-daily` a mano con el rango de días
faltantes.
