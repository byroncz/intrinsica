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
   ITSC-227 (`layers/l1_ingest/VERSION` ≥ 0.5.0) y los cuatro jobs de
   ITSC-224: `l1-backfill`, `l1-daily`, `l1-monthly-close`, `l1-seam-check`.
   Verifica que existen: `gcloud run jobs list --region <región>`.
3. **Timeout por tarea revisado.** ITSC-219 sigue en Por refinar y el módulo
   `layer` no fija `timeout`: vale el de Cloud Run Jobs, **600 s por tarea**
   con 3 reintentos. 2023-03, el mes más pesado, tardó **577 s** (23 s de
   margen). Una tarea que pasa de 600 s falla por timeout y se reintenta
   igual de lenta. Antes de lanzar, elige una de estas dos:
   - Fijar `timeout` en el módulo `layer` (card ITSC-219, PR y `apply`).
     Es lo recomendado.
   - Aceptar el riesgo y reintentar a mano los meses que fallen por timeout
     (ver "Reintentar solo lo fallido"). Anota tu decisión en "Resultados".
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
rango. Regla: no lances a mano el mismo mes y modo que Scheduler esté
ejecutando.

Al final, el run vuelca los logs de la ejecución y un resumen (estado,
tareas completadas y fallidas, inicio y fin) en el *Summary* del run.

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
OOM ("Memory limit exceeded"), timeout (tarea cortada a los 600 s) o error
de datos. Luego relanza `l1-backfill` **con el mismo rango completo** o solo
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

Los llena la card ITSC-228 al desbloquearse; el humano entrega las URLs.

**Runs**

| Paso | URL del run | Rango | Tareas OK / fallidas | Reintentos (meses) |
| --- | --- | --- | --- | --- |
| Backfill | pendiente | 2017-08 a 2026-07 | pendiente | pendiente |
| Seam-check | pendiente | 2017-08 a 2026-07 | pendiente | - |
| daily | pendiente | 2026-08-01 a 2026-08-31 | pendiente | - |
| monthly-close | pendiente | 2026-08 | pendiente | - |

**Timeout por tarea:** pendiente (¿ITSC-219 aplicado o riesgo aceptado? ¿Algún
mes por encima de 600 s?).

**seam_discontinuity**

| Borde | Explicación | ¿Hueco del proveedor? |
| --- | --- | --- |
| pendiente | pendiente | pendiente |

**Ciclo daily a monthly-close (2026-08)**

| Verificación | Resultado |
| --- | --- |
| 31 provisionales tras daily | pendiente |
| `consolidated.parquet` presente | pendiente |
| Sin provisionales tras el cierre | pendiente |
| `daily_monthly_drift` | pendiente |
| Hallazgos canonical | pendiente |

**Costo**

| Concepto | Medido | §10.3 |
| --- | --- | --- |
| Backfill, GiB-s | pendiente | - |
| Backfill, vCPU-s | pendiente | - |
| Backfill, facturado (Billing) | pendiente | ≈ 0–1 USD |
| Estado estacionario, GiB-s/mes | pendiente | ≈ 0 (free tier) |

**Estado del mes en curso:** sin provisionales hasta que el humano encienda
los schedulers (ITSC-226) o lance `l1-daily` a mano con el rango de días
faltantes.
