# Runbook: sonda de dimensionamiento de L1 (§14.1) y caracterización de header (§14.2)

Lo ejecuta el humano; ningún agente dispara `terraform.yml` ni `run-job.yml`.
Mide cuánta memoria y tiempo necesita el mes más pesado de BTCUSDT y con eso
fija la configuración de Cloud Run del job `l1-job`. Referencia:
[TRD-L1 §14.1 y §14.2](../TRD/l1.md). La medición y la configuración final
las registró la card ITSC-213 (ver "Resultados"); aquí solo se explica
cómo obtenerlas.

## Antes de empezar

1. El runbook de habilitación de [infra/README.md](../../infra/README.md)
   ("Habilitar el despliegue desde GitHub Actions") está completo: stack
   `data` aplicado, environment `gcp` creado y las variables
   `GCP_DEPLOY_SERVICE_ACCOUNT`, `GCP_PROJECT_ID` y `GCP_REGION` cargadas.
2. El stack `l1` apunta a la imagen real de `l1_ingest` (ITSC-204), no a un
   placeholder, y está aplicado con la configuración vigente de **4 vCPU y
   16 GiB**. En *Actions → Terraform → Run workflow*: `stack` = `l1`,
   `action` = `apply`; apruébalo en el environment `gcp`.

## Mes más pesado

El mes más pesado es el ZIP mensual más grande de
`data/spot/monthly/aggTrades/BTCUSDT/` en `data.binance.vision`. El tamaño del
ZIP es el proxy del número de filas.

Cómo determinarlo, desde una máquina con acceso a `data.binance.vision`:

```bash
# Opción A: listado del bucket, con tamaños en bytes
curl -s "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/spot/monthly/aggTrades/BTCUSDT/" \
  | grep -oE '<Key>[^<]*aggTrades-[0-9-]+\.zip</Key><LastModified>[^<]*</LastModified><ETag>[^<]*</ETag><Size>[0-9]+' \
  | sed -E 's#<Key>.*aggTrades-([0-9-]+)\.zip</Key>.*<Size>([0-9]+)#\2 \1#' | sort -rn | head -3

# Opción B: Content-Length de un HEAD por mes
curl -sI https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-YYYY-MM.zip | grep -i content-length
```

Anota la fecha de consulta y los tres meses mayores:

| Puesto | Mes (YYYY-MM) | Bytes del ZIP |
| --- | --- | --- |
| 1 | 2023-03 | 2.658.089.884 |
| 2 | 2023-02 | 2.624.704.281 |
| 3 | 2022-11 | 2.460.042.801 |

Fecha de consulta: 2026-09-25 (HEAD a los 109 meses, de 2017-08 a 2026-08).
Mes elegido: **2023-03**. Si un mes posterior supera esos bytes, repite la
consulta.

Ejecuta *Actions → Run job → Run workflow* con estos inputs exactos:

| Input | Valor |
| --- | --- |
| `job` | `l1-job` |
| `mode` | `backfill` |
| `from` | `2023-03` |
| `to` | `2023-03` |

## Qué leer y qué anotar

En el log que vuelca el paso "Logs de la ejecución" del run:

- La línea `sonda: unit=... mode=backfill rss_peak_mib=<MiB> wall_s=<s>`: el
  RSS pico y el tiempo de pared.
- La línea final `fin unidad=... ruta=gs://.../consolidated.parquet
  content_hash=<sha256>`, con la ruta de `consolidated.parquet` y su hash.

En el resumen del run ("Resumen de la ejecución"): el estado de la ejecución
(Exitosa o Fallida) y las tareas fallidas.

Una ejecución Fallida no siempre es un OOM. El módulo `layer` no fija
`timeout` ni `max_retries`, así que valen los de Cloud Run Jobs (600 s por
tarea y 3 reintentos). Distingue la causa en el log volcado:

- **OOM**: aparece "Memory limit of ... exceeded". El proceso muere con
  SIGKILL, el `finally` del CLI no corre y **no hay línea `sonda:`**.
- **Timeout**: la tarea termina al alcanzar el timeout máximo, sin mensaje de
  memoria.
- **Reintentos**: cada intento suma sus propias líneas. Toma la línea `sonda:`
  del primer intento exitoso.

**Costo (§14.1)**, con la configuración vigente y `s` = `wall_s`:

- GiB-s = 16 × `s`; vCPU-s = 4 × `s`.
- Backfill completo: multiplica por 96 meses, como el TRD. La página lista
  109 meses publicados: si quieres el techo, anota también el valor ×109
  (13 % más).
- Contra el cupo gratis mensual: 360.000 GiB-s y 180.000 vCPU-s.

**Regla de decisión:**

- RSS pico ≤ 75 % de 16 GiB (12.288 MiB) y sin OOM: se fija **4 vCPU y 16 GiB**.
- OOM o RSS pico mayor: **8 vCPU y 32 GiB**. El cambio es de código: en
  `infra/stacks/batch/l1/main.tf`, bloque `module "layer"`, poner
  `cpu = "8"` y `memory = "32Gi"` (los valores por defecto están en
  `infra/modules/layer/variables.tf`). Entra por una card (`task-create`) y un
  PR mergeado a `main` antes de volver a aplicar `terraform.yml`; después se
  repite la corrida. ADR-L1-09: nunca disco ni Batch antes de agotar 32 GiB.
- Timeout (no OOM): no cambies la memoria. Registra una card para fijar
  `timeout` en el módulo `layer` (§14.1 exige "dentro del timeout") y repite.

## Caracterización de header (§14.2)

Tres ejecuciones `run-job.yml` con `job` = `l1-job`, `mode` = `daily` y
`from` = `to` = un día por época:

| Época | Día |
| --- | --- |
| 2017 | 2017-08-17 |
| 2020 | 2020-01-01 |
| 2025 | 2025-01-01 |

El chequeo `header_detected` se emite siempre, con o sin header, y el log solo
muestra `check_type=header_detected`: eso confirma que corrió, no responde la
pregunta. La respuesta está en `metric_value` de la fila del Parquet de
hallazgos: `1.0` = con header, `0.0` = sin header (`details.first_line` trae la
primera línea). No bloquea la sonda: el diseño detecta el header por archivo y
funciona con o sin él.

1. En el log de cada ejecución toma el `run_id` de la línea
   `inicio unidad=... modo=... run_id=<run_id>`.
2. Desde Cloud Shell, lee el Parquet bajo
   `gs://<bucket dq-findings>/l1/detected_date=<fecha>/`, cuyo nombre empieza
   por ese `run_id`:

   ```bash
   gcloud storage ls "gs://<bucket dq-findings>/l1/detected_date=*/<run_id>-*.parquet"
   gcloud storage cp "gs://<bucket dq-findings>/l1/detected_date=<fecha>/<run_id>-<uuid>.parquet" /tmp/h.parquet
   python3 -c "
   import pyarrow.compute as pc, pyarrow.parquet as pq
   t = pq.read_table('/tmp/h.parquet')
   print(t.filter(pc.equal(t['check_type'], 'header_detected')).select(['metric_value', 'details']).to_pylist())"
   ```

Verlo en el log exigiría registrar `metric_value` en `dq/emit.py`: es un
cambio de código, va por una card aparte.

Al terminar las tres corridas, borra los objetos `provisional-day` que dejaron
en el bucket landing. Están dentro de meses cerrados y `monthly-close` aún no
existe, así que el `consolidated.parquet` del backfill quedaría duplicado con
ellos para quien lea la partición entera:

```bash
B=gs://<bucket landing>/l1/provider=binance/market=spot/asset=BTCUSDT
gcloud storage rm \
  "$B/year=2017/month=08/provisional-day=17.parquet" \
  "$B/year=2020/month=01/provisional-day=01.parquet" \
  "$B/year=2025/month=01/provisional-day=01.parquet"
```

## Resultados

Los llenó la card ITSC-213.

**Sonda (mes 2023-03)**

| Fecha de la corrida | URL del run | Config | RSS pico (MiB) | Pared (s) | Estado | GiB-s ×96 | vCPU-s ×96 | Config final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-26 | [36248786009](https://github.com/byroncz/intrinsica/actions/runs/36248786009) | 4 vCPU, 16 GiB, imagen 0.1.1 | 5622 (34 % de 16 GiB; 46 % del umbral de 12.288 MiB) | 577,0 | Exitosa, sin OOM | 886.272 | 221.568 | 4 vCPU y 16 GiB |

- Ejecución Cloud Run `l1-job-dvwz4`, 1 tarea. Salida:
  `gs://intrinsica-dc-landing/l1/provider=binance/market=spot/asset=BTCUSDT/year=2023/month=03/consolidated.parquet`,
  `content_hash=35a8f396db7f0f57ff6cad58adde412a61bb82b30cc5240ceecba60569a69996`.
- Por corrida: 16 × 577 = 9.232 GiB-s y 4 × 577 = 2.308 vCPU-s.
- **Extrapolación contra el cupo gratis mensual** (360.000 GiB-s y 180.000
  vCPU-s): ×96 meses son 886.272 GiB-s (2,46 veces el cupo) y 221.568 vCPU-s
  (1,23 veces). ×109 meses son 1.006.288 GiB-s y 251.572 vCPU-s. El backfill
  completo **no cabe** en un solo mes de cupo: el excedente se factura o el
  backfill se reparte en varios meses calendario. Es un peor caso: 2023-03 es
  el mes más pesado y los demás tardan menos.
- Regla del runbook: RSS pico 5622 MiB ≤ 12.288 MiB y sin OOM, así que se fija
  **4 vCPU y 16 GiB** (explícitos en `infra/stacks/batch/l1/main.tf`).
- Intento previo con la imagen 0.1.0: OOM a 16 GiB (run 36220593271). Lo
  resolvió ITSC-215 (memoria acotada); la 0.1.1 es la medida.
- **Margen de timeout:** 577 s de pared contra los 600 s por tarea de Cloud
  Run Jobs dejan 23 s. Es riesgo de timeout, no de memoria: hace falta una
  card para fijar `timeout` en el módulo `layer` (ver "Regla de decisión").

**Header**

Sin corrida todavía; no bloquea la sonda.

| Época | Día | Header (sí/no) |
| --- | --- | --- |
| 2017 | 2017-08-17 | pendiente |
| 2020 | 2020-01-01 | pendiente |
| 2025 | 2025-01-01 | pendiente |
