# Runbook: sonda de dimensionamiento de L1 (§14.1) y caracterización de header (§14.2)

Lo ejecuta el humano; ningún agente dispara `terraform.yml` ni `run-job.yml`.
Mide cuánta memoria y tiempo necesita el mes más pesado de BTCUSDT y con eso
fija la configuración de Cloud Run del job `l1-job`. Referencia:
[TRD-L1 §14.1 y §14.2](../TRD/l1.md). La medición y la configuración final
las registra la card hija 8; aquí solo se explica cómo obtenerlas.

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
| 1 | _pendiente_ | _pendiente_ |
| 2 | _pendiente_ | _pendiente_ |
| 3 | _pendiente_ | _pendiente_ |

Fecha de consulta: _pendiente_. Mes elegido: **el puesto 1** (`<MES>` abajo).

> Nota de la card ITSC-212: `data.binance.vision` respondió `403 Filtered` del
> proxy del contenedor, y no hubo otra fuente pública verificable, así que el
> mes no se pudo determinar al escribir este runbook. Lo completa el humano
> con el comando anterior.

Ejecuta *Actions → Run job → Run workflow* con estos inputs exactos:

| Input | Valor |
| --- | --- |
| `job` | `l1-job` |
| `mode` | `backfill` |
| `from` | `<MES>` |
| `to` | `<MES>` |

## Qué leer y qué anotar

En el log que vuelca el paso "Logs de la ejecución" del run:

- La línea `sonda: unit=... mode=backfill rss_peak_mib=<MiB> wall_s=<s>`: el
  RSS pico y el tiempo de pared.
- La línea final con la ruta de `consolidated.parquet` y su hash.

En el resumen del run ("Resumen de la ejecución"): el estado de la ejecución
(Exitosa o Fallida) y las tareas fallidas. Un OOM aparece como Fallida.

**Costo (§14.1)**, con la configuración vigente y `s` = `wall_s`:

- GiB-s = 16 × `s`; vCPU-s = 4 × `s`.
- Backfill completo: multiplica por 96 meses.
- Contra el cupo gratis mensual: 360.000 GiB-s y 180.000 vCPU-s.

**Regla de decisión:**

- RSS pico ≤ 75 % de 16 GiB (12.288 MiB) y sin OOM: se fija **4 vCPU y 16 GiB**.
- Si no: **8 vCPU y 32 GiB**, se cambia en el stack `l1`, se aplica y se repite
  la corrida. ADR-L1-09: nunca disco ni Batch antes de agotar 32 GiB.

## Caracterización de header (§14.2)

Tres ejecuciones `run-job.yml` con `job` = `l1-job`, `mode` = `daily` y
`from` = `to` = un día por época:

| Época | Día |
| --- | --- |
| 2017 | 2017-08-17 |
| 2020 | 2020-01-01 |
| 2025 | 2025-01-01 |

En cada log busca el hallazgo `header_detected`: si aparece, ese archivo trae
header; si no, no. No bloquea la sonda: el diseño detecta el header por
archivo y funciona con o sin él.

## Resultados

Los llena la card hija 8.

**Sonda (mes `<MES>`)**

| Fecha de la corrida | URL del run | Config | RSS pico (MiB) | Pared (s) | Estado | GiB-s ×96 | vCPU-s ×96 | Config final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |

**Header**

| Época | Día | Header (sí/no) |
| --- | --- | --- |
| 2017 | 2017-08-17 |  |
| 2020 | 2020-01-01 |  |
| 2025 | 2025-01-01 |  |
