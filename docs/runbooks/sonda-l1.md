# Runbook: sonda de dimensionamiento de L1 (§14.1) y caracterización de header (§14.2)

Lo ejecuta el humano; ningún agente dispara `terraform.yml` ni `run-job.yml`.
Mide cuánta memoria y tiempo necesita el mes más pesado de BTCUSDT y con eso
fija la configuración de Cloud Run de los jobs `l1-<modo>` (la sonda usa
`l1-backfill`). Referencia:
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

Sin un backfill completo previo, el mes más pesado es el ZIP mensual más grande
de `data/spot/monthly/aggTrades/BTCUSDT/` en `data.binance.vision`. El tamaño
del ZIP es el proxy del número de filas. Si ya existe un backfill completo, el
mes más pesado es el de mayor `rss_peak_mib` de sus líneas `sonda:` (hoy
2026-02), como explica el párrafo siguiente a la tabla.

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

El tamaño del ZIP no predice la memoria. En los dos backfills completos el
mayor RSS pico fue 2026-02 (9.194 y 9.390 MiB), no 2023-03 (5.712 MiB en el
último, con 596 s de pared frente a 253 s de 2026-02). Por eso, una vez hecho
un backfill completo, el "mes más pesado" para dimensionar es el de mayor
`rss_peak_mib` de sus líneas `sonda:`, no el del ZIP más grande (ver
"Resultados").

Ejecuta *Actions → Run job → Run workflow* con estos inputs exactos:

| Input | Valor |
| --- | --- |
| `job` | `l1-backfill` |
| `from` | `2026-02` (mes de mayor `rss_peak_mib`; sin backfill previo, el del ZIP más grande, `2023-03`) |
| `to` | (vacío) |

Para una sola unidad basta con `from`: `to` por defecto es igual a `from`.
Si igual completas `to` con el mismo valor de `from`, el workflow lo detecta
y no lo repite en `--args` (ITSC-232); `gcloud` rechaza un valor duplicado en
esa lista.

## Qué leer y qué anotar

En el log que vuelca el paso "Logs de la ejecución" del run:

- La línea `sonda: unit=... mode=backfill rss_peak_mib=<MiB> wall_s=<s>`: el
  RSS pico y el tiempo de pared.
- La línea final `fin unidad=... ruta=gs://.../consolidated.parquet
  content_hash=<sha256>`, con la ruta de `consolidated.parquet` y su hash.

En el resumen del run ("Resumen de la ejecución"): el estado de la ejecución
(Exitosa o Fallida) y las tareas fallidas.

Una ejecución Fallida no siempre es un OOM. Los jobs de l1 fijan `timeout`
(3600 s por tarea en backfill y monthly-close, 900 s en daily y seam-check) y
1 reintento; ver "Timeouts y reintentos de los jobs de l1" en
`infra/README.md`. Distingue la causa en el log volcado:

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
- Timeout (no OOM): no cambies la memoria. Sube el `timeout` del modo en
  `infra/stacks/batch/l1/main.tf` (§14.1 exige "dentro del timeout"), por una
  card y un PR, y repite.

**Bajar y volver a medir.** La regla solo valida que la configuración alcanza;
no busca la más barata. Por la decisión de [eficiencia de memoria ante
todo](https://app.notion.com/p/3e727957d23d811887eaf14c886b9a0c), tras validar
una configuración prueba la inmediata inferior (Cloud Run exige al menos 2 vCPU
para 8 GiB) y quédate con la más barata que cumpla las tres condiciones:

1. Sin OOM.
2. RSS pico ≤ 75 % de su memoria (para 8 GiB, 6.144 MiB).
3. Pared ≤ 1,5 × la de la configuración validada.

Mide con el mes de mayor `rss_peak_mib`, no con el del ZIP más grande. Un
backfill completo hecho con la configuración validada ya trae las líneas
`sonda:` de todos los meses y sirve solo para descartar la inferior: si su pico
supera el 75 % de la memoria inferior, esa configuración no cumple y no hay que
repetir la sonda. Si no lo supera, no basta para adoptarla: la condición 3
exige medir con menos vCPU, y el RSS también puede cambiar con menos hilos.
Despliega la configuración inferior y corre la sonda con el mes de mayor
`rss_peak_mib` antes de fijarla. Si no cumple, el stack se queda como está y el
resultado se anota en "Resultados" (ITSC-220 es el ejemplo).

## Caracterización de header (§14.2)

Tres ejecuciones `run-job.yml` con `job` = `l1-daily`, `from` = un día por
época y `to` vacío (una sola unidad no necesita `to`):

| Época | Día |
| --- | --- |
| 2017 | 2017-08-17 |
| 2020 | 2020-01-01 |
| 2025 | 2025-01-01 |

El chequeo `header_detected` se emite siempre, con o sin header, y el log solo
muestra `"check_type": "header_detected"`: eso confirma que corrió, no responde la
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
en el bucket landing. Están dentro de meses cerrados y el `consolidated.parquet`
del backfill quedaría duplicado con ellos para quien lea la partición entera.
Lo normal es correr `monthly-close` sobre esos meses: con el consolidado ya
presente no descarga nada, compara los provisionales contra él, emite
`daily_monthly_drift` y los borra. El borrado manual solo hace falta si
`monthly-close` no puede limpiar ese mes:

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
- **Fila del manifiesto:** el código no la registra en el log. Evidencia
  indirecta: `write_manifest` corre antes de la línea `fin`
  (`pipeline.py:135` frente a `:183`) y lanza excepción si falla, así que la
  línea `fin` implica la fila escrita. Para confirmarla, toma el `run_id` de
  la línea `inicio` del log (`gh run view 36248786009 --log | grep 'inicio unidad=binance/spot/BTCUSDT/2023-03'`)
  y lista el bucket:
  `gcloud storage ls gs://intrinsica-dc-manifest/l1/provider=binance/market=spot/asset=BTCUSDT/year=2023/month=03/<run_id>-*.parquet`.
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
- **Margen de timeout:** 577 s de pared contra los 600 s por defecto de
  Cloud Run Jobs dejaban 23 s. Era riesgo de timeout, no de memoria: ITSC-219
  fijó 3600 s por tarea en backfill (margen 6x).

**Bajar a 2 vCPU y 8 GiB (ITSC-220): descartado**

Con el mes de mayor RSS de los backfills completos. Se midió con 4 vCPU y
16 GiB (no se desplegó 2/8: el pico ya lo descarta, suponiendo que el RSS no
baje un 35 % con la mitad de vCPU; no se midió a 2/8).

| Backfill | Imagen | Meses | Mes de mayor RSS | RSS pico (MiB) | Pared (s) | Mediana de RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| [36283117940](https://github.com/byroncz/intrinsica/actions/runs/36283117940) (`l1-backfill-zzpft`) | anterior a 0.5.14 | 106 | 2026-02 | 9.194 | 537 | 2.568 |
| [37096045973](https://github.com/byroncz/intrinsica/actions/runs/37096045973) (`l1-backfill-m89q8`, ITSC-294) | 0.5.14 | 106 (105 exitosas; 2026-09 falla por 404 esperado) | 2026-02 | 9.390 | 253,1 | n/d |

Cinco mayores RSS de `l1-backfill-m89q8` (4 vCPU, 16 GiB):

| Mes | RSS pico (MiB) | Pared (s) |
| --- | --- | --- |
| 2026-02 | 9.390 | 253,1 |
| 2023-03 | 5.712 | 595,6 |
| 2023-02 | 5.614 | 510,1 |
| 2022-11 | 5.473 | 474,6 |
| 2023-01 | 5.304 | 476,9 |

- **2 vCPU y 8 GiB no cumple:** 9.390 MiB supera los 8.192 MiB (OOM) y el
  umbral de 6.144 MiB. Ambos backfills coinciden en el mes y el orden de
  magnitud (9.194 y 9.390 MiB), así que no es un valor suelto.
- **4 vCPU y 16 GiB sigue cumpliendo:** 9.390 MiB es 57 % de 16 GiB, bajo el
  umbral de 12.288 MiB. El stack se queda como está;
  `infra/stacks/batch/l1/main.tf` no cambia.
- **Intermedio 4 vCPU y 12 GiB no cumple la regla:** 9.390 MiB es 76,4 % de
  12 GiB, sobre el 75 %. 13 GiB sí la cumpliría (9.390 / 13.312 = 70,5 %), pero
  se descarta por costo: ahorra ≈ 3 GiB × 577 s × 0,0000025 USD ≈ 0,004 USD por
  mes, que no justifica el cambio.
- **El pico de RAM no sigue a la pared ni al tamaño del ZIP:** en
  `l1-backfill-m89q8`, 2026-02 es el mes de mayor memoria con 253 s de pared,
  frente a los 596 s de 2023-03 (mes del ZIP más grande). El pico real
  (9.390 MiB) es 1,67 veces el que midió la sonda de ITSC-213 en 2023-03
  (5.622 MiB). Cualquier cambio de memoria se valida contra 2026-02, no contra
  2023-03.
- **Costo por mes procesado** (us-east1, pared de 2023-03, 577 s): 4 vCPU y
  16 GiB ≈ 0,08 USD; 2 vCPU y 8 GiB ≈ 0,04 USD. El ahorro de ~0,04 USD por
  mes no justifica OOM en 2026-02.

**Header**

Sin corrida todavía; no bloquea la sonda.

| Época | Día | Header (sí/no) |
| --- | --- | --- |
| 2017 | 2017-08-17 | pendiente |
| 2020 | 2020-01-01 | pendiente |
| 2025 | 2025-01-01 | pendiente |
