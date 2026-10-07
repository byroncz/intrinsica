# Runbook: operación de viz 2.1 (backfill por rangos, volumen y costo, métricas de fidelidad, latest y encadenamiento)

Lo ejecuta el humano desde *Actions → Run job*, Cloud Shell y el navegador;
ningún agente dispara `terraform.yml` ni `run-job.yml`. Cierra la Épica de viz
(ITSC-303) con evidencia sobre la vista definitiva, `tiles_version` 2.1.0:
construye un rango del histórico de páginas, mide volumen, costo y las métricas
de [la decisión de fidelidad](../TRD/viz.md#614-adr-vz-14--la-página-de-un-día-lleva-sus-ticks-fidelidad-antes-que-eficiencia),
y prueba `latest.html` y el encadenamiento mensual. Referencia:
[TRD-viz §8, §10, §11 y §13](../TRD/viz.md). Mismo patrón que el
[runbook de L2](operacion-l2.md); aquí solo cambia lo que viz hace distinto.

Fechas calculadas al **2026-10-06**. Si lo ejecutas otro día, recalcula el
rango con la sección siguiente.

## Rango y alcance

- **Serie**: del `2017-08-17` (primer día de L1) al último mes que L2 cerró. A
  2026-10-06 son **110 meses** (2017-08 a 2026-09), ≈ 3 330 días.
- **Los días son independientes** (no hay carry-over que los encadene, TRD-viz
  §8.2): un rango se puede lanzar en cualquier orden y repetir. Lo que decide si
  un día se rehace es su `input_hash` (§7.8), no el rango.
- **Alcance para cerrar la card ITSC-311: un año medido**, el que elijas. Se
  recomienda **2025** (`from` = `2025-01`, `to` = `2025-12`): es el último año
  completo y, con más ticks por día que los años anteriores, sobreestima el
  histórico en vez de subestimarlo. Con esa medición se proyecta el resto (ver
  "Proyección del histórico") y **tú decides cuándo lanzar los demás rangos**;
  este runbook no exige el histórico completo.
- **Rangos de un año** para el histórico: 2017-08 a 2017-12 (5 meses), 2018 a
  2025 (12 meses cada uno) y 2026-01 a 2026-09 (9 meses): 10 lanzamientos.

## Prerrequisitos

1. **Stacks aplicados** (*Actions → Terraform*, `stack` y `action` = `apply`,
   aprobado en el environment `gcp`; `data` lo aplica el humano desde Cloud
   Shell): `data` (bucket `intrinsica-dc-viz` y el lago de DQ), `viz` (jobs
   `viz-tiles` y `viz-render`), `l1` (job `l1-monthly-close` y el workflow
   `l1-run-job` que encadena) y `l2` (job `l2-monthly`). Verifica el job:

   ```bash
   gcloud run jobs describe viz-tiles --region us-east1 \
     --format='yaml(spec.template.spec.template.spec)'
   ```

   Debe mostrar `cpu: '4'` y `memory: 4Gi` en `limits`, `timeoutSeconds: 36000`,
   `maxRetries: 1` y en `env` `VIZ_LANDING_ROOT`, `VIZ_EVENTS_ROOT`,
   `VIZ_TILES_ROOT` y `VIZ_DQ_ROOT`. Si la ruta del `--format` no coincide con la
   de tu versión de `gcloud`, mira el YAML completo.
2. **Imagen `viz_tiles` ≥ 1.1.1** (`layers/viz_tiles/VERSION`). El stack `viz`
   deriva el tag de ese archivo: sin `apply` posterior al merge, el job sigue con
   la imagen vieja. La imagen del job está en `image:` del mismo `describe`.
3. **Secret `VIZ_VIEWER`** en el repositorio, con el correo de la cuenta de Google
   que abre el bucket (`objectViewer` sobre todo `intrinsica-dc-viz`). Sin él no
   hay vista, aunque el job escriba bien.
4. **Landing de L1 y eventos de L2 completos** hasta el último mes del rango:
   `consolidated.parquet` en L1 y `events.parquet` y `carry_over.parquet` de los
   50 θ en L2. Un mes sin ellos deja `input_missing` y código 1 (TRD-viz §9.3).
5. Variables del environment `gcp` cargadas (`GCP_PROJECT_ID`, `GCP_REGION`,
   `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SERVICE_ACCOUNT`).

## Cómo abrir la vista

Con la cuenta de `VIZ_VIEWER` (Google pide iniciar sesión):

- **Último día**:
  `https://storage.cloud.google.com/intrinsica-dc-viz/tiles/latest.html`
- **Un día**:
  `https://storage.cloud.google.com/intrinsica-dc-viz/tiles/provider=binance/market=spot/asset=BTCUSDT/day=2026-09-30/index.html`

Es un solo documento (`Content-Encoding: gzip`) que no hace peticiones después de
cargar. Para compartir un día se descarga su `index.html`
([README de la capa](../../layers/viz_tiles/README.md#compartir-un-día)).

## Cómo funciona `run-job.yml` con viz

`viz-tiles` va en **una sola tarea** que recorre los meses del rango en orden. No
admite `series_start`, `script` ni `args` (el workflow los rechaza). Con `from`
vacío toma el mes anterior y revisa hacia atrás los meses con cola provisional;
**con `from` el workflow no hace esa revisión**. `force` exige `from`. No lances a
mano el mismo mes que Scheduler o el encadenamiento estén ejecutando.

Al final, el run vuelca los logs de la ejecución y un resumen (estado, tareas
completadas y fallidas, inicio y fin) en el *Summary* del run.

## Paso 1: backfill por rangos de un año

**Antes de lanzar un rango, revisa en *Facturación → Informes* el uso de Cloud Run
del mes (vCPU-s y GiB-s) y lanza solo si el rango cabe en lo que queda del cupo.**
El cupo gratis (180 000 vCPU-s y 360 000 GiB-s al mes) es uno solo para L1, L2 y
viz: comparar el año de viz contra el cupo entero engaña. Cuenta de octubre de
2026, con lo que L1 y L2 ya habían gastado al 2026-10-03 (90 044 vCPU-s y
261 358 GiB-s, [operacion-l2](operacion-l2.md#costo-real-y-volumen)):

| | vCPU-s | GiB-s |
| --- | --- | --- |
| L1 y L2 al 2026-10-03 | 90 044 (50 %) | 261 358 (73 %) |
| + viz, 2025 (medido) | 45 848 (25 %) | 46 808 (13 %) |
| **Usado** | **135 892 (75 %)** | **308 166 (86 %)** |
| **Queda** | **44 108 (25 %)** | **51 834 (14 %)** |

Otro año como 2025 en octubre (≈ 45 848 vCPU-s) **pasaría el cupo de vCPU-s** por
≈ 1 740 (≈ 0,03 USD de lista) y dejaría 5 026 GiB-s libres; un rango de seis meses
(≈ 23 000 vCPU-s) sí cabe. El 2026-10-03 es una foto: L1 y L2 siguen gastando, así
que lee el uso del día en que lanzas. Si el rango no cabe, parte el rango o espera
al mes siguiente (el cupo se renueva el día 1).

*Actions → Run job → Run workflow*, rama `main`:

| Input | Valor |
| --- | --- |
| `job` | `viz-tiles` |
| `from` | `2025-01` (inicio del rango, `YYYY-MM`) |
| `to` | `2025-12` (fin inclusivo, `YYYY-MM`; vacío = igual a `from`) |
| `force` | sin marcar |
| `series_start`, `script`, `args` | vacíos (se rechazan) |

El paso "Unidades" debe listar `1`. En el log de la ejecución, por cada día
procesado:

- **Una línea `sonda:`**:
  `sonda: unit=2025-03-14 ticks=… wall_s=… rss_mib=… ticks_bytes=… page_bytes=…`.
  `wall_s` es el tiempo del día; `rss_mib`, el pico de memoria del proceso;
  `ticks_bytes` y `page_bytes`, lo guardado de `ticks.bin` e `index.html`. Un día
  que ya estaba al día termina la línea con `skipped=true`.
- **Un hallazgo `tiles_summary`** (JSON, severidad `info`) con `skipped`,
  `objects` (4), `bytes`, `ticks_bytes`, `events_bytes`, `page_bytes`, `thetas`,
  `provisional_thetas` y `missing_thetas`.
- Ningún `input_missing` ni `price_unrepresentable` (error). Un `price_rounded`
  (warning) no detiene el día: anota cuántos hay (ítem 10 de TRD-viz §14).

Un año completo son 365 líneas `sonda:` (366 en bisiesto). **Timeout de la
tarea: 36 000 s** (10 h), fijado en el stack. Medido con 2025: **3 h 24 min**
(≈ 17 min por mes), un tercio del tope. Si la pared real se acerca al timeout, no
subas el tope por tu cuenta: anota y abre una card.

Anota: URL del run, nombre de la ejecución de Cloud Run, tareas completadas y
fallidas, inicio y fin, y los reintentos.

### Reanudar tras un fallo

Un día que falla deja su hallazgo, el rango sigue con el mes siguiente y el
proceso termina con código 1; con `max_retries = 1`, Cloud Run reintenta la tarea
una vez y el reintento salta lo que ya está al día. Si la ejecución termina
`Fallida`, lee la causa como en "Leer una ejecución fallida" y distingue:

- **OOM** ("Memory limit exceeded"): no hay línea `sonda:` del día. El pico medido
  con 2025 fue de **581 MiB** (2025-10-10, 4 501 514 ticks, el día mayor del año):
  14 % de los 4 GiB del job (4 096 ÷ 581 ≈ 7 veces de holgura). Un OOM con 4 GiB no
  es un día grande más, es un hallazgo: anota el día y abre una card. El 204 MiB de
  la sonda sintética (950 000 ticks) no sirve de referencia.
- **Timeout**: la tarea se corta a las 10 h, sin mensaje de memoria.
- **Entrada ausente** (`input_missing` con `what` = `l1`, `events`, `carry_over`
  o `ticks`): falta un archivo de L1 o de L2 de ese mes; corrígelo en su capa.

Corregida la causa, **relanza exactamente los mismos inputs**. Un día cuyo
`input_hash` y `tiles_version` coinciden con los de su `index.json` se **salta**
(`skipped=true`, no se lee ningún tick ni se escribe nada), y el resto se rehace.
Un día que murió entre `ticks.bin` y `index.json` no tiene índice y se rehace
entero. `latest.*` también se repara: si `latest.json` apunta al día pero
`latest.html` no pesa lo mismo que su `index.html`, el job lo rehace
(ITSC-320). Relanzar con todo al día termina rápido y no escribe, pero lee los
metadatos de las entradas de cada día: anota cuánto tarda.

## Paso 2: volumen y costo

Con `s` = pared de cada día: vCPU-s = 4 × Σ`s`; GiB-s = 4 × Σ`s`, sumando **todos**
los días procesados, reintentos incluidos (un intento fallido también factura).

1. **Σ de paredes y bytes desde las líneas `sonda:`.** Con el nombre de la
   ejecución del paso 1. Busca en `textPayload` y en `jsonPayload.message`
   (según cómo el job emita la línea; lee los dos):

   ```bash
   gcloud logging read 'resource.type="cloud_run_job"
     AND resource.labels.job_name="viz-tiles"
     AND labels."run.googleapis.com/execution_name"="<ejecución>"
     AND (textPayload:"sonda: unit=" OR jsonPayload.message:"sonda: unit=")' \
     --project intrinsica-dc --limit=5000 --format='value(textPayload,jsonPayload.message)' |
     awk '/skipped=true/ { al_dia++; next }
          { n++
            for (i = 1; i <= NF; i++) {
              split($i, a, "=")
              if (a[1] == "wall_s")      { s += a[2]; if (a[2] > max) max = a[2] }
              if (a[1] == "rss_mib")     { if (a[2] > rss) rss = a[2] }
              if (a[1] == "ticks_bytes") { tb += a[2] }
              if (a[1] == "page_bytes")  { pb += a[2] } } }
          END { printf "%d días construidos, %d al día, Σ wall_s = %.1f s, máx. día = %.1f s, " \
                       "RSS pico = %d MiB, Σ ticks.bin = %.2f GB, Σ index.html = %.2f GB\n",
                       n, al_dia, s, max, rss, tb / 1e9, pb / 1e9 }'
   ```

   Filtrar por `execution_name` deja fuera las corridas de prueba y el
   encadenamiento, que usan el mismo job.
2. **Pared de la tarea.** La suma de los días deja fuera el arranque y los
   listados, pero Cloud Run factura la tarea entera:

   ```bash
   gcloud run jobs executions describe <ejecución> --region us-east1 \
     --format='value(status.startTime,status.completionTime)'
   ```

   Si hubo reintento, el intento fallido también factura: suma su pared (*Cloud
   Run → Jobs → viz-tiles → ejecución → Tareas*).
3. **Contra el cupo gratis** mensual de 360 000 GiB-s y 180 000 vCPU-s,
   **compartido con L1 y L2**. A 2026-10-03 el reproceso de L1 (ITSC-294) y el
   backfill de L2 ya habían consumido 90 044 vCPU-s (50 %) y 261 358 GiB-s (73 %)
   del cupo de octubre ([operacion-l2](operacion-l2.md#costo-real-y-volumen)):
   lee en Billing lo que queda antes de lanzar más rangos este mes.
4. **Facturado según Billing:** *Facturación → Informes*, filtro por servicio
   `Cloud Run` y `Cloud Storage`, rango de las fechas de la ejecución, agrupado
   por SKU (en Cloud Run, `Jobs CPU` y `Jobs Memory` en us-east1; en GCS,
   almacenamiento y operaciones Clase A). Anota lo bruto, el crédito o cupo
   gratis y lo neto. La cuenta de facturación está en COP: convierte o anota la
   moneda. Billing tarda hasta 24 h en reflejar el consumo y **no separa por
   job**: anota qué más corrió en la misma ventana.
5. **Volumen** (con el backfill terminado y sin otro job escribiendo):

   ```bash
   B=gs://intrinsica-dc-viz/tiles/provider=binance/market=spot/asset=BTCUSDT
   gcloud storage du -s gs://intrinsica-dc-viz/tiles
   gcloud storage du -s "$B/day=2025-*" | awk '{ s += $1 } END { print s " B" }'
   gcloud storage ls "$B/day=2025-*/index.json" | wc -l
   ```

   El primero es el total de `tiles/` (incluye todo lo construido hasta hoy); el
   segundo, el del año medido; el tercero, los días con índice (365 en 2025). Con
   un comodín, `du -s` da **una línea por día** (no el total del año): el `awk` las
   suma, en bytes (sin `-h`). Con `du` solo (sin `-s`) saldría una línea por objeto.
   Para 2025, el segundo debe dar **4 561 622 351 B**, la cifra de "Resultados"; si
   difiere, otra corrida escribió en el año. Volumen por día = segundo ÷ tercero. Compáralo con
   [TRD-viz §7.3 y §10.3](../TRD/viz.md#73-ticksbin-los-ticks-del-día-sin-reducir):
   ≈ 9 MB por día (4,85 de `ticks.bin`, ≈ 0,8 de `events.bin` y ≈ 3,3 de
   página en gzip), que son **≈ 30 GB** para el histórico con los binarios
   (cota alta) y ≈ 0,6 USD al mes; la decisión original contaba solo páginas:
   **≈ 9 GB y ≈ 0,20 USD al mes**. Di cuál de las dos cifras confirma lo medido.
   Para el año medido, estima el histórico con el promedio por día × 3 330.

## Paso 3: las métricas de la decisión de fidelidad

Se miden **en el navegador**, sobre el último día disponible (hoy 2026-09-30,
abierto desde `tiles/latest.html`), con la cuenta de `VIZ_VIEWER`. Cada vez,
sesión limpia (ventana de incógnito o *Disable cache* en DevTools) y red de casa.

1. Abre DevTools (F12) → *Console* y escribe `viz:` en el filtro. Recarga con
   Ctrl+Shift+R. La vista escribe una línea por métrica:

   | Línea | Qué mide | Criterio |
   | --- | --- | --- |
   | `viz: primer trazo a … ms desde el inicio de la navegación` | **Apertura**: de pedir la página al primer dibujo | < 5 s |
   | `viz: ticks decodificados en … ms` | Decodificar `ticks.bin` y `events.bin` | informativa |
   | `viz: redibujo en … ms (… ticks en la vista)` | **Redibujo tras zoom o desplazamiento**: haz 10 zooms con la rueda y arrastres, y anota el mínimo y el máximo | < 100 ms |
   | `viz: cambio de θ en … ms` | **Cambio de θ**: cámbialo 10 veces entre θ pequeños y grandes | < 100 ms |

2. **Bytes transferidos**: pestaña *Network*, filtro *Doc*, recarga y lee la
   columna *Transferred* de `latest.html` (o `index.html`). Es la página en gzip,
   un solo documento y ninguna otra petición. **Criterio: ≤ 4 MB el 2026-09-30**
   (RNF-VZ-02). Confirma de paso, en *Headers*, `content-type: text/html;
   charset=utf-8` y `content-encoding: gzip` (ítem 12 de TRD-viz §14), o:

   ```bash
   gcloud storage objects describe gs://intrinsica-dc-viz/tiles/latest.html \
     --format='yaml(content_type,content_encoding,size)'
   ```

3. Anota los números en la tabla de "Resultados". **Si una métrica no cumple**,
   el runbook lo dice ahí y se abre una card con `task-create`; por encima de
   10 MB de página se reabre ADR-VZ-14.

Hay **un solo juego de cifras**, el de la tabla de "Resultados" (medido el
2026-10-06 sobre 2026-09-30, una vez; no se repitió el 2026-10-07). Cumple las
cuatro; la página queda a 0,34 MB del techo (la estimación era 3,3 a 3,4 MB).

## Paso 4: `latest` verificado

`tiles/latest.html` debe abrir y **pesar lo mismo** que el `index.html` del día al
que apunta `latest.json` (ITSC-318 y ITSC-320: la copia es byte a byte porque se
comprime al vuelo, nunca se relee un objeto con etiqueta gzip). Desde Cloud
Shell:

```bash
T=gs://intrinsica-dc-viz/tiles
gcloud storage cat $T/latest.json                                  # {"day": "AAAA-MM-DD", ...}
gcloud storage ls -l $T/latest.html \
  "$T/provider=binance/market=spot/asset=BTCUSDT/day=<día de latest.json>/index.html"
```

Las dos líneas deben dar **el mismo tamaño** (lo guardado, en gzip). Después, abre
`tiles/latest.html` en el navegador: debe mostrar el día de `latest.json`. El mismo
chequeo está en `viz_check_page.py` (`--tiles-root gs://intrinsica-dc-viz/tiles`
termina en `latest.html = index.html del <día>: <bytes> B` o en `FALLO
latest_distinto`). Medido el 2026-10-07: `tiles/latest.html` pesa **3 660 213 B**,
igual que el `index.html` del 2026-09-30 (la verificación de ITSC-320).

## Paso 5: encadenamiento mensual

El workflow `l1-run-job` ejecuta el job del modo sin overrides y encadena los
`next_jobs` del modo: con `monthly-close`, **`l1-monthly-close`, luego
`l2-monthly` y luego `viz-tiles`**, cada uno solo si el anterior terminó con éxito
(sin argumentos: cada job toma el mes anterior). Se lanza desde Cloud Shell:

```bash
gcloud workflows run l1-run-job --location=us-east1 --data '{"mode":"monthly-close"}'
gcloud run jobs executions list --region us-east1 --limit=6   # los tres nombres nuevos
```

El workflow espera a cada eslabón menos al último: `viz-tiles` queda ejecutándose
cuando el workflow ya terminó; sigue su ejecución en *Cloud Run → Jobs →
viz-tiles*. Anota el nombre de las tres ejecuciones (`l1-monthly-close-…`,
`l2-monthly-…`, `viz-tiles-…`) y la pared de cada una. En el repo, el
scheduler de `monthly-close` (día 8) está con `paused = true`
(`infra/stacks/batch/l1/main.tf`); verifica su estado real antes de contar con él.

**Esta prueba no se hizo en ITSC-311 y la registra ITSC-323.** El primer cierre real
que se registra es el de octubre de 2026 (Scheduler del 8 de noviembre, o
`monthly-close` lanzado por el humano). No se fuerza antes: relanzaría `l2-monthly`
sobre 2026-09, ya procesado (decisión del humano, comentario en ITSC-311,
2026-10-07). Cuando ocurra, el
humano reporta las cifras en la card y se anotan en "Encadenamiento" de
"Resultados": los tres nombres, el mes, la pared y el resultado de cada ejecución, y
que `latest.html` avanzó al último día de octubre (paso 4).

**La cadena solo prospera si L1 cerró el mes.** El 2026-09-30, `l2-monthly-xdtrf`
falló porque L1 aún no había dejado el `consolidated.parquet` del mes: L2 solo lee
consolidados (`input_missing`, `input_provisional_only`) y el consolidado de un
mes se publica el primer lunes del mes siguiente. Como L1 ya no trata un mes no
publicado como error (ITSC-295), es probable que `l1-monthly-close` termine con
éxito sin cerrar nada y que el fallo aparezca un eslabón después (es una lectura
de ese cambio; confírmala en el log de `l1-monthly-close`). Antes de lanzar la
cadena, comprueba que el mes ya está publicado y, si el segundo eslabón falla, no
relances a ciegas: lee la causa.

### Leer una ejecución fallida

```bash
# Quién la lanzó, con qué argumentos y qué condiciones dejó
gcloud run jobs executions describe <ejecución> --region us-east1 \
  --format='yaml(metadata.annotations,spec.template.spec.template.spec.containers[0].args,status.conditions,status.succeededCount,status.failedCount)'

# Qué dijo el job. Busca el mensaje en textPayload y en jsonPayload.message
gcloud logging read 'resource.type="cloud_run_job"
  AND labels."run.googleapis.com/execution_name"="<ejecución>"
  AND severity>=WARNING' \
  --project intrinsica-dc --limit=50 --format='value(timestamp,textPayload,jsonPayload.message)'
```

- **Creador**: en `metadata.annotations`, `run.googleapis.com/creator` dice si la
  lanzó el workflow (su service account), Scheduler o un humano.
- **Args**: sin ellos, el job tomó el mes por defecto. Un mes distinto del que
  esperabas explica un `input_missing`.
- **Condiciones**: `Completed` en `False` con un `message` de la tarea (OOM,
  timeout, "Container called exit(1)") dice el tipo de fallo.
- **`textPayload` y `jsonPayload.message`**: según cómo llegue la línea a Cloud
  Logging, el mensaje queda en uno u otro campo; buscar solo en `textPayload` puede
  dejar fuera justo la línea del fallo. Los hallazgos son líneas JSON con
  `finding_id`: búscalos en los dos campos.

Si el fallo es de un eslabón, el siguiente no corre: relanza **solo ese job** con
`run-job.yml` (mismos inputs) una vez corregida la causa, no la cadena entera.

## Scripts de operación

Versionados en [`layers/ops_tools/scripts/`](../../layers/ops_tools/scripts/). Hay
**dos pares con nombres parecidos y propósitos distintos**; no los confundas:

| Script | Qué hace | Dónde corre |
| --- | --- | --- |
| `viz_check_day.py` | Valida L2 contra L1: invariantes de los eventos del mes, eventos y velas de una ventana, regla de columna. No lee tiles | `ops-script` |
| `viz_probe_ticks.py` | Mide la codificación de un día de ticks. No lee tiles | `ops-script` |
| `viz_check_page.py` | Valida los archivos del día de viz (`index.json`, `ticks.bin`, `events.bin`) y `latest.html` | **local** |
| `viz_probe_page.py` | Mide `ticks.bin`, `events.bin` y la página guardada contra el presupuesto de 4 MB | **local** |

**Los dos primeros son los originales** que corrieron en `ops-script-2r2fv` y en
la sonda del 2026-10-06 (TRD-viz §14 ítem 17), versionados tal cual: no se
reformatean ni se editan (`layers/ops_tools/scripts/ruff.toml` los excluye del
formato y del lint de CI). Si hay que cambiarlos, se cambian a propósito en una card,
no por estilo. **Los dos últimos leen `tiles/` y
la cuenta de `ops-script` no tiene permiso sobre el bucket viz** (su IAM cubre
`l1/` y `l2/`: `landing`, `dc-events`, `dq-findings`, `manifest` y `ops`): como
`ops-script` fallarían con 403. Corren con `uv run` en local o desde Cloud Shell con
tu cuenta. Dar lectura a `viz/tiles/` a `ops-script` es un cambio del stack `ops`,
fuera de esta card. Los cuatro tienen pruebas en `layers/ops_tools/tests/`: los
locales (`test_viz_scripts.py`) contra un día que escribe viz de verdad, y los dos
originales (`test_ops_script_originals.py`) contra el lago de fixtures.

- **`viz_check_day.py`**: valida L2 contra L1 para un día y un θ. Imprime los
  invariantes de todos los eventos del mes (contrato §7.2 y ADR-L2-03), los eventos y
  las velas de 1 min de una ventana, y las columnas del nivel con su forma y estado;
  el resumen cuenta las que contradicen el color y la causa. Con `OPS_RESULTS_URI`
  deja además un CSV por columna en `results/`.

  ```text
  script = gs://intrinsica-dc-ops/scripts/viz_check_day.py
  args   = --theta 0.01 --day 2026-09-30 --window 12:30-13:00 --w 1024
  ```

  Un valor mayor que 0 en `barra ≥ θ contra el estado` es un defecto real; las
  violaciones de invariantes (`== Invariantes L2 sobre N eventos: K violaciones ==`)
  deben ser 0. Lee `l1/` y `l2/`: corre con `ops-script` (Run job, `job` = `ops-script`).
- **`viz_probe_ticks.py`**: cuánto pesa un día de ticks codificado en columnas planas
  (A) y en deltas varint (B), en crudo, gzip, base64 más gzip y zstd; termina en
  `== Resumen: N ticks | B en gzip … MB = … B/tick | histórico ~3 330 días ≈ … GB ==`.

  ```text
  script = gs://intrinsica-dc-ops/scripts/viz_probe_ticks.py
  args   = --day 2026-09-30
  ```

  También corre con `ops-script`. No lee tiles: mide la codificación, no lo que viz
  escribió (eso es `viz_probe_page.py`).

- **`viz_check_page.py`**: los invariantes de un día (`ticks.bin` decodifica y suma
  lo del índice, `events.bin` pesa 25 B por evento, cada punto apunta a un tick
  con su mismo tiempo, la cadena de eventos cierra, `latest.html` pesa lo que la
  página del día).

  ```bash
  uv run layers/ops_tools/scripts/viz_check_page.py --dir dia/    # un día descargado
  uv run layers/ops_tools/scripts/viz_check_page.py --tiles-root gs://intrinsica-dc-viz/tiles --day 2026-09-30
  ```

  Termina en `día AAAA-MM-DD: ticks=…; θ=…; eventos=…; fallos=0` y código 0.
- **`viz_probe_page.py`**: bytes por tick, por sección (crudos y en gzip),
  `ticks.bin` y `events.bin` en gzip y en base64 más gzip, y, con `--tiles-root`, la
  página guardada contra el presupuesto de 4 MB.

  ```bash
  uv run layers/ops_tools/scripts/viz_probe_page.py --tiles-root gs://intrinsica-dc-viz/tiles --day 2026-09-30 --page-budget-mb 4
  ```

  Termina en `sonda ticks: día=… presupuesto=ok` o `excedido` (código 1). Con
  `--dir`, sin `index.html` descargado, dice `sin_página`.

Con `--tiles-root gs://` hace falta `pyarrow` y credenciales (`gcloud auth
application-default login` en local; en Cloud Shell ya están). Para un día
descargado (sin comillas: el shell expande las llaves):

```bash
mkdir dia && gcloud storage cp \
  gs://intrinsica-dc-viz/tiles/provider=binance/market=spot/asset=BTCUSDT/day=2026-09-30/{index.json,ticks.bin,events.bin} dia/
```

**Copia a GCS de los originales.** Los de `ops-script` viven en
`gs://intrinsica-dc-ops/scripts/` (el bucket tiene versionado, ver
[operacion-ops](operacion-ops.md#trazabilidad-sin-git)); el repo es la fuente y el
bucket, la copia que el job lee. Tras cambiarlos en una card:

```bash
gcloud storage cp layers/ops_tools/scripts/viz_check_day.py gs://intrinsica-dc-ops/scripts/viz_check_day.py
gcloud storage cp layers/ops_tools/scripts/viz_probe_ticks.py gs://intrinsica-dc-ops/scripts/viz_probe_ticks.py
```

Los dos locales **no se copian** al bucket de `ops`: ahí no los puede leer nadie que
los necesite y, como `ops-script` no accede a viz, un 403 no diría nada útil.

## Proyección del histórico

Antes del año medido (2026-10-06) y con él (2026-10-07):

| Dato | Antes (un día) | Medido (2025, 365 días) | Fuente |
| --- | --- | --- | --- |
| Pared por día | 29 s (2026-09-30, 948 740 ticks; era 5 s en la 2.0) | **31,2 s** (Σ `wall_s` 11 400 s ÷ 365) | humano |
| Pared de la tarea | | 3 h 24 min, **≈ 17 min por mes** | humano |
| Histórico: 110 meses | ≈ 25 h de job (110 × 14 min) | **≈ 31 h de job** (110 × 17 min) | cálculo |
| vCPU-s y GiB-s del histórico a 4 vCPU / 4 GiB | ≈ 360 000 de cada uno | **≈ 450 000 de cada uno** (4 × 31 h × 3 600 s; el humano lo redondeó a ≈ 500 000) | cálculo |
| Volumen del histórico (3 330 días) | ≈ 30 GB (cota de TRD-viz §10.3) | **≈ 42 GB** (12,5 MB por día) | cálculo |
| Almacenamiento del histórico al mes | ≈ 0,6 USD | **≈ 0,9 USD** | cálculo |
| Cupo mensual gratis (compartido con L1 y L2) | 180 000 vCPU-s y 360 000 GiB-s | igual | Cloud Run |

**La cota de antes resultó baja, no alta.** Se creía que septiembre de 2026 era el
mes más pesado y que los años anteriores tenían menos ticks por día. 2025 cuesta
31 s por día, más que los 29 s del día de referencia, porque tiene días mucho más
grandes que él (el 2025-10-10 trae 4 501 514 ticks, 4,7× los de 2026-09-30).
Todo el histórico a ese ritmo es una estimación razonable, no una cota; los años
anteriores a 2025 pueden bajarla, y solo medirlos lo dice. Lo que la medición
confirma:

1. **No cabe en un solo mes calendario del cupo**: ≈ 450 000 vCPU-s son 2,5× el
   cupo de vCPU-s, y L1 y L2 gastan parte del mismo cupo. **Repartir el backfill en
   tres meses calendario** (≈ 150 000 vCPU-s al mes) lo deja dentro del cupo
   gratis, siempre que L1 y L2 no consuman más de lo que sobra: lee Billing antes
   de cada rango. Todo en un mes limpio costaría, de lista, ≈ 4,9 USD de vCPU-s y
   ≈ 0,2 USD de GiB-s por lo que pase del cupo (0,000018 y 0,000002 USD por
   vCPU-s y GiB-s); repartido, 0 (RVZ-04).
2. **El costo lo domina la relectura de tramos**: con la 2.1.0, el 2026-09-30 pasó
   de 5 s a 29 s porque `first_at_price` relee de `ticks.bin` los tramos que
   contienen cada confirmación (TRD-viz §7.5). Es casi **seis veces** el costo por
   día (29 ÷ 5 ≈ 5,8).

**Pendiente (ITSC-324): evaluar la eficiencia antes del backfill completo.** Tres
palancas que la medición debe decidir, sin asumir
ninguna: evitar la relectura de tramos de `ticks.bin` en `first_at_price`
(guardar la posición del primer tick por precio al escribirlo, o resolverla con
los tramos que el job ya tiene en memoria); medir si 4 vCPU hacen falta (el
proceso es de un solo hilo salvo Arrow, y facturar 1 vCPU dividiría los vCPU-s
por 4); y dejar de escribir los binarios sueltos si `render` leyera los datos de
la propia página (ítem 11 de TRD-viz §14). El humano decide cuándo lanzar los demás
rangos (ITSC-325, que depende de ITSC-324).

## Resultados

Ejecutados por el humano con el arquitecto el 2026-10-07; los números son los que
reportaron en la card. Stack `viz` con 4 vCPU / 4 GiB e imagen `viz_tiles:1.1.1`.

**Runs**

| Paso | Run de Actions | Ejecución de Cloud Run | Rango | Días construidos / al día | Pared de la tarea | Reintentos |
| --- | --- | --- | --- | --- | --- | --- |
| Backfill del año medido | 37570918667 | `viz-tiles-tc5gb` | 2025-01 a 2025-12 | 365 / 0 | 3 h 24 min (04:19 a 07:43 UTC) | 0 |
| Relanzar con los mismos inputs | 37634503346 | `viz-tiles-j85kx` | igual | 0 / 365 | 10 min | 0 |
| Encadenamiento (`l1-run-job`) | _pendiente: primer cierre real, ITSC-323_ | `l1-monthly-close-…`, `l2-monthly-…`, `viz-tiles-…` | 2026-10 | _pendiente_ | _pendiente_ | _pendiente_ |

El backfill no dejó **ningún hallazgo** fuera de `tiles_summary`. El relanzamiento
confirma la idempotencia: saltó los 365 días, no escribió y, aun así, tardó 10 min
por leer los metadatos de las entradas de cada día (RSS 154 MiB).

**Sonda del año medido** (paso 2): Σ `wall_s` **11 400 s** (3,17 h); `rss_mib` pico
**581 MiB** (de 4 GiB); páginas (`page_bytes`): media **5,0 MB**, mínima 1,20 MB,
máxima **68,76 MB** (2025-10-10, 4 501 514 ticks); **179 días de 365 pesan más de
4 MB y 25 más de 10 MB**. Los cinco mayores:

| Día | Página en gzip |
| --- | --- |
| 2025-10-10 | 68,76 MB |
| 2025-01-20 | 26,91 MB |
| 2025-02-03 | 24,41 MB |
| 2025-02-28 | 19,69 MB |
| 2025-04-07 | 18,67 MB |

El presupuesto de 4 MB (RNF-VZ-02) está medido y se cumple **solo para el
2026-09-30** (3,66 MB). Más de 10 MB es el umbral que reabre ADR-VZ-14; 25 días de
2025 lo superan. **La decisión sobre esos días no es de esta card**: es de ITSC-322
(ver "Cards abiertas").

**Costo**

| Concepto | Valor |
| --- | --- |
| vCPU-s del rango (4 × Σ `wall_s`) | 45 600 calculados; **45 847,77** en Billing (coincide) |
| GiB-s del rango | **46 807,77** en Billing (con 4 GiB, el mismo orden) |
| Contra el cupo mensual (180 000 vCPU-s y 360 000 GiB-s, compartido con L1 y L2) | 25 % de los vCPU-s y 13 % de los GiB-s |
| Billing, Cloud Run Jobs CPU us-east1 | bruto 2 756, cupo −2 756, **neto 0** |
| Billing, Cloud Run Jobs Memory us-east1 | bruto 313, cupo −313, **neto 0** |
| Billing, Cloud Storage Regional Standard, Clase A (1 303 operaciones) | bruto 22, neto 0 |
| Billing, Cloud Storage Regional Standard, Clase B (8 404 operaciones) | bruto 11, neto 0 |
| Billing, Artifact Registry (0,05 GiB-mes) | 0 |
| Billing, almacenamiento GCS | aún no facturado (GB-mes se cobra al cierre del mes) |
| **Total neto** | **0** |
| Costo de lista del año (0,000018 USD por vCPU-s + 0,000002 USD por GiB-s) | ≈ 0,91 USD (45 600 vCPU-s y GiB-s) |
| Costo proyectado del histórico | CPU y memoria: 0 si se reparte en tres meses (≈ 9 USD de lista); almacenamiento ≈ 0,9 USD al mes |

Billing del 2026-10-06 y 07 (*Facturación → Informes*, CSV, moneda de la cuenta, el
cupo gratis aparece en "Other savings"). Cuenta los dos días de las dos
ejecuciones (backfill y relanzamiento) y no separa por job.

**Volumen** (`gcloud storage du`):

| Medida | Valor | Contra lo estimado |
| --- | --- | --- |
| `tiles/` total | 4 840 908 461 B (≈ 4,84 GB) | |
| Año 2025 | 4 561 622 351 B | |
| Por día (÷ 365) | **12,5 MB** | ≈ 9 MB por día estimados (TRD-viz §10.3): **+39 %** |
| Histórico (3 330 días) | **≈ 42 GB** | ≈ 9 GB (decisión, solo páginas) y ≈ 30 GB (cota con binarios) |
| Almacenamiento del histórico al mes | **≈ 0,9 USD** | ≈ 0,20 USD (decisión) y ≈ 0,6 USD (cota) |

Lo medido **no confirma ninguna de las dos cifras**: supera la de solo páginas
(≈ 9 GB) y también la cota con binarios (≈ 30 GB). La causa es la media de página
de 5,0 MB (frente a los 3,3 MB del 2026-09-30) por los días grandes. El costo sigue
siendo bajo en términos absolutos, pero el presupuesto de TRD-viz §10.3 hay que
corregirlo.

**Métricas de la decisión de fidelidad** (Chrome, consola con `viz:`, día
2026-09-30, medidas el 2026-10-06; **un solo juego**, no se repitieron el
2026-10-07):

| Métrica | Criterio | Medido | ¿Cumple? |
| --- | --- | --- | --- |
| Apertura (primer trazo desde la navegación) | < 5 s | 1 671,9 ms | sí |
| Ticks decodificados (informativa) | | 41,3 ms (948 740 ticks) | |
| Redibujo tras zoom | < 100 ms | 0,4 a 11,2 ms | sí |
| Cambio de θ | < 100 ms | 0,4 a 1,6 ms | sí |
| Página en gzip, 2026-09-30 | ≤ 4 MB | 3,66 MB | sí |

Las cuatro cumplen, así que esta card no abre una card por incumplimiento. Pero
ninguna se midió en otro día: **el 2025-10-10 (68,76 MB, 4 501 514 ticks)
no se abrió en el navegador**, y a ese tamaño la apertura y la decodificación no
tienen evidencia. Eso queda a ITSC-322.

**`latest`**: `tiles/latest.html` pesa **3 660 213 B**, igual que el `index.html` del
2026-09-30 al que apunta `latest.json` (verificación de ITSC-320).

**Encadenamiento**: **no se probó en ITSC-311; lo registra ITSC-323.** Forzar
`monthly-close` hoy relanzaría `l2-monthly` sobre 2026-09, que ya está procesado. El
humano decidió no forzarlo (comentario en ITSC-311, 2026-10-07). El único
antecedente de la cadena es el fallo de `l2-monthly-xdtrf` del 2026-09-30 (creador
`l1-workflow`, código 1 en 60 s), que tuvo otra causa: L1 aún no había publicado el
mes (paso 5, "La cadena solo prospera si L1 cerró el mes"). Queda
pendiente del primer cierre real, el de octubre (a partir del 8 de noviembre de
2026): con L1 ya publicado, se lanza el comando del paso 5 y se anota aquí, en una
tabla, por ejecución: nombre (`l1-monthly-close-…`, `l2-monthly-…`, `viz-tiles-…`),
mes procesado, pared, resultado, y si `latest.html` avanzó al último día de octubre.
Si una ejecución falla, se lee con la guía de arriba (`describe` más `logging read`
con `jsonPayload.message`) y se abre la card que corresponda.

_Pendiente de ITSC-323 al 2026-10-07: el cierre de octubre aún no ocurre._

## Cards abiertas

Lo que esta medición deja fuera de la card, con el motivo:

- **Días de más de 10 MB** (ITSC-322; 25 de 2025, umbral de reapertura de ADR-VZ-14):
  decisión de fondo sobre la fidelidad frente al tamaño de página. Incluye abrir en el
  navegador el 2025-10-10.
- **Eficiencia de `viz-tiles`** (ITSC-324; TRD-viz §14 ítems 3, 11 y 19):
  `first_at_price` (31 s por día, casi seis veces el costo de la 2.0), si bastan menos
  vCPU y los binarios duplicados. Obliga a repartir el backfill en tres meses.
- **Backfill del histórico restante** (ITSC-325; ítems 2, 10 y 16): depende de ITSC-324.
  Cada rango se lanza con la revisión del cupo del paso 1.
- **Acceso de `ops-script` a `tiles/`** (ITSC-321; ítem 20).
- **Primer encadenamiento real y mediciones del humano** (ITSC-323; ítems 12, 13, 15 y
  16): las tres ejecuciones del cierre de octubre (paso 5), el `content-type` y
  `content-encoding` guardados (paso 3, ítem 12), `events_gzip` de `viz_probe_page.py`
  (ítem 13) y la validación de la ventana de tres eventos (ítem 15).

La entrada consolidada de la Épica la escribe `task-close.sh` al cerrarla.
