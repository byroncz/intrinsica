# Runbook: sonda de dimensionamiento de L2 (ADR-04)

Lo ejecuta el humano; ningún agente dispara `terraform.yml` ni `run-job.yml`.
Mide cuánta memoria y tiempo necesita el mes más pesado de BTCUSDT en L2 con
2, 4 y 8 vCPU en Cloud Run Jobs, y con eso alimenta la decisión ADR-04
(Cloud Run Jobs o Cloud Batch con Spot) y la configuración final del stack
`l2`. Mismo patrón que la [sonda de L1](sonda-l1.md) (TRD-L1 §14.1); aquí solo
cambia lo que L2 hace distinto. Referencia: [TRD-L2 §10.2 y §14](../TRD/l2.md).
ITSC-281 midió con la imagen 0.5.0 y ITSC-282 fijó 2 vCPU con esos números;
ITSC-286 remidió con la 0.5.2 y fijó 4 vCPU (ver "Redimensionamiento de vCPU con
la imagen 0.5.2"). Los números de ITSC-281 van en "Resultados".

## Regla de decisión (escrita antes de correr)

Se aplica sobre la corrida de **8 vCPU**, el techo de Cloud Run Jobs
(32 GiB / 8 vCPU / 168 h por tarea, ADR-L1-09). Si esa corrida no cumple, las
de 2 y 4 vCPU tampoco deciden nada.

- **Cloud Run Jobs** si se cumplen las dos:
  1. El mes más pesado cabe holgado en 8 vCPU / 32 GiB: sin OOM y
     RSS pico ≤ 75 % de 32 GiB (24.576 MiB). Es el mismo 75 % de la sonda de L1.
  2. El backfill completo cabe en una tarea de menos de 24 h: 109 × pared del
     mes más pesado < 86.400 s. L2 no usa task array (los meses se encadenan
     por carry-over), así que la pared del backfill es la suma de los meses.
     Multiplicar por 109 el mes más pesado es un techo: los demás meses tardan
     menos.
- **Cloud Batch con Spot** solo si falla alguna de las dos.

Esta regla decide *dónde* corre L2. Cuántos vCPU se fijan (2, 4 u 8) lo decide
la regla de "Redimensionamiento de vCPU con la imagen 0.5.2" (ITSC-286).

## Antes de empezar

1. El runbook de habilitación de [infra/README.md](../../infra/README.md)
   ("Habilitar el despliegue desde GitHub Actions") está completo: stack
   `data` aplicado (con `bucketIamAdmin` de `deploy-github` sobre `dc-events`),
   environment `gcp` creado y las variables `GCP_DEPLOY_SERVICE_ACCOUNT`,
   `GCP_PROJECT_ID`, `GCP_REGION` y `GCP_WIF_PROVIDER` cargadas.
2. La imagen `l2_dc_events:<versión>` está publicada en Artifact Registry
   (la versión de `layers/l2_dc_events/VERSION`; la publica el run de CI en
   `main`) y el stack `l2` está aplicado con ella: *Actions → Terraform →
   Run workflow*, `stack` = `l2`, `action` = `apply`, aprobado en el
   environment `gcp`. Debe existir el job `l2-backfill`.
3. ITSC-280 está en `main`: `run-job.yml` ofrece `l2-backfill` en `job` y el
   input `series_start`. Corre el workflow de `main` (o de una rama que lo
   incluya).
4. La landing tiene el `consolidated.parquet` de 2023-03 (lo dejó la sonda de
   L1, ITSC-213). Desde Cloud Shell:

   ```bash
   gcloud storage ls "gs://<bucket landing>/l1/provider=binance/market=spot/asset=BTCUSDT/year=2023/month=03/consolidated.parquet"
   ```

   Si no está, corre `l1-backfill` con `from` = `2023-03` antes de seguir.

## Mes más pesado

El mes más pesado es el ZIP mensual más grande de
`data/spot/monthly/aggTrades/BTCUSDT/`, el mismo criterio que L1: el tamaño del
ZIP es el proxy del número de filas, y el número de ticks es lo que mueve el
tiempo de L2. Vuelve a verificarlo con las opciones A o B de la
[sonda de L1](sonda-l1.md#mes-más-pesado) desde una máquina con acceso a
`data.binance.vision`. La última consulta (2026-09-25, 109 meses, de 2017-08 a
2026-08) dio:

| Puesto | Mes (YYYY-MM) | Bytes del ZIP |
| --- | --- | --- |
| 1 | 2023-03 | 2.658.089.884 |
| 2 | 2023-02 | 2.624.704.281 |
| 3 | 2022-11 | 2.460.042.801 |

Mes elegido: **2023-03**. Si al repetir la consulta otro mes lo supera, usa
ese mes en `from`, `to` y `series_start` de todas las corridas y en las
rutas de este runbook. La diferencia entre el puesto 1 y el 2 es de 1,3 %: si
queda así, 2023-03 sigue siendo el mes a medir.

Última verificación: 2026-09-30 (sin meses nuevos).

## Inputs de Run job

*Actions → Run job → Run workflow*, con estos inputs exactos y los mismos en
las tres corridas:

| Input | Valor |
| --- | --- |
| `job` | `l2-backfill` |
| `from` | `2023-03` |
| `to` | `2023-03` |
| `force` | marcado (`true`) |
| `series_start` | `2023-03` |

- `from` = `to` = `2023-03`: un solo mes. En `l2-backfill` el workflow pasa
  `--to` siempre que `to` venga, aunque sea igual a `from`. Con `to` vacío, en
  cambio, el backfill recorre hasta el último mes cerrado de L1 (desde la
  imagen 0.6.0, ITSC-285): la sonda no es eso, así que no lo dejes vacío. El
  2026-10-02 una corrida con `to` omitido procesó 42 meses en vez de uno
  (ITSC-292). L2 corre siempre en una sola tarea (`--tasks 1`), sin task
  array. Una corrida bien armada de un mes deja una sola línea `sonda:` y
  termina en 3 a 5 minutos.
- `series_start` = `2023-03` declara ese mes como el primero de la serie, y
  así arranca **en frío**, sin carry-over del mes anterior. Sin este input, la
  CLI toma `L2_SERIES_START` = `2017-08` del stack y el mes falla por falta del
  carry-over de 2023-02.
- `force` marcado: la primera corrida no lo necesita, pero la segunda y la
  tercera sí. La primera deja el `carry_over.parquet` de 2023-03 y la regla de
  reanudación (RF-L2-09) salta los meses que ya lo tienen: sin `force`, la
  corrida termina con "carry-over completo, se salta" en segundos, sin línea
  `sonda:`, y habrías medido nada. Marcarlo en las tres deja los mismos inputs
  y evita el error.
- La salida queda en `dc-events` bajo `year=2023/month=03`, escrita en frío, y
  cada corrida sobrescribe la anterior. No es la serie definitiva: el backfill
  de ITSC-284 (hija 8) la reescribe encadenada.

## Cambiar la CPU entre corridas

Las tres corridas cambian **solo la CPU**: 2, 4 y 8 vCPU, con la misma memoria
en las tres. La memoria es la mínima que Cloud Run exige para 8 vCPU, que son
**4 GiB**, contra un RSS local de ~300 MiB (TRD-L2 §10.2). Se mide con la misma
memoria para que la pared dependa solo de la CPU. Si `gcloud` rechaza `4Gi` con
8 vCPU, el error dice el mínimo vigente: usa ese valor en las tres.

Desde Cloud Shell, antes de cada corrida (`<cpu>` = `2`, `4` u `8`):

```bash
gcloud run jobs update l2-backfill \
  --project "<GCP_PROJECT_ID>" --region "<GCP_REGION>" \
  --cpu <cpu> --memory 4Gi --task-timeout 86400 --max-retries 0
```

Y compruébalo antes de lanzar:

```bash
gcloud run jobs describe l2-backfill --project "<GCP_PROJECT_ID>" --region "<GCP_REGION>" \
  --format='value(spec.template.spec.template.spec.containers[0].resources.limits,spec.template.spec.template.spec.timeoutSeconds,spec.template.spec.template.spec.maxRetries)'
```

- `--task-timeout 86400` (24 h) evita que el timeout provisional del stack (3600 s) corte
  una corrida de 2 vCPU que tarde más; lo que se mide es la pared real.
- `--max-retries 0` evita que un reintento sume otra corrida al run y deje
  dos líneas `sonda:`; un fallo se lee y se relanza a mano.
- **Es deriva temporal frente a Terraform.** El stack `l2` fija los valores
  finales (`infra/stacks/batch/l2/main.tf`, TRD-L2 §10.2): 4 vCPU, 4 GiB,
  1 reintento, 1200 s en `monthly` y 36.000 s en `backfill`, tras ITSC-286
  (antes, desde ITSC-282, eran 2 vCPU, 2400 s y 54.000 s). `apply` desharía los
  cambios de la sonda: no lo apliques entre corridas. El `apply` del final de
  "Redimensionamiento de vCPU con la imagen 0.5.2" cierra la deriva.

## Qué leer y qué anotar

En el log que vuelca el paso "Logs de la ejecución" del run:

- La línea `sonda: unit=binance/spot/BTCUSDT/2023-03 mode=backfill
  rss_peak_mib=<MiB> wall_s=<s> ticks=<n> cores=<n> ticks_s_core=<n>
  theta_ticks_s_core=<n>`:
  - `rss_peak_mib`: RSS pico de la tarea.
  - `wall_s`: pared de la unidad completa (lectura, 50 θ y escritura), no del
    arranque del contenedor.
  - `ticks`: ticks del mes; debe ser el mismo en las tres corridas.
  - `cores`: en las imágenes anteriores a 0.5.1, núcleos visibles de la
    máquina, **no** el límite de vCPU del job (la sonda del 2026-09-30 leyó 6,
    6 y 9 con 2, 4 y 8 vCPU configurados). Desde ITSC-289 es el límite
    efectivo y `cores_visible` conserva el dato viejo; aun así, valida la
    config con `gcloud run jobs describe`.
  - `ticks_s_core` = `ticks / wall_s / cores` y `theta_ticks_s_core` es lo
    mismo por los 50 θ (la unidad del benchmark de `dc_core`). Por lo dicho
    en `cores`, recalcúlalos dividiendo por el vCPU configurado.
- Las líneas `θ=<θ> eventos=<n> events_content_hash=... carry_over_content_hash=...`,
  una por θ (50), y `eventos cerrados por θ: min=<n> max=<n>`. Con las tres
  corridas sobre los mismos datos, eventos y hashes deben coincidir entre ellas:
  el resultado no depende de la CPU.
- Los `events_summary` (uno por θ) están también en el lago de hallazgos, no
  solo en el log. Para confirmarlos, desde Cloud Shell:

  ```bash
  gcloud storage cp -r "gs://<bucket dq-findings>/l2/detected_date=<fecha de la corrida>" /tmp/dq
  python3 -c "
  import pyarrow.compute as pc, pyarrow.parquet as pq
  t = pq.read_table('/tmp/dq')
  t = t.filter(pc.and_(pc.equal(t['check_type'], 'events_summary'),
                       pc.and_(pc.equal(t['year'], 2023), pc.equal(t['month'], 3))))
  print(t.num_rows, t.select(['run_id', 'metric_value', 'details']).to_pylist()[:3])"
  ```

  Esperas 50 filas por corrida (una por θ), todas con el mismo `run_id`. Si ese
  día hubo más de una corrida, verás 50 filas por cada `run_id`.

En el resumen del run ("Resumen de la ejecución"): el estado (Exitosa o
Fallida) y las tareas fallidas.

Una ejecución Fallida no siempre es un OOM. Distingue la causa en el log:

- **OOM**: aparece "Memory limit of ... exceeded". El proceso muere con
  SIGKILL, el `finally` de la CLI no corre y **no hay línea `sonda:`**. Con
  4 GiB sería un hallazgo grande (el RSS local es ~300 MiB): anótalo y avisa,
  no subas la memoria por tu cuenta sin dejarlo dicho.
- **Timeout**: la tarea termina al alcanzar `--task-timeout`, sin mensaje de
  memoria. Con 24 h sería una señal clara para la regla de decisión.
- **Error de uso** (código 2, sin `sonda:`): falta `series_start`, rango
  inválido o variables de entorno ausentes. Lo dice el log.
- **Falta la landing de 2023-03** (código 1): revisa el paso 4 de "Antes de
  empezar".

## Costo y extrapolación

Con `s` = `wall_s` y la config de la corrida (`c` vCPU, `m` GiB):

- vCPU-s = `c` × `s`; GiB-s = `m` × `s`.
- **Costo de la corrida a precio de lista**: vCPU-s × tarifa de vCPU-s +
  GiB-s × tarifa de GiB-s, sin descontar el cupo gratis. Toma las tarifas de
  <https://cloud.google.com/run/pricing> el día de la corrida, para la región de
  `GCP_REGION` y para jobs (se cobran como CPU asignada durante toda la
  ejecución, no por solicitud), y anótalas junto con la fecha. Sin tarifas
  anotadas, la columna de costo queda sin llenar.
- **Extrapolación a 109 meses**: pared total = 109 × `s`, en horas y contra el
  tope de 24 h (86.400 s) de una tarea; GiB-s y vCPU-s ×109. Es un techo
  (2023-03 es el mes más pesado y los demás tardan menos), con el backfill
  secuencial en una sola tarea.
- **Contra el cupo gratis mensual** (360.000 GiB-s y 180.000 vCPU-s, los mismos
  de la sonda de L1): ×109 sobre el cupo, y el excedente facturable =
  (total − cupo) × tarifa, con mínimo 0. El cupo lo comparten todos los jobs y
  servicios del mes (L1 incluido): si el backfill cae en el mismo mes que uno
  de L1, el cupo disponible es menor.

## Resultados

Medido por el humano el 2026-09-30, con `run-job.yml` desde `main` sobre
`l2-backfill`, `from` = `to` = `series_start` = `2023-03` y `force` = `true`.

Tarifas de lista de Cloud Run Jobs (Default, sin CUD), consultadas el
2026-09-30 en <https://cloud.google.com/run/pricing>: vCPU-s 0,000018 USD,
GiB-s 0,000002 USD, región `us-east1`.

**Sonda (mes 2023-03)**, imagen `l2_dc_events:0.5.0`, 4 GiB, `force` = `true`,
sin reintentos, timeout de 24 h. `ticks` = 190.227.841 en las tres corridas.
Ticks/s por core se calculó con el **vCPU configurado**, no con el `cores` del
log (ver hallazgo abajo).

| Corrida | Fecha | URL del run | Config | RSS pico (MiB) | Pared (s) | Ticks/s por core | GiB-s | vCPU-s | Costo de la corrida (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-09-30 | [36744329462](https://github.com/byroncz/intrinsica/actions/runs/36744329462) | 2 vCPU, 4 GiB | 446 | 292,2 | 325.510 | 1.168,8 | 584,4 | 0,0129 |
| 2 | 2026-09-30 | [36747509163](https://github.com/byroncz/intrinsica/actions/runs/36747509163) | 4 vCPU, 4 GiB | 446 | 261,5 | 181.862 | 1.046,0 | 1.046,0 | 0,0209 |
| 3 | 2026-09-30 | [36749897862](https://github.com/byroncz/intrinsica/actions/runs/36749897862) | 8 vCPU, 4 GiB | 445 | 301,1 | 78.972 | 1.204,4 | 2.408,8 | 0,0458 |

Ticks/s por core (θ × 50, la unidad del benchmark de `dc_core`): 16.275.483,
9.093.109 y 3.948.602. La línea `sonda:` reportó 108.503, 121.241 y 70.197
(5.425.161, 6.062.073 y 3.509.868 por θ) porque divide por `cores` (6, 6 y 9).

**Extrapolación a los 109 meses** (backfill secuencial en una tarea, techo con
el mes más pesado)

| Corrida | Pared total (h) | Contra 24 h | GiB-s ×109 | vs cupo (360.000) | vCPU-s ×109 | vs cupo (180.000) | Costo del backfill, con cupo (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 vCPU | 8,85 | 36,9 % | 127.399 | 35,4 % | 63.700 | 35,4 % | 0,00 |
| 4 vCPU | 7,92 | 33,0 % | 114.014 | 31,7 % | 114.014 | 63,3 % | 0,00 |
| 8 vCPU | 9,12 | 38,0 % | 131.280 | 36,5 % | 262.559 | 145,9 % | 1,49 |

El costo con cupo del 8 vCPU es el excedente de vCPU-s (82.559) × tarifa. A
precio de lista y sin cupo, el backfill costaría 1,40, 2,28 y 4,99 USD. Si L1
consumió cupo ese mes, el excedente sube.

**Veredicto según la regla de decisión** (sobre la corrida de 8 vCPU):
**Cloud Run Jobs**.

1. RSS pico 445 MiB de 24.576 MiB (1,8 %), sin OOM.
2. 109 × 301,1 s = 32.820 s (9,12 h) < 86.400 s: 38 % del tope de 24 h.

Cloud Batch con Spot no hace falta.

- Verificación del mes más pesado: 2026-09-30. El ranking del 2026-09-25
  cubre los 109 meses hasta 2026-08 y no hay meses nuevos; 2023-03 sigue
  siendo el mayor.
- Estado de cada ejecución: Exitosa en las tres, sin OOM ni timeout.
- Consistencia entre corridas: `ticks` igual y `events_summary` con
  `status=pass` en los 50 θ en las tres. No se anotaron los hashes por θ.
- Memoria: 4 GiB funcionó con 2, 4 y 8 vCPU (verificado con `gcloud run jobs
  describe`). El RSS no cambia con la CPU: ~445 MiB.

**Hallazgos para ITSC-282**

- **La pared no baja con más vCPU**: 292,2 s con 2, 261,5 con 4 y 301,1 con 8.
  Pasar de 2 a 8 vCPU cuadruplica vCPU-s y costo sin ganar tiempo. Con una
  sola corrida por config, la diferencia entre 261 y 301 s puede ser ruido;
  lo sólido es que más CPU no acelera. Es lo que ITSC-282 debe pesar al elegir
  los vCPU; esta sonda no explica la causa. La separa por fase la
  "Corrida por fases (ITSC-289)" del final.
- **`cores` no es el límite del job.** La línea `sonda:` reportó 6, 6 y 9
  cores (los visibles de la máquina) contra 2, 4 y 8 configurados. Por eso
  `ticks_s_core` y `theta_ticks_s_core` de la línea `sonda:` no sirven: usa
  los de la tabla. Desde ITSC-289 la sonda lee la cuota del cgroup y reporta
  `cores` (límite efectivo) y `cores_visible` por separado; estos números
  de 2026-09-30 son de la imagen anterior y conservan el `cores` visible.

## Corrida por fases (ITSC-289)

La sonda de arriba dice *cuánto* tarda el mes más pesado, no *dónde*. Desde la
versión 0.5.1 de la imagen, la línea `sonda:` y el hallazgo `unit_timing`
separan el tiempo por fase. Una sola corrida, sobre 2023-03, con la
configuración vigente del stack `l2` (no cambies CPU ni memoria: lo que se mide
es esa config).

**Cómo correrla.** Tras el `apply` del stack `l2` con la imagen 0.5.1 o
posterior, *Actions → Run job* con los inputs de "Inputs de Run job" (`job` =
`l2-backfill`, `from` = `to` = `series_start` = `2023-03`, `force` marcado).

**Qué leer en la línea `sonda:`:**

| Campo | Qué es |
| --- | --- |
| `cores` | Límite efectivo de CPU: cuota del cgroup (`cpu.max`) o, sin ella, los cores visibles. `ticks_s_core` y `theta_ticks_s_core` se calculan con él. |
| `cores_visible` | Lo que ve la máquina (`sched_getaffinity`); es el `cores` de la sonda anterior. |
| `cores_source` | Quién fijó `cores`: `cgroup-v2`, `cgroup-v1` o `affinity`. Si sale `affinity` en Cloud Run, la cuota no es legible desde el contenedor y `cores` no es fiable: vale el vCPU de `gcloud run jobs describe`. |
| `read_s` | Lo que el hilo principal esperó por la landing: abrir el archivo y, por cada row group, lo que tardó en llegar el que pidió el lector anticipado (ITSC-290; 0.5.1 lo medía en serie). |
| `decode_s` | Parquet a Arrow: CPU del hilo lector. No es pared del principal: no entra en la suma de abajo. |
| `detect_s` | El fan-out de los 50 θ, pared del hilo principal. |
| `detect_cpu_s` | CPU del hilo principal durante el fan-out (`thread_time`). Con `detect_s` distingue un detector lento (CPU ≈ pared de su parte) de uno desalojado (CPU mucho menor que la pared: lo frena la cuota o otros hilos). |
| `fanout_threads`, `write_workers` | Hilos que usó el fan-out (`ceil(cores)`, acotado a `cores_visible`) y escritores de Parquet (`round(cores)`). Los decide `cpu.py`. |
| `carry_s` | Cargar el carry-over del mes anterior (50 lecturas). Cero en frío. |
| `wait_s` | El hilo principal esperando a los escritores (incluye la publicación final). |
| `write_s` | Codificar y subir Parquet (eventos y carry-over), **sumado entre los hilos de escritura**: puede pasar de `wall_s`. |
| `other_s` | `wall_s` menos `carry_s`, `read_s`, `detect_s` y `wait_s`. Es lo que ninguna fase explica; incluye emitir los hallazgos. |
| `row_groups`, `bytes_in` | Row groups del mes y bytes comprimidos de las tres columnas que L2 lee. |
| `cpu_throttled_s` | Segundos que el cgroup estuvo frenado por agotar su cuota durante la unidad. Si es alto, parte de `read_s` es CPU estrangulada, no I/O. |

`read_s`, `detect_s`, `carry_s` y `wait_s` son pared del hilo principal y se
suman con `other_s` hasta `wall_s`; `decode_s`, `detect_cpu_s` y `write_s` no
entran en esa suma porque corren en otros hilos o miden CPU. Para saber si la escritura es el cuello,
compara `wait_s` con `wall_s`.

**Dónde queda el dato sin leer logs.** El hallazgo `unit_timing` (uno por mes,
`metric_value` = `wall_s`, el resto en `details`) va al lago `dq-findings` con
cada unidad, así que el backfill de ITSC-284 deja 109 puntos. Desde Cloud
Shell, con el `gcloud storage cp` de "Qué leer y qué anotar":

```bash
python3 -c "
import json, pyarrow.compute as pc, pyarrow.parquet as pq
t = pq.read_table('/tmp/dq')
t = t.filter(pc.equal(t['check_type'], 'unit_timing'))
for r in t.select(['year', 'month', 'metric_value', 'details']).to_pylist():
    print(r['year'], r['month'], r['metric_value'], json.loads(r['details']))"
```

**Qué anotar.** La línea `sonda:` completa va al README de la capa
("Dónde se va el tiempo en la nube") y a la card ITSC-289, con un párrafo: qué
fase domina y qué optimización propone la card siguiente.

## Corrida antes y después (ITSC-290)

ITSC-290 cambia la tubería de L2 (GIL, hilos según la cuota, lectura anticipada,
Parquet sin diccionario) sin tocar la salida. Esta corrida mide cuánto bajó la
pared de 2023-03 en la nube.

**Antes** es la línea base de 0.5.1 en el mismo host: pared 613 a 733 s en
cuatro corridas (ITSC-289). Si el host cambió o hay dudas, repite una corrida
con 0.5.1. **Después** es una corrida de la versión 0.5.2 o posterior, con los
mismos inputs de "Inputs de Run job" (`job` = `l2-backfill`, `from` = `to` =
`series_start` = `2023-03`, `force` marcado) y la config vigente del stack
`l2` (no cambies CPU ni memoria).

**Qué comparar.** `wall_s` de las dos, con `cores` y `cores_visible` al lado: la
cuota que entrega Cloud Run varía entre corridas (1,61, 1,86 y 1,97 para
2 vCPU) y un `cores_visible` distinto cambia cuántos hilos usa 0.5.1. La meta
de la card es `wall_s` ≤ 50 % de la línea base de 0.5.1 en el mismo host. Anota
también `fanout_threads` (debe ser 2), `write_workers` (2), `detect_cpu_s`
frente a `detect_s`, `wait_s` y `cpu_throttled_s`.

**Regla de decisión posterior** (no es parte de la card): si `other_s` más las
esperas (`wait_s` y `read_s`) siguen por encima del 20 % de `wall_s`, la
cáscara Python estorba y se abre la card de mover la tubería a Rust. Si no, la
frontera Rust/Python se queda como está.

**Qué anotar.** La línea `sonda:` completa de las dos corridas va al README de
la capa ("Corrida en la nube, antes y después de ITSC-290") y a la card
ITSC-290, con un párrafo: cuánto bajó la pared y qué fase domina ahora.

## Redimensionamiento de vCPU con la imagen 0.5.2 (ITSC-286)

La sonda de ITSC-281 no decidía el tamaño: con 0.5.0 la pared no cambiaba entre
2, 4 y 8 vCPU (292, 262 y 301 s) porque el detector corría con un hilo y los
escritores esperaban el GIL. Con 0.5.2 (ITSC-290) el job sí usa su cuota, así
que se repitió la sonda de 2023-03 con los tres tamaños. Solo cambia la CPU:
memoria 4 GiB, `force`, `from` = `to` = `series_start` = `2023-03`, sin
reintentos y con `--task-timeout 86400`, como en "Cambiar la CPU entre corridas".
1 vCPU se descartó sin medir: a 2 vCPU el cgroup ya frenó 99,4 s, y con uno solo
la pared se duplicaría.

### Regla de decisión

Se escribió antes de aplicarla, con un desempate que se agregó al ver los datos
(lo explica el punto 2):

1. **Candidatos:** los tamaños cuya pared del mes más pesado cumple 6× contra el
   timeout de `monthly` (el 6× de L1) y cuyo backfill de 109 meses cabe en
   menos de 86.400 s con 1,5× de holgura.
2. **Elección:** el de menor costo de lista por mes. **Desempate:** dos costos a
   menos de 10 % entre sí cuentan como iguales, porque un mes cuesta centavos y
   el backfill entero cae dentro del cupo gratis; en un empate gana el de menor
   pared. Un tamaño con `cpu_throttled_s` > 10 % de `wall_s` solo gana si no
   hay otro en el empate: está limitado por la CPU y su pared depende de la
   cuota que Cloud Run entregue (1,61 a 1,97 de 2 en 0.5.x).
3. **Techo del escalado**, para decir qué frena a los tamaños mayores: si
   `cpu_throttled_s` sigue alto a 4 vCPU, el techo es la CPU. Si la pared se
   estanca entre 4 y 8 con freno cero, el techo es el θ pesado (21 % de los
   eventos en un hilo) o la subida a GCS, y va a una card nueva; no se arregla
   en esta.

### Mediciones

Mes 2023-03, 190.227.841 ticks y 237 row groups en las tres. 2 vCPU corrió con
la imagen 0.5.2 y 4 y 8 vCPU con la 0.6.0, que no cambia el rendimiento (solo
la frontera por θ de ITSC-285). Tarifas de lista del 2026-09-30 (vCPU-s
0,000018 USD, GiB-s 0,000002 USD, `us-east1`), las mismas de la sonda de
ITSC-281.

| vCPU | Run | Imagen | `wall_s` | `cores` | `cores_visible` | `fanout_threads` | `write_workers` | `detect_s` | `detect_cpu_s` | `wait_s` | `cpu_throttled_s` | RSS pico (MiB) | vCPU-s | GiB-s | Costo de lista (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | [36956085713](https://github.com/byroncz/intrinsica/actions/runs/36956085713) | 0.5.2 | 311,7 | 1,86 | 5 | 2 | 2 | 275,1 | 231,1 | 31,2 | 99,4 | 487 | 623,4 | 1.246,8 | 0,0137 |
| 4 | [37061790059](https://github.com/byroncz/intrinsica/actions/runs/37061790059) | 0.6.0 | 184,1 | 3,944 | 5 | 4 | 4 | 135,2 | 106,3 | 43,2 | 4,5 | 489 | 736,4 | 736,4 | 0,0147 |
| 8 | [37069958118](https://github.com/byroncz/intrinsica/actions/runs/37069958118) | 0.6.0 | 133,4 | 7,441 | 9 | 8 | 7 | 75,6 | 13,8 | 51,3 | 0,0 | 491 | 1.067,2 | 533,6 | 0,0203 |

- **De la corrida de 4 vCPU solo vale la primera línea `sonda:`**, la de 2023-03.
  Esa ejecución siguió hasta 2026-08 porque `l2-backfill` no pasaba `--to` si
  era igual a `from` (ITSC-292, corregido en `main`); las demás líneas son de
  otros meses.
- **`detect_cpu_s` solo es comparable entre 2 y 4 vCPU.** Es la CPU del hilo
  principal; con 8 hilos en el fan-out ese hilo solo reparte, así que baja a
  13,8 s y no es el costo del detector.
- `write_s` (164,0, 165,9 y 173,3 s, sumados entre hilos) casi no cambia: el
  trabajo de codificar y subir es el mismo, solo se reparte entre más hilos.
- `read_s` 1,2, 1,1 y 1,9 s; `decode_s` 21,7, 24,3 y 21,1 s; `carry_s` 0 (en
  frío). La lectura anticipada de ITSC-290 dejó de ser un cuello.
- RSS pico 487 a 491 MiB: no depende de la CPU y sube ~40 MiB frente a 0.5.0
  (446), lo que permite `L2_READ_AHEAD` = 2. Son 12 % de 4 GiB.

### Lectura

| Paso | Pared | Costo de lista | `cpu_throttled_s` / `wall_s` | `wait_s` / `wall_s` |
| --- | --- | --- | --- | --- |
| 2 → 4 vCPU | −40,9 % (1,69×; ideal 2×) | +7,4 % | 31,9 % → 2,4 % | 10,0 % → 23,5 % |
| 4 → 8 vCPU | −27,5 % (1,38×; ideal 2×) | +37,8 % | 2,4 % → 0,0 % | 23,5 % → 38,5 % |

- **2 vCPU está limitado por la CPU:** el cgroup lo frenó 99,4 s de 311,7 (32 %).
  Su pared depende de la cuota entregada, y la de 0.5.1 varió 613 a 733 s entre
  corridas del mismo tamaño.
- **A 4 vCPU el freno casi desaparece** (4,5 s, 2,4 %). El techo ya no es la
  CPU.
- **De 4 a 8 el escalado se acaba sin freno.** Con `cpu_throttled_s` = 0 la
  pared solo baja 27 % por el doble de CPU, y `wait_s` pasa de 23 % a 38 % de la
  pared: el hilo principal termina de repartir y espera a los escritores. Es la
  firma de un tramo serial al final, no de falta de CPU. La medición no separa
  si es el θ pesado (21 % de los eventos en un hilo, también en el escritor) o
  la subida a GCS; es una card nueva (ver abajo), no se arregla aquí.

### Decisión: 4 vCPU y 4 GiB

| Tamaño | Costo de lista por mes (USD) | Backfill de 109 meses: pared | Backfill: vCPU-s / GiB-s, contra el cupo (180.000 / 360.000) | Backfill a precio de lista (USD) |
| --- | --- | --- | --- | --- |
| 2 vCPU | 0,0137 | 33.975 s (9,44 h) | 67.951 (37,7 %) / 135.901 (37,7 %) | 1,49 |
| 4 vCPU | 0,0147 | 20.067 s (5,57 h) | 80.268 (44,6 %) / 80.268 (22,3 %) | 1,60 |
| 8 vCPU | 0,0203 | 14.541 s (4,04 h) | 116.325 (64,6 %) / 58.162 (16,2 %) | 2,21 |

(109 × el mes más pesado, techo; el cupo es compartido con L1.)

Aplicada la regla:

1. Los tres tamaños son candidatos: 6× la pared de 8 vCPU son 800 s y 6× la de
   2 vCPU, 1.870 s, y los 109 meses caben en 9,4 h como máximo.
2. Por costo de lista gana 2 vCPU (0,0137 USD), pero por **0,001 USD al mes**,
   menos de 10 % de diferencia con 4 vCPU (0,0147): son un empate. El backfill
   entero cabe en el cupo gratis con cualquiera de los tres, así que en dinero
   real la diferencia es cero. En el empate gana la pared (−41 %) y 2 vCPU,
   además, está limitado por la CPU (32 % de freno). **Se elige 4 vCPU.**
   El desempate lo propuso el arquitecto al ver los números; sin él, la regla
   literal daba 2 vCPU. Queda escrito para que no parezca otra cosa.
3. 8 vCPU cuesta 38 % más que 4 por 27 % menos de pared, y sin freno: lo que
   frena ya no es la CPU. No se justifica hasta que la card siguiente resuelva
   el tramo serial.

**Timeouts** (`infra/stacks/batch/l2/main.tf`):

- `monthly` = **1200 s**: 6× 184,1 s = 1.105 s, redondeado hacia arriba (6,5×).
  Cubre también una cuota de 2 vCPU con freno, que es el peor caso de la
  imagen anterior.
- `backfill` = **36.000 s** (10 h): 1,79× el techo de 109 × 184,1 s (20.067 s).
  La regla de L2 pide al menos 1,5× y no más de 86.400 s.
- `max_retries` = 1, igual que antes.

**Siguiente card (no se arregla aquí):** separar el tramo serial que aparece con
8 vCPU: `wait_s` del 38 % con freno cero. Medir qué θ cierra último y cuánto
tarda su escritura, y comparar con la subida a GCS. Si es el θ pesado, partir su
escritura o repartir mejor el fan-out; si es la subida, paralelizarla.

### Deriva a cerrar

La última corrida dejó `l2-backfill` en 8 vCPU, 4 GiB, 86.400 s y sin
reintentos por `gcloud run jobs update`. El `apply` del stack `l2` lo lleva a
4 vCPU, 4 GiB, 36.000 s y 1 reintento, y deja `l2-monthly` en 4 vCPU, 4 GiB,
1.200 s y 1 reintento. Lo aplica el humano en *Actions → Terraform → `l2` →
`apply`*. El plan debe mostrar cambios de `cpu`, `timeout` y `max_retries` en
`l2-backfill` y de `cpu` y `timeout` en `l2-monthly`; la memoria no cambia.
Cualquier otro cambio es una sorpresa y se revisa antes de aprobar.
