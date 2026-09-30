# Runbook: sonda de dimensionamiento de L2 (ADR-04)

Lo ejecuta el humano; ningún agente dispara `terraform.yml` ni `run-job.yml`.
Mide cuánta memoria y tiempo necesita el mes más pesado de BTCUSDT en L2 con
2, 4 y 8 vCPU en Cloud Run Jobs, y con eso alimenta la decisión ADR-04
(Cloud Run Jobs o Cloud Batch con Spot) y la configuración final del stack
`l2`. Mismo patrón que la [sonda de L1](sonda-l1.md) (TRD-L1 §14.1); aquí solo
cambia lo que L2 hace distinto. Referencia: [TRD-L2 §10.2 y §14](../TRD/l2.md).
La decisión y la configuración final las registra la card ITSC-282 (hija 6);
esta card (ITSC-281) solo mide. Los números de las tres corridas van en
"Resultados".

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

Esta regla decide *dónde* corre L2. Cuántos vCPU se fijan (2, 4 u 8) lo elige
ITSC-282 comparando pared, GiB-s, vCPU-s y costo de las tres corridas.

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

- `from` = `to` = `2023-03`: un solo mes. El workflow omite `--to` si es igual
  a `from`; es lo esperado. L2 corre siempre en una sola tarea
  (`--tasks 1`), sin task array.
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

- `--task-timeout 86400` (24 h) evita que el timeout del stack (3600 s) corte
  una corrida de 2 vCPU que tarde más; lo que se mide es la pared real.
- `--max-retries 0` evita que un reintento sume otra corrida al run y deje
  dos líneas `sonda:`; un fallo se lee y se relanza a mano.
- **Es deriva temporal frente a Terraform.** El stack `l2` fija 4 vCPU,
  16 GiB, 3600 s y 1 reintento (provisionales, `infra/stacks/batch/l2/main.tf`).
  No lo apliques entre corridas ni al terminar: `apply` desharía estos cambios.
  El `apply` de ITSC-282 (hija 6) fija los valores finales y cierra la deriva;
  hasta entonces, el job en la nube no coincide con el código.

## Qué leer y qué anotar

En el log que vuelca el paso "Logs de la ejecución" del run:

- La línea `sonda: unit=binance/spot/BTCUSDT/2023-03 mode=backfill
  rss_peak_mib=<MiB> wall_s=<s> ticks=<n> cores=<n> ticks_s_core=<n>
  theta_ticks_s_core=<n>`:
  - `rss_peak_mib`: RSS pico de la tarea.
  - `wall_s`: pared de la unidad completa (lectura, 50 θ y escritura), no del
    arranque del contenedor.
  - `ticks`: ticks del mes; debe ser el mismo en las tres corridas.
  - `cores`: núcleos visibles de la máquina, **no** el límite de vCPU del job
    (la sonda del 2026-09-30 leyó 6, 6 y 9 con 2, 4 y 8 vCPU configurados).
    No lo uses para validar la config: usa `gcloud run jobs describe`.
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
  los vCPU; esta sonda no explica la causa.
- **`cores` no es el límite del job.** La línea `sonda:` reportó 6, 6 y 9
  cores (los visibles de la máquina) contra 2, 4 y 8 configurados. Por eso
  `ticks_s_core` y `theta_ticks_s_core` de la línea `sonda:` no sirven: usa
  los de la tabla. Falta que la sonda lea la cuota de CPU del cgroup; queda
  como card aparte.
