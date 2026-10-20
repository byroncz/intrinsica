# Runbook: operación de L3 (backfill histórico, lectura de fallos, cascada, cadena mensual, volumen y costo, perfil de una unidad, hilos de la unidad, verificación cruzada)

Lo ejecuta el humano desde *Actions → Run job* y Cloud Shell; ningún agente
dispara `terraform.yml` ni `run-job.yml`. Dice cómo se opera L3 una vez construido
su histórico: cómo se lanza, cómo se lee un fallo, cómo se relanza un mes, qué
hacer cuando L1 reprocesa, cómo corre la cadena mensual y con qué scripts se
comprueba. Referencia: [TRD-L3 §8, §9.3, §10 y §11](../TRD/l3.md). Mismo patrón que
el [runbook de L2](operacion-l2.md); aquí solo cambia lo que L3 hace distinto. Las
cifras de dimensionamiento salieron de la sonda y del histórico de ITSC-335.

Fechas calculadas al **2026-10-15**. Si lo ejecutas otro día, recalcula el
rango con la sección siguiente.

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
   `L3_SERIES_START` y `L3_THETAS_URI` van en `env`; `L3_THREADS` y `L3_PREFETCH` son opcionales
   (ver "Hilos de la unidad").
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
5. Variables del environment `gcp` cargadas (`GCP_PROJECT_ID`, `GCP_REGION`,
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

La suma de los 110 meses son 11,5 h (41 256 s), más que las 6 h que espera el
run de Actions. Cloud Run no corta la tarea (el timeout es el tope del módulo,
86 400 s), pero el run fallaría. Por eso el histórico se lanza por rangos,
**uno a la vez y cada uno con éxito antes del siguiente**. Los de ITSC-335:

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
una familia): fija cada rango por su **pared estimada**, no por su número de
meses. La suma de los `wall_s` esperados del rango debe quedar en **≈ 5 h
(18 000 s) o menos**. Estima cada mes con el ajuste de
[TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición)
(`wall_s` ≈ 68 s + 0,038 s por millón de ticks + 24,9 s por millón de eventos) o
con los `unit_timing` de `dq-findings/l3/`. La pared sigue a los **eventos**,
más que a los ticks: 2021-01 (46,0 M de eventos) tomó 1 419 s y 2023-03, el mes
con más ticks, 873 s. Los 41 meses de 2017-08 a 2020-12 promediaron ≈ 7,2 min por
mes; 2021 promedió ≈ 10,75 min. Lo más simple es repetir los rangos de la tabla.

La frontera de cada θ es la cadena **contigua desde `series_start`**: un rango
sin `from` arranca en el mes siguiente al último construido, y un mes hecho fuera
de orden (como 2023-03 y 2026-09 en ITSC-335) se reconstruye al llegar a él, con el
mismo `content_hash`. Para no repetirlo, termina el rango un mes antes.
Relanzar un rango interrumpido es seguro: la frontera retoma donde quedó.

## Leer un fallo

Un fallo detiene el rango (los meses siguientes no se tocan) y deja un hallazgo
en `gs://<proyecto>-dq-findings/l3/detected_date=<fecha>/`. Primero distingue la
causa en el log de la ejecución:

- **OOM** ("Memory limit exceeded"): no hay línea `unidad <mes>:` de ese mes. El pico
  medido es 1 267 MiB de 4 GiB: sería un hallazgo nuevo; anótalo y avisa.
- **Timeout**: la tarea se corta sin mensaje de memoria. El de `l3-monthly` es de
  10 800 s (7,6 veces el mes más pesado).
- **Un hallazgo `error`**: lee su `check_type` en la tabla siguiente.

Los 14 `check_type` de [TRD-L3 §9.3](../TRD/l3.md#93-tipos-de-chequeo-check_type),
y qué hacer con cada uno. En todos los casos de `error`, **el mes no se publica y
la frontera no avanza**: relanzar los mismos inputs retoma donde quedó. Nunca
repares un archivo de `l3/` a mano.

| `check_type` | Severidad | Qué significa | Qué hacer |
| --- | --- | --- | --- |
| `phase_quantity_mismatch` | `error` | `buy + sell` de una fase no es la suma de `quantity` de sus ticks | No debería ocurrir: es un defecto del núcleo o de los datos. Abre una card con `task-create` con `details` (θ, `confirm_agg_trade_id`, fase, los dos valores). No relances: dará lo mismo |
| `family_skeleton_mismatch` | `error` | Un archivo de familia no repite filas, orden, claves o row groups del esqueleto | Compara `details` con `events.parquet` de L2 del mes. Si L2 cambió después, relanza el mes con `force` (ver "Relanzar un mes"). Si persiste, card |
| `event_over_budget` | `warning` | Un evento supera el presupuesto de ticks (solo familias de serie completa) | `summaries` no tiene presupuesto y no lo emite. Informa; no falla la unidad |
| `l2_input_missing` | `error` | Falta `events.parquet` o `carry_over.parquet` de `(θ, M)` | Completa L2 para ese θ y mes ([runbook de L2](operacion-l2.md)); luego relanza `l3-backfill` con los mismos inputs. Ese θ salió del rango y los demás siguieron (código 1) |
| `l2_state_inconsistent` | `error` | L2 contradice a L1 o a un invariante de fila (TRD-L3 §9.2); `details` trae θ, mes, campo y los dos valores | Descarta primero L1: ids repetidos o marcas del proveedor (ver "Cascada"). Si L1 está sano, es un defecto de L2 o de L3: card con `details`. Es el hallazgo que detectó los duplicados de 2026-02 |
| `phase_ticks_mismatch` | `error` | No cuadra la conciliación de ticks de §9.1; `details` trae `N_M`, `Σ_cerrados`, `P_M`, `Cola_M` y `U_M` | Corre el seam-check de [L1](operacion-l1.md) y el de [L2](operacion-l2.md#paso-3-seam-check-de-continuidad) sobre el mes. Si ambos pasan, card con `details` |
| `input_missing` | `error` | El mes no tiene `consolidated.parquet` ni provisionales | Corre `l1-backfill` o `l1-monthly-close` del mes ([runbook de L1](operacion-l1.md)); luego relanza |
| `input_provisional_only` | `error` | El mes solo tiene `provisional-day=DD.parquet` | Espera el cierre de L1 (el consolidado sale el primer lunes del mes siguiente) o lanza `l1-monthly-close`; luego relanza |
| `theta_catalog_invalid` | `error` | El catálogo no existe, no se lee o incumple TRD-L2 §7.3. Código 2, sin escribir | Corrige `manifest/l2/thetas.yaml` ([runbook de L2](operacion-l2.md#el-catálogo-de-θ)); relanza |
| `theta_behind_frontier` | `warning` | `monthly` no procesó un θ; `details.reason` es `l3_frontier` (la serie de L3 tiene un hueco o no existe) o `l2_missing` (L2 no tiene el mes) | `l3_frontier`: lanza `l3-backfill` sin `from` ni `to`, que retoma desde la frontera de cada θ. `l2_missing`: completa L2 primero. No falla la unidad si otros θ avanzan |
| `theta_config_drift` | `info` | L3 tiene particiones de un θ que ya no está en el catálogo | Informa. Quitar un θ del catálogo no borra nada; si fue un error, restáuralo |
| `summaries_summary` | `info` | Uno por θ y mes: filas, `content_hash` del archivo y de las entradas de L2 que usó | Es el linaje contra L2: sirve para comparar corridas y para detectar un mes obsoleto |
| `unit_timing` | `info` | Uno por unidad (mes): pared, tiempo por fase, CPU efectiva y consumida (`cpu_s`, `cpu_utilization`) y bytes leídos | Es la fuente de las cifras de "Volumen y costo" |
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

Medidos sobre el histórico completo (ITSC-335, 2026-10-15; detalle en
[TRD-L3 §10.4 y §10.5](../TRD/l3.md#104-tamaño-de-l3--estimación-por-fila)).

| Medida | Valor |
| --- | --- |
| `dc-descriptors/l3/` | 46 742 260 408 B (**43,5 GiB**) en 5 500 `summaries.parquet`; 2,06 veces L2; ≈ 45 B por evento (1 042 M de filas) |
| Suma de `wall_s` de los 110 meses | 41 256 s (11,5 h); ≈ 165 000 vCPU-s y GiB-s |
| Costo de lista (0,000018 USD por vCPU-s y 0,000002 por GiB-s) | ≈ 3,3 USD, un cálculo y no una cifra de Billing |
| Mes más pesado | 2021-01: 1 419 s (46,0 M de eventos). 2023-03, el máximo por ticks: 873 s |
| RSS pico máximo | 1 267 MiB (2018-01), 31 % de 4 GiB |
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

## Hilos de la unidad: `L3_THREADS` y `L3_PREFETCH`

Cada mes usa hilos dentro de su tarea ([ADR-L3-15](../TRD/l3.md#615-adr-l3-15--lector-anticipado-y-pool-de-θ-por-row-group-hilos-no-procesos)):
un hilo lector que trae el siguiente row group de L1 mientras se procesa el actual y un pool
que agrega y escribe los θ de cada row group. Son opcionales y se dejan sin fijar:

| Variable | Qué fija | Por defecto |
|---|---|---|
| `L3_THREADS` | Hilos del pool de θ. Entero ≥ 1; nunca se usan más que θ | `ceil(3 × cores)` de `unit_timing`, nunca más que θ: 12 en `l3-backfill` (4 vCPU) y 24 a 8 vCPU. El stack `l3` no lo fija: el valor sigue a los vCPU |
| `L3_PREFETCH` | Row groups que el lector trae de ventaja, además del que está en proceso. Entero ≥ 0; 0 apaga el hilo lector | 2 (uno en proceso y dos listos). El stack `l3` no lo fija |

`L3_THREADS=1` con `L3_PREFETCH=0` es el flujo secuencial de antes, útil como línea base o
si se sospecha de los hilos: el resultado (tablas y `content_hash`) es el mismo en cualquier
combinación. Un valor que no sea un entero válido termina la corrida con código 2, sin
escribir nada. Cada `prefetch` extra cuesta ≈ 48 MB de RSS (un row group de L1 decodificado). Un pool del mismo tamaño hace la compuerta de entrada y otro el linaje (`gate_s` y `lineage_s` son la pared de cada fase).

`unit_timing` dice lo que se usó (`threads`, `prefetch`); para sobrescribir los valores por defecto hay que lanzar desde Cloud Shell, como en el ejemplo de abajo. Con hilos, `aggregate_s` y `write_s`
son **suma sobre los hilos**: pueden pasar de `wall_s`, y eso es lo esperado. Para comparar
dos corridas usa `wall_s`, `cpu_s` y `cpu_utilization`.

`run-job.yml` no admite variables de entorno: para fijarlas en una corrida suelta se
lanza **desde Cloud Shell**, como el perfil (`--update-env-vars` vale solo para esa
ejecución; el job no cambia). Por ejemplo, la línea base secuencial de 2021-01:

```bash
gcloud run jobs execute l3-backfill \
  --project <proyecto> --region <región> \
  --update-env-vars L3_THREADS=1,L3_PREFETCH=0 \
  --args="--mode,backfill,--from=2021-01,--to=2021-01,--force" \
  --wait
```

La corrida T3d de ITSC-360 (los valores por defecto: 12 hilos y prefetch 2 a 4 vCPU) se lanza con
*Actions → Run job*, `l3-backfill`, `from = to = 2021-01` y `force`; el resultado se
registra como comentario en ITSC-360 y en el cuadro de
[TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición).
La prueba de la espera al lector se lanza desde Cloud Shell con el mismo comando de arriba y
`--update-env-vars L3_PREFETCH=3` (sin `L3_THREADS`); se acepta si `rss_peak_mib` ≤ 2 048.
Las metas de T3d son `wall_s` ≤ 300 s y `gate_s + lineage_s + read_s + reconcile_s` ≤ 60 s
([TRD-L3 §10.2](../TRD/l3.md#102-dimensionamiento--hipótesis-de-partida-y-primera-medición)):
si se cumplen, T5 (ITSC-352) puede empezar; si no, no empieza. No son la condición de reversión.
La condición para conservar los valores por defecto es la de ITSC-357: si la corrida no la
cumple (50 `content_hash` idénticos, `rss_peak_mib` ≤ 2 048 y pared ≥ 15 % menor que T1),
el stack `l3` fija `L3_THREADS=1` y `L3_PREFETCH=0` por Terraform, sin revertir código.
Un perfil con hilos (`L3_PROFILE=1`) ve todos los hilos y sus tiempos son de hilo, no de pared.

## Lector de tramas: `DC_FRAMES_IO_THREADS` y `readers`

`dc_frames` lee cada row group con `pre_buffer=True`, en un archivo que abre y cierra por
row group: Arrow junta los rangos de bytes de las columnas y los pide a la vez, en vez de una
petición por columna y por página. Cuántas peticiones viajan en paralelo lo fija el pool de E/S de Arrow:

| Variable | Qué fija | Por defecto |
|---|---|---|
| `DC_FRAMES_IO_THREADS` | Hilos del pool de E/S de Arrow. Entero ≥ 1; un valor inválido hace fallar el import de `dc_frames` con `ValueError`, antes de leer nada | 8 |

Se lee al importar `dc_frames`, en cualquiera de las capas que lo usan (L1, L2, L3, viz).
Como `L3_THREADS`, solo se sobrescribe lanzando desde Cloud Shell con `--update-env-vars`.
Más hilos de E/S no aceleran un row group suelto (es un solo rango de ≈ 15 MB): ayudan
cuando varios row groups se leen a la vez (`readers` > 1). `read_row_groups(readers=R)` aún
no lo usa L3: el orquestador lo adopta en la hija 3 de E6 (hoy L3 sigue con
`readers = 1` más su lector anticipado, `L3_PREFETCH`). Cada lector de más cuesta ≈ 48 MB de
RSS ([README de dc_frames](../../shared/dc_frames/README.md#lectura-pre_buffer-pool-de-es-y-readers)).

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
`arrow_pico_mib` y `readers=`/`pre_buffer=`; la línea `pico del pool de Arrow=…` dice si quedó bajo el
umbral del lector, `(K + 1,5)` row groups (informativo); de `row groups de L1: …`, los decodificados. Los bytes salen por
metadatos (`bytes por metadatos`), porque `--sin-contar-io` no los mide. Con `readers` > 1 las
fases por hilo no se reportan (los hilos se solapan): decide la pared. Metas
([TRD-L3 §10.3](../TRD/l3.md#103-sonda-del-lector-de-tramas--umbrales-de-aceptación-se-escriben-antes-de-medir)):
`s_por_row_group` ≤ 1,0 con `readers = 1` y `pared_s` ≤ 60 con `readers = 2`. Con `readers = 2`
se decodifican a lo sumo 2 row groups más que con 1, al final.

## Scripts de operación

Dos scripts de `layers/ops_tools/scripts/`, para el job `ops-script`
([runbook de operación ad hoc](operacion-ops.md)). Se suben a
`gs://<proyecto>-ops/scripts/` y se lanzan con `job` = `ops-script`. Ambos solo
leen, toman las raíces del proyecto de `OPS_RESULTS_URI` y también aceptan rutas
locales (`uv run`).

| Script | Para qué | `args` típicos |
| --- | --- | --- |
| `l3_probe_reader.py` | Mide el lector de tramas (pared por fase, RSS, bytes, row groups) contra los umbrales de TRD-L3 §10.3 | `--theta all --from 2023-03`; con lectores, `--theta all --from 2021-01 --modo row-groups --readers 2 --sin-contar-io` |
| `l3_check_month.py` | Verificación cruzada: agrega por su cuenta las tramas de cada evento del mes y compara fila a fila con `summaries.parquet` | `--month 2026-09` |

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

### La trampa del 403 por prefijo

La service account de `ops-script` lee `l1/`, `l2/`, `l3/` y `viz/tiles/` con
condiciones IAM por prefijo. `pyarrow.dataset.dataset("gs://<bucket>/l3")` (o
`pq.read_table` sobre un prefijo) pide primero el **objeto** `l3` sin barra, y la
condición `startsWith("…/l3/")` lo niega: 403, aunque los archivos de abajo sean
legibles (run 38659287542 de `ops-script`). Los dos scripts evitan el problema:
**listan de forma recursiva** y **abren cada archivo por su ruta completa**. Un
script nuevo hace lo mismo; desde Cloud Shell, con tus credenciales, el 403 no
aparece.
