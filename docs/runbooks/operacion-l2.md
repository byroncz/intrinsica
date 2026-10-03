# Runbook: operación de L2 (backfill histórico, reproducibilidad, seam-check, costo y volumen)

Lo ejecuta el humano desde *Actions → Run job* y Cloud Shell; ningún agente
dispara `terraform.yml` ni `run-job.yml`. Cierra el criterio 1 de la Épica de
L2 con evidencia: materializa el histórico de eventos de los 50 θ, comprueba
que la corrida se reproduce, que las costuras entre meses cierran y cuánto
costó y pesa. Referencia: [TRD-L2 §7.3, §8, §10 y §14](../TRD/l2.md). La
dimensión del job salió de la [sonda](sonda-l2.md). Mismo patrón que el
[runbook de L1](operacion-l1.md); aquí solo cambia lo que L2 hace distinto.

Fechas calculadas al **2026-10-02**. Si lo ejecutas otro día, recalcula el
rango con la sección siguiente.

## Rango

- **Serie**: arranca en `2017-08` (`series_start`, `L2_SERIES_START` del
  stack). Es el único mes que se procesa sin carry-over previo.
- **Fin**: `l2-backfill` sin `to` llega hasta el último mes con
  `consolidated.parquet` en L1 (nunca provisionales). A 2026-10-02 es
  **2026-08**: el consolidado de 2026-09 se publica el primer lunes de octubre.
  Son **109 meses** (2017-08 a 2026-08) y 50 θ.
- **No hay `from` ni `to`.** La frontera de cada θ decide desde dónde avanza
  ([RF-L2-09](../TRD/l2.md)), así que el mismo lanzamiento sirve para la
  primera corrida, para reanudar y para agregar θ.

## Prerrequisitos

1. **Stacks aplicados** (*Actions → Terraform*, `stack` = `l2`, `action` =
   `apply`, aprobado en el environment `gcp`; `data` lo aplica el humano desde
   Cloud Shell) con la config de ITSC-282: **4 vCPU / 4 GiB**, timeout de
   `backfill` **36 000 s** (10 h) y de `monthly` 1 200 s, `max_retries = 1`,
   `L2_SERIES_START = 2017-08` y `L2_THETAS_URI` apuntando al catálogo. Verifica
   lo que quedó aplicado:

   ```bash
   gcloud run jobs describe l2-backfill --region <región> \
     --format='yaml(spec.template.spec.template.spec)'
   ```

   Debe mostrar `timeoutSeconds: 36000`, `maxRetries: 1`, `cpu: '4'` y
   `memory: 4Gi` en `limits`, y las tres raíces, `L2_SERIES_START` y
   `L2_THETAS_URI` en `env`. Existen `l2-backfill` y `l2-monthly`: `gcloud run jobs list --region <región>`.
   Si la ruta del `--format` no coincide con la de tu versión de `gcloud`, mira el
   YAML completo.
2. **Imagen `l2_dc_events` ≥ 0.6.0** (`layers/l2_dc_events/VERSION`). ITSC-282
   fijó 0.5.0 como piso, pero el catálogo de θ en GCS y la frontera por θ, que
   este runbook da por hechos, llegaron con la 0.6.0 (ITSC-285). El stack `l2`
   deriva el tag de ese archivo: sin `apply` posterior al merge, el job sigue
   con la imagen vieja.
3. **Catálogo de θ subido** (ver "El catálogo de θ"). Sin el objeto, la
   corrida termina con código 2 y el hallazgo `theta_catalog_invalid`.
4. **Landing completa de L1**: el `consolidated.parquet` de cada mes de la
   serie. Desde Cloud Shell:

   ```bash
   gcloud storage ls "gs://<proyecto>-landing/l1/provider=binance/market=spot/asset=BTCUSDT/year=*/month=*/consolidated.parquet" | wc -l
   ```

   Debe dar **109** (uno por mes, 2017-08 a 2026-08). Un mes sin consolidado, o
   con solo `provisional-day=*.parquet`, detiene el backfill en ese mes
   (`input_missing`, `input_provisional_only`): córrelo antes con
   [`l1-backfill` o `l1-monthly-close`](operacion-l1.md).
5. **Salida previa de las sondas.** `dc-events` ya puede tener meses de la
   sonda de ITSC-281/286/289 (2023-03 en frío con `series_start = 2023-03`) y
   los de la corrida de ITSC-292 (2023-03 a 2026-08, también en frío). No los
   borres: la frontera de cada θ es la cadena **contigua desde 2017-08**, así
   que esos meses no cuentan, el backfill parte de 2017-08 y los reescribe al
   llegar a ellos. El seam-check (paso 3) lo comprueba. El bucket no tiene
   versionado, así que lo reescrito no deja versiones viejas ocupando volumen.
6. **Cuota de Cloud Run.** Una sola tarea de 4 vCPU y 4 GiB; no hay task array
   (los meses se encadenan por carry-over), así que no hay paralelismo que
   ajustar.
7. Variables del environment `gcp` cargadas (`GCP_PROJECT_ID`, `GCP_REGION`,
   `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SERVICE_ACCOUNT`).

## El catálogo de θ

Los θ no son código sino un dato: `gs://<proyecto>-manifest/l2/thetas.yaml`
([TRD-L2 §7.3](../TRD/l2.md#73-el-catálogo-de-θ), decisión del 2026-09-30). La
cuenta de L2 solo lo lee; lo edita el humano desde Cloud Shell, sin PR.

**Los 50 θ iniciales** son la semilla del repo,
[`thetas.yaml`](../../layers/l2_dc_events/src/l2_dc_events/config/thetas.yaml):
`scale: 100000000` y 50 enteros `round(θ × 10⁸)`, log-espaciados de 0,01 %
(`10000`) a 5 % (`5000000`) por la regla de ADR-L2-10. Los valores, no la
fórmula, son la fuente de verdad. La CLI valida el objeto antes de tocar nada:
enteros únicos y dentro de `10000 ≤ θ ≤ 5000000`.

**Solo la primera vez** el objeto no existe. Desde un clon del repo en Cloud
Shell:

```bash
gcloud storage cp layers/l2_dc_events/src/l2_dc_events/config/thetas.yaml \
  gs://<proyecto>-manifest/l2/thetas.yaml
gcloud storage cat gs://<proyecto>-manifest/l2/thetas.yaml | grep -c '^  - '   # 50
```

## Cómo funciona `run-job.yml` con L2

`l2-backfill` va en **una sola tarea** (`--tasks 1`) que recorre los meses en
orden en un proceso: cada mes lee el carry-over que escribió el anterior, y
paralelizarlos rompería la cadena. Con `from` vacío el workflow no pasa
`--from`; con `to` vacío no pasa `--to` y el backfill llega al último mes
cerrado de L1. Si pasas `to`, el workflow lo manda siempre, aunque sea igual a
`from` (ITSC-292). `force` exige `from`. `series_start` vacío usa el del stack.
No lances a mano el mismo mes y modo que Scheduler esté ejecutando, ni dos
`l2-backfill` a la vez: escribirían la misma cadena.

Al final, el run vuelca los logs de la ejecución y un resumen (estado, tareas
completadas y fallidas, inicio y fin) en el *Summary* del run.

## Paso 1: backfill histórico

*Actions → Run job → Run workflow*, rama `main`:

| Input | Valor |
| --- | --- |
| `job` | `l2-backfill` |
| `from` | vacío |
| `to` | vacío |
| `force` | sin marcar |
| `series_start` | vacío |

El paso "Unidades" debe listar `1`. En el log de la ejecución:

- 50 líneas `θ=<entero> frontera=None` (un θ sin frontera arranca en
  `series_start`; el entero es `round(θ × 10⁸)`).
- **109 líneas `sonda: unit=... wall_s=...`**, una por mes, y tras cada una
  `eventos cerrados por θ: min=... max=...`. Cada mes escribe sus 50
  `events.parquet` y `carry_over.parquet` y los hallazgos `events_summary` (uno
  por θ) y `unit_timing` (uno por mes).
- Ninguna línea `ningún θ la necesita` (con las fronteras en `None`, todo mes
  lo necesita algún θ).

**Timeout de la tarea:** **36 000 s** (10 h), fijado en el stack (ITSC-282).
El techo del backfill son 109 × 184,1 s (el mes más pesado, 2023-03, con
4 vCPU) = 20 067 s (5,57 h): el timeout es 1,79× ese techo y queda bajo el tope
de 86 400 s del módulo. Si la pared real se acerca al timeout, o un mes pesa
claramente más que 2023-03, no subas el tope por tu cuenta: anótalo y abre una
card.

Anota: URL del run, nombre de la ejecución de Cloud Run, tareas completadas y
fallidas, inicio y fin.

### Reanudar tras un fallo

Un fallo detiene el rango: los meses siguientes no se tocan, el proceso
termina con código 1 y el mes deja su hallazgo en el lago de DQ. Con
`max_retries = 1`, Cloud Run reintenta la tarea una vez, y el reintento ya
retoma desde la frontera. Si la ejecución termina `Fallida`, distingue la causa
en el log, como en la [sonda](sonda-l2.md):

- **OOM** ("Memory limit exceeded"): no hay línea `sonda:` del mes. Con 4 GiB
  sería un hallazgo grande (el pico medido es ~490 MiB): anótalo y avisa.
- **Timeout**: la tarea se corta a las 10 h, sin mensaje de memoria.
- **Entrada ausente** (`input_missing`, `input_provisional_only`): falta o es
  provisional el consolidado de L1 de ese mes; corrígelo en L1.
- **Carry-over ausente o de otra versión** (`carry_over_missing`,
  `carry_over_version_mismatch`): un hallazgo por cada θ afectado, con el mes.

Corregida la causa, **relanza exactamente los mismos inputs** (todo vacío, sin
`force`). Cada θ calcula su frontera de nuevo (el último mes de su cadena de
carry-over válida), el log lo deja en `θ=<entero> frontera=<YYYY-MM>`, los
meses que ningún θ necesita se saltan sin leer L1 (`ningún θ la necesita`) y el
resto sigue desde ahí. Un mes publicado a medias no cuenta: el carry-over se
escribe después de los eventos, y sin carry-over el mes se rehace entero.
Relanzar con todo al día termina en segundos, sin escribir y con
`backfill: los 50 θ están al día en el rango`.

### Agregar θ nuevos

Sin PR ni reproceso de los θ existentes:

```bash
gcloud storage cp gs://<proyecto>-manifest/l2/thetas.yaml thetas.yaml
$EDITOR thetas.yaml    # agrega round(θ × 10⁸), p. ej. θ = 0,0003 es 30000
gcloud storage cp thetas.yaml gs://<proyecto>-manifest/l2/thetas.yaml
```

Luego lanza `l2-backfill` con los mismos inputs del paso 1 (todo vacío). Los θ
que ya llegaron al último mes cerrado se saltan; **solo los nuevos** recorren
la serie desde `2017-08`, leyendo cada mes de L1 una sola vez. Mientras eso no
termine, `l2-monthly` no avanza los nuevos y deja el hallazgo
`theta_behind_frontier` (warning). Los θ pequeños dominan el volumen de
`dc-events`: uno diminuto lo multiplica, uno grande casi no lo mueve.

## Paso 2: reproducibilidad

Una corrida del mes da los mismos archivos: la entrada es la misma y nada de la
salida depende de la ejecución. La evidencia es el `content_hash` del hallazgo
`events_summary` (uno por θ y mes, de `events.parquet` y de
`carry_over.parquet`). Relanza un mes intermedio con `force` y compáralo con el
de la primera corrida. La misma comparación existe como script del repo,
[`ops-scripts/hashes_events_summary.py`](ops-scripts/hashes_events_summary.py),
para correr con el job `ops-script` ([runbook](operacion-ops.md)) en lugar de
Cloud Shell; el snippet de abajo da lo mismo sin subir nada.

| Input | Valor |
| --- | --- |
| `job` | `l2-backfill` |
| `from` | `2020-01` |
| `to` | `2020-01` |
| `force` | **marcado** |
| `series_start` | vacío |

**`to` no se deja vacío**: con `force` y `to` vacío el backfill reprocesaría
de 2020-01 hasta el último mes. Con `to` = `from` procesa un solo mes, para los
50 θ, leyendo el carry-over de 2019-12 y reescribiendo 2020-01; el carry-over de
2020-01 no cambia, así que 2020-02 en adelante siguen válidos. Debe dejar una
sola línea `sonda:`.

Compara los hashes desde Cloud Shell (necesita `python3 -m pip install --user
pyarrow`; usa las credenciales de Cloud Shell, y si fallan, `gcloud auth
application-default login`). `SINCE` es la fecha UTC de la primera corrida, para
no leer particiones más viejas del lago:

```bash
DQ_ROOT=gs://<proyecto>-dq-findings/l2 MONTH=2020-01 SINCE=<YYYY-MM-DD> python3 - <<'PY'
import json
import os
import sys
from collections import defaultdict

import pyarrow.dataset as ds
import pyarrow.fs as pafs

root = os.environ["DQ_ROOT"]  # gs://<proyecto>-dq-findings/l2, o una ruta local
year, month = (int(x) for x in os.environ["MONTH"].split("-"))  # p. ej. 2020-01
since = os.environ.get("SINCE")  # YYYY-MM-DD: salta las particiones detected_date anteriores
fs, base = pafs.FileSystem.from_uri(root)

flt = (ds.field("check_type") == "events_summary") & (ds.field("year") == year) & (ds.field("month") == month)
if since:
    flt &= ds.field("detected_date") >= since
table = ds.dataset(base, filesystem=fs, partitioning="hive").to_table(
    filter=flt, columns=["run_id", "details"]
)

by_theta = defaultdict(list)  # θ -> [(run_id, events_hash, carry_over_hash)]
for row in table.to_pylist():
    d = json.loads(row["details"])
    by_theta[d["theta"]].append((row["run_id"], d["events_content_hash"], d["carry_over_content_hash"]))

runs = sorted({r[0] for rows in by_theta.values() for r in rows})
print(f"mes {year:04d}-{month:02d}: {len(by_theta)} θ, {len(runs)} corridas: {', '.join(runs)}")
if len(runs) < 2:
    sys.exit("falta una segunda corrida del mes para comparar")
bad = [t for t, rows in sorted(by_theta.items()) if len({r[1:] for r in rows}) != 1]
for theta in bad:
    print(f"DISTINTO θ={theta / 1e8:.8f}: {sorted({r[1:] for r in by_theta[theta]})}")
print("hashes idénticos en los θ" if not bad else f"{len(bad)} θ con hashes distintos")
sys.exit(1 if bad else 0)
PY
```

Imprime el mes, los θ y las corridas encontradas (deben ser al menos dos: la
primera y la forzada) y termina en 0 si **los 50 θ** tienen el mismo par de
hashes en todas ellas. Un θ con hashes distintos sale como `DISTINTO` y el
código es 1: es un defecto de determinismo, no se compensa; abre una card con
`task-create`.

## Paso 3: seam-check de continuidad

L2 encadena los meses por el carry-over: un evento se escribe en el mes que
confirma el evento **siguiente**, así que el evento pendiente al cierre de M va
en `carry_over.parquet` y lo completa M+1. Un borde (M, M+1) es correcto si,
para un θ:

1. el primer evento de M+1 es el evento pendiente del carry-over de M (misma
   referencia y misma confirmación), y
2. su referencia es el extremo del último evento de M.

El snippet lee el listado de `dc-events` una vez y, por θ, solo el primer y el
último row group de cada `events.parquet` y el carry-over (no los 100 GiB).
Cada borde es `pass`, `sin_eventos` (M+1 no cerró ningún evento, por ejemplo un
θ grande en un mes calmo: el pendiente sigue abierto y el carry-over debe
repetirlo) o `fail`. También marca un `fail` si falta un mes de la cadena. Los
bordes se comparan contra el último evento de **cualquier** mes anterior, así
que un mes sin eventos no oculta una costura rota. Desde Cloud Shell, con
pyarrow instalado. Con la landing completa son ~5.400 bordes y tarda ~13 min:
una sesión de Cloud Shell no es fiable para eso. El mismo chequeo está en
[`ops-scripts/seam_l2.py`](ops-scripts/seam_l2.py): súbelo y lánzalo con el job
`ops-script` ([runbook](operacion-ops.md)), que sigue sin ti y deja la salida en
el resumen del run. Así se ejecutó la primera vez.

```bash
EVENTS_ROOT=gs://<proyecto>-dc-events/l2 python3 - <<'PY'
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow.fs as pafs
import pyarrow.parquet as pq

root = os.environ["EVENTS_ROOT"]  # gs://<proyecto>-dc-events/l2, o una ruta local
asset = os.environ.get("ASSET", "BTCUSDT")
fs, base = pafs.FileSystem.from_uri(root)

EV = ["reference_price", "reference_time", "reference_agg_trade_id",
      "confirm_price", "confirm_time", "confirm_agg_trade_id",
      "extreme_price", "extreme_time", "extreme_agg_trade_id"]
PEND = ["pending_reference_price", "pending_reference_time", "pending_reference_agg_trade_id",
        "pending_confirm_price", "pending_confirm_time", "pending_confirm_agg_trade_id"]

# Un solo listado recursivo; un carry-over presente significa mes completo.
pat = re.compile(rf"asset={asset}/theta=(0\.\d{{8}})/year=(\d{{4}})/month=(\d{{2}})/(events|carry_over)\.parquet$")
found = {}
for info in fs.get_file_info(pafs.FileSelector(base, recursive=True)):
    m = pat.search(info.path)
    if m:
        found.setdefault(m[1], {}).setdefault(f"{m[2]}-{m[3]}", {})[m[4]] = info.path


def ends(path):
    """Primer y último evento del archivo (None, None si no tiene filas): lee solo
    el primer y el último row group no vacíos."""
    f = pq.ParquetFile(path, filesystem=fs)
    groups = [i for i in range(f.metadata.num_row_groups) if f.metadata.row_group(i).num_rows]
    if not groups:
        return None, None
    first = f.read_row_group(groups[0], columns=EV).slice(0, 1).to_pylist()[0]
    t = f.read_row_group(groups[-1], columns=EV)
    return first, t.slice(t.num_rows - 1, 1).to_pylist()[0]


def pending(path):
    c = pq.read_table(path, columns=["has_pending_event", *PEND], filesystem=fs).to_pylist()[0]
    return tuple(c[k] for k in PEND) if c["has_pending_event"] else None


def next_month(ym):
    y, m = int(ym[:4]), int(ym[5:])
    return f"{y + (m == 12):04d}-{m % 12 + 1:02d}"


def check(theta):
    months = sorted(mm for mm, files in found[theta].items() if "carry_over" in files)
    rows, last_extreme, prev = [], None, None  # prev = (mes, pendiente de su carry-over)
    for ym in months:
        files = found[theta][ym]
        first, last = ends(files["events"]) if "events" in files else (None, None)
        carry = pending(files["carry_over"])
        if prev is not None:
            p_ym, p_pend = prev
            if ym != next_month(p_ym):
                rows.append((theta, p_ym, ym, "fail", f"hueco: falta {next_month(p_ym)}"))
            elif first is None:
                ok = p_pend is None or carry == p_pend
                rows.append((theta, p_ym, ym, "sin_eventos" if ok else "fail",
                             "el pendiente sigue abierto" if ok else "sin eventos, pero el pendiente cambió"))
            else:
                errs = []
                head = tuple(first[k] for k in EV[:6])
                if p_pend is None:
                    if last_extreme is not None:
                        errs.append("sin pendiente en M, habiendo eventos antes")
                elif head != p_pend:
                    errs.append("el primer evento no es el pendiente del carry-over de M")
                if last_extreme is not None and tuple(first[k] for k in EV[:3]) != last_extreme:
                    errs.append("la referencia no es el extremo del último evento de M")
                rows.append((theta, p_ym, ym, "fail" if errs else "pass", "; ".join(errs)))
        if last is not None:
            last_extreme = tuple(last[k] for k in EV[6:])
        prev = (ym, carry)
    return theta, (months[0], months[-1], len(months)) if months else None, rows


with ThreadPoolExecutor(16) as pool:
    results = sorted(pool.map(check, sorted(found)))

print(f"{'theta':<12} {'meses':>5} {'rango':<17} {'pass':>5} {'sin_eventos':>11} {'fail':>5}")
tot = dict(pass_=0, sin_eventos=0, fail=0)
for theta, span, rows in results:
    n = {s: sum(r[3] == s for r in rows) for s in ("pass", "sin_eventos", "fail")}
    tot["pass_"] += n["pass"]; tot["sin_eventos"] += n["sin_eventos"]; tot["fail"] += n["fail"]
    rango = f"{span[0]}..{span[1]}" if span else "-"
    print(f"{theta:<12} {span[2] if span else 0:>5} {rango:<17} {n['pass']:>5} {n['sin_eventos']:>11} {n['fail']:>5}")
fails = [r for _, _, rows in results for r in rows if r[3] == "fail"]
for theta, a, b, _, why in fails:
    print(f"FAIL θ={theta} borde ({a}, {b}): {why}")
print(f"θ: {len(results)}; bordes: {tot['pass_'] + tot['sin_eventos'] + tot['fail']} "
      f"(pass {tot['pass_']}, sin_eventos {tot['sin_eventos']}, fail {tot['fail']})")
sys.exit(1 if fails else 0)
PY
```

Esperado sobre el rango completo: **50 θ**, cada uno con 109 meses
(`2017-08..2026-08`) y 108 bordes: **5 400 bordes, 0 `fail`**, y el código de
salida 0. Un θ con menos meses, o un rango distinto entre θ, es un backfill sin
terminar.

A diferencia de L1, aquí un `fail` **no es un hueco del proveedor**: los huecos
del origen ya se evaluaron en el seam-check de L1 y L2 los atraviesa tal cual.
La cadena de L2 es determinista, así que un borde roto es un defecto nuestro, o
un mes que quedó de una corrida anterior (el caso de las sondas, ver el
prerrequisito 5): un mes reprocesado en frío o con otra imagen rompe el borde
que lo une con el vecino. Si el `fail` es de un mes sobrante, relanza el
backfill desde ese mes con `force` (`from` = ese mes, `to` vacío) y vuelve a
correr el snippet. Si persiste, no lo repares a mano: abre una card con
`task-create`, con el θ, el borde y la salida del snippet.

## Costo real y volumen

Con la config de 4 vCPU y 4 GiB y `s` = pared de cada mes:
vCPU-s = 4 × Σ`s`; GiB-s = 4 × Σ`s`, sumando **todos** los meses procesados,
reintentos incluidos (un intento fallido también factura).

1. **Σ de paredes desde las líneas `sonda:`.** Con el nombre de la ejecución del
   paso 1 (también suma `ticks`, que se usa en el volumen):

   ```bash
   gcloud logging read 'resource.type="cloud_run_job"
     AND resource.labels.job_name="l2-backfill"
     AND labels."run.googleapis.com/execution_name"="<ejecución>"
     AND textPayload:"sonda: unit="' \
     --project <proyecto> --limit=1000 --format='value(textPayload)' |
     awk '{ for (i = 1; i <= NF; i++) {
              if ($i ~ /^wall_s=/)  { split($i, a, "="); s += a[2]; n++ }
              if ($i ~ /^ticks=/)   { split($i, a, "="); t += a[2] } }
            }
          END { printf "%d meses, Σ wall_s = %.1f s, Σ ticks = %d\n", n, s, t }'
   ```

   Filtrar por `execution_name` deja fuera las sondas y el reproceso de
   2020-01, que usan el mismo job y el mismo formato.
2. **La misma Σ desde el lago de DQ** (hallazgo `unit_timing`, `metric_value`
   = `wall_s`), por corrida, con GiB-s, vCPU-s y costo de lista (tarifas de
   [TRD-L2 §10.3](../TRD/l2.md#103-costo-de-l2): 0,000018 USD por vCPU-s y
   0,000002 USD por GiB-s). Sirve para contrastar con el log y, como
   `SINCE` acota el rango, para separar el backfill del reproceso:

   ```bash
   DQ_ROOT=gs://<proyecto>-dq-findings/l2 SINCE=<YYYY-MM-DD> python3 - <<'PY'
import os
from collections import defaultdict

import pyarrow.dataset as ds
import pyarrow.fs as pafs

root = os.environ["DQ_ROOT"]  # gs://<proyecto>-dq-findings/l2, o una ruta local
since = os.environ.get("SINCE")  # YYYY-MM-DD: salta las particiones detected_date anteriores
vcpu, gib = float(os.environ.get("VCPU", 4)), float(os.environ.get("GIB", 4))
VCPU_S_USD, GIB_S_USD = 0.000018, 0.000002  # lista, us-east1 (TRD-L2 §10.3)
fs, base = pafs.FileSystem.from_uri(root)

flt = ds.field("check_type") == "unit_timing"
if since:
    flt &= ds.field("detected_date") >= since
table = ds.dataset(base, filesystem=fs, partitioning="hive").to_table(
    filter=flt, columns=["run_id", "year", "month", "metric_value"]
)

runs = defaultdict(list)  # run_id -> [(pared, año, mes)]
for row in table.to_pylist():
    runs[row["run_id"]].append((row["metric_value"], row["year"], row["month"]))
total = 0.0
for run_id, rows in sorted(runs.items()):
    s = sum(r[0] for r in rows)
    total += s
    top = max(rows)
    print(f"{run_id}: {len(rows)} meses, Σ wall_s = {s:,.1f} s, GiB-s = {s * gib:,.0f}, "
          f"vCPU-s = {s * vcpu:,.0f}, máx. {top[0]:.1f} s ({top[1]}-{top[2]:02d})")
print(f"Total: Σ wall_s = {total:,.1f} s ({total / 3600:.2f} h), GiB-s = {total * gib:,.0f}, "
      f"vCPU-s = {total * vcpu:,.0f}, lista = {total * (vcpu * VCPU_S_USD + gib * GIB_S_USD):.2f} USD")
PY
   ```

3. **Pared de la tarea.** La suma de los meses deja fuera lo que no es un mes
   (arranque del contenedor, frontera, listados), pero Cloud Run factura la
   tarea entera:

   ```bash
   gcloud run jobs executions describe <ejecución> --region <región> \
     --format='value(status.startTime,status.completionTime)'
   ```

   Anota la diferencia: es la pared facturable. Si hubo reintento, el intento
   fallido también factura: suma su pared, que ves en *Cloud Run → Jobs →
   l2-backfill → ejecución → Tareas*.
4. **Contra el cupo gratis** mensual de 360 000 GiB-s y 180 000 vCPU-s. El cupo
   lo comparten todos los jobs del mes, L1 incluido: si el backfill cae en el
   mismo mes que uno de L1, queda menos. L1 consumió 185 846 GiB-s y
   46 462 vCPU-s en septiembre de 2026 ([resultados](operacion-l1.md)); un
   backfill de octubre empieza con el cupo limpio.
5. **Facturado según Billing:** *Facturación → Informes*, filtro por servicio
   `Cloud Run`, rango de las fechas de la ejecución, agrupado por SKU. Anota lo
   bruto y lo neto de crédito y de cupo gratis. Billing tarda hasta 24 h en
   reflejar el consumo.
6. **Contra [§10.3 del TRD-L2](../TRD/l2.md#103-costo-de-l2):** el techo del
   backfill son 80 268 vCPU-s y 80 268 GiB-s (109 × 736), 1,61 USD de lista y
   0,00 con cupo (45 % de los vCPU-s y 22 % de los GiB-s del cupo). Anota si la
   Σ real queda por debajo del techo (los meses son más livianos que 2023-03) y
   cuánto.

### Volumen de `dc-events`

Con el backfill terminado (y sin otro job escribiendo):

```bash
gcloud storage du -s -h gs://<proyecto>-dc-events/l2
gcloud storage du -s -h "gs://<proyecto>-dc-events/l2/provider=binance/market=spot/asset=BTCUSDT/theta=0.00010000"
```

(`gsutil du -s -h` da lo mismo.) El segundo comando mide el θ más pequeño, que
domina el volumen. Compáralo con la estimación de la fila "L2 (eventos, 50 θ)"
de [§7.1 del TRD maestro](../TRD/plataforma_directional_change.md): **73 a
103 GiB**, extrapolada con 30 MiB por millón de ticks (2020-01, ITSC-244). Dos
referencias para leer la diferencia: la partición de 2023-03 midió 735,47 MiB
(~3,9 MiB por millón de ticks, ITSC-281), y desde ITSC-290 los Parquet se
escriben sin diccionario y pesan ~40 % menos. Calcula MiB por millón de ticks
con la Σ de `ticks` del punto 1 (volumen en MiB ÷ Σ ticks en millones) y di si
el total real confirma la estimación, o la corrige, para que la fila de §7.1 se
actualice con el dato medido.

## Resultados

Ejecutados por el humano el 2026-10-02 y 2026-10-03 (UTC). Stack `l2` con
4 vCPU / 4 GiB e imagen `l2_dc_events:0.6.0` durante el backfill (hoy 0.7.1 por
ITSC-296 y ITSC-298: solo cambia el formato del log). `thetas.yaml` con los 50 θ
subido a `gs://intrinsica-dc-manifest/l2/` el 2026-10-02. El backfill no corrió
limpio a la primera: apareció un defecto en L1 (ver "Hallazgos").

**Runs**

| Paso | Ejecución | Rango / mes | Tareas OK / fallidas | Reintentos | Inicio → fin (UTC) |
| --- | --- | --- | --- | --- | --- |
| Backfill, 1.ª corrida | `l2-backfill-vszg7` | 2017-08 a 2017-11 OK; 2017-12 falló | 1 tarea fallida | 2 intentos en 2017-12 (37,6 s) | 2026-10-02 23:39 |
| Backfill, 2.ª corrida (inputs vacíos) | `l2-backfill-gkdlr` | 2017-12 a 2026-08 (105 meses) | 1 tarea, 0 fallidas | 0 | 2026-10-03 14:15:58 → 15:37:41 (4.903 s) |
| Reproducibilidad | `l2-backfill-h42bm` | 2020-01, `force` | OK | 0 | 2026-10-03 15:51:00 → 15:52:12 (71,5 s; `wall_s` del mes 46,7 s) |
| Seam-check | `ops-script-qdfpv` | 2017-08 a 2026-08 | OK | 0 | 2026-10-03 20:46:17 → 20:59:24 (13 min) |

No se reportaron las URLs de los runs de Actions; sí las ejecuciones de Cloud
Run. **Meses procesados:** 109 (4 en `vszg7` y 105 en `gkdlr`). **Hallazgos de
DQ distintos de `info`:** ninguno.

La tarea de `gkdlr` usó 4.903 s de los 36.000 s de timeout (14 %): el backfill
completo cabe holgado en una sola tarea.

**Reproducibilidad.** El snippet del paso 2 dio, con código de salida 0:

```text
mes 2020-01: 50 θ, 2 corridas: 0338134c-…, 815330c9-…; hashes idénticos en los θ
```

Los `content_hash` de `events.parquet` y `carry_over.parquet` coinciden en los
50 θ entre la corrida del backfill y el reproceso con `force`.

**Seam-check.** Corrió como el primer script real del job `ops-script`
(ITSC-298): `gs://intrinsica-dc-ops/scripts/seam_l2.py`, generation
`1791060231642929`, SHA-256 `85ea3a41…e9ab`. Salida, con código 0:

```text
θ: 50; bordes: 5400 (ok: 5400; fallos: 0)
```

50 θ × 108 bordes = 5.400. Ninguna costura rota en todo el rango.

**Costo**

| Tramo | Pared | vCPU-s (×4) | GiB-s (×4) |
| --- | --- | --- | --- |
| `gkdlr`, pared facturable de la tarea | 4.903 s | 19.612 | 19.612 |
| `vszg7`, 4 meses OK + 2 intentos fallidos (Σ de `wall_s`) | 116,1 s | 464 | 464 |
| **Total** | **≈ 5.019 s (1,39 h)** | **≈ 20.100** | **≈ 20.100** |

La Σ de `wall_s` de los meses es 4.878,4 s (105 meses de `gkdlr`) + 78,5 s +
37,6 s ≈ 4.995 s; la tarea de `gkdlr` factura 24,6 s más que sus meses
(arranque, frontera y listados).

- **Contra el cupo gratis mensual** (360.000 GiB-s, 180.000 vCPU-s): 11 % de los
  vCPU-s y 5,6 % de los GiB-s. A precio de lista, ≈ 0,40 USD; con cupo, 0,00.
- **Contra [§10.3 del TRD-L2](../TRD/l2.md#103-costo-de-l2)** (techo de 80.268
  vCPU-s y GiB-s): 25 % del techo. Los meses son mucho más livianos que 2023-03,
  el que fijó el techo.
- **Billing:** pendiente. Tarda hasta 24 h en reflejar el consumo; el humano lo
  anota en un comentario de la card cuando lo lea.
- **Costo fijo por mes.** Los meses de 2017 tardan 13 a 23 s con casi cero
  cómputo: el costo fijo ronda 20 s por mes, ~44 % de la pared del backfill.
  Está anotado en ITSC-293; no cambia el veredicto de costo.

**Volumen** (`gcloud storage du`):

| Medida | Valor |
| --- | --- |
| `dc-events/l2`, 50 θ | 21,14 GiB |
| θ = 0.00010000 (el que domina) | 2,68 GiB (12,7 % del total) |
| Σ ticks | 4.051.363.415 |
| MiB por millón de ticks | 5,3 |

Contra la estimación de §7.1 del TRD maestro (73 a 103 GiB, con 30 MiB por
millón de ticks): el real es 3,5 a 5 veces menor. La estimación extrapoló
2020-01 (ITSC-244), medido antes de que los Parquet se escribieran sin
diccionario (ITSC-290); con el dato medido, la fila de §7.1 se corrige a 21 GiB
y 5,3 MiB por millón de ticks.

**Hallazgos de la corrida** y la card que cubre cada uno:

| Hallazgo | Card |
| --- | --- |
| L1 conformaba tal cual las marcas de registros inválidos de Binance (`price=0`, `quantity=0`, trade ids `-1`) de los ZIP regenerados de 2017-12 y 2018-01; L2 falló con `prices[38419] no cumple 0 < price`. Se reprocesó L1 2017-12..2026-08 (`l1-backfill-m89q8`) antes de reanudar L2 | ITSC-294 (Hecha) |
| L1 trataba el 404 de un mes aún no publicado como error | ITSC-295 (Hecha) |
| Sin alerta para hallazgos con severidad ERROR | ITSC-296, ITSC-297 (Hechas) |
| Verificaciones largas en Cloud Shell (sesión inestable, lectura lenta) | ITSC-298, job `ops-script` (Hecha) |
| Costo fijo de ~20 s por mes | ITSC-293 |

La reanudación tras el fallo de 2017-12 usó el mecanismo del runbook sin
cambios: se relanzó `l2-backfill` con los inputs vacíos y la frontera de cada θ
siguió en 2017-12.

La entrada consolidada de la Épica la escribe `task-close.sh` al cerrarla.
