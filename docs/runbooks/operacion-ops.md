# Runbook: operación ad hoc con el job `ops-script`

`ops-script` corre un script Python que tú subes a GCS, dentro de Cloud Run y
cerca del lago, lanzado y monitoreado desde *Actions → Run job*. Reemplaza las
verificaciones que antes se corrían en Cloud Shell (escaneo de pies Parquet,
seam-check, comparación de hashes): allí la sesión se cae, la VM se apaga si te
vas, la lectura va a ~2 MiB/s y la salida muere en un `.log` de una VM efímera.
Aquí la ejecución sigue sin ti, lee a velocidad de región y deja la salida en el
resumen del run. Lo lanza y aprueba el humano; ningún agente dispara
`run-job.yml` (regla del proyecto).

No es una capa de datos: no escribe en el lago y no tiene scheduler.

## Los tres pasos

1. **Sube el script** desde Cloud Shell. Los scripts de este runbook están en el
   repo (`docs/runbooks/ops-scripts/`); con el checkout al día:

   ```bash
   git pull
   gcloud storage cp docs/runbooks/ops-scripts/seam_l2.py \
     gs://<proyecto>-ops/scripts/seam_l2.py
   ```

   Para uno nuevo, sin repo de por medio:

   ```bash
   cat > mi_script.py <<'EOF'
   print("hola desde ops-script")
   EOF
   gcloud storage cp mi_script.py gs://<proyecto>-ops/scripts/mi_script.py
   ```

   La ruta es fija y se sobrescribe: subir de nuevo el mismo nombre reemplaza al
   anterior (el bucket conserva las versiones, ver "Trazabilidad").

2. **Lánzalo.** *Actions → Run job* (rama `main`):

   | Input | Valor |
   | --- | --- |
   | `job` | `ops-script` |
   | `script` | `gs://<proyecto>-ops/scripts/seam_l2.py` (obligatorio) |
   | `args` | texto libre, opcional; se separa como en un shell: `--asset BTCUSDT --umbral 0` |
   | `from`, `to`, `force`, `series_start` | vacíos: `ops-script` no los admite |

3. **Aprueba** el environment `gcp` y vuelve cuando el resumen esté listo. Si
   `script` falta o no es un `gs://<bucket>/<objeto>`, el run falla en el primer
   paso, antes de gastar una ejecución.

## Qué ves en el resumen del run

Bajo la tabla de la ejecución (estado, tareas, inicio y fin) y la de hallazgos
de calidad de datos, la sección **Script**:

- **URI, generation y SHA-256** del objeto que se ejecutó. La generation
  identifica la versión exacta (ver "Trazabilidad").
- **Resultados**: la carpeta `gs://<proyecto>-ops/results/<ejecución>/` de esa
  ejecución.
- **Código de salida** del script. Un script que falla (`sys.exit(1)`, una
  excepción) deja la ejecución en rojo y aquí aparece su código. Si dice "sin
  código", el job murió antes de que el script terminara: timeout (3600 s) u
  OOM (8 GiB); el log lo dice.
- **Contenido del script**, en un bloque plegable (las primeras 300 líneas).
- **Salida del script**: las últimas 40 líneas, con el traceback si falló.

Si el job no llegó a ejecutar el script (el objeto no existe, la cuenta no puede
leerlo), la sección muestra las últimas líneas del log completo: ahí está el
motivo (`404` o `403`).

El log completo de la ejecución sigue en el paso *Logs de la ejecución*. Regla
para decidir adónde va una salida:

| Qué | Dónde |
| --- | --- |
| Una conclusión, una tabla corta, la línea final | `print`: va al resumen (las últimas 40 líneas) |
| Un detalle por fila de miles de filas, un Parquet, un CSV | `OPS_RESULTS_URI`: carpeta de la ejecución en `results/`, se borra a los 7 días |

Lo que quede en `results/` lo lees desde Cloud Shell:
`gcloud storage cat gs://<proyecto>-ops/results/<ejecución>/<archivo>`.

## Cómo escribir un script

- El entrypoint lo ejecuta como `python script.py <args>`; los args llegan en
  `sys.argv`. El código de salida del proceso es el de la ejecución.
- Variable `OPS_RESULTS_URI`: `gs://<proyecto>-ops/results/<ejecución>/`, para
  salidas grandes. Termina en `/`.
- La imagen trae Python 3.14, `pyarrow` (lee `gs://` con
  `pyarrow.fs.FileSystem.from_uri`), `google-cloud-storage`, `duckdb`, `dq`
  (esquema y lector de hallazgos), `dc_frames` (lector de tramas) y `pyutils`. Una librería nueva es una card
  sobre `layers/ops_tools/`, no un `pip install` dentro del script.
- Lee el lago con prefijos `l1/` o `l2/` (ver "Permisos").
- Escribe en `results/` con `google-cloud-storage`
  (`client.bucket(b).blob(path).upload_from_string(...)`). Solo se pueden crear
  objetos nuevos: no se pisa ni se borra lo ya escrito.
- Un script de más de 1 MiB se rechaza: no es un script.

## Permisos de la service account `ops-script`

Rol `roles/storage.objectViewer` (lectura y listado) con condición por prefijo,
y `roles/storage.objectCreator` (solo crear) en `results/`. Sin roles de IAM, sin
`run.jobs.update`: el script no puede darse permisos ni cambiar el job.

| Bucket | Prefijo | Acceso |
| --- | --- | --- |
| `<proyecto>-landing` | `l1/` | lectura |
| `<proyecto>-dc-events` | `l2/` | lectura |
| `<proyecto>-dq-findings` | `l1/`, `l2/` | lectura |
| `<proyecto>-manifest` | `l1/`, `l2/` | lectura |
| `<proyecto>-viz` | `tiles/` | lectura |
| `<proyecto>-ops` | `scripts/` | lectura |
| `<proyecto>-ops` | `results/` | solo crear objetos nuevos |

Cualquier otra escritura (el lago, `tiles/`, `scripts/`) falla con 403. El script corre
con esta identidad y tiene la imagen entera a su disposición: el riesgo
(ejecutar código arbitrario) lo acotan estos permisos, los límites de abajo, la
aprobación del environment `gcp` y que ningún agente dispara el workflow.

### Prueba negativa: escribir fuera de `results/` da 403

`permisos.py` lo comprueba dentro del job, con la identidad real: puede leer
landing, dc-events y `tiles/` de viz, puede crear en `results/`, y recibe 403 al
pisar un objeto de `results/`, escribir en `scripts/` o en cualquiera de los cinco
buckets del lago (landing, dc-events, dq-findings, manifest y viz).

```bash
gcloud storage cp docs/runbooks/ops-scripts/permisos.py gs://<proyecto>-ops/scripts/permisos.py
```

Lánzalo con `script` = ese URI y sin `args`. Esperas la línea final
`permisos: 11 de 11 como se esperaba` y el código de salida 0. Una línea `FALLO`
dice qué permiso no es el esperado; si una escritura que debía dar 403 tuvo
éxito, el script deja un objeto `ops-permisos-*` en ese bucket: bórralo y
corrige el IAM antes de seguir. Se corre una vez tras el primer apply y cada vez
que cambie el IAM del stack `ops`.

## Límites

| | |
| --- | --- |
| CPU | 4 vCPU |
| Memoria | 8 GiB |
| Tiempo | 3600 s (1 h) por ejecución; al vencer, el proceso muere sin código |
| Tareas | 1 |
| Reintentos | 0: un script ad hoc no se reintenta solo |
| Script | 1 MiB |
| `results/` | todo se borra a los 7 días |

Si un script necesita más, no es ad hoc: es código de una capa, con su sonda.

## Trazabilidad sin git

El script se sube a una ruta fija y se sobrescribe; lo que fija qué corrió es:

- El bucket tiene **versionado**: cada subida es una *generation* nueva y las
  versiones no vigentes de `scripts/` se conservan 90 días. La vigente no vence.
- El job imprime URI, generation, SHA-256 y el contenido completo antes de
  ejecutar, y el resumen los repite. Descarga el objeto atado a esa generation:
  si alguien lo sobrescribe a mitad, falla en vez de ejecutar otro contenido.

Para ver o recuperar una versión anterior:

```bash
gcloud storage ls -a gs://<proyecto>-ops/scripts/seam_l2.py       # lista las generations
gcloud storage cp 'gs://<proyecto>-ops/scripts/seam_l2.py#<generation>' viejo.py
```

## Regla de graduación

Un script ad hoc es para algo que corres una o dos veces. **Lo que se corre más
de dos veces se gradúa**: pasa a este runbook (como los de abajo) o al código de
la capa, con una card. Si lo corres a menudo, no debería depender de que alguien
suba un archivo a mano.

## Los primeros scripts

Los cuatro están en `docs/runbooks/ops-scripts/` y una prueba
(`tests/test_ops_scripts.py`) los corre contra un lago que escribe L2 de verdad,
con casos en que deben fallar. Todos toman las raíces del proyecto del bucket de
resultados; los args solo las cambian.

### `pies_parquet.py`: escaneo de pies Parquet por mínimo de `price`

Lee solo el pie (footer) de cada `consolidated.parquet` de la landing, no los
datos, y toma `min(price)` de las estadísticas de los row groups. Caza marcas
inválidas como `price = 0` ([ITSC-294](https://github.com/byroncz/intrinsica/pulls?q=ITSC-294)) en
109 meses en segundos.

- Args: `--landing-root`, `--provider`, `--market`, `--asset`, `--umbral`
  (por defecto 0: un mes con `min(price) <= umbral` es un fallo).
- Línea final: `meses: 109; min(price): <X> (<AAAA-MM>); con min <= 0: 0`.
- Sale con 1 si algún mes está en fallo o no trae estadísticas.

### `seam_l2.py`: seam-check de L2

Comprueba la costura de cada par de meses consecutivos de cada θ. L2 encadena
los meses por carry-over: el evento que queda pendiente al cierre de M viaja en
su `carry_over.parquet` y se escribe en el mes que confirma el evento siguiente
([TRD-L2 §6.6, ADR-L2-06](../TRD/l2.md)).
Por cada borde (M, M+1) verifica:

1. Existen el carry-over de M y el de M+1 (`falta_carry_over`).
2. Si M cerró con un evento pendiente: su referencia es el extremo del último
   evento de M (`cadena_rota`); el primer evento de M+1 es ese pendiente, con la
   misma referencia y confirmación (`pendiente_no_coincide`); y si M+1 no
   confirmó nada, su carry-over lo conserva (`pendiente_perdido`).

Lee de cada `events.parquet` solo la primera y la última fila, así que los 109
meses × 50 θ (5 400 bordes) salen en minutos.

- Args: `--events-root`, `--provider`, `--market`, `--asset`.
- Línea final: `θ: 50; bordes: 5400 (ok: 5400; fallos: 0)`. Imprime hasta 50
  líneas `FALLO θ=... AAAA-MM -> AAAA-MM: <motivo>` antes.
- Sale con 1 si hay algún fallo.

Este script formaliza la comprobación en este repo y se probó contra la salida
real de L2; si tienes una versión anterior en Cloud Shell, compara sus bordes
con estos.

### `hashes_events_summary.py`: comparación de hashes de `events_summary`

L2 es determinista: dos corridas del mismo mes deben dejar, por θ, los mismos
eventos y los mismos `content_hash` de `events.parquet` y `carry_over.parquet`
([sonda de L2](sonda-l2.md)). Cada corrida emite un hallazgo `events_summary`
por θ y mes; el script los agrupa por (θ, mes) y compara lo que dejó cada
`run_id`.

- Args: `--dq-root`, `--month AAAA-MM` (por defecto todos), `--asset`.
- Línea final: `θ: 50; unidades: <n> (con >= 2 corridas: <k>; discrepancias: 0)`.
- Sale con 1 si hay discrepancias, o si ninguna unidad tiene dos corridas
  (no se comprobó nada).

### `permisos.py`: la prueba negativa de 403

Descrita arriba, en "Permisos".

### Sondas del lector de tramas de L3 (ITSC-331)

Dos scripts de `layers/ops_tools/scripts/` que miden el criterio de aceptación del
lector `shared/dc_frames` ([TRD-L3 §10.3, ADR-L3-09](../TRD/l3.md)). La imagen trae
`dc_frames` desde `ops_tools` 0.2.0: sin esa versión desplegada, el `import` falla.
Súbelos como cualquier script (`gcloud storage cp layers/ops_tools/scripts/<script>.py
gs://<proyecto>-ops/scripts/`) y lánzalos con `job` = `ops-script`. Ambos solo leen y
toman las raíces de L1 y L2 del proyecto desde `OPS_RESULTS_URI`.

- **`l3_probe_reader.py`**: recorre `read_frames` y mide pared por fase (lectura,
  decodificación, selección), RSS pico, bytes leídos, row groups decodificados y
  ticks/s, con eventos y ticks por fase de cada θ, y lo compara con los umbrales de
  §10.3. Corre una pasada por ejecución (el RSS pico es del proceso), así que cada
  medición es un run:

  | Medición | `args` |
  | --- | --- |
  | Un θ (`θ_min`, uno intermedio y `θ_max`: tres runs) | `--theta 0.00100000 --from 2023-03` |
  | Los 50 θ en una pasada | `--theta all --from 2023-03` |

  Última línea: `sonda lector: … veredicto=ok|excedido|fallo`; sale con 1 si no es `ok`.
  La tabla por θ queda también en `results/` como `l3_probe_reader_<fecha>.csv`.
  Los pies de la salida (`pared por fase`, `bytes leídos`, `RSS pico`) son lo que se
  copia a la card y al runbook de L3. La sonda envuelve los archivos para contar bytes,
  lo que serializa las lecturas de un row group: si la pared queda cerca del umbral,
  repite con `--sin-contar-io` para la pared del lector sin envolver.
- **`l3_probe_ties.py`**: cuántos `transact_time` distintos hay en un milisegundo de L1
  y, para cada θ que confirma en él, el tamaño de su grupo de empate (ticks con el
  `transact_time` de la confirmación, cuántos hasta `C` y cuántos después).
  `args` = vacío para el 2026-09-30 12:40:26.980 (`--at` cambia el milisegundo).
  Última línea: `sonda empates: ms=… ticks=… distintos=… max_por_tt=… thetas=…
  max_empate=…`.

## Prueba de extremo a extremo (criterios de ITSC-298)

La hace el humano una vez que `ops_tools` está publicada y el stack aplicado:

1. `seam_l2.py` desde Actions: el resumen trae URI, generation, SHA-256, el
   contenido y la línea final `θ: 50; bordes: 5400 (ok: 5400; fallos: 0)`. Debe
   terminar en minutos, no en horas.
2. `permisos.py`: `permisos: 11 de 11 como se esperaba`.
3. Un script de una línea, `import sys; sys.exit(1)`: la ejecución queda en rojo
   y el resumen muestra `Código de salida | 1`.

## Puesta en marcha

Orden, todo por el humano (detalle en
[infra/README.md](../../infra/README.md#stack-ops-scripts-de-operación)):

1. Aplicar `data` desde Cloud Shell: crea el bucket `<proyecto>-ops` y da a
   `deploy-github` el permiso de IAM sobre él. Hazlo con el checkout de la
   rama del PR, antes de aprobar: hasta entonces el check `stack (ops)` está en
   rojo (el plan no encuentra el bucket `ops` en el estado de `data`). Tras el
   apply, re-ejecuta `stack (ops)`; su plan es la evidencia de lo que `ops`
   crea.
2. Esperar a que el run de CI en `main` publique `ops_tools:<versión>`.
3. *Actions → Terraform* → `ops` → `apply`.
4. Subir los scripts y lanzar la prueba de extremo a extremo.
