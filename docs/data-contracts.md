# Contratos de datos

Índice de los contratos de interfaz entre capas. El contenido de cada
sección lo escriben E1 y E2; por ahora cada una solo enlaza al TRD.

## Salida Parquet de L1

Contrato hacia L2. Ver [TRD-L1 §7.2](TRD/l1.md#72-salida--parquet-conformado-de-l1-contrato-hacia-l2).

## Lago de hallazgos de calidad de datos

Contrato del lago donde toda capa deja sus hallazgos de calidad de datos (DQ).
Fuente de diseño: [TRD-L1 §7.3](TRD/l1.md#73-lago-de-hallazgos-de-calidad-de-datos).
Fuente en código: `FINDING_SCHEMA` en
[`shared/dq/src/dq/schema.py`](../shared/dq/src/dq/schema.py). Las tres
listas de columnas (TRD, código y esta tabla) van en el mismo orden; una
prueba (`shared/dq/tests/test_contract_doc.py`) rompe el CI si esta tabla se
desvía del código.

### Esquema

| Columna | Tipo Arrow/Parquet | Nulable | Descripción |
|---|---|---|---|
| `finding_id` | string (UUID) | no | Identidad del hallazgo; se repite en todos los eventos del mismo hallazgo |
| `detected_at` | int64 | no | Marca del evento, en microsegundos desde la época (UTC) |
| `layer` | string | no | Capa que emite el hallazgo, p. ej. `l1` |
| `mode` | string | no | Modo de ejecución: `backfill`, `daily` o `monthly-close` |
| `check_type` | string | no | Chequeo que lo generó (ver TRD-L1 §9) |
| `severity` | string | no | `info`, `warning` o `error` |
| `stage` | string | no | `provisional` o `canonical` |
| `status` | string | no | `pass`, `fail` o `corrected` |
| `provider` | string | no | Proveedor del dato afectado, p. ej. `binance` |
| `market` | string | no | Mercado del dato afectado, p. ej. `spot` |
| `asset` | string | no | Activo del dato afectado, p. ej. `BTCUSDT` |
| `year` | int32 | no | Año del dato afectado |
| `month` | int32 | no | Mes del dato afectado (1 a 12) |
| `metric_value` | float64 | sí | Medida del chequeo, p. ej. número de huecos; nulo si no aplica |
| `details` | string (JSON) | no | Payload libre de cada chequeo, como texto JSON |
| `run_id` | string | no | Ejecución que emitió el hallazgo |
| `image_version` | string | no | semver + git SHA de la imagen que lo emitió |

Por qué estas decisiones, escritas una sola vez:

- **`year` y `month` son `int32`**: son números pequeños y `int32` alcanza de
  sobra; con `int64` cada valor ocupa el doble sin ganar nada.
- **La partición se llama `detected_date=`**, no `year=` ni `month=`: esas dos
  son columnas del esquema (el período del dato afectado). Si la partición
  usara los mismos nombres, DuckDB y Arrow verían la columna duplicada y
  mezclarían dos ideas distintas: cuándo se detectó el hallazgo y a qué mes
  del dato se refiere.

### Disposición física

- **Raíz**: una ruta local o `gs://<project_id>-dq-findings`.
- **Partición**: `<raíz>/detected_date=YYYY-MM-DD/`, con la fecha UTC de
  `detected_at`.
- **Archivo**: `<run_id>-<uuid>.parquet`. El nombre es único por llamada, así
  que una emisión nunca sobrescribe otra.
- **Formato**: Parquet con compresión ZSTD nivel 3 y estadísticas de columna.
- **Append-only**: un hallazgo nunca se actualiza. Un cambio de estado es un
  evento nuevo con el mismo `finding_id` y un `detected_at` posterior.

### Emitir

```python
from dq import Finding, emit_findings

finding = Finding(detected_at=..., layer="l1", ...)  # campos del esquema
emit_findings([finding], "gs://<project_id>-dq-findings")
```

En GCS usa Application Default Credentials; no hay credenciales en código.

### Estado actual de un hallazgo

El estado actual es el último evento por `finding_id`
([ADR-L1-08](TRD/l1.md#68-adr-l1-08--hallazgos-append-only-con-event-sourcing-sin-motor-transaccional)).
La consulta de referencia es `CURRENT_FINDINGS_SQL` en `dq.reader`
(requiere el extra `dq[reader]`):

```sql
select * exclude (rn) from (
    select
        *,
        row_number() over (
            partition by finding_id order by detected_at desc
        ) as rn
    from read_parquet($pattern, hive_partitioning = true)
)
where rn = 1
```

Sobre una raíz local: `dq.reader.current_findings(root)`.

Política provisional a canonical: `daily` emite el hallazgo con
`stage = provisional`; `monthly-close` lo revalida y emite un evento nuevo con
el mismo `finding_id` y `stage = canonical`. Como el canónico tiene un
`detected_at` mayor, la consulta lo devuelve y el provisional queda como
historia.

Sobre `gs://`, `current_findings` todavía no acepta esa raíz. Hay dos
caminos:

- **DuckDB**: instala y carga la extensión `httpfs` y crea un secreto HMAC de
  GCS (`CREATE SECRET (TYPE gcs, KEY_ID ..., SECRET ...)`). A diferencia de
  `emit_findings` y del camino pyarrow, que usan Application Default
  Credentials, `httpfs` no las acepta. Luego ejecuta `CURRENT_FINDINGS_SQL` con
  `$pattern = "gs://<project_id>-dq-findings/detected_date=*/*.parquet"`.
- **pyarrow**: lee con
  `pyarrow.dataset.dataset("<project_id>-dq-findings", filesystem=GcsFileSystem(),
  partitioning="hive")` y registra el resultado en DuckDB con
  `con.register("findings", ds)`. Con `filesystem` explícito la ruta va sin
  `gs://` (`GcsFileSystem` rechaza URIs); si omites `filesystem`, pasa la URI
  `gs://<project_id>-dq-findings` y pyarrow resuelve el sistema de archivos. La
  consulta es la misma, pero cambia `read_parquet($pattern, hive_partitioning =
  true)` por `findings`.

## Manifiesto de checksums

Ver [TRD-L1 §7.4](TRD/l1.md#74-manifiesto-de-checksums).
