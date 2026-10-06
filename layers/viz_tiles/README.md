# viz_tiles

Capa viz: codifica un día de ticks de L1 y de eventos de L2 en dos archivos,
`ticks.bin` (todos los ticks, sin reducir) y `events.bin` (los eventos exactos de
cada θ), y los lleva dentro de una página que dibuja lo que mide L1: un tick o la
envolvente exacta de los ticks de un píxel. Diseño en
[`docs/TRD/viz.md`](../../docs/TRD/viz.md); el contrato de los archivos, en
[`docs/data-contracts.md`](../../docs/data-contracts.md) ("Tiles de viz"). Una
prueba rompe el CI si ese contrato se desvía del código.

La imagen corre `python -m viz_tiles --mode tiles`: lee un mes de L1 y los
eventos de L2 (en disco o en `gs://`), codifica los archivos de cada día pedido y
los escribe junto a una página `index.html` autocontenida del día: la plantilla de
[`site/`](site/), uPlot y los dos archivos en base64 dentro de un solo documento,
sin peticiones de red (el porqué, en [TRD-viz §6.5](../../docs/TRD/viz.md#65-adr-vz-05--entrega-solo-bucket-sin-servidor)).
`--mode render` vuelve a armar esas páginas desde los archivos ya escritos, sin leer
L1 ni L2, cuando cambia la plantilla.

## Uso en local

```bash
export VIZ_LANDING_ROOT=$PWD/sandbox.local/landing   # raíz de L1
export VIZ_EVENTS_ROOT=$PWD/sandbox.local/events     # raíz de L2
export VIZ_TILES_ROOT=$PWD/sandbox.local/tiles       # salida
export VIZ_DQ_ROOT=$PWD/sandbox.local/dq             # hallazgos

uv run python -m viz_tiles --mode tiles --day 2017-08-18      # un día
uv run python -m viz_tiles --mode tiles --from 2017-08        # un mes
uv run python -m viz_tiles --mode tiles --from 2017-01 --to 2017-08
uv run python -m viz_tiles --mode tiles                       # mes anterior
uv run python -m viz_tiles --mode tiles --day 2017-08-18 --force

uv run python -m viz_tiles --mode render --day 2017-08-18     # solo la página
uv run python -m viz_tiles --mode render --from 2017-01 --to 2017-08
```

En modo `tiles` las cuatro variables son obligatorias (raíz local o `gs://`); en
modo `render` bastan `VIZ_TILES_ROOT` y `VIZ_DQ_ROOT`. Si falta una, o un
argumento es inválido, el proceso sale con código 2. `--mode` acepta `tiles` y
`render`.

| Argumento | Efecto |
|---|---|
| `--day YYYY-MM-DD` | Un solo día. No se combina con `--from` ni `--to`. |
| `--from YYYY-MM` / `--to YYYY-MM` | Todos los días de cada mes del rango; `--to` es `--from` por defecto. |
| (ninguno) | El mes anterior al actual (UTC) y la revisión hacia atrás de los meses con cola provisional (abajo). |
| `--force` | Regenera lo seleccionado aunque el `input_hash` (tiles) o la huella de la plantilla (render) no haya cambiado. |
| `--asset` | Activo; por defecto `BTCUSDT`. Debe tener `price_scale` en `contract.py`. |

Un mes completo pide los días que L1 cubre, del primer al último tick del
archivo (agosto de 2017 arranca el 17). Un día sin ticks dentro de ese rango es un
hueco: deja `input_missing` con `what = "ticks"`.

Códigos de salida: `0` éxito (incluye "todo al día"), `1` algún hallazgo `error`
o una entrada faltante, `2` error de uso. Un rango sigue con el mes siguiente si
uno falla, y termina con 1.

## Cómo decide qué rehacer

El `input_hash` de un día es el SHA-256 de un texto con `tiles_version`, el día y
una línea `ruta<TAB>tamaño<TAB>CRC32C` por archivo de entrada, ordenadas por
ruta (rutas relativas a `VIZ_LANDING_ROOT` o `VIZ_EVENTS_ROOT`). El tamaño y el
CRC32C salen de los metadatos del objeto en GCS, sin descargarlo; en disco se
calculan sobre el archivo. Si el `index.json` del día trae el mismo `input_hash` y
`tiles_version`, el día se salta y no se escribe nada; no se lee ningún tick.

Los archivos de entrada del día son el `consolidated.parquet` del mes y el
`events.parquet` y `carry_over.parquet` de cada θ. Cada `events.parquet` se lee
una vez por mes, no una por día. Los θ son los que L2 tiene en
el mes; uno sin `events.parquet` o sin `carry_over.parquet` va a
`missing_thetas`, deja `input_missing` y el día se escribe con los demás.

**Cola provisional.** Si un θ tiene un evento pendiente al cierre del mes, sus
ticks tras la confirmación se dibujan con el extremo vigente del carry-over
([TRD-viz §7.6](../../docs/TRD/viz.md#76-de-dónde-salen-los-eventos-de-un-día)).
Mientras la cadena de carry-overs de los meses siguientes siga abierta, ese
extremo es un candidato; cuando L2 cierra el evento, el definitivo sale del
`events.parquet` de ese mes. Desde el día del candidato del propio mes, el
`input_hash` suma los archivos de esa cadena, así que un mes nuevo de L2 rehace
esos días. Se mide con el candidato del carry-over del mes y no con el de la última
carry-over de la cadena: si la primera ya mueve el candidato a un mes posterior,
los eventos del día cambian (la cola deja de ser provisional) y con el candidato último el
hash seguiría igual.

**Revisión hacia atrás.** Sin `--day` ni `--from`, tras el mes anterior se revisan
los meses previos: desde `N−1`, si el `index.json` del último día trae algún
`provisional_from_s`, se retrocede por los días del mes mientras los encuentre
provisionales y se pasa al mes anterior. Se detiene en el primer mes cuyo último día
es definitivo para todos los θ. La idempotencia salta lo que no cambió.

## La página del día

Un día trae `index.html` junto a `ticks.bin` y `events.bin`, y el último día se copia también a
`latest.html` en la raíz (`tiles/latest.html`). Es un solo documento: la plantilla
(`site/index.html`, `app.js`, `style.css`), uPlot y los dos archivos del día, en
base64 bajo su nombre, en `window.VIZ_DATA`. Se abre igual desde disco (`file://`),
desde `python -m http.server` o desde `storage.cloud.google.com`; no hace
peticiones de red después de cargar. Al abrir, el navegador decodifica los ticks a
arreglos tipados una sola vez; las métricas (ticks decodificados en … ms, primer
trazo, redibujo, cambio de θ) van a la consola.

- **En un bucket** la página se guarda comprimida (`Content-Encoding: gzip`,
  `Content-Type: text/html; charset=utf-8`, `Cache-Control: no-cache`). **En disco
  local** va sin comprimir, para que abra por `file://`.
- El `index.json` la lista en `page`, y se escribe después de ella: un índice
  implica su página.
- Los demás objetos llevan sus metadatos al escribirse: binarios
  `application/octet-stream` con `Cache-Control: private, max-age=31536000,
  immutable`; `index.json` y `latest.json`, `application/json` con `no-cache`.
- La página guarda en un `<meta name="viz-render">` la `tiles_version` y el
  SHA-256 de la plantilla. `--mode render` salta los días cuya página ya los trae
  iguales y regenera el resto desde los archivos del directorio; si no coinciden
  con el `content_hash` del índice, no escribe nada y deja `input_missing`.

La plantilla va en la imagen (`VIZ_TEMPLATE_DIR=/app/site`); en el repo se lee
de `layers/viz_tiles/site/`.

### Compartir un día

Compartir un día es compartir su `index.html`; no hay exportación aparte (la de
zip se descartó: la página ya es el archivo exportable). Hay dos caminos:

1. **Abrirlo en el bucket**, con la cuenta de Google del visor (`VIZ_VIEWER`):
   `https://storage.cloud.google.com/<bucket>/tiles/provider=binance/market=spot/asset=BTCUSDT/day=YYYY-MM-DD/index.html`,
   o `.../tiles/latest.html` para el último día escrito.
2. **Descargarlo y enviarlo**:

   ```bash
   gcloud storage cp \
     "gs://<bucket>/tiles/provider=binance/market=spot/asset=BTCUSDT/day=YYYY-MM-DD/index.html" .
   ```

   El archivo descomprimido abre con doble clic en un navegador limpio, sin
   servidor ni red: lo prueba
   `test_downloaded_gzip_page_of_the_smoke_day_is_self_contained`.

`<bucket>` es `<proyecto>-viz`. En el bucket la página está en gzip
(`Content-Encoding: gzip`). Si lo descargado empieza con los bytes `1f 8b` (el
navegador no lo abre), `gunzip -c index.html > dia.html` lo deja listo para
enviar. Qué entrega cada herramienta (descomprimido o no) está sin verificar
contra el bucket real: la primera descarga del humano lo confirma.

### Re-render del histórico

Cuando cambia la plantilla (`site/`: HTML, JS, CSS o uPlot) se sube `VERSION`, se
despliega la imagen y se corre el job `viz-render` (*Actions → Run job*, ver
[infra/README.md](../../infra/README.md#stack-viz-tiles-de-visualización)) sobre
el rango que se quiera rehacer. Lee solo `ticks.bin` y `events.bin` del bucket, sin
L1 ni L2: un re-render completo son unas 3 300 lecturas de `index.json` y de dos
archivos por día y otras tantas escrituras, minutos de cómputo y centavos de operaciones,
frente a repetir el backfill desde L1 y L2. Esa diferencia es la razón de
conservar `ticks.bin` y `events.bin` como artefactos separados.

Cada día deja un `render_summary` (`skipped`, hash de la plantilla, bytes de la
página guardados y descomprimidos). En un rango, los días sin `index.json` se
omiten; `input_missing` (`what = tiles`, código 1) sale si un mes no tiene ningún
día con archivos o si un día tiene archivos rotos. Un día suelto sin `index.json`
solo se detecta con `--day`, que `run-job.yml` no expone. Un día cuya página ya
trae la misma `tiles_version` y la misma huella de plantilla se salta, salvo con
`force`. Si el rango incluye el último día, `latest.html` se actualiza con él.

### uPlot

`site/vendor/uPlot.iife.min.js` y `uPlot.min.css` son uPlot 1.6.32, de
`github.com/leeoniya/uPlot` (`dist/` de la etiqueta `1.6.32`), con licencia MIT
(© Leon Sorokin; el texto está en `site/vendor/LICENSE`). Son archivos de texto.
Para subir la versión se reemplazan los dos archivos y su `LICENSE`, se corrige
esta sección y la prueba `test_uplot_is_pinned_and_licensed`, y se sube `VERSION`:
la huella de la plantilla cambia sola y `--mode render` rehace las páginas.

### Vista

Tres paneles con eje X y cursor compartidos (precio 65 %, confirmaciones 15 %,
volumen 20 %), franja de estado fija (día, θ, navegación por eventos con «evento
k / n», última actualización, datos completos o incompletos) y tooltip por píxel.
Todo se deriva por píxel de ancho, al dibujar, de los ticks y los eventos ya
decodificados ([TRD-viz §7.4](../../docs/TRD/viz.md#74-lo-que-el-navegador-deriva-por-píxel)):

- **Precio**, nunca velas: un píxel con 1 o 2 ticks es un punto por tick; con más,
  el segmento de su mínimo a su máximo y nada más. Nada une un píxel con el
  vecino. Los ticks de un mismo milisegundo comparten píxel: el tooltip dice «n
  ticks en este ms».
- **Volumen**: una barra por píxel con la suma de la cantidad; rótulo «Volumen».
- **Confirmaciones**: barra con los θ que confirman en el píxel y marca intensa
  con el máximo de θ que confirman en el mismo instante; el tooltip lista cuáles y
  a qué hora; rótulo «θ que confirman».
- **Franjas del θ activo**, dibujadas desde `events.bin` en los instantes exactos
  de cada evento a cualquier zoom: confirmación tenue y fina, overshoot intenso y
  grueso, y una línea del color del evento en cada confirmación (no hay línea de
  extremo: el cambio de color entre franjas ya lo marca). Donde varios eventos
  enteros caen en un mismo píxel, una marca gris con el número («4 eventos»);
  `DENSE_PX` fija el ancho de ese píxel.
- **Navegación**: «Evento anterior» y «Evento siguiente» desplazan la vista a la
  ventana `[referencia(k−1), extremo(k+1)]` sin cambiar la escala; «Ajustar a la
  ventana» pone la escala en esa ventana con 5 % de margen. La referencia, la
  confirmación, el extremo, la ventana y las notas (recorte, provisional) van al
  tooltip al pasar el mouse sobre el evento.
- **Leyenda** con muestras dibujadas como en el gráfico, sin frases.

Los principios y la Evaluación ergonómica que cada cambio de vista debe traer
están en
[TRD-viz §6.7](../../docs/TRD/viz.md#67-adr-vz-07--nueve-principios-de-ergonomía-y-evaluación-ergonómica-obligatoria).

## Hallazgos

Con `layer = viz`, `mode = tiles` o `render`, `stage = canonical` y el día en `details.day`
([TRD-viz §9.3](../../docs/TRD/viz.md#93-tipos-de-chequeo-check_type)). Se emiten
en una llamada por mes procesado: una línea JSON por hallazgo en el log (la que
resume `run-job.yml`) y las filas en `VIZ_DQ_ROOT`.

| `check_type` | `severity` | Cuándo |
|---|---|---|
| `tiles_summary` | info | Uno por día, construido o saltado: `skipped`, `bytes`, `objects`, `ticks_bytes`, `events_bytes`, `page_bytes`, `input_hash`, `content_hash`, `thetas`, `provisional_thetas`, `provisional_tail` y `missing_thetas`. |
| `input_missing` | error | Sin `consolidated.parquet` (`what = l1`), sin `events.parquet` en el mes (`events`), un θ sin `events.parquet` o `carry_over.parquet` o con la cadena rota (`events`, `carry_over`, con `theta`), o un día sin ticks (`ticks`). En modo `render`, un día sin `index.json` o con archivos que faltan o no coinciden con su `content_hash` (`what = tiles`). Los de mes llevan el primer día pedido en `details.day` y `days`. |
| `price_rounded` | warning | Ticks fuera del tick del activo: se redondean y el día se escribe. |
| `price_unrepresentable` | error | El precio máximo no cabe en `int32`: el día no se escribe. |
| `render_summary` | info | Modo `render`: uno por día, regenerado o al día. `skipped`, `tiles_version`, `template_hash`, `content_hash`, `page_bytes` (lo guardado: gzip en un bucket, plano en disco) y `decoded_bytes` (el HTML ya descomprimido). |

Cada día termina con una línea sonda: `sonda: unit=<día> ticks=… wall_s=…
rss_mib=… ticks_bytes=… page_bytes=…`; `wall_s` es el tiempo por día (objetivo: un
mes en menos de 10 minutos).

## Memoria

Una sola pasada por el `consolidated.parquet` del mes, row group a row group,
saltando por las estadísticas de `transact_time` los que no tocan los días por
construir. Un día cierra cuando los ticks pasan al siguiente. Cada lote se codifica
en varint y se suelta; de un día solo viven los bytes codificados (≈ 5 B por tick,
el propio `ticks.bin`) y se escriben por tramos, sin juntar sus tres secciones.
Los `events.parquet` se leen una vez por mes y quedan como arreglos de NumPy (41 B
por evento). En RAM: un row group, los bytes del día en curso y los eventos del mes.

## Imagen

```bash
docker build -f layers/viz_tiles/Dockerfile -t viz_tiles .
layers/viz_tiles/smoke.sh viz_tiles     # el día real 2017-08-18 de los fixtures de dc_core
```

La imagen trae la plantilla en `/app/site`. Sin Rust: `viz_tiles` no importa `dc_pyo3` ni las capas L1 y L2; lee sus Parquet por
nombre de columna.

## API del paquete

```python
from viz_tiles.contract import price_scale
from viz_tiles.events import EventsBuffer, event_rows
from viz_tiles.ticks import encode_day
from viz_tiles.write import ThetaEvents, write_day

# `batches`: RecordBatch de L1 (agg_trade_id, price, quantity, transact_time),
# ordenados por transact_time; se consumen uno a uno.
ticks = encode_day(batches, day, price_scale("BTCUSDT"))

# `events`: MonthEvents.touching(first_id, last_id) del θ; `pending`: la cola del carry-over.
buffer = EventsBuffer()
buffer.add(event_rows(events, pending, provisional, day_start_us))

write_day(
    root,
    provider="binance",
    market="spot",
    asset="BTCUSDT",
    day=day,
    ticks=ticks,
    events=buffer,
    thetas=[ThetaEvents("0.00010000", n_events, None)],
    input_hash=input_hash,
    image_version="1.0.0",
)
```

- `TicksAccumulator` (y `encode_day`) guarda solo los bytes ya codificados: la RAM
  es O(lote) más ≈ 5 B por tick. `decode_ticks` es la inversa, para pruebas y
  sondas.
- `read_month_events` lee un `events.parquet` entero, una vez por mes;
  `MonthEvents.touching` saca los eventos de cada día por `agg_trade_id`.
- `write_day` escribe `events.bin`, `ticks.bin`, la página y, al final,
  `index.json` (marca de commit); `latest.json` y `latest.html` solo avanzan.
- `render_day(index, arrays)` arma el `index.html` de un día: `arrays` entrega
  `(nombre, tramos)` en orden de nombre y cada uno se codifica por tramos y se suelta.

## Pruebas

```bash
uv run pytest layers/viz_tiles/tests
```
