# dc_pyo3

Bindings [PyO3](https://pyo3.rs) sobre [`dc_core`](../dc_core/README.md): el
fan-out de N θ y el carry-over, para que Python use el núcleo Rust (ITSC-243), y
sobre [`dc_descriptors`](../dc_descriptors/README.md): el motor de agregación de
L3 (ITSC-358). No tiene lógica de detección ni de agregación: convierte tipos,
buffers y errores, nada más.

Es miembro del workspace de Cargo (`Cargo.toml` raíz) y del de uv
(`shared/*`). `uv sync` lo compila con maturin, en release, y
`uv run python -c "import dc_pyo3"` funciona sin más pasos. `uv sync`
también lo reconstruye cuando cambia el Rust suyo o el de `dc_core`
(`cache-keys` de su `pyproject.toml`).

## API

```python
import dc_pyo3

fan = dc_pyo3.FanOut([10_000, 11_352, ...])  # round(θ × 10⁸), en este orden
closed = fan.feed_batch_columns(prices, times, ids)  # list[EventColumns], uno por θ
last = fan.finish_columns()  # list[EventColumns] de 0 o 1 evento: fin de entrada
# Por objetos, para pruebas y usos pequeños:
closed = fan.feed_batch(prices, times, ids)  # list[list[Event]], uno por θ
last = fan.finish()  # list[Event | None]
carry = fan.carry_overs()  # list[CarryOver], uno por θ
fan = dc_pyo3.FanOut.from_carry_over(thetas, carry)  # el mes siguiente
```

Tipos en [`dc_pyo3.pyi`](dc_pyo3.pyi). `Event` y `CarryOver` son de solo
lectura; sus puntos son tuplas `(price, time, agg_trade_id)` con el precio
entero en escala `SCALE` (`10⁸`). `CarryOver.to_bytes()`/`from_bytes()` son el
transporte interno de `dc_core`, no el formato del TRD-L2 §7.4: el Parquet de
carry-over lo escribe la capa a partir de los campos.

Los errores de `dc_core` (θ inválido, carry-over de otro θ o versión, grupo
abierto sin `finish()`) llegan como `ValueError` con el mensaje del crate.

## `feed_batch`: buffers, sin copia

Las tres entradas son objetos con protocolo de buffer, que en la capa son los
`Buffer` de un `RecordBatch` de Arrow (ver
[`landing.py`](../../layers/l2_dc_events/src/l2_dc_events/landing.py)):

| Argumento | Contenido |
|---|---|
| `prices` | `decimal128` de Arrow: 16 bytes little-endian por tick, el entero sin escalar de `DECIMAL(18,8)` |
| `times` | `transact_time`, `int64` contiguo |
| `ids` | `agg_trade_id`, `int64` contiguo |

- **Sin segunda representación del precio.** El binding lee el `decimal128`
  directamente: valida todo el lote (`0 < price < PRICE_LIMIT`, largos
  iguales) antes de alimentar nada, y lo decodifica a `i64` por tramos de
  65 536 ticks (512 KB), no el lote entero. Si falla con `ValueError`, el
  fan-out no se tocó.
- **El GIL se suelta durante el lote** (ITSC-290): `feed_batch*` y
  `finish_columns` validan y detectan dentro de `py.detach`, así que los
  escritores de Parquet de otros hilos de Python corren mientras se detecta. Las
  vistas de los buffers se toman antes de soltarlo y el `PyBuffer` las retiene
  hasta que la llamada vuelve; los buffers de Arrow son inmutables. El fan-out
  multihilo corre dentro de la llamada. `FanOut(thetas, threads)`: la capa pasa
  `threads` explícito (`cpu.py`); sin él se usan los núcleos que reporta
  `available_parallelism`, que con cuota fraccionaria redondea hacia abajo.
- **El orden es del llamador**: los ticks van en orden estricto de
  `(time, agg_trade_id)`, como los entrega L1. El resultado no depende de cómo
  se parta la serie en lotes (lo verifican las pruebas).
- Con θ chico puede haber un evento cada pocos ticks: pasa lotes de decenas
  de miles de ticks (la capa usa 65 536), no un row group entero de 1 M, o el
  pico de RAM sale de los eventos y no de los precios.

## Eventos en columnas (`feed_batch_columns`, `finish_columns`)

`feed_batch` crea un objeto `Event` por evento (~100 B y varias tuplas). Con
12 M de eventos por mes, volcarlos a columnas desde Python era el 50 % del
tiempo de una unidad de L2 (ITSC-275). Las variantes `*_columns` reciben lo
mismo y calculan lo mismo, pero devuelven por θ un `EventColumns`:

| | |
|---|---|
| `len(block)` | número de eventos |
| `block.buffers()` | 10 `bytes`, en el orden de `events.parquet`: `reference_*`, `confirm_*`, `extreme_*` (precio, tiempo, `agg_trade_id`) y `direction` |

Cada buffer ya está en el layout de Arrow: precio `decimal128` (16 B
little-endian, el entero sin escalar), tiempo e id `int64`, `direction`
`int8`. `pyarrow.py_buffer(b)` los envuelve sin copia, y `theta`, que es
constante, lo agrega quien escribe. Se construyen directo en la memoria del
`bytes`, así que nunca conviven los eventos y una segunda copia: los
eventos de `dc_core` de un θ se sueltan en cuanto están sus columnas. Las
pruebas (`test_columns_carry_the_same_events_as_the_objects`) comprueban que
`EventColumns` y `Event` llevan exactamente lo mismo.

## `Engine`: el motor de agregación de L3 (ITSC-358)

Bindings sobre [`dc_descriptors`](../dc_descriptors/README.md), el motor que calcula los
descriptores de `summaries.parquet` en hilos nativos (ADR-L3-16). Aquí tampoco hay lógica:
solo buffers y errores. El modelo (ítems, casillas, fusión) y el contrato de entrada están
en el README del crate.

```python
engine = dc_pyo3.Engine(thetas, schema, row_groups, threads=None)
engine.push(row_group_id, rows, time, price, quantity, sold, pieces)
while (sealed := engine.pop_sealed()) is not None:
    theta, index, rows, columns, validity, null_count = sealed
engine.inflight()
engine.busy()
stats = (
    engine.finish()
)  # por θ: (rows, closed_ticks, before_month, first_reference, last_extreme)
engine.close()
```

- `schema` son las 18 columnas de datos como `(nombre, bytes por valor, nulable)`; el motor
  las compara con las suyas y falla con `ValueError` si el esquema de la capa cambió.
  `row_groups[t]` son las filas de cada row group de salida del θ `t` (el esqueleto de L2),
  y van en el constructor y no en cada `push` porque no cambian por row group.
- **Ticks sin copia.** `time`, `price`, `quantity` y `sold` son `(memoryview de bytes del
  Buffer de Arrow, Array.offset)`. El motor retiene el `memoryview` hasta terminar el row
  group (`inflight()` baja) y lo suelta sin tomar el GIL: PyO3 difiere el `decref`. Los
  buffers deben ser inmutables, como los de Arrow.
- **`pieces`**: una por θ que tiene eventos en el row group; cualquier objeto con los
  atributos de `Piece` (ver `.pyi`; la capa usa un `NamedTuple`). Los arreglos de L2 se
  copian (son pocos); `positions` es `(n, 3)` `int64` contiguo y los decimales van como
  `(bytes del buffer, offset)`.
- **Salida sin copia.** `columns` son 18 `SealedColumn`: objetos de solo lectura con
  protocolo de buffer que poseen la memoria del motor. `pyarrow.py_buffer(column)` los
  envuelve sin copiar, y `validity` es el mapa de bits de las columnas `os_*`, o `None`.
- **Errores.** `DescriptorError` (subclase de `ValueError`) con `kind` y, según el caso,
  `theta` (posición), `event`, `confirm_agg_trade_id`, `phase`, `field`, `got`, `wanted`.
  `l3_dc_descriptors.native.to_summarize_error` los convierte en las excepciones tipadas
  de L3.
- `finish()` espera con el GIL suelto.

Se prueba con `layers/l3_dc_descriptors/tests/test_l3_native*.py`, no con pytest de este
paquete: el contrato real es el que usa la capa.

## Por qué `extension-module` no es una feature por defecto

Con `pyo3/extension-module` el enlazador deja sin resolver los símbolos de
libpython, y un binario que los use (`cargo test`) no enlaza. Por eso es una
feature del crate que solo activa maturin (`[tool.maturin] features` en
`pyproject.toml`); sin ella, el crate enlaza libpython como cualquier otro.
Además, `[lib] test = false`: el crate no tiene pruebas Rust (se prueba con
pytest, como lo usa la capa), y así `cargo test` no exige un Python
instalado ni `libpython` en tiempo de ejecución.

## Herramientas Rust

`cargo fmt`, `cargo test` y `cargo test --workspace` no necesitan Python.
`cargo clippy` y `cargo build` sí compilan este crate, y `pyo3` busca un
`python3` en el `PATH`; el contenedor solo trae `python3.14`, así que:

```bash
PYO3_PYTHON=$PWD/.venv/bin/python cargo clippy --all-targets -- -D warnings
```

## Versión de PyO3

`pyo3 = "0.27"`: la 0.28 y la 0.29 exigen `rustc` 1.83 y `rust-toolchain.toml`
fija 1.82.0. Subir el toolchain para eso es una decisión del repo, no de este
crate; 0.27 ya soporta Python 3.14.

## Pruebas

`uv run pytest shared/dc_pyo3`: el fan-out contra un caso calculado a mano,
independencia del troceado del lote y del número de hilos, continuación por
carry-over, `finish`, y los errores (θ, precios fuera de rango, largos,
buffers no contiguos o de otro tipo).
