# dc_pyo3

Bindings [PyO3](https://pyo3.rs) sobre [`dc_core`](../dc_core/README.md): el
fan-out de N θ y el carry-over, para que Python use el núcleo Rust (ITSC-243).
No tiene lógica de detección: convierte tipos y errores, nada más. El
detector es `dc_core` sin tocarlo.

Es miembro del workspace de Cargo (`Cargo.toml` raíz) y del de uv
(`shared/*`). `uv sync` lo compila con maturin, en release, y
`uv run python -c "import dc_pyo3"` funciona sin más pasos. `uv sync`
también lo reconstruye cuando cambia el Rust suyo o el de `dc_core`
(`cache-keys` de su `pyproject.toml`).

## API

```python
import dc_pyo3

fan = dc_pyo3.FanOut([10_000, 11_352, ...])  # round(θ × 10⁸), en este orden
closed = fan.feed_batch(prices, times, ids)  # list[list[Event]], uno por θ
last = fan.finish()  # list[Event | None]: fin de entrada
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
- **El GIL se mantiene durante el lote**: los buffers son de Python. El
  fan-out multihilo corre dentro de la llamada.
- **El orden es del llamador**: los ticks van en orden estricto de
  `(time, agg_trade_id)`, como los entrega L1. El resultado no depende de cómo
  se parta la serie en lotes (lo verifican las pruebas).
- Los eventos que devuelve una llamada son objetos de Python (~100 B cada
  uno). Con θ chico pueden ser un evento cada pocos ticks: pasa lotes de
  decenas de miles de ticks (la capa usa 65 536), no un row group entero de
  1 M, o el pico de RAM sale de los eventos y no de los precios.

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
