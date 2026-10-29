# dc_descriptors

Motor de agregación de L3 (ITSC-358, [ADR-L3-16](../../docs/TRD/l3.md)): calcula los
descriptores por evento DC sin pasar por el intérprete de Python. Python orquesta, Arrow
transporta, Rust calcula. Tiene dos familias (`Config.family`): `summaries`
(`summaries.parquet`, la familia S) y `path` (`path.parquet`, la familia T, ITSC-393,
[ADR-L3-17](../../docs/TRD/l3.md)).

Es un crate sin libpython (`cargo test` no necesita Python). Los bindings viven en
[`dc_pyo3`](../dc_pyo3/README.md) (`dc_pyo3.Engine`). Pertenece al workspace de Cargo; no
tiene `pyproject.toml`, así que el `pyproject.toml` raíz lo excluye del workspace de uv.

## Modelo

```text
push_boundaries(θ, lote) ──► fronteras de L2 en mano (por lotes, en orden, sin copia)
push(row group k) ──► cola de ítems (θ₀,k) (θ₁,k) … (θ₄₉,k)  ──► hilos nativos
                                                                  │ fetch_add (índice atómico)
                      casilla del evento i de θ ◄─ fila cerrada ──┤
                      marca de cerrados de θ (needed) ──► suelta los lotes ya cerrados
                      almacén de parciales de θ ◄─ eventos que cruzan row groups
                      row group de salida lleno ──► sellado ──► pop_sealed()
```

- **Familias.** El motor calcula una familia por instancia (`Config.family`). Las dos usan
  los mismos ítems, casillas, ubicación de fronteras y sellado; cambian el acumulador, el
  cierre y las columnas de salida:

  | Familia | Archivo | Columnas de datos | Parciales de un evento que cruza row groups |
  |---|---|---|---|
  | `Family::Summaries` (por defecto) | `summaries.parquet` | 18 (`COLUMNS`) | uno por row group, en cualquier orden |
  | `Family::Path` | `path.parquet` | 14 (`PATH_COLUMNS`: la clave y T-01 a T-13) | corridas de row groups vecinos, fusión ordenada |

  La familia T usa además `Config.silence_us` (µs): un hueco entre ticks de una fase cuenta
  como inactividad si es estrictamente mayor. S no lo usa. Los registros de T, sus fórmulas y
  su regla de nulos están en [`docs/data-contracts.md`](../../docs/data-contracts.md) y en
  TRD-L3 §7.2.1.

- **Fronteras.** `push_boundaries` agrega un lote de filas de L2 de un θ detrás de los
  anteriores, antes o después de empezar los `push`. El lote de un evento debe llegar antes
  que el row group que lo toca: en cada θ, o llegaron todos los eventos del esqueleto o el
  último entregado empieza después del último id del row group. Si no, `push` falla con
  `Input` en vez de esperar.

- **Ítem.** `(row group, θ)`: ubica las fronteras de los eventos de ese θ que tocan el row
  group y agrega sus fases. `push` encola los ítems; los hilos los toman de la cola, ordenada por row
  group, con un `fetch_add`. Ningún hilo tiene afinidad por θ ni por row group y el
  camino por tick no toma locks.
- **Posiciones.** Nadie entrega posiciones: el ítem las calcula con los ids. Toca los eventos
  que no cerraron en el row group anterior (`E > último id del anterior`) y empiezan en este
  o antes (`R ≤ último id`), y busca `R`, `C` y `E` en `agg_trade_id` (la posición es cuántos
  ticks tienen id ≤ la frontera). Hace las tres comprobaciones del lector: una frontera dentro
  del rango de ids que no es un tick, una anterior al row group que no cae dentro de ningún
  anterior (en un hueco) y el `transact_time` del tick `C`. No guarda estado entre row groups:
  solo necesita el último id del anterior, que `push` fija en orden de ordinal.
- **Casilla.** El evento `i` de un θ es la fila `i` de su esqueleto (los row groups de
  `events.parquet` de L2). Su fila va a esa casilla, la cierre quien la cierre.
- **Sumas.** Conteos, `Σ quantity` compradora y vendedora y `Σ price·quantity` en `u128`
  (cada tick aporta menos de 2¹²⁰; un desborde es un error de fase, nunca un valor
  truncado). VWAP = división entera con redondeo half-even.
- **Parciales de S.** Un evento que cruza row groups deja un parcial por row group en el
  almacén de su θ. Todo es suma, máximo, mínimo o selección por posición sobre enteros
  (el primer y el último tiempo, como mínimo y máximo), así que los parciales se fusionan
  en cualquier orden. El evento está completo cuando se fusionaron `último − primero + 1`
  parciales, con `primero` el ordinal del row group del parcial que no empieza antes y
  `último` el del que no termina después.
- **Parciales de T y fusión ordenada.** En T los pasos entre ticks consecutivos, los huecos
  y los pares de lados dependen del par que une dos tramos, así que los parciales no se
  fusionan en cualquier orden. Cada `PhasePath` lleva su estado de borde (primer y último
  precio, tiempo y lado; `n`, `Σ|Δp|`, `Σ(Δln p)²`, `S`, `P`, hueco máximo y suma de
  huecos mayores que `silence_us`) y `PathAcc` suma, por evento, el menor tiempo de cada
  nivel de T-11 a T-13. El almacén de un θ guarda por evento abierto una lista de
  **corridas** (`Runs`): tramos de ordinales consecutivos `[primero..último]`, sin
  solaparse. Un parcial nuevo se une solo con la corrida que termina en el ordinal anterior
  y con la que empieza en el siguiente; si ninguna lo toca, abre otra. El evento cierra
  cuando queda una única corrida del row group de `R` al de `E`. `merge` es asociativo
  (enteros, mínimos y máximos), así que el orden de llegada de los row groups y de los
  hilos no cambia ni un bit. Un parcial pesa del orden de 300 B y el almacén crece con los
  eventos abiertos, no con los ticks.
- **Sumas exactas de T.** Camino, lados, pares, huecos y los niveles de T-11 a T-13 son
  enteros. La única cuenta en `f64` por paso es `Δln p = ln_1p((p_i − p_{i−1}) / p_{i−1})`;
  su cuadrado se cuantiza **una vez** a un entero de punto fijo (escala 2⁻⁹⁶, `i128`,
  redondeo al más cercano) y se suma exacto; el `f64` sale al cerrar (`× 2⁻⁹⁶`). Por eso el
  resultado no depende del tamaño de tramo ni de los hilos. Error absoluto por término
  ≤ 2⁻⁹⁷; una suma que no cabe en 128 bits es un error de fase (`PathOverflow`), nunca un
  valor truncado. Tortuosidad y autocorrelación son una sola división `f64` de dos enteros.
- **Cierre de T.** `close_path` suma el paso del ancla (`reference_price` en la confirmación,
  `confirm_price` en el overshoot) al primer tick de cada fase y deriva los 13 registros
  con la regla de nulos del contrato. Solo comprueba lo que necesita para no producir
  `±inf` ni `NaN` y para que `c_tortuosity ≥ 1` valga: dirección, anclas positivas del lado
  que la dirección dice y niveles alcanzados; los invariantes de §7.2 contra L2 son de S.
- **Cierre de S.** Al completarse, un evento verifica las cantidades de las fases y los siete
  invariantes de TRD-L3 §7.2 contra su fila de L2, en el orden de `build_row`; un fallo
  trae el campo y los valores del hallazgo `l2_state_inconsistent`. El invariante 8
  (`buy + sell` es la suma de `quantity`) vale por construcción.
- **Sellado.** Cuando todas las filas de un row group de salida están escritas, el motor
  lo sella y lo entrega con sus columnas en el layout de Arrow (`decimal128` de 16
  bytes o `int64`; los `DOUBLE` de T viajan como los 64 bits del `f64`). En S, `validity` es
  el mapa de bits de las columnas `os_*`. En T, `SealedRowGroup.masks[columna]` trae el mapa
  de validez de cada columna nulable (`ColumnMask`: `bits` y `null_count`), o `None` si la
  columna no es nulable o no tiene nulos. Cada θ sale en orden de row group de salida.
- **Errores.** Si varios eventos fallan se reporta el de menor `(row group donde termina,
  θ, evento)`. Un fallo se levanta en `push`, `pop_sealed` o `finish` cuando ya ningún
  row group anterior al suyo puede traer uno de menor clave.
- **Determinismo.** Ni el orden en que los hilos toman ítems ni el orden de `push` cambian
  un bit de la salida.
- **Memoria.** Los ticks no se copian: `Engine<B>` es genérico en el dueño `B` de los
  buffers (`AsRef<[u8]>`) y lo suelta cuando los ítems del row group terminan. Las filas
  de L2 de los eventos tampoco se copian, y el motor suelta cada lote (su `B`) en cuanto
  sus eventos y los de los lotes anteriores cerraron: las filas que un ítem escribe en
  sus casillas avanzan, en el mismo paso, la marca de cerrados del θ (`needed`). Solo se
  sueltan lotes del frente, porque un row group sin procesar no toca ningún evento cerrado;
  así las búsquedas sobre los eventos en mano dan lo mismo que sobre todos. Lo retenido va
  de la marca al último lote entregado: O(lote + eventos abiertos de los row groups en
  vuelo), no O(eventos del mes); un evento largo abierto en la marca ancla los lotes
  entregados detrás de él (**81 bytes por evento**, antes 65: `R`, `C`, `confirm_time` y
  `E` de 8 bytes, `direction` de 1 y tres `decimal128` de 16: `confirm_price`,
  `extreme_price` y `reference_price`; son 81 y no 73 porque `reference_price` llega como
  `decimal128` de Arrow y no se copia). Los buffers de
  salida se reservan al llegar la primera fila de su row group y se entregan al sellar.

## API

```rust
let engine = Engine::<B>::new(Config {
    thetas,      // round(θ × 10⁸), en el orden en que se reportan los errores
    threads,     // hilos nativos
    row_groups,  // por θ, las filas de cada row group de salida
    family,      // Family::Summaries (por defecto) o Family::Path
    silence_us,  // umbral de silencio de la familia T, en µs (S no lo usa)
})?;
// Por lotes, en orden y sin copiar: filas consecutivas de events.parquet de L2 del θ, por
// delante del row group que las toca (antes o después de empezar los push).
engine.push_boundaries(theta, EventBatch::new(rows, reference, confirm, confirm_time,
                                              extreme, direction, confirm_price,
                                              extreme_price, reference_price)?)?;
// reference_price (último argumento, la octava columna de L2) es el ancla de la confirmación
// de T: decimal128 de 16 bytes, sin copia. S no la lee, pero se entrega siempre.
engine.needed(theta)?;  // primer evento del θ sin cerrar: leer L2 por delante, y no más
engine.push(RowGroupInput {
    id,            // ordinal del row group entre los que entrega el lector: 0, 1, 2, …
    ticks,         // TickColumns: ids, time, price, quantity, is_buyer_maker (Region = buffer + offset)
    before_month,  // el row group es de un mes anterior al de los eventos
})?;
while let Some(sealed) = engine.pop_sealed()? { /* SealedRowGroup */ }
engine.inflight();   // row groups que aún anclan sus ticks
engine.busy();       // fracción del tiempo con ítems en la cola
let stats = engine.finish()?;  // por θ: filas, ticks cerrados, ticks antes del mes, primera
                               // referencia y último extremo, para la conciliación (§9.1)
```

Contrato de entrada:

- `id` son ordinales consecutivos desde 0; pueden empujarse en cualquier orden, pero un row
  group no se procesa hasta que se empuja el anterior (necesita su último id), y un evento
  solo se completa cuando llegaron todos los row groups de su rango. `finish()` falla con
  `Input` si un ordinal nunca llegó.
- `push_boundaries` entrega los eventos de un θ en orden, en lotes de cualquier tamaño (un
  θ sin eventos no necesita ninguno). El lote de un evento llega antes que el row group que
  lo toca; si no, `push` falla con `Input` (no espera). Los eventos cumplen `R < C ≤ E` y
  `E_i ≤ R_{i+1}`, también entre un lote y el anterior, y los tres precios
  (`confirm_price`, `extreme_price` y `reference_price`) caben en `i64`
  (contrato de L2); se comprueba al entregarlos, y un lote que pasa del esqueleto falla con
  `Skeleton`. El evento `i` es la fila `i` del esqueleto, sea cual sea el lote que lo trajo.
- `needed(θ)` es el primer evento del θ que no cerró; los lotes enteros anteriores ya se
  soltaron.
- Los ids de un row group crecen estrictamente y sus ticks están en el orden de la serie
  (contrato de L1).
- `before_month` marca los row groups de un mes anterior al de los eventos: sus ticks de
  `(R, E]` suman a `before_month` de las cuentas (`ticks_before_month` de §7.4).
- `engine.locate(θ, ticks, prev_high)` devuelve qué eventos toca un row group y sus
  posiciones y banderas sin empujarlo, con las fronteras en mano: es lo que calcula un
  ítem, para probar y diagnosticar.
- Los fallos de ubicación son `Boundary` (`not_a_tick` o `unseen`, con la frontera, el
  evento y el rango de ids del row group) y `ConfirmTime`; si un row group trae los dos,
  gana el de frontera, como en el lector.
- El motor valida los ticks que agrega (`0 ≤ valor < 2⁶³`), no los que ninguna fase toca.
- Un ordinal que aporta dos veces su tramo al mismo evento de T, o fuera del rango del
  evento, es un `EngineError::Internal`, no un dato.
- `finish()` espera los ítems pendientes y falla con `Incomplete` si alguna casilla quedó
  sin llenar. Los sellados siguen disponibles en `pop_sealed()` después de `finish()`.

## Pruebas y medición

```sh
cargo test -p dc_descriptors
cargo run --release -p dc_descriptors --example bench_engine -- [ticks] [row_group] [θ] [hilos…]
```

`tests/path.rs` prueba la familia T: los 13 registros calculados a mano (un upturn con
overshoot, un downturn que espeja los niveles, fases de un tick, overshoot vacío o de un
tick, cruce atómico, los casos de la autocorrelación y huecos a ambos lados de
`silence_us`), un evento repartido en tres row groups con el mismo resultado bit a bit en
**todos** los órdenes de `push`, un row group intermedio sin ticks de la fase, que el
almacén de corridas quede vacío al terminar, los eventos y ticks inválidos (son error, no
valor) y un oráculo ingenuo en `f64` sobre series aleatorias que, además, no cambia con los
cortes, el orden de `push` ni con 1 y 4 hilos. Las pruebas unitarias de `src/path.rs`
cubren la fusión de vecinos en cualquier asociación y la cuantización por término.

`tests/engine.rs` compara el motor de S con un oráculo ingenuo (suma tick a tick con otra
fórmula de redondeo) con 1 y 4 hilos, en órdenes de `push` barajados y con las fronteras en
lotes de 1 a 3 eventos intercalados con los `push`; comprueba que un lote se suelta en
cuanto sus eventos cerraron (su `B` muere) y que un row group sin sus fronteras falla, e
incluye los fallos
de frontera y de `confirm_time` (también en orden inverso), operandos
en los bordes de `DECIMAL(18,8)`, fases vacías, eventos que cruzan tres row groups, un
row group intermedio sin ticks del evento, row groups sin eventos de un θ, los invariantes
de §7.2 y el orden de los errores. El microbenchmark sobre una serie sintética da
≈ 5 ns por tick y θ y escala lineal de 1 a 4 hilos; no es la medición de la unidad.
