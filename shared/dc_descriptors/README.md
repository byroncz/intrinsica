# dc_descriptors

Motor de agregación de L3 (ITSC-358, [ADR-L3-16](../../docs/TRD/l3.md)): calcula los
descriptores de `summaries.parquet` por evento DC sin pasar por el intérprete de
Python. Python orquesta, Arrow transporta, Rust calcula.

Es un crate sin libpython (`cargo test` no necesita Python). Los bindings viven en
[`dc_pyo3`](../dc_pyo3/README.md) (`dc_pyo3.Engine`). Pertenece al workspace de Cargo; no
tiene `pyproject.toml`, así que el `pyproject.toml` raíz lo excluye del workspace de uv.

## Modelo

```text
set_boundaries(θ)  ──► fronteras de L2 del mes (una vez por θ, sin copia)
push(row group k) ──► cola de ítems (θ₀,k) (θ₁,k) … (θ₄₉,k)  ──► hilos nativos
                                                                  │ fetch_add (índice atómico)
                      casilla del evento i de θ ◄─ fila cerrada ──┤
                      almacén de parciales de θ ◄─ eventos que cruzan row groups
                      row group de salida lleno ──► sellado ──► pop_sealed()
```

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
- **Parciales.** Un evento que cruza row groups deja un parcial por row group en el
  almacén de su θ. Todo es suma, máximo, mínimo o selección por posición sobre enteros
  (el primer y el último tiempo, como mínimo y máximo), así que los parciales se fusionan
  en cualquier orden. El evento está completo cuando se fusionaron `último − primero + 1`
  parciales, con `primero` el ordinal del row group del parcial que no empieza antes y
  `último` el del que no termina después.
- **Cierre.** Al completarse, un evento verifica las cantidades de las fases y los siete
  invariantes de TRD-L3 §7.2 contra su fila de L2, en el orden de `build_row`; un fallo
  trae el campo y los valores del hallazgo `l2_state_inconsistent`. El invariante 8
  (`buy + sell` es la suma de `quantity`) vale por construcción.
- **Sellado.** Cuando todas las filas de un row group de salida están escritas, el motor
  lo sella y lo entrega con sus 18 columnas en el layout de Arrow (`decimal128` de 16
  bytes o `int64`) y el mapa de bits de validez de las columnas `os_*`. Cada θ sale en
  orden de row group de salida.
- **Errores.** Si varios eventos fallan se reporta el de menor `(row group donde termina,
  θ, evento)`. Un fallo se levanta en `push`, `pop_sealed` o `finish` cuando ya ningún
  row group anterior al suyo puede traer uno de menor clave.
- **Determinismo.** Ni el orden en que los hilos toman ítems ni el orden de `push` cambian
  un bit de la salida.
- **Memoria.** Los ticks no se copian: `Engine<B>` es genérico en el dueño `B` de los
  buffers (`AsRef<[u8]>`) y lo suelta cuando los ítems del row group terminan. Las filas
  de L2 de los eventos tampoco se copian, pero el motor las retiene todo el mes: son
  O(eventos), unos 65 bytes por evento (`R`, `C`, `confirm_time` y `E`, `direction` y dos
  `decimal128`) y no O(row groups en vuelo). Los buffers de salida se reservan al
  llegar la primera fila de su row group y se entregan al sellar.

## API

```rust
let engine = Engine::<B>::new(Config {
    thetas,      // round(θ × 10⁸), en el orden en que se reportan los errores
    threads,     // hilos nativos
    row_groups,  // por θ, las filas de cada row group de salida
})?;
// Una vez por θ, antes del primer push: sus row groups de events.parquet de L2, sin copiar.
engine.set_boundaries(theta, vec![EventBatch::new(rows, reference, confirm, confirm_time,
                                                   extreme, direction, confirm_price,
                                                   extreme_price)?, …])?;
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
- `set_boundaries` va una vez por θ y antes del primer `push` (un θ sin eventos lleva una
  lista vacía). Los eventos cumplen `R < C ≤ E` y `E_i ≤ R_{i+1}` y los precios caben en
  `i64` (contrato de L2); se comprueba al entregarlos. El evento `i` es la fila `i` del
  esqueleto.
- Los ids de un row group crecen estrictamente y sus ticks están en el orden de la serie
  (contrato de L1).
- `before_month` marca los row groups de un mes anterior al de los eventos: sus ticks de
  `(R, E]` suman a `before_month` de las cuentas (`ticks_before_month` de §7.4).
- `engine.locate(θ, ticks, prev_high)` devuelve qué eventos toca un row group y sus
  posiciones y banderas sin empujarlo: es lo que calcula un ítem, para probar y diagnosticar.
- Los fallos de ubicación son `Boundary` (`not_a_tick` o `unseen`, con la frontera, el
  evento y el rango de ids del row group) y `ConfirmTime`; si un row group trae los dos,
  gana el de frontera, como en el lector.
- El motor valida los ticks que agrega (`0 ≤ valor < 2⁶³`), no los que ninguna fase toca.
- `finish()` espera los ítems pendientes y falla con `Incomplete` si alguna casilla quedó
  sin llenar. Los sellados siguen disponibles en `pop_sealed()` después de `finish()`.

## Pruebas y medición

```sh
cargo test -p dc_descriptors
cargo run --release -p dc_descriptors --example bench_engine -- [ticks] [row_group] [θ] [hilos…]
```

`tests/engine.rs` compara el motor con un oráculo ingenuo (suma tick a tick con otra
fórmula de redondeo) con 1 y 4 hilos y en órdenes de `push` barajados, e incluye los fallos
de frontera y de `confirm_time` (también en orden inverso), operandos
en los bordes de `DECIMAL(18,8)`, fases vacías, eventos que cruzan tres row groups, un
row group intermedio sin ticks del evento, row groups sin eventos de un θ, los invariantes
de §7.2 y el orden de los errores. El microbenchmark sobre una serie sintética da
≈ 5 ns por tick y θ y escala lineal de 1 a 4 hilos; no es la medición de la unidad.
