# dc_core

Crate Rust del núcleo de Directional Change (DC). Definido en el
[TRD maestro §8.1](../../docs/TRD/plataforma_directional_change.md#81-estrategia-de-repositorio-monorepo-políglota).
Expone el detector de un solo θ, `Detector` (ITSC-239), y el fan-out de N θ
multihilo sobre lotes, `FanOut` (ITSC-241); el carry-over y los bindings se
construyen encima.

## Toolchain

`rust-toolchain.toml`, en la raíz del repo, fija canal y versión. Dos formas
de tener `cargo` en el contenedor de desarrollo:

- `apt = ["rustup"]`: instala la versión exacta de `rust-toolchain.toml` en
  el primer `cargo`, pero el toolchain vive en `~/.rustup`, que no es
  volumen, así que se vuelve a descargar de `static.rust-lang.org` en cada
  `devkit recreate`.
- `apt = ["cargo", "rustc", "rustfmt", "rust-clippy"]`: paquete de Debian,
  horneado en la imagen (no se vuelve a bajar), pero es la versión que trae
  `trixie`, no la de `rust-toolchain.toml`.

Elegimos `rustup`: el objetivo de esta card es que el repo compile "de forma
reproducible en el contenedor y en CI", y CI (`ubuntu-latest`, que ya trae
`rustup`) siempre usa la versión fijada en `rust-toolchain.toml`. Con la
ruta de paquetes Debian, `clippy`/`rustfmt` locales divergen de esa versión
y un lint puede pasar en el contenedor y fallar en CI (o al revés). El costo
—volver a bajar el toolchain tras cada `recreate`— es aceptable porque
`recreate` es poco frecuente frente a `rebuild`.

`rustup` trae el compilador pero no un linker: `cargo test` invoca `cc` para
enlazar el binario de pruebas, y Debian trixie no lo trae por defecto. `apt`
suma `gcc` (no `build-essential`) porque es lo mínimo que provee `cc`, y
`libc6-dev` porque la imagen instala sin paquetes recomendados y `gcc` solo
lo *recomienda*: sin él `cc` existe pero el enlace falla con `cannot find
Scrt1.o` / `crti.o`. No hace falta `g++` ni `make` para un crate que aún no
tiene dependencias con build scripts en C/C++.

## Detector

```rust
let mut d = Detector::new(10_000_000)?;        // θ = 10 %: round(θ × 10⁸)
for tick in ticks {                            // Point { price, time, agg_trade_id }
    if let Some(event) = d.feed(tick) { /* evento cerrado */ }
}
if let Some(event) = d.finish() { /* cierra un grupo de empate abierto */ }
```

- **Sin float, estado O(1).** Precio y θ son enteros de escala `10⁸`; el
  umbral se multiplica en `i128` (`ceil` en upturn, `floor` en downturn) y se
  compara en `i64` (TRD-L2 ADR-L2-01/02). `State` son escalares: dirección,
  ambos extremos, evento pendiente y grupo de empate abierto.
- **Un evento sale al confirmar el siguiente**, porque su extremo se conoce
  ahí (ADR-L2-05). `feed` devuelve a lo sumo uno.
- **Instante de confirmación atómico** (ADR-L2-03/04): los ticks con el mismo
  `time` que el de confirmación van al DC, el precio de confirmación es el más
  conservador que cumple el umbral, y esos ticks no mueven extremos ni
  evalúan reversión. El grupo se cierra con el primer tick de otro `time` o
  con `finish()`.
- **Contrato de entrada:** orden estricto por `(time, agg_trade_id)` y
  `0 < price < PRICE_LIMIT`; `feed` entra en pánico si el precio lo viola.
- `discarded()` cuenta los DC sin tick descartados (§9.1); con un θ válido es
  una guarda inalcanzable y debe quedar en cero.
- No serializa el estado ni lo restaura: eso es del carry-over.

## Equivalencia contra la v0

`tests/equivalence_v0.rs` compara el detector con los eventos que el kernel de
la v0 (`v0.2.0-legacy`) calcula sobre un día real de la landing, para cinco θ.
Con θ = 2 % coinciden los 12 eventos campo a campo; con los demás, las únicas
discrepancias son la divergencia declarada del instante de confirmación
atómico (ADR-L2-04) y están fijadas en la prueba. Fixture, script y cómo
regenerarlo: [`tests/fixtures/README.md`](tests/fixtures/README.md).

## Fan-out de N θ

```rust
let mut fan = FanOut::new(&thetas)?;                        // un Detector por θ
let events = fan.feed_batch(&prices, &times, &ids);         // Vec<Vec<Event>>: por θ, en orden
let last = fan.finish();                                    // Vec<Option<Event>>: cierra empates abiertos
```

- **Un tick, N detectores, un proceso.** `feed_batch` recibe el lote en tres
  columnas prestadas (`&[i64]`) y reparte los detectores en grupos contiguos,
  un hilo por grupo (`FanOut::new` usa tantos como núcleos; `with_threads`
  fija el máximo). Todos leen las mismas columnas: la serie no se copia por θ.
  Cada hilo recorre el lote por trozos de 4096 ticks y, dentro de un trozo,
  todos sus detectores, para leerlo de memoria una vez (cabe en L2) en vez de
  una vez por θ.
- **Resultado idéntico al de `Detector`.** Cada detector ve los ticks en
  orden, así que sus eventos son los mismos que alimentándolo tick a tick,
  con lotes de cualquier tamaño (un tick incluido) y con cualquier número de
  hilos. `tests/fanout.rs` lo comprueba con 50 θ, 300 000 ticks sintéticos con
  empates de `transact_time` y lotes de tamaño variable.
- **Memoria.** El fan-out retiene N `Detector` (estado O(1) cada uno). Los
  eventos cerrados salen por el valor de retorno de cada llamada y el llamador
  los suelta cuando los escribe; nada crece con la serie, y el pico es
  O(lote) más los eventos de ese lote.
- **Lotes chicos, sin hilos.** Si `ticks × θ` es menor que 250 000, el lote se
  evalúa en el hilo que llama: crear hilos por lote cuesta más que el cómputo
  (con lotes de 1000 ticks y 50 θ, 10 hilos rindieron peor que 5).
- **Por qué `std::thread::scope` y no rayon.** Los hilos duran un lote y se
  prestan las columnas sin `'static` ni copias, que es justo lo que da
  `scope`. Rayon añadiría una dependencia (y un pool residente) para repartir
  50 tareas iguales que ya se reparten con `chunks_mut`. Costo aceptado:
  crear hilos en cada lote, que a 10 000 ticks o más por lote es una fracción
  pequeña; si E4a necesitara lotes chicos con muchos hilos, el cambio es un
  pool persistente detrás de la misma firma.

## Benchmark de ticks/s por core

```sh
cargo run --release -p dc_core --example bench_fanout -- [ticks] [lote] [corridas]
```

Serie sintética determinista (camino aleatorio con ~25 % de empates de
instante; `tests/common/mod.rs`), generada lote a lote fuera del cronómetro.
Los 50 θ van de 0,05 % a 2,5 %; el caso de 1 θ usa el de 0,05 %, el que más
eventos produce. Corrida de referencia para E4a: 10 M de ticks, lotes de
65 536, mejor de 3, contenedor de desarrollo (aarch64, 10 núcleos):

| Caso | θ | Hilos | Ticks/s | Ticks/s por core | θ·ticks/s por core |
|---|---:|---:|---:|---:|---:|
| `Detector`, un hilo | 1 | 1 | 83,2 M | 83,2 M | 83,2 M |
| `FanOut` | 1 | 1 | 84,8 M | 84,8 M | 84,8 M |
| `Detector` ×50, un hilo | 50 | 1 | 2,76 M | 2,76 M | 137,8 M |
| `FanOut` | 50 | 1 | 2,80 M | 2,80 M | 139,9 M |
| `FanOut` | 50 | 2 | 5,14 M | 2,57 M | 128,5 M |
| `FanOut` | 50 | 5 | 11,23 M | 2,25 M | 112,3 M |
| `FanOut` | 50 | 10 | 13,51 M | 1,35 M | 67,6 M |

Cómo leerla:

- **Un θ: ~83 M ticks/s por core**, unas 4 a 8 veces la estimación del
  maestro (10 a 20 M, §9.2). Es una serie sintética sin E/S ni parseo; una
  cota superior del detector, no de la capa.
- **50 θ: ~2,8 M ticks/s por core**, es decir ~7 ns por tick y θ. La unidad
  que importa para dimensionar es θ·ticks/s: ~140 M por core.
- **El fan-out no suma costo**: con un hilo rinde igual que N `Detector`
  seguidos. La ganancia es el paralelismo: con 10 hilos, 13,5 M ticks/s de la
  serie (4,9× un hilo).
- **No escala 10× con 10 hilos** (4,9×; con 5 hilos, 4,0×). No investigué la
  causa (los 10 núcleos del contenedor pueden no ser independientes entre sí);
  para dimensionar E4a conviene contar con ~2 M ticks/s por core con 50 θ, no
  con el valor de un hilo, y medir en el hardware real.
- La medición con un mes real de la landing corresponde a la hija 8 de la
  épica, no a esta card.
