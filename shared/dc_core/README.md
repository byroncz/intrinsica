# dc_core

Crate Rust del núcleo de Directional Change (DC). Definido en el
[TRD maestro §8.1](../../docs/TRD/plataforma_directional_change.md#81-estrategia-de-repositorio-monorepo-políglota).
Expone el detector de un solo θ, `Detector` (ITSC-239); el fan-out de 50 θ,
el carry-over y los bindings se construyen encima.

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
