# dc_core

Crate Rust del núcleo de Directional Change (DC). Definido en el
[TRD maestro §8.1](../../docs/TRD/plataforma_directional_change.md#81-estrategia-de-repositorio-monorepo-políglota).
Hoy (ITSC-238) es un crate mínimo que solo prueba que el toolchain compila;
la lógica de detección de eventos DC llega en E4.

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
