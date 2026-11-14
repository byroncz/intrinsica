//! La cota del pico del cierre de una fase de la familia F (ADR-L3-19): la fórmula que citan
//! `spectrum_memory.rs`, que la compara con lo medido, y `spectrum.rs`, que la compara con el job.
//! Vive aparte para que `spectrum_memory.rs` siga sin más pruebas que la de la medición.

/// La longitud de la FFT interna de Bluestein para `n` primo, como la elige `rustfft` 6.4.1
/// (`plan.rs`, `design_prime`): la menor entre `3·2^(k−2)` y `2^k` que alcanza `2n − 1`.
pub fn bluestein_inner(n: usize) -> usize {
    let pow2 = (2 * n - 1).next_power_of_two();
    if pow2 / 4 * 3 >= 2 * n - 1 {
        pow2 / 4 * 3
    } else {
        pow2
    }
}

/// La cota del pico absoluto del cierre de una fase de `n` retornos, en bytes. Vale para `n` primo
/// con Bluestein, el peor plan de `rustfft`: el búfer de `n` complejos, los twiddles de longitud
/// `n` de Bluestein, y tres tramos de la longitud interna `m` (el multiplicador precomputado, el
/// scratch en el sitio, que ya es `m` más el de la FFT interna, y los twiddles de esta), cada uno
/// de 16 B. Los `f32` del búfer ya se liberaron cuando la FFT arranca.
pub fn closing_bound(n: usize) -> usize {
    16 * (2 * n + 4 * bluestein_inner(n))
}
