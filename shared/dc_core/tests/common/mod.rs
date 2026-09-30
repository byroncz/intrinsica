//! Serie sintética determinista, compartida por las pruebas del fan-out y el
//! benchmark (`examples/bench_fanout.rs` la incluye con `#[path]`).
#![allow(dead_code)]

use dc_core::SCALE;

/// Camino aleatorio de precio con empates de `transact_time` (~25 % de los
/// ticks repiten el instante anterior). Xorshift64*: sin dependencias y misma
/// serie en cada corrida para una semilla dada.
pub struct Synthetic {
    rng: u64,
    price: i64,
    time: i64,
    id: i64,
}

impl Synthetic {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: seed.max(1),
            price: 50_000 * SCALE,
            time: 1_000,
            id: 1,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Reemplaza el contenido de las tres columnas por los `n` ticks
    /// siguientes de la serie. Reutiliza su capacidad: memoria O(lote).
    pub fn fill(
        &mut self,
        n: usize,
        prices: &mut Vec<i64>,
        times: &mut Vec<i64>,
        ids: &mut Vec<i64>,
    ) {
        prices.clear();
        times.clear();
        ids.clear();
        for _ in 0..n {
            let r = self.next_u64();
            // Paso de hasta ±3,3e-4 del precio; siempre positivo y < PRICE_LIMIT.
            let step = (r % 21) as i64 - 10;
            self.price += self.price / 30_000 * step;
            // Una de cada cuatro veces el instante se repite; si no, avanza 1 a 3 µs.
            if (r >> 8) % 4 != 0 {
                self.time += 1 + ((r >> 16) % 3) as i64;
            }
            prices.push(self.price);
            times.push(self.time);
            ids.push(self.id);
            self.id += 1;
        }
    }
}

/// La serie completa de `n` ticks en tres columnas (para pruebas pequeñas).
pub fn series(n: usize, seed: u64) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (mut p, mut t, mut i) = (Vec::new(), Vec::new(), Vec::new());
    Synthetic::new(seed).fill(n, &mut p, &mut t, &mut i);
    (p, t, i)
}

/// 50 θ de 0,05 % a 2,5 %, en escala `10⁸`.
pub fn fifty_thetas() -> Vec<i64> {
    (1..=50).map(|k| k * 50_000).collect()
}
