//! La familia F (frecuencia, `spectrum.parquet`): el búfer de retornos de una fase y el cierre de
//! los registros F-01 a F-04 (ADR-L3-19).
//!
//! A diferencia de S, T y A, el estado de un evento abierto no son cientos de bytes sino los
//! retornos de sus ticks: es la excepción declarada del ADR-L3-19, acotada por el presupuesto
//! (`budget_ticks`) y por el tope global del orquestador. Un tramo de fase guarda los retornos
//! `Δ ln p` entre sus ticks consecutivos en `f32` (4 B por retorno) y los precios de sus
//! bordes. Al unir dos tramos vecinos (el de ordinal siguiente) se calcula el retorno que los
//! une con el último precio del primero y el primero del segundo: cada par de ticks consecutivos
//! de la fase se evalúa una sola vez, con la misma función, así que el búfer final es el mismo
//! con cualquier corte y cualquier orden de `push`.
//!
//! Los tramos se guardan como una lista de trozos, no como un único `Vec`: unir es concatenar
//! listas, nunca copiar retornos, y el cierre consume trozo a trozo y los libera a medida que
//! los convierte.
//!
//! Anclas: la fase de confirmación arranca en `reference_price` de L2 y la del overshoot en
//! `confirm_price` (como T). El retorno del ancla al primer tick de la fase es el primero de la
//! serie y se calcula al cerrar: la fase de `n` ticks tiene `n` retornos y el búfer guarda
//! `n − 1`.
//!
//! Cierre (por fase con `n ≥ 16` retornos): todo en `f64`, sobre los `f32` convertidos.
//!
//! * Fourier (F-01 y F-02). La serie se centra (resta de la media), se multiplica por la ventana
//!   de Hann periódica `0,5 − 0,5·cos(2π·j/n)` y pasa por una FFT exacta de longitud `n`
//!   (`rustfft`, `FftPlannerScalar`: el planificador por defecto elige instrucciones SIMD según
//!   el host y cambiaría bits del `content_hash` entre máquinas). La energía es `|X_k|²` de los
//!   bins `k = 1..⌊n/2⌋` y la fracción de cada banda de período `n/k` (en ticks: [2, 4), [4, 8),
//!   [8, 16), [16, 32), [32, 64), [64, 128) y 128 o más) es su energía sobre la total.
//! * Haar (F-03 y F-04). Sobre los últimos `2^⌊log₂ n⌋` retornos, centrados, la transformada de
//!   Haar ortonormal (`a' = (x₂ᵢ + x₂ᵢ₊₁)/√2`, `d = (x₂ᵢ − x₂ᵢ₊₁)/√2`); la energía de un nivel es
//!   `Σ d²`, el nivel `j` alinea con la banda `j` y el séptimo junta las escalas de 128 en
//!   adelante.
//!
//! * Ubicación de la energía (`c_energy_loc`, `os_energy_loc`). `Σ t_k·r_k² / Σ r_k²` con
//!   `t_k = (k − 0,5)/n`, sobre los retornos sin centrar y sin ventana: dónde se concentra la
//!   energía de la fase, en (0, 1). Se calcula antes que Haar y la FFT, en una pasada sobre el
//!   búfer `f32` con dos acumuladores compensados. Es nula solo si todos los retornos son cero:
//!   una serie de retornos iguales y no nulos no tiene Fourier pero sí ubicación (0,5).
//!
//! Una banda o nivel sin bins (la banda 5 con menos de 32 retornos, la 6 con menos de 64 y la 7
//! con menos de 128, y los niveles de Haar igual) es nula, no cero. Una serie con todos los
//! retornos iguales no tiene energía: deja nulas las columnas de su registro. Una fase con
//! menos de 16 retornos deja nulas las de sus dos registros.

use crate::close::EventErrorKind;
use crate::input::{EventIn, TickColumns};
use crate::layout::{spectrum_col as sc, SpectrumRow, SPECTRUM_WIDTH};
use crate::path::{classify, invariant};
use crate::phase::{le_u64, TickError};
use rustfft::num_complex::Complex;
use rustfft::FftPlannerScalar;
use std::f64::consts::{PI, SQRT_2};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Retornos mínimos de una fase para calcular sus registros.
pub const N_MIN: usize = 16;

/// Bandas de período (y niveles de Haar) por registro.
pub const BANDS: usize = 7;

/// Ticks por evento a partir de los cuales no se calcula (`extreme − reference` de L2).
pub const DEFAULT_BUDGET_TICKS: u64 = 130_000_000;

/// Retornos de una fase a partir de los cuales su cierre se serializa en todo el proceso. El
/// cierre tiene un transitorio que depende del plan de `rustfft`. El peor es el de `n` primo, que
/// usa Bluestein con una FFT interna de `m` complejos (`m` es `3·2^(k−2)` si alcanza `2n − 1` y
/// `2^k` si no, con `k = ⌈log₂(2n − 1)⌉`): el pico del cierre, con el búfer de `n` complejos de
/// `f64` incluido, es `16 B × (2n + 4m)`. Lo fija `tests/spectrum_memory.rs` con un asignador
/// que cuenta, y vale para cualquier plan (Rader y los compuestos pesan 48 a 64 B por retorno,
/// medidos). Con `n` = 20 M, `m` = 3·2^24 y el pico es ≈ 3,6 GiB (≈ 193 B por retorno). Con el
/// presupuesto de 130 M (ITSC-418), `3·2^26` no alcanza `2n − 1` y `m` = 2^28: la cota da
/// 21 339 869 184 B (20 351 MiB, ≈ 164 B por retorno), bajo los 24 576 MiB (75 % de 32 GiB) del
/// job de F. Cada hilo del motor puede estar cerrando un evento: sin acotarlo, el pico suma un
/// transitorio por hilo.
/// Por debajo de este umbral el peor transitorio es de 160 MiB por hilo (`n` = 1 048 573, primo)
/// y los cierres corren en paralelo.
pub const SERIAL_CLOSE_MIN: usize = 1 << 20;

/// El candado de los cierres grandes: a lo sumo uno a la vez, así el transitorio del cierre es el
/// de una sola fase, no el de una por hilo. No toca el resultado: cada cierre es una función pura
/// de su búfer, el orden en que se turnan no cambia ningún bit.
static BIG_CLOSE: Mutex<()> = Mutex::new(());

/// El turno para cerrar una fase de `n` retornos: `Some` (y se espera) si `n ≥ SERIAL_CLOSE_MIN`,
/// `None` si puede correr de inmediato. Un hilo que entró en pánico dentro del turno no lo
/// inutiliza: el candado no guarda datos.
fn close_turn(n: usize) -> Option<MutexGuard<'static, ()>> {
    (n >= SERIAL_CLOSE_MIN).then(|| BIG_CLOSE.lock().unwrap_or_else(PoisonError::into_inner))
}

/// El planificador de FFT. Es siempre el escalar: no usa instrucciones SIMD y da los mismos
/// bits en cualquier host.
pub fn planner() -> FftPlannerScalar<f64> {
    FftPlannerScalar::new()
}

/// `Δ ln p` del paso de `previous` a `price`, ambos mayores que 0, en `f32`.
///
/// `ln(1 + Δp/previous)`: el cociente de enteros y `ln_1p` no pierden dígitos cuando el paso es
/// pequeño frente al precio. Es la única función que produce un retorno.
pub fn log_return(previous: i64, price: i64) -> f32 {
    let delta = price.wrapping_sub(previous);
    (delta as f64 / previous as f64).ln_1p() as f32
}

/// Los retornos de una fase, o de un tramo de ella, con los precios de sus bordes. Con
/// `ticks == 0` el resto no significa nada.
#[derive(Debug, Default, PartialEq)]
pub struct PhaseReturns {
    pub ticks: u64,
    first_price: i64,
    last_price: i64,
    /// Los retornos entre ticks consecutivos del tramo, en orden, en trozos.
    chunks: Vec<Vec<f32>>,
}

impl PhaseReturns {
    /// Retornos guardados: uno por cada par de ticks consecutivos.
    pub fn stored(&self) -> u64 {
        self.ticks.saturating_sub(1)
    }

    /// Une el tramo `next`, que sigue a `self` sin ticks de la fase entre los dos. No
    /// conmuta; sí es asociativo, así que las corridas pueden unirse en cualquier orden de
    /// llegada mientras cada unión sea entre vecinas.
    pub fn join(&mut self, mut next: PhaseReturns) {
        if next.ticks == 0 {
            return;
        }
        if self.ticks == 0 {
            *self = next;
            return;
        }
        self.chunks
            .push(vec![log_return(self.last_price, next.first_price)]);
        self.chunks.append(&mut next.chunks);
        self.ticks += next.ticks;
        self.last_price = next.last_price;
    }

    /// Los retornos guardados, en orden (para las pruebas).
    pub fn returns(&self) -> impl Iterator<Item = f32> + '_ {
        self.chunks.iter().flatten().copied()
    }
}

/// Los retornos de un evento, o de un tramo de él, para la familia F.
#[derive(Debug, Default, PartialEq)]
pub struct SpectrumAcc {
    pub confirmation: PhaseReturns,
    pub overshoot: PhaseReturns,
    /// El evento pasa del presupuesto: no se leyó ningún tick y su fila queda nula.
    pub over_budget: bool,
}

impl SpectrumAcc {
    /// El acumulador de un evento sobre el presupuesto: vacío.
    pub fn over_budget() -> Self {
        Self {
            over_budget: true,
            ..Self::default()
        }
    }

    /// Retornos guardados en las dos fases.
    pub fn stored(&self) -> u64 {
        self.confirmation.stored() + self.overshoot.stored()
    }

    /// Une el tramo `next`, el de ordinal siguiente, a continuación de `self`.
    pub fn join(&mut self, next: SpectrumAcc) {
        self.confirmation.join(next.confirmation);
        self.overshoot.join(next.overshoot);
        self.over_budget |= next.over_budget;
    }
}

/// Los retornos de los ticks `[start, end)` de una fase.
///
/// Es el camino caliente de la familia: una pasada por tick y un solo `Vec` de `n − 1`
/// retornos.
pub fn accumulate<B: AsRef<[u8]>>(
    ticks: &TickColumns<B>,
    start: usize,
    end: usize,
) -> Result<PhaseReturns, TickError> {
    if start >= end {
        return Ok(PhaseReturns::default());
    }
    let prices = ticks.price_bytes(start, end);
    let mut returns = Vec::with_capacity(end - start - 1);
    let (mut bad, mut zero) = (0u64, false);
    let (mut first, mut previous) = (0i64, 0i64);
    for (i, price) in prices.chunks_exact(16).enumerate() {
        let p = le_u64(price);
        bad |= le_u64(&price[8..]) | (p >> 63);
        zero |= p == 0;
        let p = p as i64;
        if i == 0 {
            first = p;
        } else {
            returns.push(log_return(previous, p));
        }
        previous = p;
    }
    if bad != 0 {
        return Err(classify(ticks, start, end));
    }
    if zero {
        return Err(TickError::ZeroPrice);
    }
    Ok(PhaseReturns {
        ticks: (end - start) as u64,
        first_price: first,
        last_price: previous,
        chunks: if returns.is_empty() {
            Vec::new()
        } else {
            vec![returns]
        },
    })
}

/// Las fracciones de un registro: `None` en una banda o nivel nulo. Todo el registro es
/// `[None; 7]` si la serie no tiene energía.
type Fractions = [Option<f64>; BANDS];

/// Los dos registros de una fase y la ubicación de su energía total.
#[derive(Debug, PartialEq)]
pub struct PhaseSpectrum {
    pub fourier: Fractions,
    pub haar: Fractions,
    /// Ubicación en el tiempo de la energía total (`c_energy_loc`, `os_energy_loc`): `None` si
    /// todos los retornos son cero.
    pub location: Option<f64>,
}

/// Suma compensada (Neumaier): una serie larga no pierde dígitos, y el orden de las operaciones
/// es siempre el mismo.
#[derive(Default)]
struct Neumaier {
    sum: f64,
    compensation: f64,
}

impl Neumaier {
    fn add(&mut self, value: f64) {
        let t = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - t) + value;
        } else {
            self.compensation += (value - t) + self.sum;
        }
        self.sum = t;
    }

    fn total(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// La media de `n` valores con suma compensada.
fn mean(values: impl Iterator<Item = f64>, n: usize) -> f64 {
    let mut sum = Neumaier::default();
    for value in values {
        sum.add(value);
    }
    sum.total() / n as f64
}

/// Ubicación en el tiempo de la energía total de una fase de `n` retornos, sin centrar y sin
/// ventana: `Σ t_k·r_k² / Σ r_k²` con `t_k = (k − 0,5)/n` y `k = 1..n`. Cae en (0, 1): 0,5 si la
/// energía se reparte pareja, menos si se concentra al inicio y más si al final. `None` si la
/// energía es cero. Una sola pasada con dos acumuladores: no copia la serie.
fn energy_location(values: impl Iterator<Item = f32>, n: usize) -> Option<f64> {
    let (mut moment, mut energy) = (Neumaier::default(), Neumaier::default());
    for (k, value) in values.enumerate() {
        let square = f64::from(value) * f64::from(value);
        moment.add(square * ((k as f64 + 0.5) / n as f64));
        energy.add(square);
    }
    let energy = energy.total();
    (energy > 0.0 && energy.is_finite()).then(|| moment.total() / energy)
}

/// Los retornos de la fase en orden: el del ancla y los trozos del búfer.
fn series(first: f32, chunks: &[Vec<f32>]) -> impl Iterator<Item = f32> + '_ {
    std::iter::once(first).chain(chunks.iter().flatten().copied())
}

/// Las fracciones por nivel de Haar de la cola `tail` (`2^k` retornos, sin centrar). Consume el
/// vector: lo transforma en el sitio.
fn haar_fractions(mut tail: Vec<f64>) -> Fractions {
    let mut result = [None; BANDS];
    let length = tail.len();
    let first = tail[0];
    if tail.iter().all(|&x| x == first) {
        return result;
    }
    let center = mean(tail.iter().copied(), length);
    for x in &mut tail {
        *x -= center;
    }
    let levels = length.trailing_zeros() as usize;
    let mut energy = [0.0f64; BANDS];
    let mut size = length;
    for level in 0..levels {
        let half = size / 2;
        let mut sum = 0.0f64;
        for i in 0..half {
            let (even, odd) = (tail[2 * i], tail[2 * i + 1]);
            let detail = (even - odd) / SQRT_2;
            sum += detail * detail;
            tail[i] = (even + odd) / SQRT_2;
        }
        energy[level.min(BANDS - 1)] += sum;
        size = half;
    }
    let total: f64 = energy.iter().sum();
    if !(total > 0.0 && total.is_finite()) {
        return result;
    }
    for (level, slot) in result.iter_mut().enumerate().take(levels) {
        *slot = Some(energy[level] / total);
    }
    result
}

/// Los índices `[lo, hi)` en el arreglo de energías de los bins `k = 1..⌊n/2⌋` (índice `k − 1`)
/// de cada banda: la 1 es el período `n/k ∈ [2, 4)`, o sea `n/4 < k ≤ n/2`, y así hasta la 7,
/// `k ≤ n/128`.
fn band_slices(n: usize) -> [(usize, usize); BANDS] {
    [
        (n >> 2, n >> 1),
        (n >> 3, n >> 2),
        (n >> 4, n >> 3),
        (n >> 5, n >> 4),
        (n >> 6, n >> 5),
        (n >> 7, n >> 6),
        (0, n >> 7),
    ]
}

/// Las fracciones de energía por banda de Fourier de una serie de `n` valores centrada en `center`.
/// Consume los trozos a medida que los convierte.
fn fourier_fractions(first: f32, chunks: Vec<Vec<f32>>, n: usize, center: f64) -> Fractions {
    let mut result = [None; BANDS];
    let step = 2.0 * PI / n as f64;
    let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(n);
    let mut push = |value: f32| {
        let j = buffer.len();
        let window = 0.5 + -0.5 * (j as f64 * step).cos();
        buffer.push(Complex::new((f64::from(value) - center) * window, 0.0));
    };
    push(first);
    for chunk in chunks {
        for &value in &chunk {
            push(value);
        }
    }
    let fft = planner().plan_fft_forward(n);
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    drop(scratch);
    let mut sums = [0.0f64; BANDS];
    for (band, (lo, hi)) in band_slices(n).into_iter().enumerate() {
        // El índice `i` del arreglo de energías es el bin `i + 1`.
        sums[band] = buffer[lo + 1..hi + 1].iter().map(|c| c.norm_sqr()).sum();
    }
    let total: f64 = sums.iter().sum();
    if !(total > 0.0 && total.is_finite()) {
        return result;
    }
    for (band, (lo, hi)) in band_slices(n).into_iter().enumerate() {
        if hi > lo {
            result[band] = Some(sums[band] / total);
        }
    }
    result
}

/// Los dos registros de una fase de `n ≥ N_MIN` retornos: el del ancla y los trozos del búfer,
/// que se consumen.
fn analyse(first: f32, chunks: Vec<Vec<f32>>, n: usize) -> PhaseSpectrum {
    debug_assert!(n >= N_MIN);
    // El turno cubre todo el cierre de la fase: la copia de la cola de Haar y la FFT.
    let _turn = close_turn(n);
    // La ubicación va primero: recorre el búfer f32 tal cual, antes de que Haar y la FFT lo
    // conviertan o lo consuman.
    let location = energy_location(series(first, &chunks), n);
    let (mut low, mut high) = (first, first);
    for &value in chunks.iter().flatten() {
        low = low.min(value);
        high = high.max(value);
    }
    // Haar primero: necesita los retornos y trabaja sobre una copia en f64 de la cola.
    let length = 1usize << (usize::BITS - 1 - n.leading_zeros());
    let tail: Vec<f64> = series(first, &chunks)
        .skip(n - length)
        .map(f64::from)
        .collect();
    let haar = haar_fractions(tail);
    if low == high {
        return PhaseSpectrum {
            fourier: [None; BANDS],
            haar,
            location,
        };
    }
    let center = mean(series(first, &chunks).map(f64::from), n);
    PhaseSpectrum {
        fourier: fourier_fractions(first, chunks, n, center),
        haar,
        location,
    }
}

/// Cierra un evento de la familia F: los 28 registros y las 2 ubicaciones con la regla de nulos
/// del contrato. El
/// acumulador se consume, así que su búfer se libera aquí.
///
/// Comprueba solo lo que el cálculo necesita para no producir `inf` ni `NaN`: anclas positivas y
/// una fase de confirmación con ticks. Un evento sobre el presupuesto sale con la fila toda nula
/// (salvo la clave) y sin comprobar nada: no se leyó ningún tick.
pub fn close_spectrum(acc: SpectrumAcc, event: &EventIn) -> Result<SpectrumRow, EventErrorKind> {
    let mut cells = [0i64; SPECTRUM_WIDTH];
    let mut valid = 1u32 << sc::CONFIRM_AGG_TRADE_ID;
    cells[sc::CONFIRM_AGG_TRADE_ID] = event.confirm;
    if acc.over_budget {
        return Ok(SpectrumRow {
            cells,
            valid,
            ticks: 0,
            over_budget: Some(event.extreme.saturating_sub(event.reference).max(0) as u64),
        });
    }
    if event.reference_price <= 0 {
        return Err(invariant(
            "reference_price",
            event.reference_price,
            "más de 0",
        ));
    }
    if event.confirm_price <= 0 {
        return Err(invariant("confirm_price", event.confirm_price, "más de 0"));
    }
    let SpectrumAcc {
        confirmation,
        overshoot,
        ..
    } = acc;
    if confirmation.ticks == 0 {
        return Err(invariant("c_ticks", 0i64, "más de 0"));
    }
    let ticks = confirmation.ticks + overshoot.ticks;
    let phases = [
        (
            confirmation,
            event.reference_price,
            [sc::C_FOURIER, sc::C_HAAR],
            sc::C_ENERGY_LOC,
        ),
        (
            overshoot,
            event.confirm_price,
            [sc::OS_FOURIER, sc::OS_HAAR],
            sc::OS_ENERGY_LOC,
        ),
    ];
    for (phase, anchor, [fourier_at, haar_at], location_at) in phases {
        if phase.ticks < N_MIN as u64 {
            continue;
        }
        let n = phase.ticks as usize;
        let first = log_return(anchor, phase.first_price);
        let PhaseSpectrum {
            fourier,
            haar,
            location,
        } = analyse(first, phase.chunks, n);
        for (base, fractions) in [(fourier_at, fourier), (haar_at, haar)] {
            for (band, fraction) in fractions.into_iter().enumerate() {
                if let Some(value) = fraction {
                    cells[base + band] = value.to_bits() as i64;
                    valid |= 1 << (base + band);
                }
            }
        }
        if let Some(value) = location {
            cells[location_at] = value.to_bits() as i64;
            valid |= 1 << location_at;
        }
    }
    Ok(SpectrumRow {
        cells,
        valid,
        ticks,
        over_budget: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fracciones de una serie dada entera: el primer valor hace de retorno del ancla.
    fn fractions_of(series: &[f32]) -> PhaseSpectrum {
        analyse(series[0], vec![series[1..].to_vec()], series.len())
    }

    fn near(got: Option<f64>, wanted: Option<f64>, what: &str) {
        match (got, wanted) {
            (None, None) => {}
            (Some(g), Some(w)) => assert!((g - w).abs() < 1e-6, "{what}: {g} y debería ser {w}"),
            other => panic!("{what}: {other:?}"),
        }
    }

    fn check(got: &Fractions, wanted: [Option<f64>; BANDS], what: &str) {
        for (band, (g, w)) in got.iter().zip(wanted).enumerate() {
            near(*g, w, &format!("{what}, banda {}", band + 1));
        }
    }

    #[test]
    fn only_big_phases_wait_for_their_turn_to_close() {
        // Solo esta prueba toma el turno con un `n` grande: las demás usan series chicas.
        assert!(close_turn(SERIAL_CLOSE_MIN - 1).is_none());
        let turn = close_turn(SERIAL_CLOSE_MIN).expect("una fase grande toma el candado");
        assert!(
            BIG_CLOSE.try_lock().is_err(),
            "el turno excluye al siguiente"
        );
        assert!(close_turn(N_MIN).is_none(), "una fase chica no espera");
        drop(turn);
        assert!(
            BIG_CLOSE.try_lock().is_ok(),
            "al soltar el turno queda libre"
        );
    }

    #[test]
    fn the_planner_is_the_scalar_one() {
        // El tipo lo fija: con otro planificador esto no compila. Y la misma entrada da los
        // mismos bits que armar el plan a mano con el planificador escalar.
        let _: FftPlannerScalar<f64> = planner();
        let input: Vec<Complex<f64>> = (0..45)
            .map(|j| Complex::new((j as f64 * 0.37).sin(), 0.0))
            .collect();
        let (mut a, mut b) = (input.clone(), input);
        planner().plan_fft_forward(45).process(&mut a);
        FftPlannerScalar::<f64>::new()
            .plan_fft_forward(45)
            .process(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn alternating_returns_put_all_the_energy_in_the_first_band() {
        let series: Vec<f32> = (0..64)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let got = fractions_of(&series);
        check(
            &got.fourier,
            [
                Some(1.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                None,
            ],
            "fourier",
        );
        // Haar: todo en el nivel 1 y los niveles 7 sin bins.
        check(
            &got.haar,
            [
                Some(1.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                None,
            ],
            "haar",
        );
    }

    #[test]
    fn a_sine_of_period_8_over_64_returns_leaves_five_sixths_and_one_sixth() {
        let series: Vec<f32> = (0..64)
            .map(|j| (2.0 * PI * j as f64 / 8.0).cos() as f32)
            .collect();
        let got = fractions_of(&series).fourier;
        // Hann deja 2/3 en el bin 8 y 1/6 en cada vecino: banda 3 = 5/6 y banda 2 = 1/6.
        check(
            &got,
            [
                Some(0.0),
                Some(1.0 / 6.0),
                Some(5.0 / 6.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                None,
            ],
            "fourier",
        );
    }

    #[test]
    fn the_bands_without_bins_are_null_not_zero() {
        let wave =
            |n: usize| -> Vec<f32> { (0..n).map(|j| ((j * 7919 % 101) as f32) - 50.0).collect() };
        for (n, present) in [
            (16, 4),
            (31, 4),
            (32, 5),
            (63, 5),
            (64, 6),
            (127, 6),
            (128, 7),
        ] {
            let got = fractions_of(&wave(n));
            for (what, fractions) in [("fourier", &got.fourier), ("haar", &got.haar)] {
                let nulls: Vec<_> = fractions.iter().map(Option::is_none).collect();
                let wanted: Vec<_> = (0..BANDS).map(|b| b >= present).collect();
                assert_eq!(nulls, wanted, "{what} con n = {n}");
                let sum: f64 = fractions.iter().flatten().sum();
                assert!((sum - 1.0).abs() < 1e-12, "{what} con n = {n}: {sum}");
            }
        }
    }

    #[test]
    fn haar_by_hand_for_a_step_and_for_the_last_power_of_two() {
        // Ocho retornos de −1 y ocho de +1: solo el nivel 4 (la escala 16) ve el escalón.
        let step: Vec<f32> = (0..16).map(|j| if j < 8 { -1.0 } else { 1.0 }).collect();
        check(
            &fractions_of(&step).haar,
            [Some(0.0), Some(0.0), Some(0.0), Some(1.0), None, None, None],
            "escalón",
        );
        // Con n = 20 solo cuentan los últimos 16: el primer retorno (un pico) no entra.
        let mut tail = vec![1000.0f32; 4];
        tail.extend(&step);
        check(
            &fractions_of(&tail).haar,
            [Some(0.0), Some(0.0), Some(0.0), Some(1.0), None, None, None],
            "n = 20",
        );
    }

    #[test]
    fn a_series_of_equal_returns_has_no_energy() {
        let got = fractions_of(&[0.001f32; 40]);
        assert_eq!(got.fourier, [None; BANDS]);
        assert_eq!(got.haar, [None; BANDS]);
        // La cola de Haar sí es plana aunque la serie entera no: solo Haar queda nulo.
        let mut series = vec![0.5f32; 40];
        series[0] = 3.0;
        let got = fractions_of(&series);
        assert_eq!(got.haar, [None; BANDS]);
        assert!(got.fourier.iter().flatten().count() == 5);
    }

    #[test]
    fn the_energy_location_by_hand() {
        // LOC_X de la sonda: 41,5/80 con n = 8 (los 8 retornos son el ancla y los siete que siguen).
        let loc_x = [6.0f32, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, -6.0];
        let got = energy_location(loc_x.iter().copied(), 8).unwrap();
        assert!((got - 41.5 / 80.0).abs() < 1e-15, "{got}");
        // Toda la energía en el primer retorno y en el último: ubicación de 1/(2n) y de 1 − 1/(2n).
        let mut series = [0.0f32; 16];
        series[0] = 1.0;
        assert_eq!(energy_location(series.into_iter(), 16), Some(0.5 / 16.0));
        series.reverse();
        assert_eq!(
            energy_location(series.into_iter(), 16),
            Some(1.0 - 0.5 / 16.0)
        );
    }

    #[test]
    fn the_energy_location_of_equal_returns_is_one_half_and_of_zeros_is_null() {
        // Retornos constantes y no nulos: sin Fourier ni Haar, pero con ubicación.
        let got = fractions_of(&[0.001f32; 40]);
        assert_eq!(got.fourier, [None; BANDS]);
        let loc = got.location.expect("la ubicación existe");
        assert!((loc - 0.5).abs() < 1e-15, "{loc}");
        // Todos en cero: la energía es cero y la ubicación es nula.
        assert_eq!(fractions_of(&[0.0f32; 40]).location, None);
    }

    #[test]
    fn joining_neighbours_gives_the_buffer_of_the_whole_in_any_association() {
        // Precios con pasos de distinto tamaño; cada tramo se arma como `accumulate`.
        let prices: Vec<i64> = vec![100, 103, 101, 101, 110, 95, 95, 97, 120, 119];
        let phase = |slice: &[i64]| PhaseReturns {
            ticks: slice.len() as u64,
            first_price: slice[0],
            last_price: slice[slice.len() - 1],
            chunks: if slice.len() > 1 {
                vec![slice.windows(2).map(|w| log_return(w[0], w[1])).collect()]
            } else {
                Vec::new()
            },
        };
        let whole: Vec<f32> = phase(&prices).returns().collect();
        assert_eq!(whole.len(), 9);
        for cuts in [[1, 2], [1, 5], [2, 4], [3, 9], [1, 9]] {
            let part = |a: usize, b: usize| phase(&prices[a..b]);
            let mut left = part(0, cuts[0]);
            left.join(part(cuts[0], cuts[1]));
            left.join(part(cuts[1], prices.len()));
            let mut right = part(cuts[0], cuts[1]);
            right.join(part(cuts[1], prices.len()));
            let mut other = part(0, cuts[0]);
            other.join(right);
            for (what, got) in [("(a b) c", &left), ("a (b c)", &other)] {
                let flat: Vec<f32> = got.returns().collect();
                assert_eq!(flat, whole, "{what} con los cortes {cuts:?}");
                assert_eq!((got.first_price, got.last_price), (100, 119));
                assert_eq!(got.stored(), 9);
            }
        }
    }

    #[test]
    fn the_returns_are_steps_of_the_log_price_from_the_anchor() {
        let r = log_return(100, 103);
        assert_eq!(r, (103.0f64 / 100.0).ln() as f32);
        assert_eq!(log_return(7, 7), 0.0);
        assert!(log_return(103, 100) < 0.0);
    }
}
