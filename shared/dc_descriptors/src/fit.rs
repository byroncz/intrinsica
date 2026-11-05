//! La familia A (forma, `fit.parquet`): los momentos de una fase en coordenadas normalizadas
//! y el cierre de los 9 registros A-01 a A-09 (ADR-L3-18).
//!
//! Un evento tiene tres sistemas de coordenadas, uno por fase que se mide: la confirmación
//! `c` (ancla `(reference_time, reference_price)`, cierre `(confirm_time, confirm_price)`),
//! el overshoot `os` (de `confirm` a `extreme`) y el evento entero `ev` (de `reference` a
//! `extreme`). En cada uno, un tick `(t, p)` es el punto
//!
//! ```text
//!   τ = (t − t₀) / (t_fin − t₀)        y = (p − p₀) / (p_fin − p₀)
//! ```
//!
//! y la fase guarda los momentos `n`, `Σ τ^k` (k = 1..4) y `Σ τ^k·y` (k = 0..2). Cada término
//! se calcula en `f64` **una vez**, con la misma cuenta en cualquier tramo, y se cuantiza a
//! punto fijo de escala 2⁻⁸⁰ en `i128` antes de sumarse: la suma de enteros es exacta,
//! asociativa y conmutativa, así que un evento que cruza row groups se fusiona en cualquier
//! orden y el resultado no depende de cómo se cortó ni de qué hilo lo sumó. La escala no es
//! la 2⁻⁹⁶ de T: en el overshoot `y` sale de `[0, 1]` (con `p_E − p_C` de un paso de precio
//! y una excursión de hasta `θ·p_C`, `|y|` llega a miles), y con 2⁻⁹⁶ la suma de 10⁷ ticks
//! se desbordaría; con 2⁻⁸⁰ el rango es de 2⁴⁷ y el error por término, menor que 10⁻²⁴.
//!
//! Lo único que no es una suma es `p_prev`, el precio del tick con mayor `(t, agg_trade_id)`
//! entre los de la fase con `t` anterior al instante de cierre: se fusiona por máximo, que
//! también es asociativo y conmutativo. Del mismo modo se fusionan los rangos de tiempos que
//! deciden los nulos (el mínimo y el máximo de `t`, y los de los `t > t₀`).
//!
//! Al cerrar, los momentos entran en productos de enteros de 256 bits exactos
//! ([`Wide`](crate::arith::Wide)); la única cuenta en `f64` que queda es la división final.

use crate::arith::Wide;
use crate::close::{EventErrorKind, PhaseName};
use crate::input::{EventIn, TickColumns};
use crate::layout::{fit_col as fc, FitRow, FIT_WIDTH};
use crate::path::{classify, invariant};
use crate::phase::{le_u64, TickError};

/// Bits de fracción del punto fijo: la escala es 2⁻⁸⁰.
const FRACTION_BITS: u32 = 80;
/// 2⁸⁰: multiplicar por una potencia de dos no redondea.
const SCALE: f64 = 1_208_925_819_614_629_174_706_176.0;
/// Desde 2⁵² todo `f64` es un entero.
const INTEGER_FROM: f64 = 4_503_599_627_370_496.0;
const TWO_64: f64 = 18_446_744_073_709_551_616.0;
const UNIT_64: f64 = 1.0 / TWO_64;
const TWO_32: f64 = 4_294_967_296.0;
const UNIT_32: f64 = 1.0 / TWO_32;
/// Un término cuantizado de 2¹²⁴ o más (o no finito) cuenta como desborde: queda margen
/// para sumar 16 de ellos sin envolver.
const TERM_LIMIT: f64 = 21_267_647_932_558_653_966_460_912_964_485_513_216.0;

/// Posición de cada momento en `Moments::sums`.
const T1: usize = 0;
const T2: usize = 1;
const T3: usize = 2;
const T4: usize = 3;
const Y: usize = 4;
const TY: usize = 5;
const T2Y: usize = 6;
/// Cuántos momentos de suma guarda una fase (`n` aparte).
const MOMENTS: usize = 7;

/// El sistema de coordenadas de una fase de un evento: ancla `(t₀, p₀)`, cierre
/// `(t_fin, p_fin)` y las constantes con que se normaliza cada tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    t0: i64,
    p0: i64,
    t_fin: i64,
    p_fin: i64,
    /// `t_fin − t₀` como `f64`.
    span: f64,
    /// El divisor de `y`: `p_fin − p₀`, o `p₀` si el cierre vale lo mismo que el ancla (el
    /// eje `y` ya no está normalizado, pero la pendiente en precio no cambia; la aceleración
    /// y la participación del salto son nulas en ese caso).
    scale: f64,
    /// Hay coordenadas: `t_fin > t₀` y `scale ≠ 0`. Sin ellas solo se cuentan ticks y se
    /// guardan los rangos y `p_prev`.
    usable: bool,
}

impl Frame {
    fn new(t0: i64, p0: i64, t_fin: i64, p_fin: i64) -> Self {
        let scale = if p_fin == p0 {
            p0 as f64
        } else {
            (i128::from(p_fin) - i128::from(p0)) as f64
        };
        Self {
            t0,
            p0,
            t_fin,
            p_fin,
            span: (i128::from(t_fin) - i128::from(t0)) as f64,
            scale,
            usable: t_fin > t0 && scale != 0.0,
        }
    }

    /// Los tres sistemas de un evento: confirmación, overshoot y evento.
    pub fn of(event: &EventIn) -> [Frame; 3] {
        [
            Frame::new(
                event.reference_time,
                event.reference_price,
                event.confirm_time,
                event.confirm_price,
            ),
            Frame::new(
                event.confirm_time,
                event.confirm_price,
                event.extreme_time,
                event.extreme_price,
            ),
            Frame::new(
                event.reference_time,
                event.reference_price,
                event.extreme_time,
                event.extreme_price,
            ),
        ]
    }

    /// `p_fin = p₀`: sin `y` normalizado, aceleración y participación del salto no existen.
    fn flat(&self) -> bool {
        self.p_fin == self.p0
    }
}

/// `v` en punto fijo de escala 2⁻⁸⁰. Un valor no finito o de más de 2¹²⁴ marca desborde.
#[inline(always)]
fn quantise(v: f64, overflow: &mut bool) -> i128 {
    let scaled = v * SCALE;
    if scaled.abs() >= TERM_LIMIT || scaled.is_nan() {
        *overflow = true;
        return 0;
    }
    // `scaled as i128` y `round()` son llamadas a la biblioteca de ejecución; el camino caliente
    // usa tres conversiones de `f64` a `i64` (una instrucción cada una). Cada paso resta
    // un múltiplo entero de 2^32 que cabe exacto, así que las tres partes tienen el signo de
    // `scaled` y suman su redondeo al entero más cercano (empates, lejos del cero).
    // Desde 2^52 el valor ya es un entero y sumar 0,5 redondearía al par.
    let rounded = if scaled.abs() < INTEGER_FROM {
        scaled + 0.5f64.copysign(scaled)
    } else {
        scaled
    };
    let high = (rounded * UNIT_64) as i64;
    let rest = rounded - high as f64 * TWO_64;
    let middle = (rest * UNIT_32) as i64;
    let low = (rest - middle as f64 * TWO_32) as i64;
    (i128::from(high) << 64) + (i128::from(middle) << 32) + i128::from(low)
}

fn range(current: Option<(i64, i64)>, other: Option<(i64, i64)>) -> Option<(i64, i64)> {
    match (current, other) {
        (Some((a, b)), Some((c, d))) => Some((a.min(c), b.max(d))),
        (one, other) => one.or(other),
    }
}

/// Los momentos de una fase, o de un tramo de ella, en su sistema de coordenadas. Con
/// `n == 0` el resto no significa nada.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Moments {
    pub n: u64,
    /// `Στ, Στ², Στ³, Στ⁴, Σy, Στ·y, Στ²·y` en escala 2⁻⁸⁰.
    pub sums: [i128; MOMENTS],
    /// Algún término o alguna suma no cupo (error de fase, nunca un valor).
    pub overflow: bool,
    /// El menor y el mayor `t` de los ticks de la fase.
    pub times: Option<(i64, i64)>,
    /// El menor y el mayor `t` de los ticks con `t > t₀` (`τ > 0`).
    pub later: Option<(i64, i64)>,
    /// `(t, agg_trade_id, price)` del tick de mayor `(t, id)` con `t` anterior al cierre.
    pub before: Option<(i64, i64, i64)>,
}

impl Moments {
    /// Suma un tick de la fase. Es el camino caliente de la familia: una pasada por tick, sin
    /// locks ni asignaciones.
    #[inline]
    pub fn push(&mut self, frame: &Frame, time: i64, id: i64, price: i64) {
        self.n += 1;
        self.times = range(self.times, Some((time, time)));
        if time > frame.t0 {
            self.later = range(self.later, Some((time, time)));
        }
        if time < frame.t_fin && self.before.is_none_or(|b| (time, id) > (b.0, b.1)) {
            self.before = Some((time, id, price));
        }
        if !frame.usable {
            return;
        }
        let tau = time.wrapping_sub(frame.t0) as f64 / frame.span;
        let y = price.wrapping_sub(frame.p0) as f64 / frame.scale;
        let tau2 = tau * tau;
        let terms = [tau, tau2, tau2 * tau, tau2 * tau2, y, tau * y, tau2 * y];
        let mut overflow = false;
        for (sum, term) in self.sums.iter_mut().zip(terms) {
            let q = quantise(term, &mut overflow);
            match sum.checked_add(q) {
                Some(total) => *sum = total,
                None => {
                    overflow = true;
                    *sum = sum.saturating_add(q);
                }
            }
        }
        self.overflow |= overflow;
    }

    /// Une otro tramo de la misma fase, en cualquier orden y asociación.
    pub fn merge(&mut self, other: &Moments) {
        self.n += other.n;
        self.overflow |= other.overflow;
        for (sum, add) in self.sums.iter_mut().zip(other.sums) {
            match sum.checked_add(add) {
                Some(total) => *sum = total,
                None => {
                    self.overflow = true;
                    *sum = sum.saturating_add(add);
                }
            }
        }
        self.times = range(self.times, other.times);
        self.later = range(self.later, other.later);
        self.before = match (self.before, other.before) {
            (Some(a), Some(b)) => Some(if (b.0, b.1) > (a.0, a.1) { b } else { a }),
            (one, other) => one.or(other),
        };
    }
}

/// Los momentos de un evento, o de un tramo de él: sus tres fases.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FitAcc {
    pub confirmation: Moments,
    pub overshoot: Moments,
    pub event: Moments,
}

impl FitAcc {
    pub fn merge(&mut self, other: &FitAcc) {
        self.confirmation.merge(&other.confirmation);
        self.overshoot.merge(&other.overshoot);
        self.event.merge(&other.event);
    }
}

/// Suma los ticks `[from, to)` de una fase a su `Moments` y a los del evento.
fn scan<B: AsRef<[u8]>>(
    ticks: &TickColumns<B>,
    (from, to): (usize, usize),
    (phase_frame, event_frame): (&Frame, &Frame),
    (phase, event): (&mut Moments, &mut Moments),
) -> Result<(), TickError> {
    if from >= to {
        return Ok(());
    }
    let times = ticks.time_bytes(from, to);
    let prices = ticks.price_bytes(from, to);
    let mut bad = 0u64;
    for (k, (price, time)) in prices
        .chunks_exact(16)
        .zip(times.chunks_exact(8))
        .enumerate()
    {
        let p = le_u64(price);
        bad |= le_u64(&price[8..]) | (p >> 63);
        let time = i64::from_le_bytes(time.try_into().expect("8 bytes"));
        let id = ticks.keys().id(from + k);
        phase.push(phase_frame, time, id, p as i64);
        event.push(event_frame, time, id, p as i64);
    }
    if bad != 0 {
        return Err(classify(ticks, from, to));
    }
    Ok(())
}

/// Agrega los ticks `[start, split)` (la confirmación) y `[split, end)` (el overshoot) de un
/// evento en sus tres sistemas. Los ticks del evento entero van a `event` aparte, con sus
/// propias coordenadas.
pub fn accumulate<B: AsRef<[u8]>>(
    ticks: &TickColumns<B>,
    (start, split, end): (usize, usize, usize),
    frames: &[Frame; 3],
) -> Result<FitAcc, TickError> {
    let mut acc = FitAcc::default();
    let [c, o, v] = frames;
    scan(
        ticks,
        (start, split),
        (c, v),
        (&mut acc.confirmation, &mut acc.event),
    )?;
    scan(
        ticks,
        (split, end),
        (o, v),
        (&mut acc.overshoot, &mut acc.event),
    )?;
    Ok(acc)
}

/// Los tres registros de una fase cerrada.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Registers {
    slope: Option<f64>,
    accel: Option<f64>,
    jump_share: Option<f64>,
}

/// Una suma de enteros de 256 bits no cupo.
struct TooBig;

/// `a·b − c·d` exacto, o el desborde.
fn cross(a: i128, b: i128, c: i128, d: i128) -> Result<Wide, TooBig> {
    Wide::product(a, b)
        .checked_sub(Wide::product(c, d))
        .ok_or(TooBig)
}

/// `numerator / denominator` si el denominador es positivo y el cociente finito.
fn quotient(numerator: Wide, denominator: Wide) -> Option<f64> {
    if denominator.is_negative() || denominator.is_zero() {
        return None;
    }
    let q = numerator.to_f64() / denominator.to_f64();
    q.is_finite().then_some(q)
}

/// Pendiente de mínimos cuadrados de `y` sobre `τ` con intercepto, en fracción del precio de
/// referencia por segundo: `b · scale / (p_R · Δt_s)`. Nula con menos de dos ticks o con todos
/// en el mismo `t`.
fn slope(m: &Moments, f: &Frame, reference_price: i64) -> Result<Option<f64>, TooBig> {
    if !f.usable || m.n < 2 || m.times.is_none_or(|(low, high)| low == high) {
        return Ok(None);
    }
    let n = i128::from(m.n);
    let s = &m.sums;
    let numerator = Wide::product(n, s[TY])
        .shift_left(FRACTION_BITS)
        .and_then(|scaled| scaled.checked_sub(Wide::product(s[T1], s[Y])))
        .ok_or(TooBig)?;
    let denominator = Wide::product(n, s[T2])
        .shift_left(FRACTION_BITS)
        .and_then(|scaled| scaled.checked_sub(Wide::product(s[T1], s[T1])))
        .ok_or(TooBig)?;
    Ok(quotient(numerator, denominator)
        .map(|b| b * f.scale * 1e6 / (reference_price as f64 * f.span))
        .filter(|v| v.is_finite()))
}

/// El coeficiente `a` del ajuste `y = a·τ² + b·τ` sin intercepto, desde los momentos. Nulo
/// donde la pendiente lo es, con `t_fin = t₀`, con `p_fin = p₀` y con el sistema singular: un
/// solo `τ` positivo distinto, o `det ≤ 0` tras cuantizar (lo decide `quotient`).
fn acceleration(m: &Moments, f: &Frame) -> Result<Option<f64>, TooBig> {
    let distinct = m.later.is_some_and(|(low, high)| low < high);
    if !f.usable
        || f.flat()
        || m.n < 2
        || m.times.is_none_or(|(low, high)| low == high)
        || !distinct
    {
        return Ok(None);
    }
    let s = &m.sums;
    let numerator = cross(s[T2Y], s[T2], s[TY], s[T3])?;
    let denominator = cross(s[T4], s[T2], s[T3], s[T3])?;
    Ok(quotient(numerator, denominator))
}

/// `(p_fin − p_prev) / (p_fin − p₀)`, con `p_prev = p₀` si ningún tick es anterior al cierre.
/// Nula con `p_fin = p₀`.
fn jump_share(m: &Moments, f: &Frame) -> Option<f64> {
    if f.flat() {
        return None;
    }
    let prev = m.before.map_or(f.p0, |(_, _, price)| price);
    let share = (i128::from(f.p_fin) - i128::from(prev)) as f64
        / (i128::from(f.p_fin) - i128::from(f.p0)) as f64;
    share.is_finite().then_some(share)
}

fn close_phase(
    m: &Moments,
    f: &Frame,
    reference_price: i64,
    phase: PhaseName,
) -> Result<Registers, EventErrorKind> {
    let overflow = EventErrorKind::FitOverflow { phase };
    if m.overflow {
        return Err(overflow);
    }
    let slope = slope(m, f, reference_price).map_err(|_| overflow.clone())?;
    let accel = acceleration(m, f).map_err(|_| overflow)?;
    Ok(Registers {
        slope,
        accel,
        jump_share: jump_share(m, f),
    })
}

/// Cierra un evento de la familia A: los 9 registros con la regla de nulos del contrato.
///
/// Comprueba solo lo que el cálculo necesita para no producir `inf` ni `NaN`: dirección,
/// anclas positivas, tiempos que no decrecen de `R` a `C` y a `E` y una confirmación con
/// ticks. Los invariantes de §7.2 contra L2 los comprueba S.
pub fn close_fit(acc: &FitAcc, event: &EventIn) -> Result<FitRow, EventErrorKind> {
    if event.direction != 1 && event.direction != -1 {
        return Err(EventErrorKind::Direction {
            got: event.direction,
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
    if event.confirm_time < event.reference_time {
        return Err(invariant(
            "confirm_time",
            event.confirm_time,
            format!("al menos {}, el reference_time", event.reference_time),
        ));
    }
    if event.extreme_time < event.confirm_time {
        return Err(invariant(
            "extreme_time",
            event.extreme_time,
            format!("al menos {}, el confirm_time", event.confirm_time),
        ));
    }
    let (c, o, v) = (&acc.confirmation, &acc.overshoot, &acc.event);
    if c.n == 0 {
        return Err(invariant("c_ticks", 0i64, "más de 0"));
    }
    let [frame_c, frame_o, frame_v] = Frame::of(event);
    let reference = event.reference_price;
    let confirmation = close_phase(c, &frame_c, reference, PhaseName::Confirmation)?;
    let whole = close_phase(v, &frame_v, reference, PhaseName::Event)?;
    let overshoot = if o.n > 0 {
        close_phase(o, &frame_o, reference, PhaseName::Overshoot)?
    } else {
        Registers::default()
    };

    let mut cells = [0i64; FIT_WIDTH];
    cells[fc::CONFIRM_AGG_TRADE_ID] = event.confirm;
    let mut valid = 1u16 << fc::CONFIRM_AGG_TRADE_ID;
    let mut set = |column: usize, value: Option<f64>| {
        if let Some(value) = value {
            cells[column] = value.to_bits() as i64;
            valid |= 1 << column;
        }
    };
    set(fc::C_SLOPE, confirmation.slope);
    set(fc::OS_SLOPE, overshoot.slope);
    set(fc::EV_SLOPE, whole.slope);
    set(fc::C_ACCEL, confirmation.accel);
    set(fc::OS_ACCEL, overshoot.accel);
    set(fc::EV_ACCEL, whole.accel);
    set(fc::C_JUMP_SHARE, confirmation.jump_share);
    set(fc::OS_JUMP_SHARE, overshoot.jump_share);
    set(fc::EV_JUMP_SHARE, whole.jump_share);
    Ok(FitRow {
        cells,
        valid,
        ticks: c.n + o.n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Un tick: `(t, agg_trade_id, price)`.
    type Tick = (i64, i64, i64);

    fn moments(frame: &Frame, ticks: &[Tick]) -> Moments {
        let mut m = Moments::default();
        for &(t, id, p) in ticks {
            m.push(frame, t, id, p);
        }
        m
    }

    fn near(got: Option<f64>, wanted: f64, what: &str) {
        let got = got.unwrap_or_else(|| panic!("{what} es nulo"));
        assert!(
            (got - wanted).abs() <= 1e-12 * wanted.abs().max(1.0),
            "{what}: {got} y debería ser {wanted}"
        );
    }

    #[test]
    fn the_fast_quantiser_rounds_to_the_nearest_integer_of_the_scaled_value() {
        // Contra la definición directa `(v · 2⁸⁰).round() as i128`, con valores de todos los
        // tamaños y signos (xorshift: sin dependencias y siempre los mismos).
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let mut overflow = false;
        for _ in 0..200_000 {
            let mantissa = (next() >> 11) as f64 / (1u64 << 53) as f64;
            let exponent = (next() % 90) as i32 - 60;
            let sign = if next() % 2 == 0 { 1.0 } else { -1.0 };
            let v = sign * mantissa * 2.0f64.powi(exponent);
            let direct = (v * SCALE).round() as i128;
            assert_eq!(quantise(v, &mut overflow), direct, "{v:e}");
        }
        assert!(!overflow);
        // Empates: lejos del cero, como `round`.
        for tie in [0.5, 1.5, 2.5, 123_456_789.5] {
            for v in [tie, -tie] {
                assert_eq!(quantise(v / SCALE, &mut overflow), v.round() as i128, "{v}");
            }
        }
        // El límite: justo debajo cabe, desde ahí es un desborde.
        assert_eq!(
            quantise(TERM_LIMIT * 0.99 / SCALE, &mut overflow),
            (TERM_LIMIT * 0.99) as i128
        );
        assert!(!overflow);
        quantise(TERM_LIMIT / SCALE, &mut overflow);
        assert!(overflow);
    }

    #[test]
    fn a_term_is_quantised_once_to_a_power_of_two_scale() {
        let mut overflow = false;
        assert_eq!(quantise(1.0, &mut overflow), 1i128 << 80);
        assert_eq!(quantise(-0.5, &mut overflow), -(1i128 << 79));
        assert!(!overflow);
        quantise(f64::NAN, &mut overflow);
        assert!(overflow);
        let mut overflow = false;
        quantise(1e300, &mut overflow);
        assert!(overflow);
        // 2⁻⁸¹ y menos se redondea a 0 o a 1 unidad: el error por término es ≤ 2⁻⁸¹.
        let mut overflow = false;
        assert_eq!(quantise(2.0f64.powi(-82), &mut overflow), 0);
    }

    #[test]
    fn merging_the_pieces_in_any_order_gives_the_same_bits() {
        let frame = Frame::new(100, 1_000, 160, 1_060);
        let ticks: Vec<Tick> = vec![
            (100, 1, 1_000),
            (110, 2, 1_012),
            (110, 3, 1_020),
            (125, 4, 1_015),
            (150, 5, 1_040),
            (160, 6, 1_060),
        ];
        let whole = moments(&frame, &ticks);
        for cuts in [[1, 2], [1, 5], [2, 4], [3, 4], [1, 3], [2, 5]] {
            let parts = [
                moments(&frame, &ticks[..cuts[0]]),
                moments(&frame, &ticks[cuts[0]..cuts[1]]),
                moments(&frame, &ticks[cuts[1]..]),
            ];
            for order in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let mut merged = Moments::default();
                for k in order {
                    merged.merge(&parts[k]);
                }
                assert_eq!(merged, whole, "cortes {cuts:?}, orden {order:?}");
            }
            // (a b) c y a (b c).
            let mut right = parts[1];
            right.merge(&parts[2]);
            let mut other = parts[0];
            other.merge(&right);
            assert_eq!(other, whole, "a (b c) con los cortes {cuts:?}");
        }
    }

    #[test]
    fn p_prev_is_the_greatest_time_and_id_before_the_close() {
        let frame = Frame::new(100, 1_000, 160, 1_060);
        // Dos ticks en t = 150; gana el de mayor id. El de t = 160 no es anterior al cierre.
        let ticks = [
            (100, 1, 1_000),
            (150, 4, 1_020),
            (150, 5, 1_030),
            (160, 6, 1_060),
        ];
        assert_eq!(moments(&frame, &ticks).before, Some((150, 5, 1_030)));
        // Ningún tick es anterior: p_prev queda sin definir y jump_share vale 1.
        let late = moments(&frame, &[(160, 6, 1_060)]);
        assert_eq!(late.before, None);
        assert_eq!(jump_share(&late, &frame), Some(1.0));
    }

    #[test]
    fn a_linear_phase_has_exact_slope_and_zero_acceleration() {
        // y = τ: precios y tiempos en línea recta de (100, 1 000) a (160, 1 060) con 4 ticks.
        let frame = Frame::new(100, 1_000, 160, 1_060);
        let ticks = [
            (100, 1, 1_000),
            (120, 2, 1_020),
            (140, 3, 1_040),
            (160, 4, 1_060),
        ];
        let m = moments(&frame, &ticks);
        // (Δp / p_R) / (Δt en s) = (60 / 1 000) / (60 / 10⁶) = 1 000 por segundo.
        near(slope(&m, &frame, 1_000).ok().flatten(), 1_000.0, "slope");
        let a = acceleration(&m, &frame).ok().flatten().unwrap();
        assert!(a.abs() < 1e-12, "{a}");
        // El último tick es el cierre (t = t_fin): p_prev = 1 040, salto = 20 / 60.
        near(jump_share(&m, &frame), 1.0 / 3.0, "jump_share");
    }

    #[test]
    fn a_parabolic_phase_has_the_exact_coefficient() {
        // y = 0,75·τ² + 0,25·τ → a = 0,75, b = 0,25 en τ = 0, ¼, ½, ¾, 1 con 5 ticks.
        let frame = Frame::new(0, 0, 400, 1_600);
        let y = |tau: f64| 0.75 * tau * tau + 0.25 * tau;
        let ticks: Vec<Tick> = (0..5)
            .map(|k| {
                let tau = k as f64 / 4.0;
                (100 * k, k + 1, (y(tau) * 1_600.0).round() as i64)
            })
            .collect();
        // 1 600·y(τ) es entero en cada τ: 0, 700·… se comprueba que no se redondeó.
        for (k, &(_, _, p)) in ticks.iter().enumerate() {
            let tau = k as f64 / 4.0;
            assert_eq!(p as f64, y(tau) * 1_600.0, "tick {k}");
        }
        let m = moments(&frame, &ticks);
        near(acceleration(&m, &frame).ok().flatten(), 0.75, "accel");
    }

    #[test]
    fn the_nulls_follow_the_rule() {
        let frame = Frame::new(100, 1_000, 160, 1_060);
        // Un tick, o varios en el mismo t: sin pendiente ni aceleración.
        let one = moments(&frame, &[(130, 1, 1_030)]);
        assert_eq!(slope(&one, &frame, 1_000).ok().flatten(), None);
        assert_eq!(acceleration(&one, &frame).ok().flatten(), None);
        let same = moments(&frame, &[(130, 1, 1_030), (130, 2, 1_031)]);
        assert_eq!(slope(&same, &frame, 1_000).ok().flatten(), None);
        assert_eq!(acceleration(&same, &frame).ok().flatten(), None);
        // Dos ticks en t = t₀ y t > t₀: la pendiente existe, la aceleración (un solo τ > 0) no.
        let two = moments(&frame, &[(100, 1, 1_000), (130, 2, 1_030)]);
        assert!(slope(&two, &frame, 1_000).ok().flatten().is_some());
        assert_eq!(acceleration(&two, &frame).ok().flatten(), None);
        // t_fin = t₀: nada de coordenadas.
        let instant = Frame::new(100, 1_000, 100, 1_060);
        let m = moments(&instant, &[(100, 1, 1_000), (100, 2, 1_060)]);
        assert_eq!(slope(&m, &instant, 1_000).ok().flatten(), None);
        assert_eq!(acceleration(&m, &instant).ok().flatten(), None);
        // Ningún tick es anterior al cierre (t = t_fin = t₀): el salto vale 1.
        assert_eq!(jump_share(&m, &instant), Some(1.0));
        // p_fin = p₀: pendiente sí (en precio), aceleración y salto no.
        let flat = Frame::new(100, 1_000, 160, 1_000);
        let m = moments(&flat, &[(100, 1, 1_000), (130, 2, 1_030), (160, 3, 1_000)]);
        assert!(flat.usable && flat.flat());
        assert!(slope(&m, &flat, 1_000).ok().flatten().is_some());
        assert_eq!(acceleration(&m, &flat).ok().flatten(), None);
        assert_eq!(jump_share(&m, &flat), None);
    }

    #[test]
    fn the_slope_is_the_least_squares_one_of_price_over_reference_per_second() {
        let ticks = [(0, 1, 2_000), (100, 2, 2_010), (200, 3, 2_030)];
        let a = Frame::new(0, 2_000, 200, 2_030);
        let m = moments(&a, &ticks);
        // Pendiente de mínimos cuadrados de p/2 000 sobre t en s: Σ(t−t̄)(p−p̄)/Σ(t−t̄)².
        // t̄ = 100, p̄ = 2 013,33…: (−100·−13,33 + 0 + 100·16,66)/20 000 = 0,15 por µs.
        let by_hand = 0.15 / 2_000.0 * 1e6;
        near(slope(&m, &a, 2_000).ok().flatten(), by_hand, "slope");
    }

    #[test]
    fn a_term_beyond_the_range_is_an_overflow_not_a_wrap() {
        let frame = Frame::new(0, 1_000, 10, 1_001);
        let mut m = moments(&frame, &[(5, 1, 1_000)]);
        m.sums[Y] = i128::MAX - 1;
        m.push(&frame, 5, 2, 1_001);
        assert!(m.overflow);
        let mut other = Moments::default();
        other.push(&frame, 5, 3, 1_000);
        other.sums[T1] = i128::MAX;
        let mut fresh = moments(&frame, &[(5, 1, 1_000)]);
        fresh.merge(&other);
        assert!(fresh.overflow);
        assert!(slope(&fresh, &frame, 1_000).is_ok());
    }
}
