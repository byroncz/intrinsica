//! Los agregados de una fase, o de un tramo de ella: enteros exactos y asociativos.
//!
//! Conteos y sumas se suman, máximo y mínimo se comparan, y el primer y el último tiempo
//! son el mínimo y el máximo de los tiempos (`transact_time` no decrece en el orden de
//! la serie). Fusionar los tramos de un evento en cualquier orden da el mismo resultado
//! (ADR-L3-16). `Σ price·quantity` se acumula en `u128`: un tick aporta menos de `2^120`.

use crate::input::TickColumns;
use std::fmt;

/// Los agregados de una fase. Con `ticks == 0` el resto no significa nada.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhaseAcc {
    pub ticks: u64,
    pub buy: u128,
    pub sell: u128,
    /// `Σ price·quantity`, escala 10⁻¹⁶.
    pub notional: u128,
    /// La suma no cupo en 128 bits (se trata como error de fase, nunca como valor).
    pub overflow: bool,
    pub high: i64,
    pub low: i64,
    pub first_time: i64,
    pub last_time: i64,
}

impl PhaseAcc {
    /// Suma otro tramo de la misma fase, en cualquier orden.
    pub fn merge(&mut self, other: &PhaseAcc) {
        if other.ticks == 0 {
            return;
        }
        if self.ticks == 0 {
            *self = *other;
            return;
        }
        self.ticks += other.ticks;
        self.buy += other.buy;
        self.sell += other.sell;
        self.overflow |= other.overflow;
        match self.notional.checked_add(other.notional) {
            Some(sum) => self.notional = sum,
            None => {
                self.overflow = true;
                self.notional = u128::MAX;
            }
        }
        self.high = self.high.max(other.high);
        self.low = self.low.min(other.low);
        self.first_time = self.first_time.min(other.first_time);
        self.last_time = self.last_time.max(other.last_time);
    }
}

/// Los ticks de la confirmación con `transact_time = confirm_time` (ADR-L3-01).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InstantAcc {
    pub ticks: u64,
    pub high: i64,
    pub low: i64,
}

impl InstantAcc {
    pub fn merge(&mut self, other: &InstantAcc) {
        if other.ticks == 0 {
            return;
        }
        if self.ticks == 0 {
            *self = *other;
            return;
        }
        self.ticks += other.ticks;
        self.high = self.high.max(other.high);
        self.low = self.low.min(other.low);
    }
}

/// Los agregados de un evento, o de un tramo de él.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EventAcc {
    pub confirmation: PhaseAcc,
    pub overshoot: PhaseAcc,
    pub instant: InstantAcc,
}

impl EventAcc {
    pub fn merge(&mut self, other: &EventAcc) {
        self.confirmation.merge(&other.confirmation);
        self.overshoot.merge(&other.overshoot);
        self.instant.merge(&other.instant);
    }
}

/// Un tick que no cumple el contrato de L1 (`0 ≤ valor < 2^63`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickError {
    Negative,
    TooLarge,
}

impl fmt::Display for TickError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Negative => "un row group de L1 trae price o quantity negativos",
            Self::TooLarge => "un row group de L1 trae price o quantity que no caben en 63 bits",
        })
    }
}

impl std::error::Error for TickError {}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"))
}

/// Agrega los ticks `[start, end)` de una fase. Con `instant_time` cuenta además los
/// ticks de ese instante (solo en la confirmación).
///
/// Es el camino caliente: una pasada por tick, sin locks ni asignaciones.
pub fn aggregate<B: AsRef<[u8]>>(
    ticks: &TickColumns<B>,
    start: usize,
    end: usize,
    instant_time: Option<i64>,
) -> Result<(PhaseAcc, InstantAcc), TickError> {
    let mut acc = PhaseAcc::default();
    let mut instant = InstantAcc::default();
    if start >= end {
        return Ok((acc, instant));
    }
    let times = ticks.time_bytes(start, end);
    let (bits, first_bit) = ticks.sold_bits(start);
    let wanted = instant_time.unwrap_or(i64::MIN);
    let mut buy = 0u128;
    let mut sell = 0u128;
    let mut notional = 0u128;
    let mut overflow = false;
    let mut bad = 0u64;
    let (mut high, mut low) = (i64::MIN, i64::MAX);
    let (mut inst_high, mut inst_low) = (i64::MIN, i64::MAX);
    let mut inst_ticks = 0u64;
    let rows = ticks
        .price_bytes(start, end)
        .chunks_exact(16)
        .zip(ticks.quantity_bytes(start, end).chunks_exact(16))
        .zip(times.chunks_exact(8));
    for (i, ((price, quantity), time)) in rows.enumerate() {
        let (p, q) = (le_u64(price), le_u64(quantity));
        bad |= le_u64(&price[8..]) | le_u64(&quantity[8..]) | ((p | q) >> 63);
        let bit = first_bit + i;
        let sold = u64::from((bits[bit >> 3] >> (bit & 7)) & 1);
        let mask = 0u64.wrapping_sub(sold);
        sell += u128::from(q & mask);
        buy += u128::from(q & !mask);
        let (sum, over) = notional.overflowing_add(u128::from(p) * u128::from(q));
        notional = sum;
        overflow |= over;
        let price = p as i64;
        high = high.max(price);
        low = low.min(price);
        if i64::from_le_bytes(time.try_into().expect("8 bytes")) == wanted {
            inst_ticks += 1;
            inst_high = inst_high.max(price);
            inst_low = inst_low.min(price);
        }
    }
    if bad != 0 {
        return Err(classify(ticks, start, end));
    }
    acc.ticks = (end - start) as u64;
    acc.buy = buy;
    acc.sell = sell;
    acc.notional = notional;
    acc.overflow = overflow;
    acc.high = high;
    acc.low = low;
    acc.first_time = i64::from_le_bytes(times[..8].try_into().expect("8 bytes"));
    acc.last_time = i64::from_le_bytes(times[times.len() - 8..].try_into().expect("8 bytes"));
    if inst_ticks > 0 {
        instant = InstantAcc {
            ticks: inst_ticks,
            high: inst_high,
            low: inst_low,
        };
    }
    Ok((acc, instant))
}

/// Camino frío: distingue un valor negativo de uno que no cabe en 63 bits.
fn classify<B: AsRef<[u8]>>(ticks: &TickColumns<B>, start: usize, end: usize) -> TickError {
    let negative = [
        ticks.price_bytes(start, end),
        ticks.quantity_bytes(start, end),
    ]
    .iter()
    .any(|bytes| {
        bytes
            .chunks_exact(16)
            .any(|value| i128::from_le_bytes(value.try_into().expect("16 bytes")) < 0)
    });
    if negative {
        TickError::Negative
    } else {
        TickError::TooLarge
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc(ticks: u64, buy: u128, sell: u128, high: i64, low: i64, t0: i64, t1: i64) -> PhaseAcc {
        PhaseAcc {
            ticks,
            buy,
            sell,
            notional: u128::from(ticks) * 7,
            overflow: false,
            high,
            low,
            first_time: t0,
            last_time: t1,
        }
    }

    #[test]
    fn merge_is_commutative_and_empty_is_the_identity() {
        let a = acc(3, 5, 6, 90, 10, 100, 130);
        let b = acc(2, 1, 0, 95, 20, 140, 150);
        let (mut ab, mut ba) = (a, b);
        ab.merge(&b);
        ba.merge(&a);
        assert_eq!(ab, ba);
        assert_eq!((ab.ticks, ab.high, ab.low), (5, 95, 10));
        assert_eq!((ab.first_time, ab.last_time), (100, 150));
        let mut empty = PhaseAcc::default();
        empty.merge(&a);
        assert_eq!(empty, a);
        let mut same = a;
        same.merge(&PhaseAcc::default());
        assert_eq!(same, a);
    }

    #[test]
    fn a_notional_that_does_not_fit_is_flagged_not_wrapped() {
        let mut big = acc(1, 1, 0, 1, 1, 0, 0);
        big.notional = u128::MAX - 1;
        let mut other = big;
        other.notional = 5;
        big.merge(&other);
        assert!(big.overflow);
        assert_eq!(big.notional, u128::MAX);
        // El indicador es pegajoso: una fusión posterior no lo borra.
        big.merge(&acc(1, 1, 0, 1, 1, 0, 0));
        assert!(big.overflow);
    }
}
