//! Aritmética entera exacta del cierre de una fase.

/// `numerator / denominator` redondeado al entero más cercano, empates al par.
///
/// Es `divide_half_even` de `l3_dc_descriptors.phase`, sobre `u128`. El resto se
/// compara con `denominator - resto` y no con `2 · resto`, que desbordaría con un
/// divisor cercano a `u128::MAX`.
///
/// # Panics
/// Si `denominator` es 0.
pub fn divide_half_even(numerator: u128, denominator: u128) -> u128 {
    let quotient = numerator / denominator;
    let rest = numerator % denominator;
    let complement = denominator - rest;
    if rest > complement || (rest == complement && quotient % 2 == 1) {
        quotient + 1
    } else {
        quotient
    }
}

/// 2¹²⁸ como `f64` (exacto: es una potencia de dos).
const TWO_128: f64 = 340_282_366_920_938_463_463_374_607_431_768_211_456.0;

/// Un entero con signo de hasta 256 bits, en signo y magnitud: lo justo para multiplicar dos
/// `i128` y restar dos productos sin perder un bit. Cada operación que no cabe devuelve
/// `None` (nunca envuelve).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wide {
    negative: bool,
    high: u128,
    low: u128,
}

/// `a · b` en 256 bits, como `(parte alta, parte baja)`.
fn multiply(a: u128, b: u128) -> (u128, u128) {
    const HALF: u128 = u64::MAX as u128;
    let (a0, a1) = (a & HALF, a >> 64);
    let (b0, b1) = (b & HALF, b >> 64);
    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
    let middle = (p00 >> 64) + (p01 & HALF) + (p10 & HALF);
    let low = (p00 & HALF) | (middle << 64);
    let high = p11 + (p01 >> 64) + (p10 >> 64) + (middle >> 64);
    (high, low)
}

impl Wide {
    pub const ZERO: Wide = Wide {
        negative: false,
        high: 0,
        low: 0,
    };

    /// `a · b`, exacto.
    pub fn product(a: i128, b: i128) -> Self {
        let (high, low) = multiply(a.unsigned_abs(), b.unsigned_abs());
        Self {
            negative: (a < 0) != (b < 0) && (high | low) != 0,
            high,
            low,
        }
    }

    /// `self · 2^bits` con `0 ≤ bits < 128`; `None` si no cabe en 256 bits.
    pub fn shift_left(self, bits: u32) -> Option<Self> {
        assert!(bits < 128, "el desplazamiento es de menos de 128 bits");
        if bits == 0 {
            return Some(self);
        }
        if self.high >> (128 - bits) != 0 {
            return None;
        }
        Some(Self {
            negative: self.negative,
            high: (self.high << bits) | (self.low >> (128 - bits)),
            low: self.low << bits,
        })
    }

    fn compare_magnitude(&self, other: &Self) -> std::cmp::Ordering {
        (self.high, self.low).cmp(&(other.high, other.low))
    }

    /// `self + other`; `None` si la magnitud no cabe en 256 bits.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        if self.negative == other.negative {
            let (low, carry) = self.low.overflowing_add(other.low);
            let high = self
                .high
                .checked_add(other.high)?
                .checked_add(u128::from(carry))?;
            return Some(Self {
                negative: self.negative,
                high,
                low,
            });
        }
        let (big, small) = if self.compare_magnitude(&other).is_ge() {
            (self, other)
        } else {
            (other, self)
        };
        let (low, borrow) = big.low.overflowing_sub(small.low);
        let high = big.high - small.high - u128::from(borrow);
        Some(Self {
            negative: big.negative && (high | low) != 0,
            high,
            low,
        })
    }

    /// `self − other`; `None` si la magnitud no cabe en 256 bits.
    pub fn checked_sub(self, other: Self) -> Option<Self> {
        self.checked_add(Self {
            negative: !other.negative && (other.high | other.low) != 0,
            ..other
        })
    }

    pub fn is_negative(&self) -> bool {
        self.negative
    }

    pub fn is_zero(&self) -> bool {
        (self.high | self.low) == 0
    }

    /// El valor como `f64`: la magnitud se convierte por partes (un error de un par de
    /// unidades en el último lugar) y el signo se aplica al final.
    pub fn to_f64(self) -> f64 {
        let magnitude = self.high as f64 * TWO_128 + self.low as f64;
        if self.negative {
            -magnitude
        } else {
            magnitude
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{divide_half_even, Wide};

    fn wide(a: i128, b: i128) -> Wide {
        Wide::product(a, b)
    }

    #[test]
    fn a_product_of_two_i128_is_exact() {
        assert_eq!(wide(6, -7).to_f64(), -42.0);
        assert_eq!(wide(0, -7), Wide::ZERO);
        // (2^127 − 1)² = 2^254 − 2^128 + 1: la parte alta es 2^126 − 1.
        let big = i128::MAX;
        let square = wide(big, big);
        assert_eq!((square.high, square.low), ((1u128 << 126) - 1, 1));
        assert!(!square.is_negative());
        // i128::MIN² = 2^254.
        let min = wide(i128::MIN, i128::MIN);
        assert_eq!((min.high, min.low), (1u128 << 126, 0));
        assert_eq!(wide(i128::MIN, 1).to_f64(), -(2.0f64.powi(127)));
    }

    #[test]
    fn the_difference_of_two_products_has_the_right_sign_and_size() {
        let a = wide(10, 10);
        let b = wide(3, 33);
        assert_eq!(a.checked_sub(b).unwrap().to_f64(), 1.0);
        assert_eq!(b.checked_sub(a).unwrap().to_f64(), -1.0);
        assert!(a.checked_sub(a).unwrap().is_zero());
        assert!(!a.checked_sub(a).unwrap().is_negative());
        // Resta que cruza el límite de 128 bits en la parte baja: 2^200 − 1.
        let x = wide(1 << 100, 1 << 100);
        let y = wide(1, 1);
        let d = x.checked_sub(y).unwrap();
        assert_eq!((d.high, d.low), ((1 << 72) - 1, u128::MAX));
        // Suma de dos negativos y de dos grandes que desborda.
        assert_eq!(
            wide(-2, 3).checked_add(wide(-4, 5)).unwrap().to_f64(),
            -26.0
        );
        let max = wide(i128::MIN, i128::MIN);
        assert!(max.checked_add(max).is_some());
        let near_top = Wide {
            negative: false,
            high: u128::MAX,
            low: 0,
        };
        assert!(near_top.checked_add(near_top).is_none());
    }

    #[test]
    fn shifting_left_multiplies_by_a_power_of_two_or_reports_the_overflow() {
        assert_eq!(
            wide(3, -5).shift_left(80).unwrap().to_f64(),
            -15.0 * 2.0f64.powi(80)
        );
        assert_eq!(wide(3, 5).shift_left(0).unwrap(), wide(3, 5));
        let top = Wide {
            negative: false,
            high: 1 << 127,
            low: 0,
        };
        assert!(top.shift_left(1).is_none());
        let edge = wide(1 << 63, 1 << 63);
        assert_eq!(edge.shift_left(127).unwrap().to_f64(), 2.0f64.powi(253));
    }

    #[test]
    fn ties_go_to_the_even_neighbour() {
        // x.5 con x par baja, con x impar sube.
        assert_eq!(divide_half_even(1, 2), 0);
        assert_eq!(divide_half_even(3, 2), 2);
        assert_eq!(divide_half_even(5, 2), 2);
        assert_eq!(divide_half_even(7, 2), 4);
        assert_eq!(divide_half_even(25, 10), 2);
        assert_eq!(divide_half_even(35, 10), 4);
    }

    #[test]
    fn rounds_to_nearest_away_from_ties() {
        assert_eq!(divide_half_even(7, 3), 2);
        assert_eq!(divide_half_even(8, 3), 3);
        assert_eq!(divide_half_even(10, 5), 2);
        assert_eq!(divide_half_even(0, 7), 0);
    }

    #[test]
    fn exact_at_the_edges_of_u128() {
        assert_eq!(divide_half_even(u128::MAX, 1), u128::MAX);
        assert_eq!(divide_half_even(u128::MAX, u128::MAX), 1);
        // MAX = 2^128 - 1 es impar: MAX / 2 = 2^127 - 0,5 → empate, 2^127 - 1 es impar, sube.
        assert_eq!(divide_half_even(u128::MAX, 2), 1 << 127);
        assert_eq!(divide_half_even(u128::MAX - 1, 2), (1 << 127) - 1);
        // Un divisor mayor que 2^127: 2 · resto desbordaría.
        let d = (1u128 << 127) + 3;
        assert_eq!(divide_half_even(d - 1, d), 1);
        assert_eq!(divide_half_even(d / 2 - 1, d), 0);
    }
}
