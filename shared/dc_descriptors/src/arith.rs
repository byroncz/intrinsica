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

#[cfg(test)]
mod tests {
    use super::divide_half_even;

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
