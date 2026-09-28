//! Núcleo de Directional Change (DC): detección de puntos de inflexión en
//! series de precios. Crate mínimo (ITSC-238) que fija el toolchain de Rust
//! del repo; la lógica completa de detección llega en E4.

/// Verdadero si el cambio relativo de precio alcanza el umbral que define un
/// evento DC.
pub fn is_directional_change(price_change: f64, threshold: f64) -> bool {
    price_change.abs() >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_change_at_threshold() {
        assert!(is_directional_change(0.02, 0.02));
        assert!(!is_directional_change(0.01, 0.02));
    }
}
