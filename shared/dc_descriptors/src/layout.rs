//! Las 18 columnas de datos de `summaries.parquet` (TRD-L3 §7.2) y la fila de un evento.
//!
//! Es el contrato con `l3_dc_descriptors.schema` (`_COLUMNS`): mismo orden, mismos
//! tipos. Los decimales (`DECIMAL(18,8)`) viajan sin escalar: el entero de 64 bits es la
//! palabra baja del decimal de 128 bits y la alta es su extensión de signo.

/// Columnas de datos por fila (sin `theta`, que es constante por row group).
pub const WIDTH: usize = 18;

/// Los valores de `DECIMAL(18,8)` son menores que `10^18` sin escalar.
pub const LIMIT: i64 = 1_000_000_000_000_000_000;

/// Una columna de datos: su nombre, si es `decimal128` (16 bytes) o `int64` (8) y si
/// admite nulos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnSpec {
    pub name: &'static str,
    pub wide: bool,
    pub nullable: bool,
}

impl ColumnSpec {
    /// Bytes por fila de la columna.
    pub const fn bytes(&self) -> usize {
        if self.wide {
            16
        } else {
            8
        }
    }
}

const fn int(name: &'static str) -> ColumnSpec {
    ColumnSpec {
        name,
        wide: false,
        nullable: false,
    }
}

const fn dec(name: &'static str) -> ColumnSpec {
    ColumnSpec {
        name,
        wide: true,
        nullable: false,
    }
}

const fn nullable(spec: ColumnSpec) -> ColumnSpec {
    ColumnSpec {
        nullable: true,
        ..spec
    }
}

pub const COLUMNS: [ColumnSpec; WIDTH] = [
    int("confirm_agg_trade_id"),
    int("c_ticks"),
    dec("c_quantity_buy"),
    dec("c_quantity_sell"),
    dec("c_price_max"),
    dec("c_price_min"),
    int("c_duration"),
    dec("c_vwap"),
    int("os_ticks"),
    dec("os_quantity_buy"),
    dec("os_quantity_sell"),
    nullable(dec("os_price_max")),
    nullable(dec("os_price_min")),
    nullable(int("os_duration")),
    nullable(dec("os_vwap")),
    int("confirm_instant_ticks"),
    dec("confirm_instant_price_best"),
    dec("confirm_instant_price_worst"),
];

/// Posición de cada columna en la fila.
pub mod col {
    pub const CONFIRM_AGG_TRADE_ID: usize = 0;
    pub const C_TICKS: usize = 1;
    pub const C_QUANTITY_BUY: usize = 2;
    pub const C_QUANTITY_SELL: usize = 3;
    pub const C_PRICE_MAX: usize = 4;
    pub const C_PRICE_MIN: usize = 5;
    pub const C_DURATION: usize = 6;
    pub const C_VWAP: usize = 7;
    pub const OS_TICKS: usize = 8;
    pub const OS_QUANTITY_BUY: usize = 9;
    pub const OS_QUANTITY_SELL: usize = 10;
    pub const OS_PRICE_MAX: usize = 11;
    pub const OS_PRICE_MIN: usize = 12;
    pub const OS_DURATION: usize = 13;
    pub const OS_VWAP: usize = 14;
    pub const CONFIRM_INSTANT_TICKS: usize = 15;
    pub const CONFIRM_INSTANT_PRICE_BEST: usize = 16;
    pub const CONFIRM_INSTANT_PRICE_WORST: usize = 17;
}

/// La fila de un evento cerrado: `WIDTH` enteros, los decimales sin escalar. Las
/// columnas `os_*` nulables van en 0 cuando el overshoot es vacío (`os_ticks` = 0).
pub type Row = [i64; WIDTH];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nullable_columns_are_the_overshoot_ones() {
        let names: Vec<_> = COLUMNS
            .iter()
            .filter(|c| c.nullable)
            .map(|c| c.name)
            .collect();
        assert_eq!(
            names,
            ["os_price_max", "os_price_min", "os_duration", "os_vwap"]
        );
        assert_eq!(COLUMNS[col::OS_TICKS].name, "os_ticks");
        assert_eq!(
            COLUMNS[col::CONFIRM_INSTANT_PRICE_WORST].name,
            COLUMNS[WIDTH - 1].name
        );
    }
}
