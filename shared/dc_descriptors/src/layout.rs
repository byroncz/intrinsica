//! Las 18 columnas de datos de `summaries.parquet` (TRD-L3 §7.2), las 14 de `path.parquet`
//! (la clave y los 13 registros de la familia T) y la fila de un evento de cada una.
//!
//! Es el contrato con `l3_dc_descriptors.schema` (`_COLUMNS` y `PATH_SCHEMA`): mismo orden,
//! mismos tipos. Los decimales (`DECIMAL(18,8)`) viajan sin escalar: el entero de 64 bits es
//! la palabra baja del decimal de 128 bits y la alta es su extensión de signo. Los `DOUBLE`
//! de `path.parquet` viajan como los 64 bits del `f64`.

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

/// Qué familia calcula el motor: la de resumen (`summaries.parquet`, S) o la de trayectoria
/// (`path.parquet`, T).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Family {
    #[default]
    Summaries,
    Path,
}

impl Family {
    /// Las columnas de datos que la familia entrega, en orden.
    pub const fn columns(self) -> &'static [ColumnSpec] {
        match self {
            Self::Summaries => &COLUMNS,
            Self::Path => &PATH_COLUMNS,
        }
    }
}

/// Columnas de datos por fila de `path.parquet` (sin `theta`): la clave y los 13 registros.
pub const PATH_WIDTH: usize = 14;

pub const PATH_COLUMNS: [ColumnSpec; PATH_WIDTH] = [
    int("confirm_agg_trade_id"),
    int("c_realized_var"),
    nullable(int("os_realized_var")),
    int("c_tortuosity"),
    nullable(int("os_tortuosity")),
    nullable(int("c_sign_autocorr")),
    nullable(int("os_sign_autocorr")),
    int("c_max_gap"),
    nullable(int("os_max_gap")),
    nullable(int("c_idle_fraction")),
    nullable(int("os_idle_fraction")),
    nullable(int("c_t25")),
    nullable(int("c_t50")),
    nullable(int("c_t75")),
];

/// Posición de cada columna de `path.parquet` en la fila.
pub mod path_col {
    pub const CONFIRM_AGG_TRADE_ID: usize = 0;
    pub const C_REALIZED_VAR: usize = 1;
    pub const OS_REALIZED_VAR: usize = 2;
    pub const C_TORTUOSITY: usize = 3;
    pub const OS_TORTUOSITY: usize = 4;
    pub const C_SIGN_AUTOCORR: usize = 5;
    pub const OS_SIGN_AUTOCORR: usize = 6;
    pub const C_MAX_GAP: usize = 7;
    pub const OS_MAX_GAP: usize = 8;
    pub const C_IDLE_FRACTION: usize = 9;
    pub const OS_IDLE_FRACTION: usize = 10;
    pub const C_T25: usize = 11;
    pub const C_T50: usize = 12;
    pub const C_T75: usize = 13;
}

/// La fila de un evento de `path.parquet`: `cells[c]` es la columna `c` (un `int64`, o los
/// bits de un `f64`), `valid` tiene el bit `c` en 1 si la columna `c` no es nula (una
/// celda nula vale 0) y `ticks` son los ticks de sus dos fases, para las cuentas del θ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathRow {
    pub cells: [i64; PATH_WIDTH],
    pub valid: u16,
    pub ticks: u64,
}

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
    fn the_path_columns_are_the_key_and_the_thirteen_registers() {
        assert_eq!(PATH_COLUMNS[path_col::C_T75].name, "c_t75");
        assert_eq!(PATH_COLUMNS[PATH_WIDTH - 1].name, "c_t75");
        assert!(PATH_COLUMNS.iter().all(|c| !c.wide));
        let required: Vec<_> = PATH_COLUMNS
            .iter()
            .filter(|c| !c.nullable)
            .map(|c| c.name)
            .collect();
        assert_eq!(
            required,
            [
                "confirm_agg_trade_id",
                "c_realized_var",
                "c_tortuosity",
                "c_max_gap"
            ]
        );
    }

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
