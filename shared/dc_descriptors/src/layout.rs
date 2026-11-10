//! Las 18 columnas de datos de `summaries.parquet` (TRD-L3 §7.2), las 14 de `path.parquet`
//! (la clave y los 13 registros de la familia T), las 10 de `fit.parquet` (la clave y los 9
//! de la familia A), las 29 de `spectrum.parquet` (la clave y las 28 de la familia F) y la
//! fila de un evento de cada una.
//!
//! Es el contrato con `l3_dc_descriptors.schema` (`_COLUMNS`, `PATH_SCHEMA`, `FIT_SCHEMA` y
//! `SPECTRUM_SCHEMA`): mismo orden, mismos tipos. Los decimales (`DECIMAL(18,8)`) viajan sin escalar: el entero de 64 bits es
//! la palabra baja del decimal de 128 bits y la alta es su extensión de signo. Los `DOUBLE`
//! de `path.parquet` y `fit.parquet` viajan como los 64 bits del `f64`.

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

/// Qué familia calcula el motor: la de resumen (`summaries.parquet`, S), la de trayectoria
/// (`path.parquet`, T), la de forma (`fit.parquet`, A) o la de frecuencia (`spectrum.parquet`, F).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Family {
    #[default]
    Summaries,
    Path,
    Fit,
    Spectrum,
}

impl Family {
    /// Las columnas de datos que la familia entrega, en orden.
    pub const fn columns(self) -> &'static [ColumnSpec] {
        match self {
            Self::Summaries => &COLUMNS,
            Self::Path => &PATH_COLUMNS,
            Self::Fit => &FIT_COLUMNS,
            Self::Spectrum => &SPECTRUM_COLUMNS,
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

/// Columnas de datos por fila de `fit.parquet` (sin `theta`): la clave y los 9 registros.
pub const FIT_WIDTH: usize = 10;

pub const FIT_COLUMNS: [ColumnSpec; FIT_WIDTH] = [
    int("confirm_agg_trade_id"),
    nullable(int("c_slope")),
    nullable(int("os_slope")),
    nullable(int("ev_slope")),
    nullable(int("c_accel")),
    nullable(int("os_accel")),
    nullable(int("ev_accel")),
    nullable(int("c_jump_share")),
    nullable(int("os_jump_share")),
    nullable(int("ev_jump_share")),
];

/// Columnas de datos por fila de `spectrum.parquet` (sin `theta`): la clave y 28 de F, una por
/// banda o nivel (7) y por registro (F-01 a F-04).
pub const SPECTRUM_WIDTH: usize = 29;

const fn band(name: &'static str) -> ColumnSpec {
    nullable(int(name))
}

pub const SPECTRUM_COLUMNS: [ColumnSpec; SPECTRUM_WIDTH] = [
    int("confirm_agg_trade_id"),
    band("c_fourier_band_1"),
    band("c_fourier_band_2"),
    band("c_fourier_band_3"),
    band("c_fourier_band_4"),
    band("c_fourier_band_5"),
    band("c_fourier_band_6"),
    band("c_fourier_band_7"),
    band("os_fourier_band_1"),
    band("os_fourier_band_2"),
    band("os_fourier_band_3"),
    band("os_fourier_band_4"),
    band("os_fourier_band_5"),
    band("os_fourier_band_6"),
    band("os_fourier_band_7"),
    band("c_haar_level_1"),
    band("c_haar_level_2"),
    band("c_haar_level_3"),
    band("c_haar_level_4"),
    band("c_haar_level_5"),
    band("c_haar_level_6"),
    band("c_haar_level_7"),
    band("os_haar_level_1"),
    band("os_haar_level_2"),
    band("os_haar_level_3"),
    band("os_haar_level_4"),
    band("os_haar_level_5"),
    band("os_haar_level_6"),
    band("os_haar_level_7"),
];

/// Posición de cada columna de `spectrum.parquet` en la fila: la banda o nivel `b` (1 a 7) de
/// cada registro está en `BASE + b - 1`.
pub mod spectrum_col {
    pub const CONFIRM_AGG_TRADE_ID: usize = 0;
    pub const C_FOURIER: usize = 1;
    pub const OS_FOURIER: usize = 8;
    pub const C_HAAR: usize = 15;
    pub const OS_HAAR: usize = 22;
}

/// Posición de cada columna de `fit.parquet` en la fila.
pub mod fit_col {
    pub const CONFIRM_AGG_TRADE_ID: usize = 0;
    pub const C_SLOPE: usize = 1;
    pub const OS_SLOPE: usize = 2;
    pub const EV_SLOPE: usize = 3;
    pub const C_ACCEL: usize = 4;
    pub const OS_ACCEL: usize = 5;
    pub const EV_ACCEL: usize = 6;
    pub const C_JUMP_SHARE: usize = 7;
    pub const OS_JUMP_SHARE: usize = 8;
    pub const EV_JUMP_SHARE: usize = 9;
}

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

/// La fila de un evento de `spectrum.parquet`: como [`PathRow`], con 29 columnas (el mapa de
/// presentes es un `u32`). `ticks` son los retornos de sus dos fases (0 si el evento está sobre
/// el presupuesto) y `over_budget` trae, si lo está, los ticks que L2 dice que tiene
/// (`extreme_agg_trade_id − reference_agg_trade_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpectrumRow {
    pub cells: [i64; SPECTRUM_WIDTH],
    pub valid: u32,
    pub ticks: u64,
    pub over_budget: Option<u64>,
}

/// La fila de un evento de `fit.parquet`, con el mismo significado de sus campos que
/// [`PathRow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitRow {
    pub cells: [i64; FIT_WIDTH],
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
