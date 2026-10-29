//! La familia T (trayectoria, `path.parquet`): los acumuladores de una fase y el cierre de
//! los 13 registros T-01 a T-13 (ADR-L3-17).
//!
//! A diferencia de S, un tramo de fase no se fusiona en cualquier orden: la suma de los
//! pasos entre ticks consecutivos, los huecos, los pares de lados y el camino necesitan el
//! par que une dos tramos. Un [`PhasePath`] guarda por eso su estado de borde (primer y
//! último precio, tiempo y lado) y [`PhasePath::merge`] une un tramo con el **siguiente**,
//! el de ordinal consecutivo. Cada par de ticks consecutivos de la fase se evalúa una sola
//! vez, dentro de un tramo o al unir dos, con la misma función, así que el resultado no
//! depende de cómo se cortó la fase ni de en qué orden se unieron las corridas.
//!
//! Todo se suma en enteros. La única cuenta en `f64` por paso es `Δln p`, y se cuantiza a
//! un entero de punto fijo (escala 2⁻⁹⁶, en `i128`) antes de sumarse: la suma de enteros es
//! asociativa y la de `f64` no. El `f64` sale una vez, al cerrar.
//!
//! Anclas: la fase de confirmación arranca en `reference_price` de L2 y la del overshoot en
//! `confirm_price`. Camino y suma de cuadrados incluyen el paso del ancla al primer tick de
//! la fase; huecos, duraciones y lados, solo los ticks de la fase.

use crate::close::{EventErrorKind, PhaseName};
use crate::input::{EventIn, TickColumns};
use crate::layout::{path_col as pc, PathRow, PATH_WIDTH};
use crate::phase::{le_u64, TickError};

/// 2⁹⁶, la escala del punto fijo de `Σ(Δln p)²`.
const SCALE: f64 = 79_228_162_514_264_337_593_543_950_336.0;
/// 2⁻⁹⁶: multiplicar por una potencia de dos no redondea.
const UNSCALE: f64 = 1.0 / SCALE;
/// Sin tiempo: el nivel todavía no se alcanzó.
const NEVER: i64 = i64::MAX;

/// Un tick de la fase: precio, tiempo y lado (+1 compró el agresor, −1 vendió).
type Tick = (i64, i64, i8);

/// Los agregados de una fase, o de un tramo de ella, con su estado de borde. Con
/// `ticks == 0` el resto no significa nada.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhasePath {
    pub ticks: u64,
    pub first_price: i64,
    pub last_price: i64,
    pub first_time: i64,
    pub last_time: i64,
    pub first_side: i8,
    pub last_side: i8,
    /// `Σ|Δp|` sobre los pares consecutivos de la fase.
    pub length: u128,
    /// `Σ(Δln p)²` sobre los mismos pares, en punto fijo de escala 2⁻⁹⁶.
    pub squares: i128,
    /// La suma de cuadrados no cupo en 128 bits (error de fase, nunca un valor).
    pub overflow: bool,
    /// `Σ s_i`.
    pub sides: i64,
    /// `Σ s_i·s_{i−1}` sobre los pares consecutivos.
    pub pairs: i64,
    /// El mayor hueco entre ticks consecutivos de la fase, en µs.
    pub max_gap: i64,
    /// La suma de los huecos mayores que `silence_us`, en µs.
    pub idle: i64,
}

/// `|Δp|` y `(Δln p)²` cuantizado del paso de `previous` a `price`, ambos mayores que 0.
///
/// `Δln p = ln(1 + Δp/previous)`: el cociente de enteros y `ln_1p` no pierden dígitos
/// cuando el paso es pequeño frente al precio, que es lo normal. La escala del precio se
/// cancela.
fn step(previous: i64, price: i64) -> (u64, i128) {
    let delta = price.wrapping_sub(previous);
    let change = (delta as f64 / previous as f64).ln_1p();
    (
        delta.unsigned_abs(),
        (change * change * SCALE).round() as i128,
    )
}

impl PhasePath {
    fn edge_last(&self) -> Tick {
        (self.last_price, self.last_time, self.last_side)
    }

    fn edge_first(&self) -> Tick {
        (self.first_price, self.first_time, self.first_side)
    }

    /// Suma el par `(before, after)` de ticks consecutivos de la fase.
    fn add_pair(&mut self, before: Tick, after: Tick, silence_us: i64) {
        let (abs, square) = step(before.0, after.0);
        self.length += u128::from(abs);
        self.add_squares(square);
        self.pairs += i64::from(before.2 * after.2);
        let gap = after.1 - before.1;
        self.max_gap = self.max_gap.max(gap);
        if gap > silence_us {
            self.idle += gap;
        }
    }

    fn add_squares(&mut self, square: i128) {
        match self.squares.checked_add(square) {
            Some(sum) => self.squares = sum,
            None => {
                self.overflow = true;
                self.squares = i128::MAX;
            }
        }
    }

    /// Une el tramo `next`, que sigue a `self` en la serie sin ticks de la fase entre los
    /// dos. No conmuta; sí es asociativo, así que las corridas pueden unirse en cualquier
    /// orden de llegada mientras cada unión sea entre vecinas.
    pub fn merge(&mut self, next: &PhasePath, silence_us: i64) {
        if next.ticks == 0 {
            return;
        }
        if self.ticks == 0 {
            *self = *next;
            return;
        }
        self.add_pair(self.edge_last(), next.edge_first(), silence_us);
        self.ticks += next.ticks;
        self.length += next.length;
        self.overflow |= next.overflow;
        self.add_squares(next.squares);
        self.sides += next.sides;
        self.pairs += next.pairs;
        self.max_gap = self.max_gap.max(next.max_gap);
        self.idle += next.idle;
        self.last_price = next.last_price;
        self.last_time = next.last_time;
        self.last_side = next.last_side;
    }
}

/// Los agregados de un evento, o de un tramo de él, para la familia T.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathAcc {
    pub confirmation: PhasePath,
    pub overshoot: PhasePath,
    /// El menor tiempo de un tick de la confirmación que alcanza el 25, 50 y 75 % del camino
    /// de `reference_price` a `confirm_price` (`NEVER` si ninguno lo alcanzó).
    pub levels: [i64; 3],
}

impl Default for PathAcc {
    fn default() -> Self {
        Self {
            confirmation: PhasePath::default(),
            overshoot: PhasePath::default(),
            levels: [NEVER; 3],
        }
    }
}

impl PathAcc {
    /// Une el tramo `next`, el de ordinal siguiente, a continuación de `self`.
    pub fn merge(&mut self, next: &PathAcc, silence_us: i64) {
        self.confirmation.merge(&next.confirmation, silence_us);
        self.overshoot.merge(&next.overshoot, silence_us);
        for (level, other) in self.levels.iter_mut().zip(next.levels) {
            *level = (*level).min(other);
        }
    }
}

/// Cuándo un precio alcanzó el 25, 50 y 75 % del camino de `reference` a `confirm`:
/// `4·(p − reference)·d ≥ k·need`, con `d` la dirección y `need = (confirm − reference)·d`.
/// Todo en enteros.
#[derive(Debug, Clone, Copy)]
pub struct Reach {
    pub reference: i64,
    pub direction: i128,
    pub need: i128,
}

impl Reach {
    pub fn of(event: &EventIn) -> Self {
        let direction = if event.direction == -1 { -1 } else { 1 };
        Self {
            reference: event.reference_price,
            direction,
            need: (i128::from(event.confirm_price) - i128::from(event.reference_price)) * direction,
        }
    }
}

/// Agrega los ticks `[start, end)` de una fase. Con `reach` calcula además los tiempos de
/// los tres niveles (solo en la confirmación).
///
/// Es el camino caliente de la familia: una pasada por tick, sin locks ni asignaciones.
pub fn accumulate<B: AsRef<[u8]>>(
    ticks: &TickColumns<B>,
    start: usize,
    end: usize,
    silence_us: i64,
    reach: Option<&Reach>,
) -> Result<(PhasePath, [i64; 3]), TickError> {
    let mut acc = PhasePath::default();
    let mut levels = [NEVER; 3];
    if start >= end {
        return Ok((acc, levels));
    }
    let times = ticks.time_bytes(start, end);
    let prices = ticks.price_bytes(start, end);
    let (bits, first_bit) = ticks.sold_bits(start);
    let mut bad = 0u64;
    let mut zero = false;
    let mut previous: Option<Tick> = None;
    for (i, (price, time)) in prices
        .chunks_exact(16)
        .zip(times.chunks_exact(8))
        .enumerate()
    {
        let p = le_u64(price);
        bad |= le_u64(&price[8..]) | (p >> 63);
        zero |= p == 0;
        let time = i64::from_le_bytes(time.try_into().expect("8 bytes"));
        let bit = first_bit + i;
        let sold = ((bits[bit >> 3] >> (bit & 7)) & 1) as i8;
        let tick: Tick = (p as i64, time, 1 - 2 * sold);
        match previous {
            Some(before) => acc.add_pair(before, tick, silence_us),
            None => {
                acc.first_price = tick.0;
                acc.first_time = tick.1;
                acc.first_side = tick.2;
            }
        }
        acc.sides += i64::from(tick.2);
        previous = Some(tick);
        if let Some(reach) = reach {
            if levels[2] == NEVER {
                let gain = (i128::from(tick.0) - i128::from(reach.reference)) * reach.direction * 4;
                for (k, level) in levels.iter_mut().enumerate() {
                    if *level == NEVER && gain >= (k as i128 + 1) * reach.need {
                        *level = time;
                    }
                }
            }
        }
    }
    if bad != 0 {
        return Err(classify(ticks, start, end));
    }
    if zero {
        return Err(TickError::ZeroPrice);
    }
    let last = previous.expect("la fase tiene ticks");
    acc.ticks = (end - start) as u64;
    acc.last_price = last.0;
    acc.last_time = last.1;
    acc.last_side = last.2;
    Ok((acc, levels))
}

/// Camino frío: distingue un precio negativo de uno que no cabe en 63 bits.
fn classify<B: AsRef<[u8]>>(ticks: &TickColumns<B>, start: usize, end: usize) -> TickError {
    let negative = ticks
        .price_bytes(start, end)
        .chunks_exact(16)
        .any(|value| i128::from_le_bytes(value.try_into().expect("16 bytes")) < 0);
    if negative {
        TickError::Negative
    } else {
        TickError::TooLarge
    }
}

fn invariant(
    field: &'static str,
    got: impl TryInto<i128>,
    wanted: impl Into<String>,
) -> EventErrorKind {
    EventErrorKind::Invariant {
        field,
        got: got.try_into().unwrap_or(i128::MAX),
        wanted: wanted.into(),
    }
}

/// Lo que una fase cerrada aporta a la fila.
struct Closed {
    variance: f64,
    tortuosity: Option<f64>,
    autocorr: Option<f64>,
    max_gap: i64,
    duration: i64,
    idle: Option<f64>,
}

/// La autocorrelación de rezago 1 de los lados de la fase con la media de la fase:
/// `[n²P − nS(2S − s₁ − sₙ) + (n−1)S²] / [n(n² − S²)]`, dos enteros y una división.
/// Nula con menos de dos ticks o con todos del mismo lado (`|S| = n`).
fn autocorrelation(phase: &PhasePath) -> Option<f64> {
    let n = i128::from(phase.ticks);
    let sum = i128::from(phase.sides);
    let denominator = n * (n * n - sum * sum);
    if phase.ticks < 2 || denominator == 0 {
        return None;
    }
    let numerator = n * n * i128::from(phase.pairs)
        - n * sum * (2 * sum - i128::from(phase.first_side) - i128::from(phase.last_side))
        + (n - 1) * sum * sum;
    Some(numerator as f64 / denominator as f64)
}

/// Cierra una fase con ticks: suma el paso del ancla al primer tick y deriva sus valores.
/// `net` es el desplazamiento neto contra el que se mide el camino.
fn close_phase(
    phase: &PhasePath,
    name: PhaseName,
    anchor: i64,
    net: u64,
    tortuosity_field: &'static str,
) -> Result<Closed, EventErrorKind> {
    let (abs, square) = step(anchor, phase.first_price);
    let overflow = EventErrorKind::PathOverflow { phase: name };
    if phase.overflow {
        return Err(overflow);
    }
    let squares = phase.squares.checked_add(square).ok_or(overflow)?;
    let length = phase.length + u128::from(abs);
    let tortuosity = if net == 0 {
        None
    } else {
        if length < u128::from(net) {
            return Err(invariant(
                tortuosity_field,
                length,
                format!("al menos {net}, el desplazamiento neto"),
            ));
        }
        Some(length as f64 / net as f64)
    };
    let duration = phase.last_time - phase.first_time;
    Ok(Closed {
        variance: squares as f64 * UNSCALE,
        tortuosity,
        autocorr: autocorrelation(phase),
        max_gap: phase.max_gap,
        duration,
        idle: (duration > 0).then(|| phase.idle as f64 / duration as f64),
    })
}

/// Cierra un evento de la familia T: los 13 registros con la regla de nulos del contrato.
///
/// Comprueba solo lo que el cálculo necesita para no producir `inf` ni `NaN` y para que
/// `c_tortuosity ≥ 1` y `os_tortuosity ≥ 1` valgan: dirección, anclas positivas y del lado
/// que la dirección dice, una fase de confirmación con ticks y niveles alcanzados. Los
/// invariantes de §7.2 contra L2 los comprueba S.
pub fn close_path(acc: &PathAcc, event: &EventIn) -> Result<PathRow, EventErrorKind> {
    if event.direction != 1 && event.direction != -1 {
        return Err(EventErrorKind::Direction {
            got: event.direction,
        });
    }
    let (reference, confirm, extreme) = (
        event.reference_price,
        event.confirm_price,
        event.extreme_price,
    );
    if reference <= 0 {
        return Err(invariant("reference_price", reference, "más de 0"));
    }
    if confirm <= 0 {
        return Err(invariant("confirm_price", confirm, "más de 0"));
    }
    let (c, o) = (&acc.confirmation, &acc.overshoot);
    if c.ticks == 0 {
        return Err(invariant("c_ticks", 0i64, "más de 0"));
    }
    let up = event.direction == 1;
    if (confirm > reference) != up || confirm == reference {
        return Err(invariant(
            "confirm_price",
            confirm,
            if up {
                format!("más de {reference}, la referencia")
            } else {
                format!("menos de {reference}, la referencia")
            },
        ));
    }
    let net_c = confirm.abs_diff(reference);
    let confirmation = close_phase(c, PhaseName::Confirmation, reference, net_c, "c_tortuosity")?;

    let mut cells = [0i64; PATH_WIDTH];
    let mut valid = 0u16;
    let mut set = |column: usize, bits: i64| {
        cells[column] = bits;
        valid |= 1 << column;
    };
    set(pc::CONFIRM_AGG_TRADE_ID, event.confirm);
    set(pc::C_REALIZED_VAR, confirmation.variance.to_bits() as i64);
    set(
        pc::C_TORTUOSITY,
        confirmation
            .tortuosity
            .expect("la confirmación avanza")
            .to_bits() as i64,
    );
    if let Some(r) = confirmation.autocorr {
        set(pc::C_SIGN_AUTOCORR, r.to_bits() as i64);
    }
    set(pc::C_MAX_GAP, confirmation.max_gap);
    if let Some(fraction) = confirmation.idle {
        set(pc::C_IDLE_FRACTION, fraction.to_bits() as i64);
    }
    if confirmation.duration > 0 {
        for (k, column) in [pc::C_T25, pc::C_T50, pc::C_T75].into_iter().enumerate() {
            let name = ["c_t25", "c_t50", "c_t75"][k];
            if acc.levels[k] == NEVER {
                return Err(invariant(
                    name,
                    -1i64,
                    "un tick de la confirmación que alcance el nivel",
                ));
            }
            let fraction = (acc.levels[k] - c.first_time) as f64 / confirmation.duration as f64;
            set(column, fraction.to_bits() as i64);
        }
    }

    if o.ticks > 0 {
        if extreme <= 0 {
            return Err(invariant("extreme_price", extreme, "más de 0"));
        }
        let overshoot = close_phase(
            o,
            PhaseName::Overshoot,
            confirm,
            extreme.abs_diff(confirm),
            "os_tortuosity",
        )?;
        set(pc::OS_REALIZED_VAR, overshoot.variance.to_bits() as i64);
        if let Some(t) = overshoot.tortuosity {
            set(pc::OS_TORTUOSITY, t.to_bits() as i64);
        }
        if let Some(r) = overshoot.autocorr {
            set(pc::OS_SIGN_AUTOCORR, r.to_bits() as i64);
        }
        set(pc::OS_MAX_GAP, overshoot.max_gap);
        if let Some(fraction) = overshoot.idle {
            set(pc::OS_IDLE_FRACTION, fraction.to_bits() as i64);
        }
    }
    Ok(PathRow {
        cells,
        valid,
        ticks: c.ticks + o.ticks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Una fase armada tick a tick con la misma `add_pair` que `accumulate`.
    fn phase(ticks: &[Tick], silence_us: i64) -> PhasePath {
        let mut acc = PhasePath {
            ticks: ticks.len() as u64,
            first_price: ticks[0].0,
            first_time: ticks[0].1,
            first_side: ticks[0].2,
            last_price: ticks[ticks.len() - 1].0,
            last_time: ticks[ticks.len() - 1].1,
            last_side: ticks[ticks.len() - 1].2,
            sides: ticks.iter().map(|t| i64::from(t.2)).sum(),
            ..PhasePath::default()
        };
        for pair in ticks.windows(2) {
            acc.add_pair(pair[0], pair[1], silence_us);
        }
        acc
    }

    const TICKS: [Tick; 6] = [
        (100, 10, 1),
        (103, 11, 1),
        (101, 40, -1),
        (101, 40, -1),
        (110, 41, 1),
        (95, 500, -1),
    ];

    #[test]
    fn merging_neighbours_gives_the_phase_of_the_whole_in_any_association() {
        let whole = phase(&TICKS, 100);
        for cuts in [[1, 2], [1, 5], [2, 4], [3, 4], [1, 3]] {
            let parts: Vec<_> = [
                &TICKS[..cuts[0]],
                &TICKS[cuts[0]..cuts[1]],
                &TICKS[cuts[1]..],
            ]
            .iter()
            .map(|ticks| phase(ticks, 100))
            .collect();
            let mut left = parts[0];
            left.merge(&parts[1], 100);
            left.merge(&parts[2], 100);
            let mut right = parts[1];
            right.merge(&parts[2], 100);
            let mut other = parts[0];
            other.merge(&right, 100);
            assert_eq!(left, whole, "(a b) c con los cortes {cuts:?}");
            assert_eq!(other, whole, "a (b c) con los cortes {cuts:?}");
        }
    }

    #[test]
    fn the_registers_of_a_phase_by_hand() {
        let whole = phase(&TICKS, 100);
        // Caminos |Δp|: 3 + 2 + 0 + 9 + 15 = 29.
        assert_eq!(whole.length, 29);
        // Huecos: 1, 29, 0, 1, 459; el máximo es 459 y solo ese supera 100.
        assert_eq!((whole.max_gap, whole.idle), (459, 459));
        // Lados: + + − − + −; pares: +1 −1 +1 −1 −1 −1 → P = −2... a mano:
        // (+,+)=1, (+,−)=−1, (−,−)=1, (−,+)=−1, (+,−)=−1 → −1.
        assert_eq!((whole.sides, whole.pairs), (0, -1));
        // n = 6, S = 0, P = −1: r = n²P / (n·n²) = P/n = −1/6.
        let r = autocorrelation(&whole).unwrap();
        assert!((r + 1.0 / 6.0).abs() < 1e-15, "{r}");
        // Σ(Δln p)²: términos ln(103/100)², ln(101/103)², 0, ln(110/101)², ln(95/110)².
        let expected: f64 = [
            (103.0f64, 100.0f64),
            (101.0, 103.0),
            (101.0, 101.0),
            (110.0, 101.0),
            (95.0, 110.0),
        ]
        .iter()
        .map(|(a, b)| (a / b).ln().powi(2))
        .sum();
        let got = whole.squares as f64 * UNSCALE;
        assert!(
            (got - expected).abs() <= 1e-12 * expected,
            "{got} {expected}"
        );
    }

    #[test]
    fn the_autocorrelation_is_the_centred_lag_one_estimator() {
        let alternating: Vec<Tick> = (0..4)
            .map(|i| (100, i, if i % 2 == 0 { 1 } else { -1 }))
            .collect();
        assert_eq!(autocorrelation(&phase(&alternating, 10)), Some(-0.75));
        let runs: Vec<Tick> = [1i8, 1, -1, -1]
            .iter()
            .enumerate()
            .map(|(i, s)| (100, i as i64, *s))
            .collect();
        assert_eq!(autocorrelation(&phase(&runs, 10)), Some(0.25));
        // Todos del mismo lado, y un solo tick: nula.
        let same: Vec<Tick> = (0..5).map(|i| (100, i, -1)).collect();
        assert_eq!(autocorrelation(&phase(&same, 10)), None);
        assert_eq!(autocorrelation(&phase(&same[..1], 10)), None);
        // Una racha con un solo tick del otro lado: S = 3, P = 2 − 1 = 1... a mano con n = 5:
        // lados + + − + +: S = 3, P = 1 − 1 − 1 + 1 = 0; s₁ = sₙ = +1.
        let one: Vec<Tick> = [1i8, 1, -1, 1, 1]
            .iter()
            .enumerate()
            .map(|(i, s)| (100, i as i64, *s))
            .collect();
        // num = 25·0 − 5·3·(6 − 2) + 4·9 = −60 + 36 = −24; den = 5·(25 − 9) = 80.
        assert_eq!(autocorrelation(&phase(&one, 10)), Some(-0.3));
    }

    #[test]
    fn the_quantised_term_is_close_to_the_float_one_for_a_minimum_tick_step() {
        // Un paso de 0,01 sobre un precio de 50 000 (escala 10⁻⁸): el cociente es 2·10⁻⁷.
        let (abs, square) = step(5_000_000_000_000, 5_000_001_000_000);
        assert_eq!(abs, 1_000_000);
        let float = (1_000_000.0f64 / 5_000_000_000_000.0).ln_1p().powi(2);
        let got = square as f64 * UNSCALE;
        assert!((got - float).abs() <= 1e-15 * float, "{got} {float}");
        // Sin movimiento no hay cuadrado.
        assert_eq!(step(7, 7), (0, 0));
    }
}
