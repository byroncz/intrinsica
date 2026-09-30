//! Detector DC de un θ: entero de punta a punta, estado O(1).

use std::fmt;

mod carry_over;

pub use carry_over::{CarryOver, CarryOverError, CarryPending, STATE_VERSION};

/// Escala compartida por precio y θ (ADR-L2-01): `DECIMAL(18,8)` → `10⁸`.
pub const SCALE: i64 = 100_000_000;

/// Cota exclusiva del precio entero: `DECIMAL(18,8)` tiene a lo sumo 18
/// dígitos, así que `0 < price < 10¹⁸`. Con `0 < θ < SCALE` el umbral
/// (`price × (SCALE ± θ) / SCALE`) es menor que `2 × 10¹⁸` y cabe en `i64`.
pub const PRICE_LIMIT: i64 = 1_000_000_000_000_000_000;

/// Un punto de la serie: precio entero (escala `SCALE`), `transact_time` en
/// µs y `agg_trade_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub price: i64,
    pub time: i64,
    pub agg_trade_id: i64,
}

/// Lo que recibe [`Detector::feed`]: un tick es un punto de la serie.
pub type Tick = Point;

/// Dirección de un evento DC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Upturn: el precio sube θ desde un mínimo.
    Up,
    /// Downturn: el precio baja θ desde un máximo.
    Down,
}

impl Direction {
    /// Valor de la columna `direction` de `events.parquet` (§7.2).
    pub const fn as_i8(self) -> i8 {
        match self {
            Direction::Up => 1,
            Direction::Down => -1,
        }
    }
}

/// Evento DC cerrado: se emite cuando confirma el siguiente, que es cuando
/// se conoce su extremo (ADR-L2-05).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub reference: Point,
    pub confirm: Point,
    /// Cierra el Overshoot. Siempre `extreme.agg_trade_id >= confirm.agg_trade_id`;
    /// si es igual a `confirm`, el Overshoot es vacío (§7.2).
    pub extreme: Point,
    pub direction: Direction,
}

/// Evento confirmado cuyo extremo todavía no se conoce.
///
/// Guarda su propia dirección (el TRD §7.4 la deriva de `State::direction`):
/// solo difieren en la rama de DC sin tick, que es inalcanzable con un θ
/// válido, y así el evento no cambia de signo si esa guarda llegara a
/// dispararse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingEvent {
    pub reference: Point,
    pub confirm: Point,
    pub direction: Direction,
}

/// Extremos vigentes. Se actualizan en cada tick, sea cual sea la tendencia.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extremes {
    pub high: Point,
    pub low: Point,
}

/// Confirmación en grupo abierto (§8.1 paso 4): el umbral ya se cruzó y se
/// esperan más ticks con el mismo `transact_time`. El instante es atómico:
/// mientras el grupo está abierto sus ticks solo alimentan la regla de
/// empates, no mueven extremos ni evalúan reversión.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Group {
    pub direction: Direction,
    pub time: i64,
    /// Umbral que cruzó el primer tick del grupo.
    pub threshold: i64,
    /// Precio más conservador entre los que cumplen el umbral.
    pub best_price: i64,
    /// `agg_trade_id` del último tick visto del grupo.
    pub last_id: i64,
}

/// Estado completo del detector. Es O(1): escalares, ningún vector de ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct State {
    /// `None` = indefinida: aún no se confirmó ninguna reversión.
    pub direction: Option<Direction>,
    /// `None` solo antes del primer tick.
    pub extremes: Option<Extremes>,
    pub pending: Option<PendingEvent>,
    pub group: Option<Group>,
}

/// θ fuera de `0 < θ < SCALE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThetaError(pub i64);

impl fmt::Display for ThetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "theta_int {} fuera de 0 < theta < {SCALE}", self.0)
    }
}

impl std::error::Error for ThetaError {}

/// Detector DC de un θ.
///
/// Entrada: ticks en orden estricto de `(transact_time, agg_trade_id)`, con
/// `0 < price < PRICE_LIMIT` (contrato de L1). Al terminar una unidad de
/// trabajo hay que llamar [`Detector::finish`] para cerrar un grupo de
/// empate todavía abierto.
#[derive(Debug, Clone)]
pub struct Detector {
    theta: i64,
    state: State,
    discarded: u64,
}

impl Detector {
    /// `theta` es `round(θ × 10⁸)`; debe cumplir `0 < theta < SCALE`.
    pub fn new(theta: i64) -> Result<Self, ThetaError> {
        if theta <= 0 || theta >= SCALE {
            return Err(ThetaError(theta));
        }
        Ok(Self {
            theta,
            state: State::default(),
            discarded: 0,
        })
    }

    pub fn theta(&self) -> i64 {
        self.theta
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    /// Eventos descartados por la validación "un DC tiene al menos un tick"
    /// (§9.1). Con un θ válido es una guarda inalcanzable; distinto de cero
    /// señala un defecto del detector.
    pub fn discarded(&self) -> u64 {
        self.discarded
    }

    /// Alimenta un tick. Devuelve el evento que este tick dejó cerrado: el
    /// grupo de empate de la confirmación anterior se cierra con el primer
    /// tick de un `transact_time` distinto, y ahí se conoce el extremo del
    /// evento previo.
    ///
    /// # Panics
    /// Si `price` no cumple `0 < price < PRICE_LIMIT`: viola el contrato de
    /// L1 y el umbral podría desbordar `i64`.
    pub fn feed(&mut self, tick: Tick) -> Option<Event> {
        assert!(
            tick.price > 0 && tick.price < PRICE_LIMIT,
            "price {} fuera de 0 < price < {PRICE_LIMIT}",
            tick.price
        );
        let closed = match &mut self.state.group {
            Some(group) if group.time == tick.time => {
                group.absorb(tick);
                return None;
            }
            Some(_) => self.close_group(),
            None => None,
        };
        self.step(tick);
        closed
    }

    /// Señal de fin de entrada (RF-L2-12): cierra un grupo de empate abierto
    /// igual que un cambio de `transact_time`. El detector sigue usable; un
    /// tick posterior con el mismo `transact_time` se trataría como un
    /// instante nuevo, cosa que el orden de la serie descarta.
    pub fn finish(&mut self) -> Option<Event> {
        self.close_group()
    }

    /// Cierra el grupo abierto: fija la confirmación, reinicia el extremo
    /// opuesto con su terna y, si el DC es válido, adopta la tendencia.
    fn close_group(&mut self) -> Option<Event> {
        let group = self.state.group.take()?;
        let extremes = self
            .state
            .extremes
            .as_mut()
            .expect("un grupo abierto implica extremos");
        let confirm = Point {
            price: group.best_price,
            time: group.time,
            agg_trade_id: group.last_id,
        };
        let (reference, opposite) = match group.direction {
            Direction::Up => (extremes.low, &mut extremes.high),
            Direction::Down => (extremes.high, &mut extremes.low),
        };
        *opposite = confirm;
        self.state.direction = Some(group.direction);

        // Un DC tiene al menos un tick: la confirmación queda estrictamente
        // después de la referencia. Si no, no hay evento, pero la tendencia
        // ya se adoptó y el evento pendiente sigue abierto.
        if confirm.agg_trade_id <= reference.agg_trade_id {
            self.discarded += 1;
            return None;
        }
        let opened = PendingEvent {
            reference,
            confirm,
            direction: group.direction,
        };
        // La referencia del evento nuevo es el extremo del pendiente.
        self.state.pending.replace(opened).map(|prev| Event {
            reference: prev.reference,
            confirm: prev.confirm,
            extreme: reference,
            direction: prev.direction,
        })
    }

    /// Ciclo normal para un tick fuera de grupo: actualiza extremos, evalúa
    /// reversión y, si cruza el umbral, abre un grupo de empate.
    fn step(&mut self, tick: Tick) {
        let extremes = self.state.extremes.get_or_insert(Extremes {
            high: tick,
            low: tick,
        });
        if tick.price > extremes.high.price {
            extremes.high = tick;
        }
        if tick.price < extremes.low.price {
            extremes.low = tick;
        }

        let theta = self.theta;
        let up = |low: &Point| up_threshold(low.price, theta);
        let down = |high: &Point| down_threshold(high.price, theta);
        let price = tick.price;
        let crossing = match self.state.direction {
            None => {
                let threshold = up(&extremes.low);
                if price >= threshold {
                    Some((Direction::Up, threshold))
                } else {
                    let threshold = down(&extremes.high);
                    (price <= threshold).then_some((Direction::Down, threshold))
                }
            }
            Some(Direction::Up) => {
                let threshold = down(&extremes.high);
                (price <= threshold).then_some((Direction::Down, threshold))
            }
            Some(Direction::Down) => {
                let threshold = up(&extremes.low);
                (price >= threshold).then_some((Direction::Up, threshold))
            }
        };
        if let Some((direction, threshold)) = crossing {
            self.state.group = Some(Group {
                direction,
                time: tick.time,
                threshold,
                best_price: price,
                last_id: tick.agg_trade_id,
            });
        }
    }
}

impl Group {
    /// Suma un tick del mismo instante: el precio solo cuenta si cumple el
    /// umbral y es más conservador (mínimo en upturn, máximo en downturn);
    /// el `agg_trade_id` siempre avanza al del último tick.
    fn absorb(&mut self, tick: Tick) {
        self.last_id = tick.agg_trade_id;
        let better = match self.direction {
            Direction::Up => tick.price >= self.threshold && tick.price < self.best_price,
            Direction::Down => tick.price <= self.threshold && tick.price > self.best_price,
        };
        if better {
            self.best_price = tick.price;
        }
    }
}

/// `ceil(low × (SCALE + θ) / SCALE)`: producto en `i128`, resultado en `i64`
/// (ADR-L2-02). Redondear hacia el extremo garantiza `TMV ≥ θ`.
pub(crate) fn up_threshold(low: i64, theta: i64) -> i64 {
    let product = i128::from(low) * i128::from(SCALE + theta);
    narrow((product + i128::from(SCALE) - 1) / i128::from(SCALE))
}

/// `floor(high × (SCALE − θ) / SCALE)`, igual criterio que [`up_threshold`].
pub(crate) fn down_threshold(high: i64, theta: i64) -> i64 {
    let product = i128::from(high) * i128::from(SCALE - theta);
    narrow(product / i128::from(SCALE))
}

fn narrow(value: i128) -> i64 {
    i64::try_from(value).expect("el umbral cabe en i64 con 0 < price < PRICE_LIMIT y 0 < θ < SCALE")
}

#[cfg(test)]
mod tests;
