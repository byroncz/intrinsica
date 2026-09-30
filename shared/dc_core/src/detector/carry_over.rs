//! Carry-over de un θ (TRD-L2 §7.4, ADR-L2-07): el estado que el detector
//! entrega al cierre de un mes y retoma el mes siguiente.
//!
//! Solo escalares: dirección, ambos extremos y el evento pendiente sin
//! extremo. No hay ticks huérfanos ni grupo de empate: el fin de la entrada
//! cierra el grupo ([`Detector::finish`]) antes de tomar el estado.

use super::{
    Detector, Direction, Extremes, PendingEvent, Point, State, ThetaError, PRICE_LIMIT, SCALE,
};
use std::fmt;

/// Versión del esquema de carry-over que esta imagen sabe escribir y leer
/// (`state_version` del TRD-L2 §7.4). Se compara por igualdad, sin inferir
/// compatibilidad (ADR-L2-08).
pub const STATE_VERSION: &str = "1.0.0";

/// Bytes de un carry-over: `theta` (8) + `direction` (1) + 2 extremos (48) +
/// `has_pending_event` (1) + 2 puntos pendientes (48).
const BODY_LEN: usize = 8 + 1 + 2 * POINT_LEN + 1 + 2 * POINT_LEN;
const POINT_LEN: usize = 3 * 8;

/// Evento confirmado cuyo extremo aún no se conoce. Su dirección es
/// [`CarryOver::direction`], por eso no se repite (TRD-L2 §7.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CarryPending {
    pub reference: Point,
    pub confirm: Point,
}

/// Una fila del carry-over de un θ, con las columnas de TRD-L2 §7.4 que
/// describen el estado del detector. Coordenadas del dato (`provider`,
/// `market`, `asset`, `year`, `month`) las agrega quien escribe el Parquet.
///
/// Los precios van enteros en escala [`SCALE`], igual que en el detector.
///
/// Mapeo 1:1 a las columnas de §7.4, que escribe la capa:
///
/// - `theta` y cada `price` son el entero sin escalar de su `DECIMAL`
///   (`theta` a `DECIMAL(9,8)`, los precios a `DECIMAL(18,8)`);
/// - `time` y `agg_trade_id` van a `INT64` y `direction` a `INT8`;
/// - `ext_high` y `ext_low` son las columnas `ext_high_*` y `ext_low_*`;
/// - `pending = Some(p)` es `has_pending_event = true` con `p.reference` en
///   `pending_reference_*` y `p.confirm` en `pending_confirm_*`;
///   `pending = None` es `has_pending_event = false` con las seis columnas
///   `pending_*` en nulo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarryOver {
    /// `round(θ × 10⁸)`, el θ de esta cadena.
    pub theta: i64,
    pub state_version: String,
    /// `0` indefinida, `1` upturn, `-1` downturn.
    pub direction: i8,
    pub ext_high: Point,
    pub ext_low: Point,
    /// `has_pending_event` del TRD es `pending.is_some()`.
    pub pending: Option<CarryPending>,
}

/// Por qué no se pudo tomar, decodificar o retomar un carry-over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CarryOverError {
    /// El θ no cumple `0 < θ < SCALE`.
    Theta(ThetaError),
    /// El carry-over es de otro θ.
    ThetaMismatch { expected: i64, found: i64 },
    /// `state_version` distinta de [`STATE_VERSION`] (fail-closed, ADR-L2-08).
    VersionMismatch {
        expected: &'static str,
        found: String,
    },
    /// Hay un grupo de empate abierto: falta llamar a [`Detector::finish`].
    OpenGroup,
    /// El detector no vio ningún tick: no hay extremos que entregar.
    NoTicks,
    /// El estado rompe una invariante del detector.
    Inconsistent(&'static str),
    /// Los bytes no tienen el formato de [`CarryOver::to_bytes`].
    Malformed(&'static str),
}

impl fmt::Display for CarryOverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Theta(e) => e.fmt(f),
            Self::ThetaMismatch { expected, found } => write!(
                f,
                "carry-over de otro theta: el detector es {expected}, el estado es {found}"
            ),
            Self::VersionMismatch { expected, found } => write!(
                f,
                "state_version {found:?} no coincide con la que esta versión lee ({expected:?})"
            ),
            Self::OpenGroup => f.write_str(
                "hay un grupo de empate abierto: llama a finish() antes de tomar el carry-over",
            ),
            Self::NoTicks => {
                f.write_str("el detector no ha visto ningún tick: no hay estado que guardar")
            }
            Self::Inconsistent(why) => write!(f, "carry-over inconsistente: {why}"),
            Self::Malformed(why) => write!(f, "carry-over mal formado: {why}"),
        }
    }
}

impl std::error::Error for CarryOverError {}

impl From<ThetaError> for CarryOverError {
    fn from(e: ThetaError) -> Self {
        Self::Theta(e)
    }
}

impl CarryOver {
    /// Bytes canónicos: `state_version` (largo en un byte + UTF-8) y luego los
    /// campos en el orden de la struct, enteros de 8 bytes little-endian,
    /// `has_pending_event` en un byte y los puntos pendientes en cero si no
    /// hay evento. Mismo estado, mismos bytes.
    ///
    /// Es un transporte interno entre el detector y quien lo llama, no el
    /// formato del TRD: el carry-over de TRD-L2 §7.4 es un Parquet de una fila
    /// que escribe la capa a partir de los campos de [`CarryOver`]. Estos
    /// bytes tampoco son la base del `content_hash`, que el TRD calcula sobre
    /// el `carry_over.parquet` comprimido (§6.7, §8.1 paso 8).
    ///
    /// Falla solo si `state_version` no cabe en el prefijo de un byte. No
    /// valida el estado: eso lo hace [`Detector::from_carry_over`].
    pub fn to_bytes(&self) -> Result<Vec<u8>, CarryOverError> {
        let version = self.state_version.as_bytes();
        let len = u8::try_from(version.len())
            .map_err(|_| CarryOverError::Malformed("state_version de más de 255 bytes"))?;
        let mut out = Vec::with_capacity(1 + version.len() + BODY_LEN);
        out.push(len);
        out.extend_from_slice(version);
        out.extend_from_slice(&self.theta.to_le_bytes());
        out.extend_from_slice(&self.direction.to_le_bytes());
        put_point(&mut out, self.ext_high);
        put_point(&mut out, self.ext_low);
        out.push(u8::from(self.pending.is_some()));
        let (reference, confirm) = self
            .pending
            .map_or((ZERO, ZERO), |p| (p.reference, p.confirm));
        put_point(&mut out, reference);
        put_point(&mut out, confirm);
        Ok(out)
    }

    /// Inversa de [`CarryOver::to_bytes`]. Lee primero la versión: si no es
    /// [`STATE_VERSION`] devuelve `VersionMismatch` sin interpretar el resto,
    /// porque otra versión puede tener otro formato. Comprueba la forma
    /// (largo exacto, bandera 0/1, ceros sin evento pendiente) pero no las
    /// invariantes del estado.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CarryOverError> {
        let mut r = Reader(bytes);
        let len = usize::from(r.byte()?);
        let version = r.take(len)?;
        if version != STATE_VERSION.as_bytes() {
            return Err(CarryOverError::VersionMismatch {
                expected: STATE_VERSION,
                found: String::from_utf8_lossy(version).into_owned(),
            });
        }
        if r.0.len() != BODY_LEN {
            return Err(CarryOverError::Malformed("largo distinto del esperado"));
        }
        let theta = r.i64()?;
        let direction = i8::from_le_bytes([r.byte()?]);
        let ext_high = r.point()?;
        let ext_low = r.point()?;
        let has_pending = r.byte()?;
        let (reference, confirm) = (r.point()?, r.point()?);
        let pending = match has_pending {
            0 if reference == ZERO && confirm == ZERO => None,
            0 => return Err(CarryOverError::Malformed("evento pendiente sin bandera")),
            1 => Some(CarryPending { reference, confirm }),
            _ => return Err(CarryOverError::Malformed("has_pending_event no es 0 ni 1")),
        };
        Ok(Self {
            theta,
            state_version: STATE_VERSION.to_owned(),
            direction,
            ext_high,
            ext_low,
            pending,
        })
    }

    /// Comprueba que este estado puede retomarlo un detector de `theta`.
    /// Las invariantes salen de cómo el detector construye su estado, no de
    /// heurísticas:
    ///
    /// - los precios cumplen `0 < price < PRICE_LIMIT` (contrato de L1);
    /// - `ext_high.price ≥ ext_low.price`: ambos se actualizan con cada tick;
    /// - un evento pendiente exige tendencia definida, la confirmación es
    ///   estrictamente posterior a la referencia (un DC tiene al menos un
    ///   tick) y su precio está del lado de la tendencia (upturn: más alto);
    /// - la confirmación de un evento es el extremo de la tendencia desde ahí,
    ///   y solo se reemplaza por otro más alto y más reciente (upturn; a la
    ///   inversa en downturn): el extremo vigente no puede quedar antes.
    fn validate(&self, theta: i64) -> Result<(), CarryOverError> {
        if self.state_version != STATE_VERSION {
            return Err(CarryOverError::VersionMismatch {
                expected: STATE_VERSION,
                found: self.state_version.clone(),
            });
        }
        if theta <= 0 || theta >= SCALE {
            return Err(ThetaError(theta).into());
        }
        if self.theta != theta {
            return Err(CarryOverError::ThetaMismatch {
                expected: theta,
                found: self.theta,
            });
        }
        let direction = direction_from_i8(self.direction)?;
        check_price(
            self.ext_high,
            "precio de ext_high fuera de 0 < price < PRICE_LIMIT",
        )?;
        check_price(
            self.ext_low,
            "precio de ext_low fuera de 0 < price < PRICE_LIMIT",
        )?;
        if self.ext_high.price < self.ext_low.price {
            return Err(CarryOverError::Inconsistent(
                "ext_high por debajo de ext_low",
            ));
        }
        let Some(pending) = self.pending else {
            return Ok(());
        };
        let Some(direction) = direction else {
            return Err(CarryOverError::Inconsistent(
                "evento pendiente con dirección indefinida",
            ));
        };
        check_price(
            pending.reference,
            "precio de la referencia pendiente fuera de 0 < price < PRICE_LIMIT",
        )?;
        check_price(
            pending.confirm,
            "precio de la confirmación pendiente fuera de 0 < price < PRICE_LIMIT",
        )?;
        let (reference, confirm) = (pending.reference, pending.confirm);
        if confirm.agg_trade_id <= reference.agg_trade_id || confirm.time < reference.time {
            return Err(CarryOverError::Inconsistent(
                "la confirmación pendiente no es posterior a su referencia",
            ));
        }
        let (moved, extreme) = match direction {
            Direction::Up => (confirm.price > reference.price, self.ext_high),
            Direction::Down => (confirm.price < reference.price, self.ext_low),
        };
        if !moved {
            return Err(CarryOverError::Inconsistent(
                "el precio de la confirmación pendiente no va en el sentido de la tendencia",
            ));
        }
        let beyond = match direction {
            Direction::Up => extreme.price >= confirm.price,
            Direction::Down => extreme.price <= confirm.price,
        };
        if !beyond || extreme.agg_trade_id < confirm.agg_trade_id {
            return Err(CarryOverError::Inconsistent(
                "el extremo vigente queda antes de la confirmación pendiente",
            ));
        }
        Ok(())
    }
}

impl Detector {
    /// Estado a entregar al cierre de una unidad de trabajo (un mes de un θ).
    ///
    /// Antes hay que llamar a [`Detector::finish`] (y escribir el evento que
    /// devuelva): un grupo de empate abierto no tiene lugar en el carry-over,
    /// porque ningún grupo cruza el borde de mes (TRD-L2 §8.1 paso 4).
    /// Devuelve `OpenGroup` si quedó uno y `NoTicks` si el detector no vio
    /// ningún tick.
    pub fn carry_over(&self) -> Result<CarryOver, CarryOverError> {
        if self.state.group.is_some() {
            return Err(CarryOverError::OpenGroup);
        }
        let extremes = self.state.extremes.ok_or(CarryOverError::NoTicks)?;
        let direction = self.state.direction;
        if let Some(pending) = self.state.pending {
            // Inalcanzable con un θ válido (ver `PendingEvent`); si ocurriera,
            // el TRD no tiene dónde guardar una dirección distinta.
            if Some(pending.direction) != direction {
                return Err(CarryOverError::Inconsistent(
                    "la dirección del evento pendiente difiere de la del detector",
                ));
            }
        }
        let carry = CarryOver {
            theta: self.theta,
            state_version: STATE_VERSION.to_owned(),
            direction: direction.map_or(0, Direction::as_i8),
            ext_high: extremes.high,
            ext_low: extremes.low,
            pending: self.state.pending.map(|p| CarryPending {
                reference: p.reference,
                confirm: p.confirm,
            }),
        };
        carry.validate(self.theta)?;
        Ok(carry)
    }

    /// Detector de `theta` que continúa donde quedó `carry`, como si hubiera
    /// visto todos los ticks anteriores. Rechaza un estado de otro θ, de otra
    /// versión o que rompa las invariantes del detector.
    ///
    /// El contador de [`Detector::discarded`] arranca en cero: cuenta por
    /// unidad de trabajo (θ y mes, TRD-L2 §9.1), no por cadena.
    pub fn from_carry_over(theta: i64, carry: &CarryOver) -> Result<Self, CarryOverError> {
        carry.validate(theta)?;
        let direction = direction_from_i8(carry.direction)?;
        Ok(Self {
            theta,
            state: State {
                direction,
                extremes: Some(Extremes {
                    high: carry.ext_high,
                    low: carry.ext_low,
                }),
                pending: carry
                    .pending
                    .zip(direction)
                    .map(|(p, direction)| PendingEvent {
                        reference: p.reference,
                        confirm: p.confirm,
                        direction,
                    }),
                group: None,
            },
            discarded: 0,
        })
    }
}

const ZERO: Point = Point {
    price: 0,
    time: 0,
    agg_trade_id: 0,
};

fn direction_from_i8(value: i8) -> Result<Option<Direction>, CarryOverError> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(Direction::Up)),
        -1 => Ok(Some(Direction::Down)),
        _ => Err(CarryOverError::Inconsistent("direction no es -1, 0 ni 1")),
    }
}

fn check_price(point: Point, out_of_range: &'static str) -> Result<(), CarryOverError> {
    if point.price > 0 && point.price < PRICE_LIMIT {
        Ok(())
    } else {
        Err(CarryOverError::Inconsistent(out_of_range))
    }
}

fn put_point(out: &mut Vec<u8>, point: Point) {
    out.extend_from_slice(&point.price.to_le_bytes());
    out.extend_from_slice(&point.time.to_le_bytes());
    out.extend_from_slice(&point.agg_trade_id.to_le_bytes());
}

/// Lector de bytes con error en vez de pánico si faltan.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], CarryOverError> {
        if self.0.len() < n {
            return Err(CarryOverError::Malformed("bytes insuficientes"));
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }

    fn byte(&mut self) -> Result<u8, CarryOverError> {
        Ok(self.take(1)?[0])
    }

    fn i64(&mut self) -> Result<i64, CarryOverError> {
        let bytes = self.take(8)?;
        Ok(i64::from_le_bytes(
            bytes.try_into().expect("take(8) devuelve 8 bytes"),
        ))
    }

    fn point(&mut self) -> Result<Point, CarryOverError> {
        Ok(Point {
            price: self.i64()?,
            time: self.i64()?,
            agg_trade_id: self.i64()?,
        })
    }
}
