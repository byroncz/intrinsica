//! Fan-out de N θ: un lote de ticks, N detectores, hilos dentro del proceso.

use crate::detector::{CarryOver, CarryOverError, Detector, Event, Point, ThetaError};
use std::fmt;
use std::num::NonZeroUsize;
use std::thread;

/// Por qué el fan-out no pudo tomar o retomar los carry-over de sus θ.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FanOutCarryError {
    /// Hay otra cantidad de carry-over que de θ.
    Count { expected: usize, found: usize },
    /// El θ de la posición `index` (en el orden de `new`) falló.
    Detector {
        index: usize,
        source: CarryOverError,
    },
}

impl fmt::Display for FanOutCarryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count { expected, found } => {
                write!(f, "{found} carry-over para {expected} theta")
            }
            Self::Detector { index, source } => write!(f, "theta en la posición {index}: {source}"),
        }
    }
}

impl std::error::Error for FanOutCarryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Count { .. } => None,
            Self::Detector { source, .. } => Some(source),
        }
    }
}

/// Ticks que un hilo procesa por detector antes de pasar al siguiente. Con
/// 4096 ticks (96 KB en tres columnas) el trozo cabe en la caché L2: el hilo
/// lo lee de memoria una vez y lo reutiliza para todos sus detectores, en vez
/// de recorrer el lote entero una vez por θ.
const TILE: usize = 4096;

/// Por debajo de este trabajo (`ticks × θ`) crear hilos cuesta más que el
/// cómputo: el lote se evalúa en el hilo que llama, con el mismo resultado.
/// Medido en el contenedor (~7 ns por tick y θ): con 50 000 de trabajo
/// (350 µs) 10 hilos rinden peor que 5 y apenas mejoran al hilo solo; con
/// 250 000 (~1,8 ms) crear los hilos ya es una fracción pequeña del lote.
const PARALLEL_MIN_WORK: usize = 250_000;

/// N detectores DC, uno por θ, que evalúan cada lote de ticks en paralelo.
///
/// El patrón es "un tick, N detectores": los detectores no comparten estado,
/// así que se reparten en grupos contiguos, un hilo por grupo, y todos leen
/// las mismas columnas del lote (`&[i64]` prestadas, sin copiar la serie por
/// θ). Los hilos son `std::thread::scope`: viven lo que dura el lote y no
/// queda nada en segundo plano.
///
/// Memoria: el fan-out retiene N [`Detector`] (estado O(1) cada uno). Los
/// eventos cerrados salen por el valor de retorno de cada llamada; nada
/// crece con la serie.
#[derive(Debug, Clone)]
pub struct FanOut {
    detectors: Vec<Detector>,
    threads: usize,
}

impl FanOut {
    /// Un detector por cada `theta` (`round(θ × 10⁸)`), en ese orden; usa
    /// tantos hilos como núcleos disponibles. Falla si algún θ no cumple
    /// `0 < theta < SCALE`.
    pub fn new(thetas: &[i64]) -> Result<Self, ThetaError> {
        let cores = thread::available_parallelism().map_or(1, NonZeroUsize::get);
        Self::with_threads(thetas, cores)
    }

    /// Igual que [`FanOut::new`] con un máximo de `threads` hilos (mínimo 1;
    /// nunca más que detectores). Sirve para medir cómo escala con los núcleos.
    pub fn with_threads(thetas: &[i64], threads: usize) -> Result<Self, ThetaError> {
        let detectors = thetas
            .iter()
            .map(|&theta| Detector::new(theta))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_detectors(detectors, threads))
    }

    /// Fan-out que retoma el carry-over de cada θ (uno por `thetas`, mismo
    /// orden), con tantos hilos como núcleos disponibles. Falla si la cuenta
    /// no coincide o si algún estado no lo acepta [`Detector::from_carry_over`].
    pub fn from_carry_over(thetas: &[i64], carry: &[CarryOver]) -> Result<Self, FanOutCarryError> {
        let cores = thread::available_parallelism().map_or(1, NonZeroUsize::get);
        Self::from_carry_over_with_threads(thetas, carry, cores)
    }

    /// Igual que [`FanOut::from_carry_over`] con un máximo de `threads` hilos.
    pub fn from_carry_over_with_threads(
        thetas: &[i64],
        carry: &[CarryOver],
        threads: usize,
    ) -> Result<Self, FanOutCarryError> {
        if thetas.len() != carry.len() {
            return Err(FanOutCarryError::Count {
                expected: thetas.len(),
                found: carry.len(),
            });
        }
        let detectors = thetas
            .iter()
            .zip(carry)
            .enumerate()
            .map(|(index, (&theta, carry))| {
                Detector::from_carry_over(theta, carry)
                    .map_err(|source| FanOutCarryError::Detector { index, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_detectors(detectors, threads))
    }

    fn from_detectors(detectors: Vec<Detector>, threads: usize) -> Self {
        let threads = threads.clamp(1, detectors.len().max(1));
        Self { detectors, threads }
    }

    /// El carry-over de cada θ, en el orden de `new`, para entregarlo al
    /// cerrar la unidad de trabajo. Igual que [`Detector::carry_over`], exige
    /// haber llamado a [`FanOut::finish`]. Falla con el índice del primer θ
    /// que no pueda entregarlo.
    pub fn carry_overs(&self) -> Result<Vec<CarryOver>, FanOutCarryError> {
        self.detectors
            .iter()
            .enumerate()
            .map(|(index, d)| {
                d.carry_over()
                    .map_err(|source| FanOutCarryError::Detector { index, source })
            })
            .collect()
    }

    /// Los detectores, en el orden de los θ de entrada.
    pub fn detectors(&self) -> &[Detector] {
        &self.detectors
    }

    /// Alimenta un lote a todos los detectores. Las tres columnas son la
    /// misma serie (`prices[i]`, `times[i]`, `ids[i]` forman el tick `i`) en
    /// orden estricto de `(time, agg_trade_id)`, continuando el lote anterior.
    ///
    /// Devuelve, por θ y en el orden de `new`, los eventos que este lote
    /// dejó cerrados, en orden. Un lote puede tener un solo tick o cortar un
    /// grupo de empate: el resultado no depende de cómo se parta la serie.
    ///
    /// # Panics
    /// Si las columnas no miden lo mismo, o si un precio viola el contrato
    /// de [`Detector::feed`]. Tras un pánico los detectores quedan a medio
    /// lote y el fan-out no debe reutilizarse.
    pub fn feed_batch(&mut self, prices: &[i64], times: &[i64], ids: &[i64]) -> Vec<Vec<Event>> {
        assert!(
            prices.len() == times.len() && prices.len() == ids.len(),
            "columnas de distinto largo: prices {}, times {}, ids {}",
            prices.len(),
            times.len(),
            ids.len()
        );
        let n = self.detectors.len();
        let mut events = vec![Vec::new(); n];
        let batch = Batch { prices, times, ids };
        if n == 0 || prices.is_empty() {
            return events;
        }

        let threads = if prices.len().saturating_mul(n) < PARALLEL_MIN_WORK {
            1
        } else {
            self.threads
        };
        let group = n.div_ceil(threads);
        let mut groups = self
            .detectors
            .chunks_mut(group)
            .zip(events.chunks_mut(group));
        // El último grupo lo corre el hilo que llama: un hilo menos que crear.
        let own = groups.next_back();
        thread::scope(|scope| {
            for (detectors, out) in groups {
                scope.spawn(move || feed_group(detectors, out, batch));
            }
            if let Some((detectors, out)) = own {
                feed_group(detectors, out, batch);
            }
        });
        events
    }

    /// Fin de entrada (RF-L2-12): cierra el grupo de empate abierto de cada
    /// θ, en el orden de `new`.
    pub fn finish(&mut self) -> Vec<Option<Event>> {
        self.detectors.iter_mut().map(Detector::finish).collect()
    }
}

/// Las tres columnas de un lote; solo referencias, `Copy` para dársela a cada hilo.
#[derive(Clone, Copy)]
struct Batch<'a> {
    prices: &'a [i64],
    times: &'a [i64],
    ids: &'a [i64],
}

/// Un hilo: su grupo de detectores contra el lote completo, por trozos.
/// Cada detector ve los ticks en orden, así que el resultado es el mismo que
/// recorrerlo entero de una vez.
fn feed_group(detectors: &mut [Detector], out: &mut [Vec<Event>], batch: Batch<'_>) {
    let Batch { prices, times, ids } = batch;
    for ((prices, times), ids) in prices
        .chunks(TILE)
        .zip(times.chunks(TILE))
        .zip(ids.chunks(TILE))
    {
        for (detector, out) in detectors.iter_mut().zip(out.iter_mut()) {
            for ((&price, &time), &agg_trade_id) in prices.iter().zip(times).zip(ids) {
                let tick = Point {
                    price,
                    time,
                    agg_trade_id,
                };
                if let Some(event) = detector.feed(tick) {
                    out.push(event);
                }
            }
        }
    }
}
