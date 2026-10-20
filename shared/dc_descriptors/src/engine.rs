//! El motor: una cola de ítems `(row group, θ)` que consumen hilos nativos.
//!
//! **Fronteras.** Antes del primer `push`, `set_boundaries` entrega una vez por θ las
//! columnas de L2 de sus eventos del mes (buffers de Arrow, sin copia). El motor decide
//! solo qué eventos toca un row group y dónde cae cada frontera: dentro del ítem, con
//! búsqueda binaria sobre los ids de los ticks y de los eventos, sin estado entre row
//! groups. Solo necesita saber cuál es el último id del row group anterior, que `push`
//! fija en orden de ordinal.
//!
//! **Ítems.** `push` convierte un row group en un trabajo con un ítem por θ que tiene
//! eventos. Los hilos toman ítems de la cola con un índice atómico (`fetch_add`):
//! ningún hilo es de un θ ni de un row group, y el camino por tick no toma locks. Un ítem
//! resuelve las posiciones de los eventos que toca su θ y agrega sus fases
//! (`phase::aggregate`).
//!
//! **Casillas.** El evento `i` de un θ es la fila `i` de su esqueleto: su casilla. Un
//! evento que cabe en un row group se cierra en el ítem y su fila va a la casilla. Uno
//! que cruza row groups deja un parcial en el almacén de su θ, que cada ítem fusiona con
//! su tramo (asociativo, en cualquier orden); el ítem que completa los parciales lo
//! cierra. Cuántos son se sabe sin esperar: van del row group de la primera pieza (la que
//! no empieza antes) al de la última (la que no termina después).
//!
//! **Sellado.** Cuando todas las casillas de un row group de salida están llenas, se
//! sella y se entrega en orden de row group de salida (`pop_sealed`).
//!
//! Los únicos locks son el almacén de parciales de un θ (una vez por ítem, para dos
//! eventos como mucho), el de una casilla (una vez por tramo de filas) y los de la cola.

use crate::close::{close_event, EventError, EventErrorKind};
use crate::input::{
    Boundaries, EventBatch, EventIn, InputError, KeyColumns, RowGroupInput, TickColumns,
};
use crate::layout::{col, Row};
use crate::output::{Delivery, OutTable, OutputError, SealedRowGroup};
use crate::phase::{aggregate, EventAcc, TickError};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Cómo se arma un motor.
#[derive(Debug, Clone)]
pub struct Config {
    /// Los θ de la unidad, `round(θ × 10⁸)`, en el orden en que se reportan los errores.
    pub thetas: Vec<i64>,
    /// Hilos nativos (al menos 1).
    pub threads: usize,
    /// Por θ, las filas de cada row group de salida (el esqueleto de L2).
    pub row_groups: Vec<Vec<usize>>,
}

/// Por qué el motor falló.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// La configuración o un `push` rompen el contrato de entrada.
    Input(String),
    /// Un tick de L1 no cumple su contrato.
    Ticks(TickError),
    /// Un evento no se puede cerrar o contradice a L2.
    Event(EventError),
    /// Una frontera de un evento no es un tick de L1 o no se vio antes.
    Boundary(BoundaryError),
    /// El tick `C` de un evento no tiene el `transact_time` que L2 dice.
    ConfirmTime(ConfirmTimeError),
    /// Hay más eventos que filas en el esqueleto del θ.
    Skeleton { theta: usize, event: u64, rows: u64 },
    /// Al terminar faltan filas: llegaron menos eventos que el esqueleto.
    Incomplete {
        theta: usize,
        missing: u64,
        expected: u64,
    },
    /// Un defecto del motor (un hilo entró en pánico, un parcial de más).
    Internal(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Internal(message) => f.write_str(message),
            Self::Ticks(error) => error.fmt(f),
            Self::Event(error) => error.fmt(f),
            Self::Boundary(error) => error.fmt(f),
            Self::ConfirmTime(error) => error.fmt(f),
            Self::Skeleton { theta, event, rows } => write!(
                f,
                "el θ {theta} tiene {rows} filas en el esqueleto y llega el evento {event}"
            ),
            Self::Incomplete {
                theta,
                missing,
                expected,
            } => write!(
                f,
                "al terminar faltan {missing} de las {expected} filas del θ {theta}"
            ),
        }
    }
}

impl std::error::Error for EngineError {}

/// Por qué una frontera no vale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind {
    /// Cae dentro del rango de ids del row group y ningún tick tiene ese id.
    NotATick,
    /// Cae antes del row group y no dentro de ningún row group anterior: en un hueco.
    Unseen,
}

/// Una frontera de un evento que no existe como tick de L1 (el lector falla igual).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryError {
    pub theta: usize,
    pub event: u64,
    /// Cuál: 0 `R`, 1 `C` o 2 `E`.
    pub field: usize,
    pub kind: BoundaryKind,
    /// `R`, `C` y `E` del evento.
    pub bounds: [i64; 3],
    /// Primer y último id del row group donde se encontró.
    pub low: i64,
    pub high: i64,
}

impl fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "el id {} del evento {} del θ {} no es un tick de L1 (row group con ids {}..{})",
            self.bounds[self.field], self.event, self.theta, self.low, self.high
        )
    }
}

/// El tick `C` de un evento tiene un `transact_time` distinto del `confirm_time` de L2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfirmTimeError {
    pub theta: usize,
    pub event: u64,
    pub bounds: [i64; 3],
    pub got: i64,
    pub wanted: i64,
}

impl fmt::Display for ConfirmTimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "el tick de confirmación {} del evento {} del θ {} tiene transact_time {} y L2 
             dice confirm_time {}",
            self.bounds[1], self.event, self.theta, self.got, self.wanted
        )
    }
}

impl From<InputError> for EngineError {
    fn from(error: InputError) -> Self {
        Self::Input(error.0)
    }
}

/// Lo que el motor lleva contado de un θ para la conciliación de §9.1.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThetaStats {
    /// Eventos cerrados.
    pub rows: u64,
    /// `Σ (c_ticks + os_ticks)` de los eventos cerrados.
    pub closed_ticks: u64,
    /// `Σ ticks_before_month` de los eventos cerrados.
    pub before_month: u64,
    /// `reference_agg_trade_id` del primer evento del θ.
    pub first_reference: Option<i64>,
    /// `extreme_agg_trade_id` del último evento del θ.
    pub last_extreme: Option<i64>,
}

/// Qué row groups se empujaron y cuáles siguen en proceso.
#[derive(Default)]
struct Groups {
    /// Todos los ordinales menores se empujaron.
    watermark: u64,
    /// Empujados con ordinal mayor o igual que `watermark` (llegaron antes de tiempo).
    ahead: BTreeSet<u64>,
    /// Empujados y aún sin terminar.
    open: BTreeSet<u64>,
}

impl Groups {
    /// Registra un row group; `false` si ya se había empujado.
    fn push(&mut self, id: u64) -> bool {
        if id < self.watermark || !self.ahead.insert(id) {
            return false;
        }
        while self.ahead.remove(&self.watermark) {
            self.watermark += 1;
        }
        self.open.insert(id);
        true
    }
}

/// Un error con su clave de orden: `(row group donde el evento termina, θ, evento)`, la
/// misma en que lo habría encontrado el recorrido secuencial. Se reporta el de menor
/// clave, así que no depende de qué hilo llegó primero.
#[derive(Debug, Clone)]
struct Failure {
    key: (u64, usize, u64),
    error: EngineError,
}

struct Partial {
    acc: EventAcc,
    merged: u64,
    first_group: Option<u64>,
    last_group: Option<u64>,
}

struct Job<B> {
    id: u64,
    ticks: TickColumns<B>,
    before_month: bool,
    /// Último id del row group.
    high: i64,
    /// Último id del row group anterior (ordinal `id - 1`); `None` en el primero.
    prev_high: Option<i64>,
    /// Ítems del trabajo: uno por θ con eventos.
    items: usize,
    /// Siguiente ítem por tomar. Puede pasar de `items`.
    next: AtomicUsize,
    remaining: AtomicUsize,
}

/// Los row groups empujados que esperan a su anterior, y el último id del que salió.
struct Order<B> {
    /// Ordinal del siguiente row group por soltar a la cola.
    next: u64,
    last_high: Option<i64>,
    held: BTreeMap<u64, Job<B>>,
}

#[derive(Default)]
struct Queue<B> {
    jobs: VecDeque<Arc<Job<B>>>,
    shutdown: bool,
}

/// Mide la fracción del tiempo con ítems esperando en la cola.
#[derive(Default)]
struct BusyClock {
    queued: AtomicI64,
    times: Mutex<ClockTimes>,
}

#[derive(Default)]
struct ClockTimes {
    started: Option<Instant>,
    busy: Duration,
    since: Option<Instant>,
}

impl BusyClock {
    fn add(&self, items: i64) {
        let mut times = lock(&self.times);
        let now = Instant::now();
        times.started.get_or_insert(now);
        if self.queued.fetch_add(items, Ordering::AcqRel) == 0 {
            times.since.get_or_insert(now);
        }
    }

    /// Un ítem salió de la cola. Solo el que la deja vacía toma el lock.
    fn take(&self) {
        if self.queued.fetch_sub(1, Ordering::AcqRel) == 1 {
            let mut times = lock(&self.times);
            if self.queued.load(Ordering::Acquire) == 0 {
                if let Some(since) = times.since.take() {
                    times.busy += since.elapsed();
                }
            }
        }
    }

    fn fraction(&self) -> f64 {
        let times = lock(&self.times);
        let Some(started) = times.started else {
            return 0.0;
        };
        let open = times.since.map_or(Duration::ZERO, |since| since.elapsed());
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            return 0.0;
        }
        ((times.busy + open).as_secs_f64() / elapsed).min(1.0)
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Un evento que toca un row group, con las posiciones de sus fronteras en él.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Located {
    /// Índice del evento en el θ (la fila `index` del esqueleto).
    pub index: u64,
    pub event: EventIn,
    /// `R`, `C` y `E` como cuenta de ticks del row group con id ≤ la frontera: la
    /// confirmación son los ticks `[R, C)` y el overshoot `[C, E)`. Una frontera fuera del
    /// row group queda en 0 (anterior) o en su largo (posterior).
    pub positions: [usize; 3],
    /// El evento empieza antes del row group (`R` < primer id).
    pub starts_before: bool,
    /// El evento termina después del row group (`E` > último id).
    pub ends_after: bool,
}

/// Qué eventos de un θ toca un row group y dónde caen sus fronteras en él, con las
/// comprobaciones del lector (`dc_frames.track.locate`): una frontera dentro del rango de
/// ids que no es un tick, una anterior al row group que no cae dentro de ninguno anterior
/// (cae en un hueco) y el `transact_time` del tick `C`.
///
/// `prev_high` es el último id del row group anterior (`None` en el primero). Toca los
/// eventos que no cerraron en el anterior (`E > prev_high`) y empiezan en este o antes
/// (`R <= último id`). No guarda estado entre row groups: una frontera anterior a este row
/// group vale si `<= prev_high`, porque entonces cayó dentro de un row group anterior
/// (que la comprobó como tick) o en un hueco antes de él (que el ítem de ese row group ya
/// rechazó). Un fallo de frontera pesa más que uno de `confirm_time`, como en el lector.
pub fn locate_events<B: AsRef<[u8]>>(
    theta: usize,
    bounds: &Boundaries<B>,
    keys: &KeyColumns<B>,
    prev_high: Option<i64>,
) -> Result<Vec<Located>, EngineError> {
    let (low, high) = (keys.id(0), keys.id(keys.rows() - 1));
    let first = prev_high.map_or(0, |high| bounds.extremes_up_to(high));
    let end = bounds.references_up_to(high);
    if first >= end {
        return Ok(Vec::new());
    }
    let mut located = Vec::with_capacity(end - first);
    let mut late = None;
    for index in first..end {
        let event = bounds.event(index);
        let ids = [event.reference, event.confirm, event.extreme];
        let mut positions = [0usize; 3];
        for (field, &id) in ids.iter().enumerate() {
            let rank = keys.rank(id);
            positions[field] = rank;
            let kind = if (low..=high).contains(&id) {
                (rank == 0 || keys.id(rank - 1) != id).then_some(BoundaryKind::NotATick)
            } else if id < low && prev_high.map_or(true, |known| id > known) {
                Some(BoundaryKind::Unseen)
            } else {
                None
            };
            if let Some(kind) = kind {
                return Err(EngineError::Boundary(BoundaryError {
                    theta,
                    event: index as u64,
                    field,
                    kind,
                    bounds: ids,
                    low,
                    high,
                }));
            }
        }
        if late.is_none() && (low..=high).contains(&event.confirm) {
            let got = keys.time_at(positions[1] - 1);
            if got != event.confirm_time {
                late = Some(EngineError::ConfirmTime(ConfirmTimeError {
                    theta,
                    event: index as u64,
                    bounds: ids,
                    got,
                    wanted: event.confirm_time,
                }));
            }
        }
        located.push(Located {
            index: index as u64,
            event,
            positions,
            starts_before: event.reference < low,
            ends_after: event.extreme > high,
        });
    }
    late.map_or(Ok(located), Err)
}

/// El índice del evento de un error de ubicación, para su clave de orden.
fn located_event(error: &EngineError) -> u64 {
    match error {
        EngineError::Boundary(e) => e.event,
        EngineError::ConfirmTime(e) => e.event,
        _ => 0,
    }
}

struct Shared<B> {
    thetas: usize,
    out: OutTable,
    delivery: Mutex<Delivery>,
    queue: Mutex<Queue<B>>,
    work: Condvar,
    /// Row groups empujados y aún sin terminar, por id.
    groups: Mutex<Groups>,
    settled: Condvar,
    /// Las fronteras de L2 de cada θ, fijadas una vez antes del primer `push`.
    bounds: Vec<OnceLock<Boundaries<B>>>,
    /// Los θ con al menos un evento, que se fija en el primer `push`; un ítem es uno de ellos.
    active: OnceLock<Vec<usize>>,
    order: Mutex<Order<B>>,
    partials: Vec<Mutex<HashMap<u64, Partial>>>,
    stats: Vec<Mutex<ThetaStats>>,
    failure: Mutex<Option<Failure>>,
    failed: AtomicBool,
    first_failed_group: AtomicU64,
    busy: BusyClock,
}

/// El motor de agregación de L3. `B` es el dueño de la memoria de los ticks de Arrow.
pub struct Engine<B: AsRef<[u8]> + Send + Sync + 'static> {
    shared: Arc<Shared<B>>,
    workers: Vec<JoinHandle<()>>,
}

impl<B: AsRef<[u8]> + Send + Sync + 'static> Engine<B> {
    /// Arma el motor y lanza sus hilos.
    pub fn new(config: Config) -> Result<Self, EngineError> {
        let thetas = config.thetas.len();
        if thetas == 0 {
            return Err(EngineError::Input("el motor necesita al menos un θ".into()));
        }
        if config.row_groups.len() != thetas {
            return Err(EngineError::Input(format!(
                "hay {} esqueletos para {thetas} θ",
                config.row_groups.len()
            )));
        }
        let shared = Arc::new(Shared {
            thetas,
            out: OutTable::new(&config.row_groups),
            delivery: Mutex::new(Delivery::new(thetas)),
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                shutdown: false,
            }),
            work: Condvar::new(),
            groups: Mutex::new(Groups::default()),
            settled: Condvar::new(),
            bounds: (0..thetas).map(|_| OnceLock::new()).collect(),
            active: OnceLock::new(),
            order: Mutex::new(Order {
                next: 0,
                last_high: None,
                held: BTreeMap::new(),
            }),
            partials: (0..thetas).map(|_| Mutex::new(HashMap::new())).collect(),
            stats: (0..thetas).map(|_| Mutex::default()).collect(),
            failure: Mutex::new(None),
            failed: AtomicBool::new(false),
            first_failed_group: AtomicU64::new(u64::MAX),
            busy: BusyClock::default(),
        });
        let workers = (0..config.threads.max(1))
            .map(|n| {
                let shared = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("dc-descriptors-{n}"))
                    .spawn(move || shared.worker())
                    .map_err(|e| EngineError::Internal(format!("no se pudo lanzar un hilo: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { shared, workers })
    }

    /// Fija una vez las filas de L2 de los eventos del θ `theta` en el mes: sus row groups de
    /// `events.parquet`, en orden y sin copiar. Hay que llamarla para cada θ antes del
    /// primer `push` (un θ sin eventos lleva una lista vacía).
    ///
    /// Falla si el θ no existe, ya tiene fronteras, ya empezaron los `push`, un buffer no
    /// alcanza, un precio no cabe en `i64` o los eventos no cumplen `R < C ≤ E` y
    /// `E_i ≤ R_{i+1}`.
    pub fn set_boundaries(
        &self,
        theta: usize,
        batches: Vec<EventBatch<B>>,
    ) -> Result<(), EngineError> {
        let Some(slot) = self.shared.bounds.get(theta) else {
            return Err(EngineError::Input(format!("el θ {theta} no existe")));
        };
        if self.shared.active.get().is_some() {
            return Err(EngineError::Input(format!(
                "las fronteras del θ {theta} llegan después del primer push"
            )));
        }
        let boundaries = Boundaries::new(batches)
            .map_err(|e| EngineError::Input(format!("fronteras del θ {theta}: {e}")))?;
        slot.set(boundaries)
            .map_err(|_| EngineError::Input(format!("el θ {theta} ya tiene fronteras")))
    }

    /// Encola un row group. Vuelve en cuanto los ítems están en la cola: el trabajo corre
    /// en los hilos. El row group se libera (se suelta `B`) cuando sus ítems terminan.
    ///
    /// Un row group espera a que se empuje el anterior (su ordinal menos uno): los ítems
    /// necesitan su último id para saber qué eventos cerraron antes. Con los ordinales en
    /// orden no espera nada.
    ///
    /// Falla si el contrato de entrada se rompe o si el motor ya falló.
    pub fn push(&self, input: RowGroupInput<B>) -> Result<(), EngineError> {
        self.shared.poll_failure()?;
        let RowGroupInput {
            id,
            ticks,
            before_month,
        } = input;
        if let Some(theta) = self.shared.bounds.iter().position(|b| b.get().is_none()) {
            return Err(EngineError::Input(format!(
                "el row group {id} llega antes de las fronteras del θ {theta}"
            )));
        }
        let active = self.shared.active.get_or_init(|| {
            (0..self.shared.thetas)
                .filter(|&t| self.shared.bounds[t].get().is_some_and(|b| !b.is_empty()))
                .collect()
        });
        if !lock(&self.shared.groups).push(id) {
            return Err(EngineError::Input(format!(
                "el row group {id} ya se empujó"
            )));
        }
        let rows = ticks.rows();
        let job = Job {
            id,
            high: ticks.keys().id(rows - 1),
            ticks,
            before_month,
            prev_high: None,
            items: active.len(),
            next: AtomicUsize::new(0),
            remaining: AtomicUsize::new(active.len()),
        };
        let mut ready = Vec::new();
        {
            let mut guard = lock(&self.shared.order);
            let order = &mut *guard;
            order.held.insert(id, job);
            while let Some(mut job) = order.held.remove(&order.next) {
                job.prev_high = order.last_high;
                order.last_high = Some(job.high);
                order.next += 1;
                ready.push(job);
            }
        }
        for job in ready {
            self.shared.release(job);
        }
        Ok(())
    }

    /// Dónde caen en un row group las fronteras de los eventos que lo tocan, sin empujarlo:
    /// lo mismo que calcula un ítem (ver [`locate_events`]), para probar y diagnosticar.
    /// `prev_high` es el último id del row group anterior. Falla si el θ no tiene fronteras.
    pub fn locate(
        &self,
        theta: usize,
        keys: &KeyColumns<B>,
        prev_high: Option<i64>,
    ) -> Result<Vec<Located>, EngineError> {
        let bounds = self
            .shared
            .bounds
            .get(theta)
            .and_then(OnceLock::get)
            .ok_or_else(|| EngineError::Input(format!("el θ {theta} no tiene fronteras")))?;
        locate_events(theta, bounds, keys, prev_high)
    }

    /// El siguiente row group de salida sellado, o `None`. Cada θ sale en orden de row
    /// group de salida. Falla si el motor ya falló.
    pub fn pop_sealed(&self) -> Result<Option<SealedRowGroup>, EngineError> {
        self.shared.poll_failure()?;
        Ok(lock(&self.shared.delivery).pop())
    }

    /// Row groups empujados cuyos ítems no terminaron: los que aún anclan sus ticks.
    pub fn inflight(&self) -> usize {
        lock(&self.shared.groups).open.len()
    }

    /// Fracción del tiempo, desde el primer `push`, con ítems esperando en la cola.
    pub fn busy(&self) -> f64 {
        self.shared.busy.fraction()
    }

    /// Fin de la entrada: espera los ítems pendientes y comprueba que cada casilla se
    /// llenó. Devuelve las cuentas de cada θ; los row groups sellados siguen en
    /// `pop_sealed`.
    pub fn finish(&self) -> Result<Vec<ThetaStats>, EngineError> {
        {
            // Un row group que espera a su anterior no se procesará nunca: la entrada no
            // trajo todos los ordinales.
            let order = lock(&self.shared.order);
            if !order.held.is_empty() {
                return Err(EngineError::Input(format!(
                    "falta el row group {}: hay {} esperándolo",
                    order.next,
                    order.held.len()
                )));
            }
        }
        {
            let mut groups = lock(&self.shared.groups);
            while !groups.open.is_empty() {
                groups = self
                    .shared
                    .settled
                    .wait(groups)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
        if let Some(failure) = lock(&self.shared.failure).clone() {
            return Err(failure.error);
        }
        for theta in 0..self.shared.thetas {
            let missing = self.shared.out.missing(theta);
            if missing > 0 || !lock(&self.shared.partials[theta]).is_empty() {
                return Err(EngineError::Incomplete {
                    theta,
                    missing,
                    expected: self.shared.out.rows(theta),
                });
            }
        }
        // Todas las casillas se llenaron: el primer evento del θ abrió con su referencia y
        // el último cerró en su extremo.
        Ok((0..self.shared.thetas)
            .map(|theta| {
                let mut stats = *lock(&self.shared.stats[theta]);
                if let Some(bounds) = self.shared.bounds[theta].get().filter(|b| !b.is_empty()) {
                    stats.first_reference = Some(bounds.reference(0));
                    stats.last_extreme = Some(bounds.extreme(bounds.len() - 1));
                }
                stats
            })
            .collect())
    }
}

impl<B: AsRef<[u8]> + Send + Sync + 'static> Drop for Engine<B> {
    fn drop(&mut self) {
        lock(&self.shared.queue).shutdown = true;
        self.shared.work.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl<B: AsRef<[u8]> + Send + Sync + 'static> Shared<B> {
    fn worker(&self) {
        let mut current: Option<Arc<Job<B>>> = None;
        loop {
            let Some(job) = current.as_ref() else {
                current = self.wait_for_job();
                if current.is_none() {
                    return;
                }
                continue;
            };
            let item = job.next.fetch_add(1, Ordering::Relaxed);
            if item >= job.items {
                current = None;
                continue;
            }
            self.busy.take();
            self.run(job, item);
            if job.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.finish_group(job.id);
            }
        }
    }

    /// El primer trabajo de la cola con ítems por tomar; duerme si no hay. `None` al
    /// apagar el motor.
    fn wait_for_job(&self) -> Option<Arc<Job<B>>> {
        let mut queue = lock(&self.queue);
        loop {
            while let Some(front) = queue.jobs.front() {
                if front.next.load(Ordering::Relaxed) < front.items {
                    return Some(Arc::clone(front));
                }
                queue.jobs.pop_front();
            }
            if queue.shutdown {
                return None;
            }
            queue = self
                .work
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn finish_group(&self, id: u64) {
        lock(&self.groups).open.remove(&id);
        self.settled.notify_all();
    }

    /// Suelta un trabajo a la cola (cuando su row group anterior ya salió de `held`).
    fn release(&self, job: Job<B>) {
        let (id, items) = (job.id, job.items);
        if items == 0 {
            self.finish_group(id);
            return;
        }
        self.busy.add(items as i64);
        lock(&self.queue).jobs.push_back(Arc::new(job));
        self.work.notify_all();
    }

    /// Tras un fallo se siguen procesando los row groups anteriores al del fallo (podrían
    /// tener uno de menor clave) y se descartan los posteriores.
    fn run(&self, job: &Job<B>, item: usize) {
        if self.failed.load(Ordering::Acquire)
            && job.id > self.first_failed_group.load(Ordering::Acquire)
        {
            return;
        }
        let theta = self.active.get().map_or(0, |active| active[item]);
        let outcome = catch_unwind(AssertUnwindSafe(|| self.process(job, theta)));
        let failure = match outcome {
            Ok(Ok(())) => return,
            Ok(Err(failure)) => *failure,
            Err(panic) => Failure {
                key: (job.id, theta, 0),
                error: EngineError::Internal(format!(
                    "un hilo del motor entró en pánico: {}",
                    panic
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("sin mensaje")
                )),
            },
        };
        let mut slot = lock(&self.failure);
        if slot.as_ref().map_or(true, |known| failure.key < known.key) {
            self.first_failed_group
                .fetch_min(failure.key.0, Ordering::AcqRel);
            *slot = Some(failure);
        }
        self.failed.store(true, Ordering::Release);
    }

    /// El fallo, una vez que todos los row groups anteriores al suyo se empujaron y
    /// terminaron: antes, uno de menor clave aún podría reemplazarlo.
    fn poll_failure(&self) -> Result<(), EngineError> {
        if !self.failed.load(Ordering::Acquire) {
            return Ok(());
        }
        let group = self.first_failed_group.load(Ordering::Acquire);
        {
            let groups = lock(&self.groups);
            if groups.watermark < group || groups.open.range(..=group).next().is_some() {
                return Ok(());
            }
        }
        match lock(&self.failure).clone() {
            Some(failure) => Err(failure.error),
            None => Ok(()),
        }
    }

    /// Un ítem: resuelve qué eventos de un θ toca el row group, los agrega, cierra los que
    /// quedan completos y deja el resto en el almacén de parciales.
    ///
    /// El ítem toca los eventos que no cerraron en el row group anterior (`E > prev_high`)
    /// y empiezan en este o antes (`R <= high`). Un evento con `R` antes del row group
    /// empieza antes (`starts_before`) y uno con `E` después termina después
    /// (`ends_after`): sus parciales se fusionan con los de los otros row groups.
    fn process(&self, job: &Job<B>, theta: usize) -> Result<(), Box<Failure>> {
        let fail = |group: u64, event: u64, error: EngineError| {
            Box::new(Failure {
                key: (group, theta, event),
                error,
            })
        };
        let event_error = |group: u64, event: u64, input: &EventIn, kind: EventErrorKind| {
            fail(
                group,
                event,
                EngineError::Event(EventError {
                    theta,
                    event,
                    confirm_agg_trade_id: input.confirm,
                    kind,
                }),
            )
        };
        let Some(bounds) = self.bounds[theta].get() else {
            return Err(fail(
                job.id,
                0,
                EngineError::Internal(format!("el θ {theta} no tiene fronteras")),
            ));
        };
        let located = locate_events(theta, bounds, job.ticks.keys(), job.prev_high)
            .map_err(|e| fail(job.id, located_event(&e), e))?;
        if located.is_empty() {
            return Ok(());
        }
        let first = located[0].index;
        let mut done: Vec<(u64, Row)> = Vec::with_capacity(located.len());
        let mut crossing: Vec<(usize, EventAcc)> = Vec::new();
        let mut before_month = 0u64;
        for (k, at) in located.iter().enumerate() {
            let (index, event, positions) = (at.index, &at.event, &at.positions);
            let acc = self
                .aggregate_event(&job.ticks, event, positions)
                .map_err(|e| fail(job.id, index, EngineError::Ticks(e)))?;
            if job.before_month {
                before_month += (positions[2] - positions[0]) as u64;
            }
            if at.starts_before || at.ends_after {
                crossing.push((k, acc));
            } else {
                let row = close_event(&acc, event)
                    .map_err(|kind| event_error(job.id, index, event, kind))?;
                done.push((index, row));
            }
        }
        if !crossing.is_empty() {
            let mut completed = Vec::new();
            {
                let mut store = lock(&self.partials[theta]);
                for (k, acc) in crossing {
                    let at = &located[k];
                    let index = at.index;
                    let partial = store.entry(index).or_insert_with(|| Partial {
                        acc: EventAcc::default(),
                        merged: 0,
                        first_group: None,
                        last_group: None,
                    });
                    partial.acc.merge(&acc);
                    partial.merged += 1;
                    if !at.starts_before {
                        partial.first_group = Some(job.id);
                    }
                    if !at.ends_after {
                        partial.last_group = Some(job.id);
                    }
                    let (Some(first_group), Some(last_group)) =
                        (partial.first_group, partial.last_group)
                    else {
                        continue;
                    };
                    let expected =
                        (last_group >= first_group).then(|| last_group - first_group + 1);
                    if expected.map_or(true, |n| partial.merged > n) {
                        return Err(fail(
                            job.id,
                            index,
                            EngineError::Internal(format!(
                                "el evento {index} del θ {theta} tiene {} parciales y sus row \
                                 groups van del {first_group} al {last_group}",
                                partial.merged
                            )),
                        ));
                    }
                    if Some(partial.merged) == expected {
                        let partial = store.remove(&index).expect("recién consultado");
                        completed.push((k, index, partial.acc, last_group));
                    }
                }
            }
            for (k, index, acc, last_group) in completed {
                let event = &located[k].event;
                let row = close_event(&acc, event)
                    .map_err(|kind| event_error(last_group, index, event, kind))?;
                done.push((index, row));
            }
        }
        done.sort_unstable_by_key(|(index, _)| *index);
        let closed_ticks: u64 = done
            .iter()
            .map(|(_, row)| (row[col::C_TICKS] + row[col::OS_TICKS]) as u64)
            .sum();
        let sealed = self.out.write(theta, &done).map_err(|e| match e {
            OutputError::Beyond { event, rows } => {
                fail(job.id, event, EngineError::Skeleton { theta, event, rows })
            }
            e @ OutputError::Overfull { .. } => {
                fail(job.id, first, EngineError::Internal(e.to_string()))
            }
        })?;
        {
            let mut stats = lock(&self.stats[theta]);
            stats.rows += done.len() as u64;
            stats.closed_ticks += closed_ticks;
            stats.before_month += before_month;
        }
        if !sealed.is_empty() {
            let mut delivery = lock(&self.delivery);
            for row_group in sealed {
                delivery.insert(row_group);
            }
        }
        Ok(())
    }

    fn aggregate_event(
        &self,
        ticks: &TickColumns<B>,
        event: &EventIn,
        positions: &[usize; 3],
    ) -> Result<EventAcc, TickError> {
        let [r, c, e] = *positions;
        let (confirmation, instant) = aggregate(ticks, r, c, Some(event.confirm_time))?;
        let (overshoot, _) = aggregate(ticks, c, e, None)?;
        Ok(EventAcc {
            confirmation,
            overshoot,
            instant,
        })
    }
}
