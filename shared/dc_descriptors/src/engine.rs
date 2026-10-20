//! El motor: una cola de ítems `(row group, θ)` que consumen hilos nativos.
//!
//! **Ítems.** `push` convierte un row group en un trabajo con un ítem por θ que tiene
//! eventos en él. Los hilos toman ítems de la cola con un índice atómico (`fetch_add`):
//! ningún hilo es de un θ ni de un row group, y el camino por tick no toma locks. Un ítem
//! agrega las fases de todos los eventos de su θ en el row group (`phase::aggregate`).
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
use crate::input::{EventIn, InputError, RowGroupInput, ThetaPiece, TickColumns};
use crate::layout::{col, Row};
use crate::output::{Delivery, OutTable, OutputError, SealedRowGroup};
use crate::phase::{aggregate, EventAcc, TickError};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
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

/// Las cuentas de un θ y el índice del evento que fijó `last_extreme`.
#[derive(Default)]
struct Counters {
    stats: ThetaStats,
    last_index: Option<u64>,
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
    pieces: Vec<ThetaPiece>,
    /// Siguiente ítem por tomar. Puede pasar de `pieces.len()`.
    next: AtomicUsize,
    remaining: AtomicUsize,
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

struct Shared<B> {
    thetas: usize,
    out: OutTable,
    delivery: Mutex<Delivery>,
    queue: Mutex<Queue<B>>,
    work: Condvar,
    /// Row groups empujados y aún sin terminar, por id.
    groups: Mutex<Groups>,
    settled: Condvar,
    partials: Vec<Mutex<HashMap<u64, Partial>>>,
    stats: Vec<Mutex<Counters>>,
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

    /// Encola un row group. Vuelve en cuanto los ítems están en la cola: el trabajo corre
    /// en los hilos. El row group se libera (se suelta `B`) cuando sus ítems terminan.
    ///
    /// Falla si el contrato de entrada se rompe o si el motor ya falló.
    pub fn push(&self, input: RowGroupInput<B>) -> Result<(), EngineError> {
        self.shared.poll_failure()?;
        let RowGroupInput { id, ticks, pieces } = input;
        let rows = ticks.rows();
        let mut seen = vec![false; self.shared.thetas];
        for piece in &pieces {
            if piece.theta >= self.shared.thetas || seen[piece.theta] {
                return Err(EngineError::Input(format!(
                    "el row group {id} trae el θ {} fuera de rango o repetido",
                    piece.theta
                )));
            }
            seen[piece.theta] = true;
            for event in &piece.events {
                let [r, c, e] = event.positions;
                if !(r <= c && c <= e && e <= rows) {
                    return Err(EngineError::Input(format!(
                        "el row group {id}, θ {}: posiciones {r}, {c}, {e} fuera de orden o de \
                         los {rows} ticks",
                        piece.theta
                    )));
                }
            }
        }
        let pieces: Vec<_> = pieces
            .into_iter()
            .filter(|p| !p.events.is_empty())
            .collect();
        if !lock(&self.shared.groups).push(id) {
            return Err(EngineError::Input(format!(
                "el row group {id} ya se empujó"
            )));
        }
        let items = pieces.len();
        if items == 0 {
            self.shared.finish_group(id);
            return Ok(());
        }
        let job = Arc::new(Job {
            id,
            ticks,
            pieces,
            next: AtomicUsize::new(0),
            remaining: AtomicUsize::new(items),
        });
        self.shared.busy.add(items as i64);
        lock(&self.shared.queue).jobs.push_back(job);
        self.shared.work.notify_all();
        Ok(())
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
        Ok(self.shared.stats.iter().map(|s| lock(s).stats).collect())
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
            if item >= job.pieces.len() {
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
                if front.next.load(Ordering::Relaxed) < front.pieces.len() {
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

    /// Tras un fallo se siguen procesando los row groups anteriores al del fallo (podrían
    /// tener uno de menor clave) y se descartan los posteriores.
    fn run(&self, job: &Job<B>, item: usize) {
        if self.failed.load(Ordering::Acquire)
            && job.id > self.first_failed_group.load(Ordering::Acquire)
        {
            return;
        }
        let piece = &job.pieces[item];
        let outcome = catch_unwind(AssertUnwindSafe(|| self.process(job, piece)));
        let failure = match outcome {
            Ok(Ok(())) => return,
            Ok(Err(failure)) => *failure,
            Err(panic) => Failure {
                key: (job.id, piece.theta, piece.first_event),
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

    /// Un ítem: agrega los eventos de un θ en el row group, cierra los que quedan
    /// completos y deja el resto en el almacén de parciales.
    fn process(&self, job: &Job<B>, piece: &ThetaPiece) -> Result<(), Box<Failure>> {
        let theta = piece.theta;
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
        let mut done: Vec<(u64, Row)> = Vec::with_capacity(piece.events.len());
        let mut crossing: Vec<(usize, EventAcc)> = Vec::new();
        let mut before_month = 0u64;
        let mut first_reference = None;
        let mut last = None;
        for (k, event) in piece.events.iter().enumerate() {
            let index = piece.first_event + k as u64;
            let acc = self
                .aggregate_event(&job.ticks, event)
                .map_err(|e| fail(job.id, index, EngineError::Ticks(e)))?;
            if index == 0 {
                first_reference = Some(event.reference);
            }
            if !event.ends_after {
                before_month += u64::try_from(event.ticks_before_month).unwrap_or(0);
                last = Some((index, event.extreme));
            }
            if event.starts_before || event.ends_after {
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
                    let event = &piece.events[k];
                    let index = piece.first_event + k as u64;
                    let partial = store.entry(index).or_insert_with(|| Partial {
                        acc: EventAcc::default(),
                        merged: 0,
                        first_group: None,
                        last_group: None,
                    });
                    partial.acc.merge(&acc);
                    partial.merged += 1;
                    if !event.starts_before {
                        partial.first_group = Some(job.id);
                    }
                    if !event.ends_after {
                        partial.last_group = Some(job.id);
                    }
                    let (Some(first), Some(last_group)) = (partial.first_group, partial.last_group)
                    else {
                        continue;
                    };
                    let expected = (last_group >= first).then(|| last_group - first + 1);
                    if expected.map_or(true, |n| partial.merged > n) {
                        return Err(fail(
                            job.id,
                            index,
                            EngineError::Internal(format!(
                                "el evento {index} del θ {theta} tiene {} parciales y sus row \
                                 groups van del {first} al {last_group}",
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
                let event = &piece.events[k];
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
            e @ OutputError::Overfull { .. } => fail(
                job.id,
                piece.first_event,
                EngineError::Internal(e.to_string()),
            ),
        })?;
        {
            let mut counters = lock(&self.stats[theta]);
            let stats = &mut counters.stats;
            stats.rows += done.len() as u64;
            stats.closed_ticks += closed_ticks;
            stats.before_month += before_month;
            if first_reference.is_some() {
                stats.first_reference = first_reference;
            }
            if let Some((index, extreme)) = last {
                // Solo el evento de mayor índice fija el último extremo.
                if counters.last_index.map_or(true, |known| index >= known) {
                    counters.last_index = Some(index);
                    counters.stats.last_extreme = Some(extreme);
                }
            }
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
    ) -> Result<EventAcc, TickError> {
        let [r, c, e] = event.positions;
        let (confirmation, instant) = aggregate(ticks, r, c, Some(event.confirm_time))?;
        let (overshoot, _) = aggregate(ticks, c, e, None)?;
        Ok(EventAcc {
            confirmation,
            overshoot,
            instant,
        })
    }
}
