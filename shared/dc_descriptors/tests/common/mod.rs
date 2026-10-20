//! Serie sintética, eventos de L2 consistentes y un oráculo ingenuo, compartidos por las
//! pruebas del motor. El oráculo suma tick a tick sobre la serie completa con `u128` y
//! redondea con otra fórmula que `divide_half_even`: no comparte código con el motor.
#![allow(dead_code)]

use dc_descriptors::{
    col, Config, Engine, EngineError, EventBatch, Region, Row, RowGroupInput, SealedRowGroup,
    ThetaStats, TickColumns, WIDTH,
};
use std::sync::{Arc, Weak};

/// Xorshift64*: sin dependencias y la misma serie en cada corrida para una semilla.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn percent(&mut self, p: u64) -> bool {
        self.below(100) < p
    }
}

/// Los ticks de un mes sintético: columnas por tick, en orden de la serie.
#[derive(Clone)]
pub struct Series {
    pub ids: Vec<i64>,
    pub time: Vec<i64>,
    pub price: Vec<i64>,
    pub qty: Vec<i64>,
    pub sold: Vec<bool>,
}

impl Series {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Un random walk con empates de tiempo, huecos de id y cantidades en cero.
    pub fn random(rng: &mut Rng, n: usize, gaps: bool) -> Self {
        let mut s = Series {
            ids: Vec::new(),
            time: Vec::new(),
            price: Vec::new(),
            qty: Vec::new(),
            sold: Vec::new(),
        };
        let (mut id, mut time, mut price) = (0i64, 1_000i64, 5_000_000_000_000i64);
        for _ in 0..n {
            id += 1 + if gaps && rng.percent(15) {
                1 + rng.below(3) as i64
            } else {
                0
            };
            time += if rng.percent(35) {
                0
            } else {
                1 + rng.below(4) as i64
            };
            price += if rng.percent(25) {
                0
            } else {
                rng.below(2001) as i64 - 1000
            };
            s.ids.push(id);
            s.time.push(time);
            s.price.push(price);
            s.qty.push(if rng.percent(8) {
                0
            } else {
                1 + rng.below(300_000_000) as i64
            });
            s.sold.push(rng.percent(50));
        }
        s
    }
}

/// Un evento de L2 con sus fronteras como cuentas de ticks: la confirmación son los ticks
/// `[r, c)` y el overshoot `[c, e)` (índices globales de la serie).
#[derive(Clone, Debug)]
pub struct Ev {
    pub r: usize,
    pub c: usize,
    pub e: usize,
    pub direction: i64,
    pub confirm_price: i64,
    pub extreme_price: i64,
}

impl Ev {
    pub fn reference_id(&self, s: &Series) -> i64 {
        if self.r == 0 {
            0
        } else {
            s.ids[self.r - 1]
        }
    }

    pub fn confirm_id(&self, s: &Series) -> i64 {
        s.ids[self.c - 1]
    }

    pub fn extreme_id(&self, s: &Series) -> i64 {
        s.ids[self.e - 1]
    }

    pub fn confirm_time(&self, s: &Series) -> i64 {
        s.time[self.c - 1]
    }
}

/// Eventos encadenados (`E_i = R_{i+1}`) que cumplen los invariantes de §7.2 por
/// construcción: el tick `C` es el extremo de su confirmación, y el extremo del
/// overshoot sale de los datos.
pub fn events(rng: &mut Rng, s: &mut Series, stop_percent: u64, overshoot_percent: u64) -> Vec<Ev> {
    let n = s.len();
    let mut out = Vec::new();
    let mut pos = 1 + rng.below(4) as usize;
    'events: loop {
        let r = pos;
        let (mut high, mut low) = (i64::MIN, i64::MAX);
        let mut k = r;
        let (c, up, down) = loop {
            if k >= n {
                break 'events;
            }
            high = high.max(s.price[k]);
            low = low.min(s.price[k]);
            k += 1;
            if rng.percent(stop_percent) {
                let (up, down) = (s.price[k - 1] == high, s.price[k - 1] == low);
                if up || down {
                    break (k, up, down);
                }
            }
        };
        let direction = if up && (!down || rng.percent(50)) {
            1
        } else {
            -1
        };
        let overshoot = if rng.percent(overshoot_percent) {
            (1 + rng.below(6) as usize).min(n - c)
        } else {
            0
        };
        let e = c + overshoot;
        // Una fase con ticks necesita cantidad: se le da al tick que cierra cada fase.
        if s.qty[r..c].iter().all(|&q| q == 0) {
            s.qty[c - 1] = 1;
        }
        if overshoot > 0 && s.qty[c..e].iter().all(|&q| q == 0) {
            s.qty[c] = 1;
        }
        let confirm_price = s.price[c - 1];
        let extreme_price = match (overshoot, direction) {
            (0, _) => confirm_price,
            (_, 1) => *s.price[c..e].iter().max().unwrap(),
            _ => *s.price[c..e].iter().min().unwrap(),
        };
        out.push(Ev {
            r,
            c,
            e,
            direction,
            confirm_price,
            extreme_price,
        });
        pos = e;
    }
    out
}

/// División entera con redondeo half-even por otra fórmula que la del motor:
/// `floor((2n + d) / 2d)`, y en un empate exacto, el par.
fn half_even(n: u128, d: u128) -> u128 {
    let nearest = (2 * n + d) / (2 * d);
    let tie = (2 * n + d) % (2 * d) == 0;
    if tie && nearest % 2 == 1 {
        nearest - 1
    } else {
        nearest
    }
}

/// La fila de un evento por la suma ingenua sobre la serie.
pub fn expected_row(s: &Series, ev: &Ev) -> Row {
    let up = ev.direction == 1;
    let phase = |from: usize, to: usize| {
        let (mut buy, mut sell, mut notional) = (0u128, 0u128, 0u128);
        let (mut high, mut low) = (i64::MIN, i64::MAX);
        for k in from..to {
            let q = s.qty[k] as u128;
            if s.sold[k] {
                sell += q;
            } else {
                buy += q;
            }
            notional += s.price[k] as u128 * q;
            high = high.max(s.price[k]);
            low = low.min(s.price[k]);
        }
        let ticks = (to - from) as i64;
        let vwap = if ticks > 0 {
            half_even(notional, buy + sell) as i64
        } else {
            0
        };
        let duration = if ticks > 0 {
            s.time[to - 1] - s.time[from]
        } else {
            0
        };
        (ticks, buy as i64, sell as i64, high, low, duration, vwap)
    };
    let (ct, cb, cs, chi, clo, cd, cv) = phase(ev.r, ev.c);
    let (ot, ob, os, ohi, olo, od, ov) = phase(ev.c, ev.e);
    let at = ev.confirm_time(s);
    let instant: Vec<i64> = (ev.r..ev.c)
        .filter(|&k| s.time[k] == at)
        .map(|k| s.price[k])
        .collect();
    let (ihi, ilo) = (
        *instant.iter().max().unwrap(),
        *instant.iter().min().unwrap(),
    );
    let mut row = [0i64; WIDTH];
    row[col::CONFIRM_AGG_TRADE_ID] = ev.confirm_id(s);
    row[col::C_TICKS] = ct;
    row[col::C_QUANTITY_BUY] = cb;
    row[col::C_QUANTITY_SELL] = cs;
    row[col::C_PRICE_MAX] = chi;
    row[col::C_PRICE_MIN] = clo;
    row[col::C_DURATION] = cd;
    row[col::C_VWAP] = cv;
    row[col::OS_TICKS] = ot;
    row[col::OS_QUANTITY_BUY] = ob;
    row[col::OS_QUANTITY_SELL] = os;
    if ot > 0 {
        row[col::OS_PRICE_MAX] = ohi;
        row[col::OS_PRICE_MIN] = olo;
        row[col::OS_DURATION] = od;
        row[col::OS_VWAP] = ov;
    }
    row[col::CONFIRM_INSTANT_TICKS] = instant.len() as i64;
    row[col::CONFIRM_INSTANT_PRICE_BEST] = if up { ihi } else { ilo };
    row[col::CONFIRM_INSTANT_PRICE_WORST] = if up { ilo } else { ihi };
    row
}

/// Un buffer compartido entre los row groups de una serie: cada row group lo lee con su
/// desfase, como los `Buffer` de Arrow de un archivo con varios row groups.
#[derive(Clone)]
pub struct Shared(pub Arc<Vec<u8>>);

impl AsRef<[u8]> for Shared {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Las columnas de ticks de toda la serie en el layout de Arrow.
pub struct Columns {
    ids: Shared,
    time: Shared,
    price: Shared,
    quantity: Shared,
    sold: Shared,
}

impl Columns {
    pub fn new(s: &Series) -> Self {
        let le = |values: &[i64], wide: bool| {
            let mut bytes = Vec::new();
            for &v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
                if wide {
                    bytes.extend_from_slice(&(i128::from(v) >> 64).to_le_bytes()[..8]);
                }
            }
            Shared(Arc::new(bytes))
        };
        let mut bits = vec![0u8; s.len().div_ceil(8)];
        for (i, &sold) in s.sold.iter().enumerate() {
            bits[i >> 3] |= u8::from(sold) << (i & 7);
        }
        Self {
            ids: le(&s.ids, false),
            time: le(&s.time, false),
            price: le(&s.price, true),
            quantity: le(&s.qty, true),
            sold: Shared(Arc::new(bits)),
        }
    }

    pub fn share_ids(&self) -> Shared {
        self.ids.clone()
    }

    pub fn share_time(&self) -> Shared {
        self.time.clone()
    }

    pub fn share_quantity(&self) -> Shared {
        self.quantity.clone()
    }

    pub fn share_sold(&self) -> Shared {
        self.sold.clone()
    }

    /// Los ticks `[lo, hi)` como un row group, sin copiar.
    pub fn row_group(&self, lo: usize, hi: usize) -> TickColumns<Shared> {
        let region = |buffer: &Shared| Region {
            buffer: buffer.clone(),
            offset: lo,
        };
        TickColumns::new(
            hi - lo,
            region(&self.ids),
            region(&self.time),
            region(&self.price),
            region(&self.quantity),
            region(&self.sold),
        )
        .expect("buffers completos")
    }
}

/// Los row groups de L1 de una serie: `[lo, hi)` por row group.
pub fn split(rng: &mut Rng, n: usize, max: u64) -> Vec<(usize, usize)> {
    let mut bounds = Vec::new();
    let mut lo = 0;
    while lo < n {
        let hi = (lo + 1 + rng.below(max) as usize).min(n);
        bounds.push((lo, hi));
        lo = hi;
    }
    bounds
}

/// Las filas de L2 de un θ como los row groups de `events.parquet` que serían (`sizes`),
/// en el layout de Arrow. Cada lote lleva un desfase distinto de 0 a 2 elementos, como un
/// `Array` cortado de Arrow.
pub fn batches(s: &Series, evs: &[Ev], sizes: &[usize]) -> Vec<EventBatch<Shared>> {
    watched_batches(s, evs, sizes).0
}

/// Como `batches`, y además una referencia débil al buffer de `reference_agg_trade_id` de
/// cada lote: muere cuando el motor suelta el lote.
pub fn watched_batches(
    s: &Series,
    evs: &[Ev],
    sizes: &[usize],
) -> (Vec<EventBatch<Shared>>, Vec<Weak<Vec<u8>>>) {
    let wide = |values: &mut Vec<u8>, v: i64| {
        values.extend_from_slice(&v.to_le_bytes());
        values.extend_from_slice(&(i128::from(v) >> 64).to_le_bytes()[..8]);
    };
    let mut out = Vec::new();
    let mut watched = Vec::new();
    let mut from = 0;
    for (b, &size) in sizes.iter().enumerate() {
        let offset = b % 3;
        let rows = &evs[from..from + size];
        from += size;
        let ints = |value: &dyn Fn(&Ev) -> i64| {
            let mut bytes = vec![0u8; offset * 8];
            for ev in rows {
                bytes.extend_from_slice(&value(ev).to_le_bytes());
            }
            Region {
                buffer: Shared(Arc::new(bytes)),
                offset,
            }
        };
        let prices = |value: &dyn Fn(&Ev) -> i64| {
            let mut bytes = vec![0u8; offset * 16];
            for ev in rows {
                wide(&mut bytes, value(ev));
            }
            Region {
                buffer: Shared(Arc::new(bytes)),
                offset,
            }
        };
        let mut direction = vec![0u8; offset];
        direction.extend(rows.iter().map(|ev| ev.direction as i8 as u8));
        let reference = ints(&|ev| ev.reference_id(s));
        watched.push(Arc::downgrade(&reference.buffer.0));
        out.push(
            EventBatch::new(
                size,
                reference,
                ints(&|ev| ev.confirm_id(s)),
                ints(&|ev| ev.confirm_time(s)),
                ints(&|ev| ev.extreme_id(s)),
                Region {
                    buffer: Shared(Arc::new(direction)),
                    offset,
                },
                prices(&|ev| ev.confirm_price),
                prices(&|ev| ev.extreme_price),
            )
            .expect("buffers completos"),
        );
    }
    (out, watched)
}

/// La `reference_agg_trade_id` del último evento de cada lote de `sizes`.
pub fn last_references(s: &Series, evs: &[Ev], sizes: &[usize]) -> Vec<i64> {
    let mut end = 0;
    sizes
        .iter()
        .map(|&size| {
            end += size;
            evs[end - 1].reference_id(s)
        })
        .collect()
}

/// Cuántos ticks de `(R, E]` del evento caen en los row groups `[lo, hi)` que `flagged`
/// marca como de un mes anterior: la cuenta que el motor llama `before_month`.
pub fn before_month(evs: &[Ev], bounds: &[(usize, usize)], flagged: &dyn Fn(usize) -> bool) -> i64 {
    let mut total = 0;
    for ev in evs {
        for (k, &(lo, hi)) in bounds.iter().enumerate() {
            if flagged(k) {
                total += ev.e.min(hi).saturating_sub(ev.r.max(lo)) as i64;
            }
        }
    }
    total
}

/// Reparte `rows` filas en row groups de salida de 1 a `max` filas.
pub fn skeleton(rng: &mut Rng, rows: usize, max: u64) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut left = rows;
    while left > 0 {
        let size = (1 + rng.below(max) as usize).min(left);
        sizes.push(size);
        left -= size;
    }
    sizes
}

/// Lo que el motor deja de una corrida: las filas de cada casilla y las cuentas.
pub struct Outcome {
    pub rows: Vec<Vec<Row>>,
    pub order: Vec<(usize, usize)>,
    pub stats: Vec<ThetaStats>,
}

pub fn decode(sealed: &SealedRowGroup) -> Vec<Row> {
    (0..sealed.rows)
        .map(|r| {
            let mut row = [0i64; WIDTH];
            for (c, spec) in dc_descriptors::COLUMNS.iter().enumerate() {
                let at = r * spec.bytes();
                row[c] = i64::from_le_bytes(sealed.columns[c][at..at + 8].try_into().unwrap());
                if spec.wide {
                    let high =
                        i64::from_le_bytes(sealed.columns[c][at + 8..at + 16].try_into().unwrap());
                    assert_eq!(high, row[c] >> 63, "extensión de signo de la columna {c}");
                }
            }
            row
        })
        .collect()
}

/// Entrega todas las fronteras de cada θ, empuja `order` (índices de row group) al motor y
/// recoge todo lo que sella.
pub fn run(
    engine: &Engine<Shared>,
    columns: &Columns,
    bounds: &[(usize, usize)],
    batches: Vec<Vec<EventBatch<Shared>>>,
    flagged: &dyn Fn(usize) -> bool,
    order: &[usize],
    sizes: &[Vec<usize>],
) -> Result<Outcome, EngineError> {
    for (theta, batches) in batches.into_iter().enumerate() {
        for batch in batches {
            engine.push_boundaries(theta, batch)?;
        }
    }
    run_with(
        engine,
        columns,
        bounds,
        flagged,
        order,
        sizes,
        &mut |_, _| Ok(()),
    )
}

/// Como `run`, pero entrega las fronteras de a lotes justo antes del row group que las
/// necesita, y no más: antes del row group `k`, los lotes de cada θ hasta que el último
/// entregado empieza después del último id del row group. `lots[t]` son los lotes del θ
/// `t` con la referencia de su último evento; `ids` son los ids de la serie.
#[allow(clippy::too_many_arguments)]
pub fn run_interleaved(
    engine: &Engine<Shared>,
    columns: &Columns,
    ids: &[i64],
    bounds: &[(usize, usize)],
    lots: Vec<Vec<(EventBatch<Shared>, i64)>>,
    flagged: &dyn Fn(usize) -> bool,
    order: &[usize],
    sizes: &[Vec<usize>],
) -> Result<Outcome, EngineError> {
    let mut pending: Vec<_> = lots.into_iter().map(|l| l.into_iter().peekable()).collect();
    let mut last: Vec<Option<i64>> = vec![None; pending.len()];
    let mut feed = |engine: &Engine<Shared>, k: usize| -> Result<(), EngineError> {
        let high = ids[bounds[k].1 - 1];
        for (theta, lots) in pending.iter_mut().enumerate() {
            while last[theta].map_or(true, |r| r <= high) {
                let Some((batch, reference)) = lots.next() else {
                    break;
                };
                engine.push_boundaries(theta, batch)?;
                last[theta] = Some(reference);
            }
        }
        Ok(())
    };
    run_with(engine, columns, bounds, flagged, order, sizes, &mut feed)
}

/// Lo que se hace antes de empujar el row group `k`: entregar sus fronteras, por ejemplo.
pub type Feed<'a> = dyn FnMut(&Engine<Shared>, usize) -> Result<(), EngineError> + 'a;

/// Empuja `order` al motor, con `feed(engine, k)` antes de cada row group `k`, y recoge
/// todo lo que sella.
pub fn run_with(
    engine: &Engine<Shared>,
    columns: &Columns,
    bounds: &[(usize, usize)],
    flagged: &dyn Fn(usize) -> bool,
    order: &[usize],
    sizes: &[Vec<usize>],
    feed: &mut Feed<'_>,
) -> Result<Outcome, EngineError> {
    let mut rows: Vec<Vec<Row>> = sizes.iter().map(|_| Vec::new()).collect();
    let mut seen_order = Vec::new();
    let mut take = |engine: &Engine<Shared>| -> Result<(), EngineError> {
        while let Some(sealed) = engine.pop_sealed()? {
            assert_eq!(
                rows[sealed.theta].len(),
                sealed.index_start(sizes),
                "orden de entrega"
            );
            seen_order.push((sealed.theta, sealed.index));
            rows[sealed.theta].extend(decode(&sealed));
        }
        Ok(())
    };
    for &k in order {
        let (lo, hi) = bounds[k];
        feed(engine, k)?;
        engine.push(RowGroupInput {
            id: k as u64,
            ticks: columns.row_group(lo, hi),
            before_month: flagged(k),
        })?;
        take(engine)?;
    }
    let stats = engine.finish()?;
    take(engine)?;
    Ok(Outcome {
        rows,
        order: seen_order,
        stats,
    })
}

trait IndexStart {
    fn index_start(&self, sizes: &[Vec<usize>]) -> usize;
}

impl IndexStart for SealedRowGroup {
    /// Filas de los row groups de salida anteriores del mismo θ.
    fn index_start(&self, sizes: &[Vec<usize>]) -> usize {
        sizes[self.theta][..self.index].iter().sum()
    }
}

pub fn engine(threads: usize, sizes: &[Vec<usize>]) -> Engine<Shared> {
    Engine::new(Config {
        thetas: (0..sizes.len() as i64).map(|t| 100_000 * (t + 1)).collect(),
        threads,
        row_groups: sizes.to_vec(),
    })
    .expect("configuración válida")
}
