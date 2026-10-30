//! El motor contra un oráculo ingenuo, con 1 y 4 hilos, en varios órdenes de `push`, con
//! eventos que cruzan row groups de todas las formas, y los errores y bordes del contrato.

mod common;

use common::*;
use dc_descriptors::{
    col, BoundaryKind, Config, Engine, EngineError, EventBatch, EventErrorKind, PhaseName, Region,
    Row, RowGroupInput, TickColumns, TickError,
};
use std::sync::Weak;

/// Las fronteras de cada θ en lotes: cada lote con la referencia de su último evento, sus
/// tamaños y una referencia débil a un buffer de cada lote.
struct Lots {
    lots: Vec<Vec<(EventBatch<Shared>, i64)>>,
    sizes: Vec<Vec<usize>>,
    watched: Vec<Vec<Weak<Vec<u8>>>>,
}

struct Case {
    series: Series,
    columns: Columns,
    bounds: Vec<(usize, usize)>,
    events: Vec<Vec<Ev>>,
    sizes: Vec<Vec<usize>>,
}

impl Case {
    fn new(seed: u64, ticks: usize, thetas: usize, rg_max: u64, gaps: bool) -> Self {
        let mut rng = Rng::new(seed);
        let mut series = Series::random(&mut rng, ticks, gaps);
        let events: Vec<_> = (0..thetas)
            .map(|t| {
                events(
                    &mut rng,
                    &mut series,
                    20 + 25 * t as u64,
                    30 + 20 * t as u64,
                )
            })
            .collect();
        let sizes = events
            .iter()
            .map(|evs| skeleton(&mut rng, evs.len(), 7))
            .collect();
        let bounds = split(&mut rng, ticks, rg_max);
        Self {
            columns: Columns::new(&series),
            series,
            bounds,
            events,
            sizes,
        }
    }

    /// Los row groups de L2 de cada θ, en el layout de Arrow.
    fn batches(&self) -> Vec<Vec<EventBatch<Shared>>> {
        (0..self.events.len())
            .map(|t| batches(&self.series, &self.events[t], &self.sizes[t]))
            .collect()
    }

    /// Los row groups de L1 de un mes anterior al de los eventos: uno de cada tres.
    fn flagged(k: usize) -> bool {
        k % 3 == 1
    }

    /// El evento toca el row group `k`: empieza en él o antes y no cerró en uno anterior.
    fn touches(&self, theta: usize, k: usize) -> bool {
        let (lo, hi) = self.bounds[k];
        self.events[theta].iter().any(|e| e.r <= hi && e.e > lo)
    }

    fn run(&self, threads: usize, order: &[usize]) -> Result<Outcome, EngineError> {
        let engine = engine(threads, &self.sizes);
        run(
            &engine,
            &self.columns,
            &self.bounds,
            self.batches(),
            &Self::flagged,
            order,
            &self.sizes,
        )
    }

    /// Los eventos de cada θ en lotes de 1 a `max` eventos, no en los row groups de L2.
    fn lots(&self, rng: &mut Rng, max: u64) -> Lots {
        let mut out = Lots {
            lots: Vec::new(),
            sizes: Vec::new(),
            watched: Vec::new(),
        };
        for evs in &self.events {
            let sizes = skeleton(rng, evs.len(), max);
            let (batches, watched) = watched_batches(&self.series, evs, &sizes);
            let references = last_references(&self.series, evs, &sizes);
            out.lots.push(batches.into_iter().zip(references).collect());
            out.sizes.push(sizes);
            out.watched.push(watched);
        }
        out
    }

    fn expected(&self, theta: usize) -> Vec<Row> {
        self.events[theta]
            .iter()
            .map(|ev| expected_row(&self.series, ev))
            .collect()
    }

    fn check(&self, outcome: &Outcome) {
        for (t, evs) in self.events.iter().enumerate() {
            assert_eq!(outcome.rows[t], self.expected(t), "filas del θ {t}");
            let stats = outcome.stats[t];
            assert_eq!(stats.rows, evs.len() as u64);
            let ticks: i64 = outcome.rows[t]
                .iter()
                .map(|r| r[col::C_TICKS] + r[col::OS_TICKS])
                .sum();
            assert_eq!(stats.closed_ticks as i64, ticks);
            let before = before_month(evs, &self.bounds, &Self::flagged);
            assert_eq!(stats.before_month as i64, before);
            assert_eq!(
                stats.first_reference,
                evs.first().map(|e| e.reference_id(&self.series))
            );
            assert_eq!(
                stats.last_extreme,
                evs.last().map(|e| e.extreme_id(&self.series))
            );
        }
        // Cada θ sale en orden de row group de salida.
        for t in 0..self.sizes.len() {
            let indices: Vec<_> = outcome
                .order
                .iter()
                .filter(|(x, _)| *x == t)
                .map(|(_, j)| *j)
                .collect();
            assert_eq!(indices, (0..self.sizes[t].len()).collect::<Vec<_>>());
        }
    }
}

fn shuffled(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut order: Vec<_> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.below(i as u64 + 1) as usize);
    }
    order
}

#[test]
fn matches_the_naive_oracle_with_one_and_four_threads() {
    for seed in 1..=24 {
        let case = Case::new(seed, 300, 3, 25, seed % 2 == 0);
        assert!(case.events.iter().all(|e| !e.is_empty()), "seed {seed}");
        let forward: Vec<_> = (0..case.bounds.len()).collect();
        for threads in [1, 4] {
            case.check(&case.run(threads, &forward).expect("corrida limpia"));
        }
    }
}

#[test]
fn the_push_order_does_not_change_the_result() {
    for seed in 100..112 {
        let case = Case::new(seed, 260, 3, 12, true);
        let forward: Vec<_> = (0..case.bounds.len()).collect();
        let baseline = case.run(1, &forward).unwrap();
        let mut rng = Rng::new(seed);
        // Dos row groups vecinos intercambiados, el orden inverso y uno al azar.
        let mut swapped = forward.clone();
        let at = rng.below(swapped.len() as u64 - 1) as usize;
        swapped.swap(at, at + 1);
        let mut reversed = forward.clone();
        reversed.reverse();
        for order in [swapped, reversed, shuffled(&mut rng, forward.len())] {
            for threads in [1, 4] {
                let outcome = case.run(threads, &order).unwrap();
                assert_eq!(outcome.rows, baseline.rows, "seed {seed}, orden {order:?}");
                assert_eq!(outcome.stats, baseline.stats);
                case.check(&outcome);
            }
        }
    }
}

#[test]
fn boundaries_in_small_lots_interleaved_with_the_pushes_give_the_same_table() {
    // Lotes de 1 a 3 eventos entregados justo antes del row group que los necesita, y no
    // más: mismas filas y cuentas que con todas las fronteras de una vez.
    for seed in 400..412 {
        let case = Case::new(seed, 260, 3, 12, seed % 2 == 0);
        let forward: Vec<_> = (0..case.bounds.len()).collect();
        let baseline = case.run(1, &forward).unwrap();
        let mut rng = Rng::new(seed);
        let mut swapped = forward.clone();
        let at = rng.below(swapped.len() as u64 - 1) as usize;
        swapped.swap(at, at + 1);
        for order in [forward.clone(), swapped] {
            for threads in [1, 4] {
                let lots = case.lots(&mut rng, 3);
                let engine = engine(threads, &case.sizes);
                let outcome = run_interleaved(
                    &engine,
                    &case.columns,
                    &case.series.ids,
                    &case.bounds,
                    lots.lots,
                    &Case::flagged,
                    &order,
                    &case.sizes,
                )
                .unwrap();
                assert_eq!(outcome.rows, baseline.rows, "seed {seed}, orden {order:?}");
                assert_eq!(outcome.stats, baseline.stats);
                case.check(&outcome);
                for (t, evs) in case.events.iter().enumerate() {
                    assert_eq!(engine.needed(t).unwrap(), evs.len() as u64);
                }
            }
        }
    }
}

#[test]
fn a_lot_of_boundaries_is_released_as_soon_as_its_events_closed() {
    // Tras cada row group, con el motor quieto: un lote entregado sigue vivo si y solo si
    // tiene un evento desde `needed` en adelante. Los anteriores ya no los ancla nadie.
    let case = Case::new(11, 400, 2, 10, false);
    let mut rng = Rng::new(11);
    let Lots {
        lots,
        sizes,
        watched,
    } = case.lots(&mut rng, 2);
    let ends: Vec<Vec<u64>> = sizes
        .iter()
        .map(|s| {
            s.iter()
                .scan(0u64, |end, &n| {
                    *end += n as u64;
                    Some(*end)
                })
                .collect()
        })
        .collect();
    let engine = common::engine(4, &case.sizes);
    let mut pending: Vec<_> = lots.into_iter().map(|l| l.into_iter().peekable()).collect();
    let mut delivered = vec![0usize; pending.len()];
    let mut last: Vec<Option<i64>> = vec![None; pending.len()];
    let (mut released, mut held, mut early) = (0, 0, 0);
    let make = |k: usize| {
        let (lo, hi) = case.bounds[k];
        RowGroupInput {
            id: k as u64,
            ticks: case.columns.row_group(lo, hi),
            before_month: false,
        }
    };
    for k in 0..case.bounds.len() {
        let high = case.series.ids[case.bounds[k].1 - 1];
        // Si a un θ le faltan fronteras para este row group, `push` falla y no espera,
        // también después de los primeros `push`.
        let short = pending
            .iter_mut()
            .enumerate()
            .any(|(t, lots)| last[t].map_or(true, |r| r <= high) && lots.peek().is_some());
        if short {
            match engine.push(make(k)) {
                Err(EngineError::Input(m)) => assert!(m.contains("antes de las fronteras")),
                other => panic!("se esperaba un error de entrada y vino {other:?}"),
            }
            early += 1;
        }
        for (t, lots) in pending.iter_mut().enumerate() {
            while last[t].map_or(true, |r| r <= high) {
                let Some((batch, reference)) = lots.next() else {
                    break;
                };
                engine.push_boundaries(t, batch).unwrap();
                last[t] = Some(reference);
                delivered[t] += 1;
            }
        }
        engine.push(make(k)).unwrap();
        while engine.inflight() > 0 {
            std::thread::yield_now();
        }
        while engine.pop_sealed().unwrap().is_some() {}
        for t in 0..pending.len() {
            let needed = engine.needed(t).unwrap();
            for j in 0..delivered[t] {
                let alive = watched[t][j].upgrade().is_some();
                assert_eq!(alive, ends[t][j] > needed, "θ {t}, lote {j}, row group {k}");
                if alive {
                    held += 1;
                } else {
                    released += 1;
                }
            }
        }
    }
    engine.finish().unwrap();
    assert!(watched.iter().flatten().all(|w| w.upgrade().is_none()));
    assert!(
        released > 0 && held > 0 && early > 0,
        "{released} {held} {early}"
    );
}

#[test]
fn events_crossing_three_row_groups_and_empty_middle_pieces() {
    // Row groups de 1 a 3 ticks y eventos largos: cruzan muchos row groups, y en el
    // medio hay tramos de un tick o sin ticks de la fase (fases del evento contiguas).
    let mut crossed = 0;
    let mut without_events = 0;
    for seed in 200..230 {
        let case = Case::new(seed, 160, 2, 3, false);
        let forward: Vec<_> = (0..case.bounds.len()).collect();
        let outcome = case.run(4, &forward).unwrap();
        case.check(&outcome);
        for evs in &case.events {
            crossed += evs
                .iter()
                .filter(|e| {
                    let group = |pos: usize| {
                        case.bounds
                            .iter()
                            .position(|&(lo, hi)| lo < pos.max(1) && pos.max(1) <= hi)
                            .unwrap()
                    };
                    group(e.e) - group(e.r) >= 2
                })
                .count();
        }
        without_events += (0..case.bounds.len())
            .filter(|&k| (0..case.events.len()).any(|t| !case.touches(t, k)))
            .count();
    }
    assert!(crossed > 20, "{crossed} eventos cruzan tres row groups");
    assert!(without_events > 0, "algún row group sin eventos de un θ");
}

/// Un solo evento de ticks escritos a mano, en el row group `[0, n)`.
struct Tiny {
    time: Vec<i64>,
    price: Vec<i64>,
    qty: Vec<i64>,
    sold: Vec<bool>,
}

impl Tiny {
    fn series(&self) -> Series {
        Series {
            ids: (1..=self.time.len() as i64).collect(),
            time: self.time.clone(),
            price: self.price.clone(),
            qty: self.qty.clone(),
            sold: self.sold.clone(),
        }
    }
}

fn single(
    series: &Series,
    ev: &Ev,
    cuts: &[usize],
    threads: usize,
) -> Result<Outcome, EngineError> {
    let mut bounds = Vec::new();
    let mut lo = 0;
    for &cut in cuts.iter().chain([&series.len()]) {
        bounds.push((lo, cut));
        lo = cut;
    }
    let events = vec![ev.clone()];
    let sizes = vec![vec![1]];
    let engine = engine(threads, &sizes);
    let columns = Columns::new(series);
    let order: Vec<_> = (0..bounds.len()).collect();
    run(
        &engine,
        &columns,
        &bounds,
        vec![batches(series, &events, &sizes[0])],
        &|_| false,
        &order,
        &sizes,
    )
}

const MAX_DECIMAL: i64 = 999_999_999_999_999_999;

#[test]
fn operands_at_the_edges_of_decimal_18_8() {
    // Un tick con price y quantity en el máximo de DECIMAL(18,8): el producto (~10^36)
    // no cabe en 64 bits y el VWAP vuelve a ser el precio.
    let tiny = Tiny {
        time: vec![5, 6],
        price: vec![MAX_DECIMAL, 7],
        qty: vec![0, MAX_DECIMAL],
        sold: vec![true, false],
    };
    let series = tiny.series();
    let ev = Ev {
        r: 1,
        c: 2,
        e: 2,
        direction: -1,
        confirm_price: 7,
        extreme_price: 7,
    };
    // R=1: confirmación = tick 1 (precio 7, cantidad máxima); sin overshoot.
    let row = single(&series, &ev, &[], 1).unwrap().rows[0][0];
    assert_eq!(row[col::C_VWAP], 7);
    assert_eq!(row[col::C_QUANTITY_BUY], MAX_DECIMAL);
    assert_eq!(row[col::C_QUANTITY_SELL], 0);
    assert_eq!(row[col::OS_TICKS], 0);
    assert_eq!(row[col::OS_VWAP], 0);

    // Precio máximo con cantidad máxima: Σ price·quantity = (10^18 - 1)^2.
    let tiny = Tiny {
        time: vec![5, 6],
        price: vec![1, MAX_DECIMAL],
        qty: vec![1, MAX_DECIMAL],
        sold: vec![false, true],
    };
    let series = tiny.series();
    let ev = Ev {
        r: 1,
        c: 2,
        e: 2,
        direction: 1,
        confirm_price: MAX_DECIMAL,
        extreme_price: MAX_DECIMAL,
    };
    let row = single(&series, &ev, &[], 1).unwrap().rows[0][0];
    assert_eq!(row[col::C_VWAP], MAX_DECIMAL);
    assert_eq!(row[col::C_QUANTITY_SELL], MAX_DECIMAL);
    assert_eq!(row[col::C_PRICE_MAX], MAX_DECIMAL);
}

#[test]
fn vwap_rounds_half_even_across_row_groups() {
    // Σ pq / Σ q = (10·1 + 11·1) / 2 = 10,5 → 10 (par); (11·1 + 12·1)/2 = 11,5 → 12.
    for (prices, wanted) in [([10, 11], 10), ([11, 12], 12)] {
        let tiny = Tiny {
            time: vec![1, 2, 3, 3],
            price: vec![1, prices[0], prices[1], 99],
            qty: vec![1, 1, 1, 1],
            sold: vec![false; 4],
        };
        let series = tiny.series();
        // Confirmación = ticks 1 y 2 (índices); el tick 2 es el instante y el máximo.
        let ev = Ev {
            r: 1,
            c: 3,
            e: 3,
            direction: if prices[1] > prices[0] { 1 } else { -1 },
            confirm_price: prices[1],
            extreme_price: prices[1],
        };
        for cuts in [vec![], vec![2], vec![2, 3], vec![1, 2, 3]] {
            let row = single(&series, &ev, &cuts, 4).unwrap().rows[0][0];
            assert_eq!(row[col::C_VWAP], wanted, "cortes {cuts:?}");
        }
    }
}

fn event_error(result: Result<Outcome, EngineError>) -> dc_descriptors::EventError {
    match result {
        Err(EngineError::Event(error)) => error,
        Err(other) => panic!("se esperaba un error de evento y vino {other}"),
        Ok(_) => panic!("se esperaba un error de evento"),
    }
}

#[test]
fn a_phase_without_quantity_or_too_large_is_an_error_not_a_value() {
    let base = |qty: Vec<i64>| Tiny {
        time: vec![1, 2, 3, 4],
        price: vec![5, 5, 5, 5],
        qty,
        sold: vec![false; 4],
    };
    let ev = |overshoot: usize| Ev {
        r: 1,
        c: 3,
        e: 3 + overshoot,
        direction: 1,
        confirm_price: 5,
        extreme_price: 5,
    };
    // Confirmación sin cantidad.
    let series = base(vec![9, 0, 0, 0]).series();
    let error = event_error(single(&series, &ev(0), &[2], 1));
    assert_eq!(
        error.kind,
        EventErrorKind::NoQuantity {
            phase: PhaseName::Confirmation
        }
    );
    assert_eq!(
        (error.theta, error.event, error.confirm_agg_trade_id),
        (0, 0, 3)
    );
    // Overshoot sin cantidad.
    let series = base(vec![9, 1, 1, 0]).series();
    let error = event_error(single(&series, &ev(1), &[], 1));
    assert_eq!(
        error.kind,
        EventErrorKind::NoQuantity {
            phase: PhaseName::Overshoot
        }
    );
    // Dos ticks de 5·10^17 + 5·10^17 = 10^18 en un lado no caben en DECIMAL(18,8)...
    let half = 500_000_000_000_000_000;
    let series = base(vec![9, half, half, 1]).series();
    let error = event_error(single(&series, &ev(0), &[2], 4));
    assert_eq!(
        error.kind,
        EventErrorKind::QuantityTooLarge {
            phase: PhaseName::Confirmation
        }
    );
    // ... pero repartidos en los dos lados sí: 5·10^17 comprando y 5·10^17 vendiendo.
    let mut tiny = base(vec![9, half, half, 1]);
    tiny.sold = vec![false, false, true, false];
    let row = single(&tiny.series(), &ev(0), &[2], 4).unwrap().rows[0][0];
    assert_eq!(row[col::C_QUANTITY_BUY], half);
    assert_eq!(row[col::C_QUANTITY_SELL], half);
}

#[test]
fn the_seven_invariants_fail_with_the_field_and_the_values_of_the_reference() {
    let tiny = Tiny {
        time: vec![1, 2, 3, 3, 4],
        price: vec![10, 12, 15, 14, 13],
        qty: vec![1; 5],
        sold: vec![false; 5],
    };
    let series = tiny.series();
    let good = Ev {
        r: 1,
        c: 4,
        e: 5,
        direction: 1,
        confirm_price: 14,
        extreme_price: 13,
    };
    // La fila sana pasa... salvo que el máximo de la confirmación (15) no está en el
    // instante (14 y 15 son del instante: sí está). Es consistente.
    let row = single(&series, &good, &[3], 1).unwrap().rows[0][0];
    assert_eq!(row[col::CONFIRM_INSTANT_TICKS], 2);

    let invariant = |ev: Ev| match event_error(single(&series, &ev, &[], 1)).kind {
        EventErrorKind::Invariant { field, got, wanted } => (field, got, wanted),
        other => panic!("se esperaba un invariante y vino {other:?}"),
    };
    // Dirección inválida.
    let error = event_error(single(
        &series,
        &Ev {
            direction: 0,
            ..good.clone()
        },
        &[],
        1,
    ));
    assert_eq!(error.kind, EventErrorKind::Direction { got: 0 });
    // 2. confirm_price fuera de [peor, mejor] = [14, 15].
    assert_eq!(
        invariant(Ev {
            confirm_price: 16,
            ..good.clone()
        }),
        ("confirm_price", 16, "entre 14 y 15".to_owned())
    );
    // 4. extreme_price distinto del extremo del overshoot.
    assert_eq!(
        invariant(Ev {
            extreme_price: 12,
            ..good.clone()
        }),
        ("os_price_max", 13, "12".to_owned())
    );
    // 5/4 sin overshoot: extreme_price debe ser confirm_price.
    assert_eq!(
        invariant(Ev {
            e: 4,
            extreme_price: 13,
            ..good.clone()
        }),
        ("extreme_price", 13, "14".to_owned())
    );
    // 3 en un downturn: el mínimo de la confirmación (12) no es el mejor del instante.
    assert_eq!(
        invariant(Ev {
            direction: -1,
            confirm_price: 14,
            ..good.clone()
        }),
        ("c_price_min", 12, "14".to_owned())
    );
}

/// El row group (índice) que contiene el tick `count - 1`.
fn group_of(case: &Case, count: usize) -> usize {
    case.bounds
        .iter()
        .position(|&(lo, hi)| lo < count && count <= hi)
        .unwrap()
}

#[test]
fn the_reported_error_is_the_one_of_least_key_whatever_the_order() {
    for seed in 300..312 {
        let mut case = Case::new(seed, 240, 3, 15, false);
        // Dos eventos con dirección inválida, en θ y row groups distintos.
        let mut rng = Rng::new(seed);
        let mut broken = Vec::new();
        for t in [2usize, 0] {
            let i = rng.below(case.events[t].len() as u64) as usize;
            case.events[t][i].direction = 0;
            broken.push((group_of(&case, case.events[t][i].e), t, i));
        }
        let (_, theta, event) = *broken.iter().min().unwrap();
        let forward: Vec<_> = (0..case.bounds.len()).collect();
        let mut reversed = forward.clone();
        reversed.reverse();
        for order in [forward, reversed] {
            for threads in [1, 4] {
                let error = event_error(case.run(threads, &order));
                assert_eq!(
                    (error.theta, error.event as usize, &error.kind),
                    (theta, event, &EventErrorKind::Direction { got: 0 }),
                    "seed {seed}, hilos {threads}"
                );
            }
        }
    }
}

#[test]
fn the_contract_of_push_and_finish() {
    let case = Case::new(7, 120, 2, 20, false);
    let order: Vec<_> = (0..case.bounds.len()).collect();
    let make = |k: usize| {
        let (lo, hi) = case.bounds[k];
        RowGroupInput {
            id: k as u64,
            ticks: case.columns.row_group(lo, hi),
            before_month: false,
        }
    };
    let input = |error: EngineError| match error {
        EngineError::Input(message) => message,
        other => panic!("se esperaba un error de entrada y vino {other}"),
    };

    // Las fronteras que un row group toca van antes que él, en cada θ: si faltan, `push`
    // falla en vez de esperar.
    let engine = common::engine(1, &case.sizes);
    assert!(input(engine.push(make(0)).unwrap_err()).contains("antes de las fronteras del θ 0"));
    let mut batches = case.batches();
    for batch in batches.remove(0) {
        engine.push_boundaries(0, batch).unwrap();
    }
    assert!(input(engine.push(make(0)).unwrap_err()).contains("antes de las fronteras del θ 1"));
    // Un θ que no existe.
    let one = || common::batches(&case.series, &case.events[1][..1], &[1]).remove(0);
    assert!(input(engine.push_boundaries(9, one()).unwrap_err()).contains("no existe"));
    for batch in batches.remove(0) {
        engine.push_boundaries(1, batch).unwrap();
    }

    // Un row group no se empuja dos veces, y un lote que pasa del esqueleto es un error.
    engine.push(make(0)).unwrap();
    assert!(input(engine.push(make(0)).unwrap_err()).contains("ya se empujó"));
    assert!(matches!(
        engine.push_boundaries(1, one()),
        Err(EngineError::Skeleton { theta: 1, .. })
    ));

    // Sin los demás row groups, al terminar faltan filas.
    match engine.finish() {
        Err(EngineError::Incomplete {
            missing, expected, ..
        }) => assert!(0 < missing && missing <= expected),
        other => panic!("se esperaba Incomplete y vino {other:?}"),
    }

    // Un ordinal que nunca llega deja esperando a los siguientes: `finish` lo dice en vez
    // de esperar para siempre.
    let engine = common::engine(1, &case.sizes);
    for (t, b) in case.batches().into_iter().enumerate() {
        for batch in b {
            engine.push_boundaries(t, batch).unwrap();
        }
    }
    engine.push(make(0)).unwrap();
    engine.push(make(2)).unwrap();
    assert!(input(engine.finish().unwrap_err()).contains("falta el row group 1"));

    // Un esqueleto más corto que los eventos.
    let mut short = case.sizes.clone();
    short[0].pop();
    let engine = common::engine(4, &short);
    let outcome = run(
        &engine,
        &case.columns,
        &case.bounds,
        case.batches(),
        &Case::flagged,
        &order,
        &short,
    );
    assert!(
        matches!(outcome, Err(EngineError::Skeleton { theta: 0, .. })),
        "{:?}",
        outcome.err()
    );

    // Sin ningún θ, o con esqueletos de más o de menos, no hay motor.
    for config in [
        Config {
            thetas: vec![],
            threads: 1,
            row_groups: vec![],
            family: Default::default(),
            silence_us: 0,
        },
        Config {
            thetas: vec![1, 2],
            threads: 1,
            row_groups: vec![vec![1]],
            family: Default::default(),
            silence_us: 0,
        },
    ] {
        assert!(matches!(
            Engine::<Shared>::new(config),
            Err(EngineError::Input(_))
        ));
    }
}

#[test]
fn the_events_of_l2_must_be_ordered_and_have_prices_that_fit() {
    let mut rng = Rng::new(5);
    let mut series = Series::random(&mut rng, 80, false);
    let evs = events(&mut rng, &mut series, 30, 40);
    assert!(evs.len() >= 3);
    let sizes = vec![vec![evs.len()]];
    let reason = |evs: &[Ev], series: &Series| {
        let engine = common::engine(1, &sizes);
        match engine.push_boundaries(0, batches(series, evs, &[evs.len()]).remove(0)) {
            Err(EngineError::Input(message)) => message,
            other => panic!("se esperaba un error de entrada y vino {other:?}"),
        }
    };
    // R < C ≤ E.
    let mut broken = evs.clone();
    broken[1].c = broken[1].r;
    assert!(reason(&broken, &series).contains("R < C"));
    // E_i ≤ R_{i+1}: dos eventos que se pisan.
    let mut broken = evs.clone();
    broken[1].r = broken[0].r;
    assert!(reason(&broken, &series).contains("antes del extremo"));
    // También entre lotes: el segundo lote no puede empezar antes del extremo del primero,
    // y un lote rechazado no cambia nada.
    let engine = common::engine(1, &sizes);
    let mut lots = batches(&series, &evs, &[1, 1, evs.len() - 2]);
    let (second, third) = (lots.remove(1), lots.remove(1));
    engine.push_boundaries(0, second).unwrap();
    match engine.push_boundaries(0, lots.remove(0)) {
        Err(EngineError::Input(message)) => {
            assert!(message.contains("evento 1") && message.contains("antes del extremo"))
        }
        other => panic!("se esperaba un error de entrada y vino {other:?}"),
    }
    engine.push_boundaries(0, third).unwrap();
    // Un precio que no cabe en i64 (la palabra alta distinta de la extensión de signo).
    let engine = common::engine(1, &sizes);
    let mut ok = batches(&series, &evs, &[evs.len()]);
    let one = ok.remove(0);
    drop(one);
    let wide = |v: i64, high: i64| {
        let mut bytes = v.to_le_bytes().to_vec();
        bytes.extend_from_slice(&high.to_le_bytes());
        bytes
    };
    let region = |bytes: Vec<u8>| Region {
        buffer: Shared(std::sync::Arc::new(bytes)),
        offset: 0,
    };
    let ints = |v: i64| region(v.to_le_bytes().to_vec());
    let batch = EventBatch::new(
        1,
        ints(1),
        ints(2),
        ints(2),
        ints(3),
        region(vec![1]),
        region(wide(5, 1)),
        region(wide(5, 0)),
        region(wide(4, 0)),
    )
    .unwrap();
    match engine.push_boundaries(0, batch) {
        Err(EngineError::Input(message)) => {
            assert!(message.contains("no cabe en i64"), "{message}")
        }
        other => panic!("se esperaba un error de entrada y vino {other:?}"),
    }
}

/// Un evento escrito a mano sobre `series`, con las fronteras que dice `declared` (la
/// misma serie con algún id o tiempo cambiado) y los row groups cortados en `cuts`.
fn declared(
    series: &Series,
    declared: &Series,
    ev: &Ev,
    cuts: &[usize],
    threads: usize,
    reversed: bool,
) -> Result<Outcome, EngineError> {
    let mut bounds = Vec::new();
    let mut lo = 0;
    for &cut in cuts.iter().chain([&series.len()]) {
        bounds.push((lo, cut));
        lo = cut;
    }
    let sizes = vec![vec![1]];
    let engine = engine(threads, &sizes);
    let columns = Columns::new(series);
    let mut order: Vec<_> = (0..bounds.len()).collect();
    if reversed {
        order.reverse();
    }
    run(
        &engine,
        &columns,
        &bounds,
        vec![batches(declared, std::slice::from_ref(ev), &sizes[0])],
        &|_| false,
        &order,
        &sizes,
    )
}

fn boundary(result: Result<Outcome, EngineError>) -> dc_descriptors::BoundaryError {
    match result {
        Err(EngineError::Boundary(error)) => error,
        Err(other) => panic!("se esperaba un error de frontera y vino {other}"),
        Ok(_) => panic!("se esperaba un error de frontera"),
    }
}

/// Seis ticks con hueco en los ids 3 y 6: 1, 2, 4, 5, 7, 8.
fn gapped() -> Series {
    Series {
        ids: vec![1, 2, 4, 5, 7, 8],
        time: vec![10, 11, 12, 13, 14, 15],
        price: vec![50, 51, 52, 53, 54, 55],
        qty: vec![1; 6],
        sold: vec![false; 6],
    }
}

fn gapped_event() -> Ev {
    // R = tick 0 (id 1), C = tick 3 (id 5), E = tick 3 (id 5): sube hasta el 5.
    Ev {
        r: 1,
        c: 4,
        e: 4,
        direction: 1,
        confirm_price: 53,
        extreme_price: 53,
    }
}

#[test]
fn a_boundary_that_is_not_a_tick_or_falls_in_a_gap_fails_like_the_reader() {
    let series = gapped();
    let ev = gapped_event();
    let sane = declared(&series, &series, &ev, &[2], 1, false).unwrap();
    assert_eq!(sane.rows[0].len(), 1);
    for (threads, reversed) in [(1, false), (4, false), (1, true), (4, true)] {
        // C = 3 está dentro del rango del row group y ningún tick lo tiene.
        let mut other = series.clone();
        other.ids[3] = 3;
        let error = boundary(declared(&series, &other, &ev, &[], threads, reversed));
        assert_eq!(
            (error.kind, error.field, error.event, error.bounds),
            (BoundaryKind::NotATick, 1, 0, [1, 3, 3])
        );
        assert_eq!((error.low, error.high), (1, 8));

        // C = 3 cae entre el row group [1, 2] y el [4, 5, 7, 8]: en un hueco.
        let error = boundary(declared(&series, &other, &ev, &[2], threads, reversed));
        assert_eq!(
            (error.kind, error.field, error.bounds),
            (BoundaryKind::Unseen, 1, [1, 3, 3])
        );
        assert_eq!((error.low, error.high), (4, 8));

        // R = 0 cae antes del primer row group.
        let mut other = series.clone();
        other.ids[0] = 0;
        let error = boundary(declared(&series, &other, &ev, &[2], threads, reversed));
        assert_eq!((error.kind, error.field), (BoundaryKind::Unseen, 0));

        // Con el tick de la frontera en un row group anterior no hay hueco: sin error.
        declared(&series, &series, &ev, &[2], threads, reversed).unwrap();
        declared(&series, &series, &ev, &[1, 2, 3], threads, reversed).unwrap();
    }
}

#[test]
fn the_tick_c_must_have_the_confirm_time_of_l2() {
    let series = gapped();
    let ev = gapped_event();
    let mut other = series.clone();
    other.time[3] += 1;
    for (threads, reversed) in [(1, false), (4, true)] {
        match declared(&series, &other, &ev, &[2], threads, reversed) {
            Err(EngineError::ConfirmTime(error)) => {
                assert_eq!((error.got, error.wanted), (13, 14));
                assert_eq!((error.theta, error.event, error.bounds), (0, 0, [1, 5, 5]));
            }
            other => panic!("se esperaba ConfirmTime y vino {:?}", other.err()),
        }
    }
}

#[test]
fn a_boundary_error_wins_over_a_confirm_time_error_in_the_same_row_group() {
    // Dos eventos en el mismo row group: el primero con el `confirm_time` mal y el
    // segundo con un id que no es tick. El lector (`locate`) comprueba todas las
    // fronteras antes que los tiempos, así que reporta el segundo.
    let mut series = gapped();
    series.price = vec![50, 51, 50, 49, 55, 56];
    series.ids = vec![1, 2, 4, 5, 7, 8];
    let first = Ev {
        r: 1,
        c: 2,
        e: 2,
        direction: 1,
        confirm_price: 51,
        extreme_price: 51,
    };
    let second = Ev {
        r: 2,
        c: 6,
        e: 6,
        direction: 1,
        confirm_price: 56,
        extreme_price: 56,
    };
    let evs = vec![first, second];
    let mut other = series.clone();
    other.time[1] += 1;
    other.ids[5] = 6;
    let sizes = vec![vec![2]];
    let engine = engine(1, &sizes);
    let columns = Columns::new(&series);
    let result = run(
        &engine,
        &columns,
        &[(0, 6)],
        vec![batches(&other, &evs, &sizes[0])],
        &|_| false,
        &[0],
        &sizes,
    );
    let error = boundary(result);
    assert_eq!((error.event, error.field, error.bounds[2]), (1, 1, 6));
}

#[test]
fn ticks_outside_the_contract_of_l1_are_input_errors() {
    for (price_high, wanted) in [(u64::MAX, TickError::Negative), (1, TickError::TooLarge)] {
        let series = Tiny {
            time: vec![1, 2, 3],
            price: vec![5, 5, 5],
            qty: vec![1, 1, 1],
            sold: vec![false; 3],
        }
        .series();
        let columns = Columns::new(&series);
        let mut price = vec![0u8; 3 * 16];
        for tick in 0..3 {
            price[tick * 16..tick * 16 + 8].copy_from_slice(&5i64.to_le_bytes());
        }
        // Un precio con la palabra alta ≠ 0: o negativo o más grande que 63 bits.
        price[16 + 8..16 + 16].copy_from_slice(&price_high.to_le_bytes());
        let ticks = TickColumns::new(
            3,
            Region {
                buffer: columns.share_ids(),
                offset: 0,
            },
            Region {
                buffer: columns.share_time(),
                offset: 0,
            },
            Region {
                buffer: Shared(std::sync::Arc::new(price)),
                offset: 0,
            },
            Region {
                buffer: columns.share_quantity(),
                offset: 0,
            },
            Region {
                buffer: columns.share_sold(),
                offset: 0,
            },
        )
        .unwrap();
        let engine = common::engine(1, &[vec![1]]);
        let ev = Ev {
            r: 1,
            c: 3,
            e: 3,
            direction: 1,
            confirm_price: 5,
            extreme_price: 5,
        };
        let series = Series {
            ids: vec![1, 2, 3],
            time: vec![1, 2, 3],
            price: vec![5, 5, 5],
            qty: vec![1, 1, 1],
            sold: vec![false; 3],
        };
        engine
            .push_boundaries(0, batches(&series, &[ev], &[1]).remove(0))
            .unwrap();
        engine
            .push(RowGroupInput {
                id: 0,
                ticks,
                before_month: false,
            })
            .unwrap();
        assert_eq!(engine.finish().unwrap_err(), EngineError::Ticks(wanted));
    }
}

#[test]
fn inflight_drops_to_zero_and_busy_is_a_fraction() {
    let case = Case::new(9, 400, 3, 8, false);
    let engine = common::engine(1, &case.sizes);
    for (t, b) in case.batches().into_iter().enumerate() {
        for batch in b {
            engine.push_boundaries(t, batch).unwrap();
        }
    }
    assert_eq!(engine.busy(), 0.0);
    for (k, &(lo, hi)) in case.bounds.iter().enumerate() {
        engine
            .push(RowGroupInput {
                id: k as u64,
                ticks: case.columns.row_group(lo, hi),
                before_month: false,
            })
            .unwrap();
    }
    assert!(engine.inflight() <= case.bounds.len());
    engine.finish().unwrap();
    assert_eq!(engine.inflight(), 0);
    let busy = engine.busy();
    assert!(0.0 < busy && busy <= 1.0, "{busy}");
}
