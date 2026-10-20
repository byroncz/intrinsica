//! El motor contra un oráculo ingenuo, con 1 y 4 hilos, en varios órdenes de `push`, con
//! eventos que cruzan row groups de todas las formas, y los errores y bordes del contrato.

mod common;

use common::*;
use dc_descriptors::{
    col, Config, Engine, EngineError, EventErrorKind, EventIn, PhaseName, Region, Row,
    RowGroupInput, ThetaPiece, TickColumns, TickError,
};

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

    fn pieces(&self, k: usize) -> Vec<ThetaPiece> {
        let (lo, hi) = self.bounds[k];
        (0..self.events.len())
            .filter_map(|t| {
                piece(&self.series, t, &self.events[t], lo, hi, &|i| {
                    before_month(t, i)
                })
            })
            .collect()
    }

    fn run(&self, threads: usize, order: &[usize]) -> Result<Outcome, EngineError> {
        let engine = engine(threads, &self.sizes);
        run(
            &engine,
            &self.columns,
            &self.bounds,
            &|k| self.pieces(k),
            order,
            &self.sizes,
        )
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
            let before: i64 = (0..evs.len()).map(|i| before_month(t, i)).sum();
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
            .filter(|&k| case.pieces(k).len() < 2)
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
        &|k| {
            let (lo, hi) = bounds[k];
            piece(series, 0, &events, lo, hi, &|_| 0)
                .into_iter()
                .collect()
        },
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
            pieces: case.pieces(k),
        }
    };
    let input = |error: EngineError| match error {
        EngineError::Input(message) => message,
        other => panic!("se esperaba un error de entrada y vino {other}"),
    };

    // Un row group no se empuja dos veces.
    let engine = common::engine(1, &case.sizes);
    engine.push(make(0)).unwrap();
    assert!(input(engine.push(make(0)).unwrap_err()).contains("ya se empujó"));
    // Un θ fuera de rango o repetido, y posiciones fuera de orden.
    let mut bad = make(1);
    bad.pieces[0].theta = 9;
    assert!(input(engine.push(bad).unwrap_err()).contains("fuera de rango"));
    let mut bad = make(1);
    if let Some(first) = bad.pieces.first().cloned() {
        bad.pieces.push(first);
    }
    assert!(input(engine.push(bad).unwrap_err()).contains("repetido"));
    let mut bad = make(1);
    bad.pieces[0].events[0].positions = [3, 2, 1];
    assert!(input(engine.push(bad).unwrap_err()).contains("fuera de orden"));

    // Sin los demás row groups, al terminar faltan filas.
    match engine.finish() {
        Err(EngineError::Incomplete {
            missing, expected, ..
        }) => assert!(0 < missing && missing <= expected),
        other => panic!("se esperaba Incomplete y vino {other:?}"),
    }

    // Un esqueleto más corto que los eventos.
    let mut short = case.sizes.clone();
    short[0].pop();
    let engine = common::engine(4, &short);
    let outcome = run(
        &engine,
        &case.columns,
        &case.bounds,
        &|k| case.pieces(k),
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
        },
        Config {
            thetas: vec![1, 2],
            threads: 1,
            row_groups: vec![vec![1]],
        },
    ] {
        assert!(matches!(
            Engine::<Shared>::new(config),
            Err(EngineError::Input(_))
        ));
    }
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
        let ev = EventIn {
            positions: [0, 3, 3],
            starts_before: false,
            ends_after: false,
            ticks_before_month: 0,
            reference: 0,
            confirm: 3,
            confirm_time: 3,
            extreme: 3,
            direction: 1,
            confirm_price: 5,
            extreme_price: 5,
        };
        engine
            .push(RowGroupInput {
                id: 0,
                ticks,
                pieces: vec![ThetaPiece {
                    theta: 0,
                    first_event: 0,
                    events: vec![ev],
                }],
            })
            .unwrap();
        assert_eq!(engine.finish().unwrap_err(), EngineError::Ticks(wanted));
    }
}

#[test]
fn inflight_drops_to_zero_and_busy_is_a_fraction() {
    let case = Case::new(9, 400, 3, 8, false);
    let engine = common::engine(1, &case.sizes);
    assert_eq!(engine.busy(), 0.0);
    for (k, &(lo, hi)) in case.bounds.iter().enumerate() {
        engine
            .push(RowGroupInput {
                id: k as u64,
                ticks: case.columns.row_group(lo, hi),
                pieces: case.pieces(k),
            })
            .unwrap();
    }
    assert!(engine.inflight() <= case.bounds.len());
    engine.finish().unwrap();
    assert_eq!(engine.inflight(), 0);
    let busy = engine.busy();
    assert!(0.0 < busy && busy <= 1.0, "{busy}");
}
