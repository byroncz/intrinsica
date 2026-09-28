//! Pruebas con series sintéticas. Los eventos esperados están calculados a
//! mano; cada prueba deja el cálculo en comentarios.
//!
//! Los precios de las series son enteros pequeños (100, 110, …) y
//! `THETA = 10 %` (`10_000_000` en escala `10⁸`), para que los umbrales se
//! lean a simple vista: el detector solo ve enteros, no le importa que
//! `100` en escala `10⁸` sea `0,000001`.

use super::*;

const THETA: i64 = 10_000_000;

fn pt(price: i64, time: i64, agg_trade_id: i64) -> Point {
    Point {
        price,
        time,
        agg_trade_id,
    }
}

/// Alimenta toda la serie y devuelve los eventos, con `finish` al final.
fn run(theta: i64, ticks: &[Tick]) -> (Detector, Vec<Event>) {
    let mut detector = Detector::new(theta).unwrap();
    let mut events: Vec<Event> = ticks.iter().filter_map(|t| detector.feed(*t)).collect();
    events.extend(detector.finish());
    (detector, events)
}

#[test]
fn theta_must_be_in_open_interval() {
    assert_eq!(Detector::new(0).unwrap_err(), ThetaError(0));
    assert_eq!(Detector::new(-1).unwrap_err(), ThetaError(-1));
    assert_eq!(Detector::new(SCALE).unwrap_err(), ThetaError(SCALE));
    assert!(Detector::new(10_000).is_ok());
    assert!(Detector::new(SCALE - 1).is_ok());
}

#[test]
fn first_tick_is_initial_extreme() {
    let mut d = Detector::new(THETA).unwrap();
    assert_eq!(d.state().extremes, None);
    assert_eq!(d.feed(pt(100, 1, 1)), None);
    let first = pt(100, 1, 1);
    let state = d.state();
    assert_eq!(
        state.extremes,
        Some(Extremes {
            high: first,
            low: first
        })
    );
    assert_eq!(state.direction, None);
    assert_eq!(state.pending, None);
    assert_eq!(state.group, None);
}

#[test]
fn series_without_events() {
    // Máximo 104 → umbral de bajada floor(93,6) = 93; mínimo 98 → umbral de
    // subida ceil(107,8) = 108. Ningún precio los cruza.
    let ticks = [pt(100, 1, 1), pt(104, 2, 2), pt(98, 3, 3), pt(103, 4, 4)];
    let (d, events) = run(THETA, &ticks);
    assert!(events.is_empty());
    assert_eq!(
        *d.state(),
        State {
            direction: None,
            extremes: Some(Extremes {
                high: pt(104, 2, 2),
                low: pt(98, 3, 3)
            }),
            pending: None,
            group: None,
        }
    );
}

#[test]
fn simple_upturn_then_downturn() {
    // 100 abre. 105 sube el máximo. 95 baja el mínimo: subida = ceil(104,5) = 105.
    // 110 ≥ 105 confirma un upturn (grupo abierto). 112 (otro instante) cierra
    // el grupo: referencia = mínimo (95,3,3), confirmación (110,4,4). No hay
    // pendiente previo, no se emite nada. 120 sube el máximo; bajada
    // = 108. 108 ≤ 108 abre un downturn; 109 cierra el grupo: referencia =
    // máximo (120,6,6), y ahí se conoce el extremo del upturn.
    let ticks = [
        pt(100, 1, 1),
        pt(105, 2, 2),
        pt(95, 3, 3),
        pt(110, 4, 4),
        pt(112, 5, 5),
        pt(120, 6, 6),
        pt(108, 7, 7),
        pt(109, 8, 8),
    ];
    let mut d = Detector::new(THETA).unwrap();
    let emitted: Vec<Option<Event>> = ticks.iter().map(|t| d.feed(*t)).collect();
    let expected = Event {
        reference: pt(95, 3, 3),
        confirm: pt(110, 4, 4),
        extreme: pt(120, 6, 6),
        direction: Direction::Up,
    };
    for (i, e) in emitted.iter().enumerate() {
        assert_eq!(*e, (i == 7).then_some(expected), "tick {}", i + 1);
    }
    assert_eq!(d.finish(), None);
    assert_eq!(d.state().direction, Some(Direction::Down));
    assert_eq!(
        d.state().pending,
        Some(PendingEvent {
            reference: pt(120, 6, 6),
            confirm: pt(108, 7, 7),
            direction: Direction::Down
        })
    );
    assert_eq!(d.discarded(), 0);
}

#[test]
fn simple_downturn_closed_by_finish() {
    // Bajada desde 100: floor(90) = 90; 90 ≤ 90 abre el downturn en (90,3,3);
    // 85 cierra el grupo (sin pendiente previo). 85 baja el mínimo a (85,4,4);
    // subida = ceil(93,5) = 94, y 94 abre un upturn. `finish` lo cierra:
    // confirmación (94,5,5), y emite el downturn con la referencia en el
    // primer tick (100,1,1) y el extremo (85,4,4).
    let ticks = [
        pt(100, 1, 1),
        pt(95, 2, 2),
        pt(90, 3, 3),
        pt(85, 4, 4),
        pt(94, 5, 5),
    ];
    let (d, events) = run(THETA, &ticks);
    assert_eq!(
        events,
        [Event {
            reference: pt(100, 1, 1),
            confirm: pt(90, 3, 3),
            extreme: pt(85, 4, 4),
            direction: Direction::Down,
        }]
    );
    assert_eq!(d.state().direction, Some(Direction::Up));
    assert_eq!(d.state().group, None);
}

#[test]
fn threshold_rounds_toward_the_extreme() {
    // Subida desde 101: 101 × 1,1 = 111,1 → ceil = 112. 111 no cruza, 112 sí.
    let mut d = Detector::new(THETA).unwrap();
    d.feed(pt(101, 1, 1));
    d.feed(pt(111, 2, 2));
    assert_eq!(d.state().group, None);
    d.feed(pt(112, 3, 3));
    assert!(d.state().group.is_some());

    // Bajada desde 109: 109 × 0,9 = 98,1 → floor = 98. 99 no cruza, 98 sí.
    let mut d = Detector::new(THETA).unwrap();
    d.feed(pt(109, 1, 1));
    d.feed(pt(99, 2, 2));
    assert_eq!(d.state().group, None);
    let mut d = Detector::new(THETA).unwrap();
    d.feed(pt(109, 1, 1));
    d.feed(pt(98, 2, 2));
    assert!(d.state().group.is_some());
}

#[test]
fn threshold_does_not_overflow_i64_arithmetic() {
    // price × (SCALE + θ) ≈ 2 × 10²⁶ desborda i64 (≈ 9,2 × 10¹⁸) pero cabe en i128.
    let price = PRICE_LIMIT - 1;
    let theta = SCALE - 1;
    // 999_999_999_999_999_999 × 199_999_999 / 10⁸ = 1_999_999_989_999_999_998,00000001 → ceil.
    assert_eq!(up_threshold(price, theta), 1_999_999_989_999_999_999);
    // 999_999_999_999_999_999 × 1 / 10⁸ = 9_999_999_999,99999999 → floor.
    assert_eq!(down_threshold(price, theta), 9_999_999_999);

    let mut d = Detector::new(theta).unwrap();
    assert_eq!(d.feed(pt(price, 1, 1)), None);
    assert_eq!(d.feed(pt(price, 2, 2)), None);
    assert_eq!(d.state().group, None);
}

#[test]
#[should_panic(expected = "fuera de 0 < price")]
fn price_out_of_contract_panics() {
    Detector::new(THETA).unwrap().feed(pt(PRICE_LIMIT, 1, 1));
}

#[test]
fn ties_upturn_use_lowest_qualifying_price() {
    // Cruce desde 100 (umbral 110). Instante 3: 120 cruza primero; 111 cumple
    // y es menor → mejor; 109 no cumple (solo avanza el id); 115 cumple pero
    // no es menor. Confirmación = (111, 3, id del último = 5). 130 (instante 4)
    // cierra el grupo. El instante es atómico: el máximo se reinicia en la
    // confirmación (111,3,5), no en 120; solo 130 lo mueve.
    let mut d = Detector::new(THETA).unwrap();
    for t in [
        pt(100, 1, 1),
        pt(120, 3, 2),
        pt(111, 3, 3),
        pt(109, 3, 4),
        pt(115, 3, 5),
    ] {
        assert_eq!(d.feed(t), None);
    }
    assert_eq!(
        d.state().group,
        Some(Group {
            direction: Direction::Up,
            time: 3,
            threshold: 110,
            best_price: 111,
            last_id: 5
        })
    );
    assert_eq!(d.feed(pt(130, 4, 6)), None);
    assert_eq!(
        d.state().pending,
        Some(PendingEvent {
            reference: pt(100, 1, 1),
            confirm: pt(111, 3, 5),
            direction: Direction::Up
        })
    );
    assert_eq!(
        d.state().extremes,
        Some(Extremes {
            high: pt(130, 4, 6),
            low: pt(100, 1, 1)
        })
    );

    // Bajada 130 → 117: el extremo del evento es 130, no el 120 del instante
    // de confirmación (que no cuenta).
    assert_eq!(d.feed(pt(117, 5, 7)), None);
    assert_eq!(
        d.feed(pt(100, 6, 8)),
        Some(Event {
            reference: pt(100, 1, 1),
            confirm: pt(111, 3, 5),
            extreme: pt(130, 4, 6),
            direction: Direction::Up,
        })
    );
}

#[test]
fn ties_downturn_use_highest_qualifying_price() {
    // Bajada desde 100 (umbral 90). Instante 2: 80 cruza; 89 cumple y es mayor
    // → mejor; 91 no cumple; 85 cumple pero no es mayor. Confirmación =
    // (89, 2, 5). 70 (instante 3) cierra el grupo.
    let mut d = Detector::new(THETA).unwrap();
    for t in [
        pt(100, 1, 1),
        pt(80, 2, 2),
        pt(89, 2, 3),
        pt(91, 2, 4),
        pt(85, 2, 5),
    ] {
        assert_eq!(d.feed(t), None);
    }
    assert_eq!(d.feed(pt(70, 3, 6)), None);
    assert_eq!(d.finish(), None);
    assert_eq!(
        d.state().pending,
        Some(PendingEvent {
            reference: pt(100, 1, 1),
            confirm: pt(89, 2, 5),
            direction: Direction::Down
        })
    );
    assert_eq!(d.state().extremes.unwrap().low, pt(70, 3, 6));
}

#[test]
fn confirmation_instant_is_atomic_and_overshoot_can_be_empty() {
    // 110 cruza el upturn en el instante 2. 85 (mismo instante) no cumple el
    // umbral 110: solo avanza el id, no evalúa la reversión. Confirmación =
    // (110, 2, 3). 86 (instante 3) cierra el grupo, reinicia el máximo en
    // (110,2,3) y cruza la bajada de inmediato: floor(99) = 99. Al confirmar
    // el downturn, el upturn queda con extremo = confirmación: Overshoot vacío.
    let ticks = [pt(100, 1, 1), pt(110, 2, 2), pt(85, 2, 3), pt(86, 3, 4)];
    let (_, events) = run(THETA, &ticks);
    assert_eq!(
        events,
        [Event {
            reference: pt(100, 1, 1),
            confirm: pt(110, 2, 3),
            extreme: pt(110, 2, 3),
            direction: Direction::Up,
        }]
    );
}

#[test]
fn zero_tick_dc_is_not_emitted_but_trend_is_adopted() {
    // La guarda es inalcanzable por `feed` con θ > 0 (el umbral queda
    // estrictamente más allá del extremo), así que se arma el estado a mano:
    // downturn vigente con evento pendiente, y un grupo de upturn cuyo último
    // id (5) no supera el del mínimo de referencia (5).
    let mut d = Detector::new(THETA).unwrap();
    let pending = PendingEvent {
        reference: pt(100, 1, 1),
        confirm: pt(90, 2, 4),
        direction: Direction::Down,
    };
    d.state = State {
        direction: Some(Direction::Down),
        extremes: Some(Extremes {
            high: pt(100, 1, 1),
            low: pt(90, 2, 5),
        }),
        pending: Some(pending),
        group: Some(Group {
            direction: Direction::Up,
            time: 3,
            threshold: 99,
            best_price: 99,
            last_id: 5,
        }),
    };
    assert_eq!(d.finish(), None);
    assert_eq!(d.discarded(), 1);
    let state = d.state();
    assert_eq!(state.direction, Some(Direction::Up));
    assert_eq!(state.pending, Some(pending));
    assert_eq!(state.group, None);
    assert_eq!(
        state.extremes,
        Some(Extremes {
            high: pt(99, 3, 5),
            low: pt(90, 2, 5)
        })
    );
}

/// Serie de reversiones repetidas, con un tick de empate y un Overshoot vacío.
fn repeated_reversals() -> Vec<Tick> {
    vec![
        pt(100, 1, 1),
        pt(120, 2, 2),
        pt(130, 3, 3),
        pt(117, 4, 4),
        pt(100, 5, 5),
        pt(110, 6, 6),
        pt(105, 7, 7),
        pt(99, 8, 8),
        pt(99, 8, 9),
        pt(90, 9, 10),
    ]
}

#[test]
fn repeated_reversals_events_and_state() {
    // t2: 120 ≥ 110 abre upturn; t3 (130) lo cierra: ref (100,1,1), conf (120,2,2).
    // t4: 117 ≤ floor(117) abre downturn; t5 lo cierra: emite E0 con extremo
    // (130,3,3); ref del downturn = (130,3,3), conf (117,4,4).
    // t6: 110 ≥ ceil(110) abre upturn (mínimo 100); t7 lo cierra: emite E1 con
    // extremo (100,5,5); pendiente: ref (100,5,5), conf (110,6,6).
    // t8: 99 ≤ floor(99) abre downturn; t8b (99) empata; t10 lo cierra:
    // referencia = máximo (110,6,6) = confirmación del upturn → E2 con
    // Overshoot vacío. Queda pendiente el downturn (110,6,6) → (99,8,9).
    let (d, events) = run(THETA, &repeated_reversals());
    assert_eq!(
        events,
        [
            Event {
                reference: pt(100, 1, 1),
                confirm: pt(120, 2, 2),
                extreme: pt(130, 3, 3),
                direction: Direction::Up,
            },
            Event {
                reference: pt(130, 3, 3),
                confirm: pt(117, 4, 4),
                extreme: pt(100, 5, 5),
                direction: Direction::Down,
            },
            Event {
                reference: pt(100, 5, 5),
                confirm: pt(110, 6, 6),
                extreme: pt(110, 6, 6),
                direction: Direction::Up,
            },
        ]
    );
    assert_eq!(
        *d.state(),
        State {
            direction: Some(Direction::Down),
            extremes: Some(Extremes {
                high: pt(110, 6, 6),
                low: pt(90, 9, 10)
            }),
            pending: Some(PendingEvent {
                reference: pt(110, 6, 6),
                confirm: pt(99, 8, 9),
                direction: Direction::Down
            }),
            group: None,
        }
    );
    assert_eq!(d.discarded(), 0);
}

#[test]
fn consecutive_events_chain_extreme_to_reference() {
    let (_, events) = run(THETA, &repeated_reversals());
    for pair in events.windows(2) {
        assert_eq!(pair[0].extreme, pair[1].reference);
    }
}

#[test]
fn same_series_twice_gives_identical_events() {
    let ticks = repeated_reversals();
    let (d1, e1) = run(THETA, &ticks);
    let (d2, e2) = run(THETA, &ticks);
    assert_eq!(e1, e2);
    assert_eq!(d1.state(), d2.state());
}

/// Camino aleatorio entero y determinista (LCG de Knuth), con empates de
/// `transact_time` cada tanto.
fn random_walk(n: usize) -> Vec<Tick> {
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut price: i64 = 6_000_000_000_000;
    let mut time: i64 = 1_000;
    let mut ticks = Vec::with_capacity(n);
    for id in 1..=n as i64 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let r = seed >> 33;
        // Paso de hasta ±0,05 % del precio.
        let step = (r % 1_001) as i64 - 500;
        price += price / 1_000_000 * step;
        if r % 4 != 0 {
            time += 1 + (r % 7) as i64;
        }
        ticks.push(pt(price, time, id));
    }
    ticks
}

#[test]
fn result_does_not_depend_on_feeding_groups() {
    let ticks = random_walk(20_000);
    for theta in [10_000, 100_000, 1_000_000] {
        let (whole, expected) = run(theta, &ticks);
        assert!(
            !expected.is_empty(),
            "la serie debe generar eventos (θ={theta})"
        );
        for chunk_size in [1, 2, 3, 7, 1_000, 19_999] {
            let mut d = Detector::new(theta).unwrap();
            let mut events = Vec::new();
            for chunk in ticks.chunks(chunk_size) {
                events.extend(chunk.iter().filter_map(|t| d.feed(*t)));
                // Leer el estado entre lotes no lo altera.
                let _ = *d.state();
            }
            events.extend(d.finish());
            assert_eq!(events, expected, "θ={theta}, lote={chunk_size}");
            assert_eq!(d.state(), whole.state());
        }
    }
}

#[test]
fn random_walk_events_keep_invariants() {
    let (d, events) = run(100_000, &random_walk(20_000));
    assert_eq!(d.discarded(), 0);
    for e in &events {
        assert!(e.extreme.agg_trade_id >= e.confirm.agg_trade_id);
        assert!(e.extreme.time >= e.confirm.time);
        assert!(e.confirm.agg_trade_id > e.reference.agg_trade_id);
        match e.direction {
            Direction::Up => assert!(e.confirm.price > e.reference.price),
            Direction::Down => assert!(e.confirm.price < e.reference.price),
        }
    }
    for pair in events.windows(2) {
        assert_eq!(pair[0].extreme, pair[1].reference);
        assert_ne!(pair[0].direction, pair[1].direction);
    }
}
