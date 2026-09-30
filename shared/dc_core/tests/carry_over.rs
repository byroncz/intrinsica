//! Carry-over (ITSC-242, R-03): una serie partida en tramos, con el estado
//! pasando por bytes entre uno y otro, da los mismos eventos y el mismo estado
//! final que la serie entera.

mod common;

use common::{fifty_thetas, series};
use dc_core::{
    CarryOver, CarryOverError, CarryPending, Detector, Direction, Event, FanOut, FanOutCarryError,
    Point, State, ThetaError, Tick, PRICE_LIMIT, SCALE, STATE_VERSION,
};

/// θ = 10 %; los precios enteros pequeños dejan leer los umbrales a ojo.
const THETA: i64 = 10_000_000;

fn pt(price: i64, time: i64, agg_trade_id: i64) -> Point {
    Point {
        price,
        time,
        agg_trade_id,
    }
}

/// La serie entera: eventos (con `finish` al final) y estado final.
fn whole(theta: i64, ticks: &[Tick]) -> (Vec<Event>, State) {
    let mut d = Detector::new(theta).unwrap();
    let mut events: Vec<Event> = ticks.iter().filter_map(|t| d.feed(*t)).collect();
    events.extend(d.finish());
    (events, *d.state())
}

/// Qué pasó en un corte, para comprobar que las pruebas cubren los casos.
#[derive(Debug)]
struct Cut {
    /// Había un grupo de empate abierto al cortar; `finish` lo cerró.
    open_group: bool,
    /// El carry-over trae un evento pendiente sin extremo.
    pending: bool,
    /// Además, el extremo vigente ya se movió más allá de su confirmación:
    /// el corte cae en medio del overshoot.
    mid_overshoot: bool,
}

/// Parte `ticks` en `cut`, pasa el estado por bytes y compara con la serie
/// entera. `None` si el corte cae dentro de un grupo de empate que sigue
/// abierto con el tick siguiente: ese corte no es un borde de mes válido
/// (ningún grupo cruza el borde, TRD-L2 §8.1 paso 4) y `carry_over` lo rechaza.
fn check_cut(theta: i64, ticks: &[Tick], cut: usize) -> Option<Cut> {
    let (expected_events, expected_state) = whole(theta, ticks);

    let mut first = Detector::new(theta).unwrap();
    let mut events: Vec<Event> = ticks[..cut].iter().filter_map(|t| first.feed(*t)).collect();
    let open_group = first.state().group.is_some();
    if open_group {
        assert_eq!(first.carry_over().unwrap_err(), CarryOverError::OpenGroup);
        if ticks
            .get(cut)
            .is_some_and(|next| next.time == ticks[cut - 1].time)
        {
            return None;
        }
        events.extend(first.finish());
    }
    let carry = first.carry_over().unwrap();
    let after_finish = *first.state();

    // Cruza los bytes: lo que lee el mes siguiente es lo que escribió este.
    let bytes = carry.to_bytes().unwrap();
    let carry = CarryOver::from_bytes(&bytes).unwrap();
    let mut second = Detector::from_carry_over(theta, &carry).unwrap();
    assert_eq!(
        *second.state(),
        after_finish,
        "corte {cut}: el estado retomado"
    );
    assert_eq!(second.carry_over().unwrap(), carry);

    events.extend(ticks[cut..].iter().filter_map(|t| second.feed(*t)));
    events.extend(second.finish());

    assert_eq!(events, expected_events, "corte {cut}: eventos");
    assert_eq!(*second.state(), expected_state, "corte {cut}: estado final");

    let mid_overshoot = carry.pending.is_some_and(|p| match carry.direction {
        1 => carry.ext_high != p.confirm,
        _ => carry.ext_low != p.confirm,
    });
    Some(Cut {
        open_group,
        pending: carry.pending.is_some(),
        mid_overshoot,
    })
}

/// Serie a mano con θ = 10 %. Cálculo:
/// - 100, 95 (mínimo, umbral de subida ceil(104,5) = 105).
/// - 105 cruza en t=3: abre grupo; 106 (t=3) empata: no mejora el precio
///   (105 es el más bajo que cumple) pero avanza el id → confirmación (105, t3, id4).
/// - 108 (t=4) cierra el grupo: upturn con referencia (95, id2), evento
///   pendiente; 108 y 112 estiran el overshoot (máximo 112).
/// - 100 (t=6) cruza la bajada floor(100,8) = 100: grupo; 99 (t=6) empata →
///   confirmación (100, t6, id8).
/// - 98 (t=7) cierra: emite el upturn con extremo (112, id6); nuevo pendiente.
/// - 90 baja el mínimo; 100 (t=9) cruza la subida ceil(99) = 99 y deja un
///   grupo abierto al terminar; `finish` lo cierra y emite el downturn.
fn hand_series() -> Vec<Tick> {
    vec![
        pt(100, 1, 1),
        pt(95, 2, 2),
        pt(105, 3, 3),
        pt(106, 3, 4),
        pt(108, 4, 5),
        pt(112, 5, 6),
        pt(100, 6, 7),
        pt(99, 6, 8),
        pt(98, 7, 9),
        pt(90, 8, 10),
        pt(100, 9, 11),
    ]
}

#[test]
fn hand_series_events_are_the_expected_ones() {
    let (events, _) = whole(THETA, &hand_series());
    assert_eq!(
        events,
        [
            Event {
                reference: pt(95, 2, 2),
                confirm: pt(105, 3, 4),
                extreme: pt(112, 5, 6),
                direction: Direction::Up,
            },
            Event {
                reference: pt(112, 5, 6),
                confirm: pt(100, 6, 8),
                extreme: pt(90, 8, 10),
                direction: Direction::Down,
            },
        ]
    );
}

#[test]
fn every_cut_of_the_hand_series_matches_the_whole() {
    let ticks = hand_series();
    let cuts: Vec<Option<Cut>> = (1..=ticks.len())
        .map(|cut| check_cut(THETA, &ticks, cut))
        .collect();
    let at = |cut: usize| cuts[cut - 1].as_ref();

    // Dentro de un grupo de empate que sigue abierto: no es un borde válido.
    assert!(at(3).is_none());
    assert!(at(7).is_none());
    // El grupo cierra con el último tick del instante (siguiente en otro t).
    let tie_end = at(4).unwrap();
    assert!(tie_end.open_group && tie_end.pending && !tie_end.mid_overshoot);
    // Justo tras la confirmación (la cierra el primer tick del instante
    // siguiente): pendiente sin extremo.
    let just_confirmed = at(5).unwrap();
    assert!(!just_confirmed.open_group && just_confirmed.pending);
    // En medio del overshoot: el máximo ya pasó la confirmación.
    assert!(at(6).unwrap().mid_overshoot);
    assert!(at(10).unwrap().mid_overshoot);
    // Grupo abierto al final y con evento pendiente.
    let end = at(11).unwrap();
    assert!(end.open_group && end.pending);
    // Sin tendencia ni evento todavía.
    let cold = at(2).unwrap();
    assert!(!cold.open_group && !cold.pending);
}

#[test]
fn every_cut_of_synthetic_series_matches_the_whole() {
    let mut open_group = 0;
    let mut pending = 0;
    let mut mid_overshoot = 0;
    let mut rejected = 0;
    for (seed, theta) in [(1, 50_000), (2, 250_000), (3, 1_000_000), (4, 2_500_000)] {
        let (p, t, i) = series(3_000, seed);
        let ticks: Vec<Tick> = (0..p.len()).map(|k| pt(p[k], t[k], i[k])).collect();
        for cut in 1..=ticks.len() {
            match check_cut(theta, &ticks, cut) {
                None => rejected += 1,
                Some(c) => {
                    open_group += usize::from(c.open_group);
                    pending += usize::from(c.pending);
                    mid_overshoot += usize::from(c.mid_overshoot);
                }
            }
        }
    }
    // Los cuatro casos del borde aparecen de sobra en las series sintéticas.
    assert!(open_group > 50, "cortes con grupo abierto: {open_group}");
    assert!(pending > 5_000, "cortes con evento pendiente: {pending}");
    assert!(
        mid_overshoot > 1_000,
        "cortes en overshoot: {mid_overshoot}"
    );
    assert!(
        rejected > 50,
        "cortes dentro de un instante abierto: {rejected}"
    );
}

#[test]
fn a_chain_of_many_cuts_matches_the_whole() {
    // Muchos tramos seguidos, como meses: cada uno retoma el carry-over del
    // anterior por bytes. Los bordes caen donde cambia `transact_time`.
    let (p, t, i) = series(60_000, 9);
    let ticks: Vec<Tick> = (0..p.len()).map(|k| pt(p[k], t[k], i[k])).collect();
    for theta in [50_000, 100_000, 300_000] {
        let (expected, expected_state) = whole(theta, &ticks);
        let mut events = Vec::new();
        let mut detector = Detector::new(theta).unwrap();
        let mut start = 0;
        for size in [1, 2, 977, 5_000, 12_345, 20_000].iter().cycle() {
            let mut end = (start + size).min(ticks.len());
            while end < ticks.len() && ticks[end].time == ticks[end - 1].time {
                end += 1;
            }
            events.extend(ticks[start..end].iter().filter_map(|t| detector.feed(*t)));
            if end == ticks.len() {
                break;
            }
            events.extend(detector.finish());
            let bytes = detector.carry_over().unwrap().to_bytes().unwrap();
            detector =
                Detector::from_carry_over(theta, &CarryOver::from_bytes(&bytes).unwrap()).unwrap();
            start = end;
        }
        events.extend(detector.finish());
        assert!(expected.len() > 20, "θ {theta}: {} eventos", expected.len());
        assert_eq!(events, expected, "θ {theta}");
        assert_eq!(*detector.state(), expected_state, "θ {theta}");
    }
}

#[test]
fn fifty_thetas_resume_through_the_fanout() {
    let thetas = fifty_thetas();
    let (p, t, i) = series(150_000, 5);

    let mut reference = FanOut::new(&thetas).unwrap();
    let mut expected = reference.feed_batch(&p, &t, &i);
    for (acc, last) in expected.iter_mut().zip(reference.finish()) {
        acc.extend(last);
    }

    // Tres "meses" de largo desigual, con el borde donde cambia `transact_time`.
    let edge = |mut k: usize| {
        while t[k] == t[k - 1] {
            k += 1;
        }
        k
    };
    let bounds = [0, edge(31_111), edge(64_000), p.len()];
    let mut fan = FanOut::new(&thetas).unwrap();
    let mut events = vec![Vec::new(); thetas.len()];
    for (month, w) in bounds.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        let batch = fan.feed_batch(&p[a..b], &t[a..b], &i[a..b]);
        for (acc, new) in events.iter_mut().zip(batch) {
            acc.extend(new);
        }
        for (acc, last) in events.iter_mut().zip(fan.finish()) {
            acc.extend(last);
        }
        if month == 2 {
            break;
        }
        let bytes: Vec<Vec<u8>> = fan
            .carry_overs()
            .unwrap()
            .iter()
            .map(|c| c.to_bytes().unwrap())
            .collect();
        let carry: Vec<CarryOver> = bytes
            .iter()
            .map(|b| CarryOver::from_bytes(b).unwrap())
            .collect();
        // Otro número de hilos que el original: no cambia el resultado.
        fan = FanOut::from_carry_over_with_threads(&thetas, &carry, 3 + month).unwrap();
    }
    assert!(expected.iter().filter(|e| e.len() > 10).count() >= 40);
    assert_eq!(events, expected);
    for (d, r) in fan.detectors().iter().zip(reference.detectors()) {
        assert_eq!(d.state(), r.state());
    }
}

/// Un carry-over válido: upturn con evento pendiente y overshoot en curso.
fn valid_carry() -> CarryOver {
    let mut d = Detector::new(THETA).unwrap();
    for t in &hand_series()[..6] {
        d.feed(*t);
    }
    d.carry_over().unwrap()
}

#[test]
fn carry_over_carries_the_trd_fields() {
    let c = valid_carry();
    assert_eq!(c.theta, THETA);
    assert_eq!(c.state_version, STATE_VERSION);
    assert_eq!(c.direction, 1);
    assert_eq!(c.ext_high, pt(112, 5, 6));
    assert_eq!(c.ext_low, pt(95, 2, 2));
    assert_eq!(
        c.pending,
        Some(CarryPending {
            reference: pt(95, 2, 2),
            confirm: pt(105, 3, 4),
        })
    );
}

#[test]
fn bytes_are_canonical_and_bounded() {
    let c = valid_carry();
    let bytes = c.to_bytes().unwrap();
    // 1 (largo) + 5 ("1.0.0") + 8 + 1 + 48 + 1 + 48, sin importar cuántos
    // ticks vio el detector: nada crece con la serie.
    assert_eq!(bytes.len(), 1 + STATE_VERSION.len() + 106);
    assert_eq!(bytes[0] as usize, STATE_VERSION.len());
    assert_eq!(CarryOver::from_bytes(&bytes).unwrap(), c);
    assert_eq!(c.to_bytes().unwrap(), bytes);

    let (p, t, i) = series(200_000, 21);
    let mut d = Detector::new(500_000).unwrap();
    for k in 0..p.len() {
        d.feed(pt(p[k], t[k], i[k]));
    }
    d.finish();
    assert_eq!(
        d.carry_over().unwrap().to_bytes().unwrap().len(),
        bytes.len()
    );

    // Sin evento pendiente el largo es el mismo y los puntos van en cero.
    let mut cold = Detector::new(THETA).unwrap();
    cold.feed(pt(100, 1, 1));
    let cold = cold.carry_over().unwrap();
    assert_eq!((cold.direction, cold.pending), (0, None));
    let cold_bytes = cold.to_bytes().unwrap();
    assert_eq!(cold_bytes.len(), bytes.len());
    assert!(cold_bytes[cold_bytes.len() - 48..].iter().all(|&b| b == 0));
    assert_eq!(CarryOver::from_bytes(&cold_bytes).unwrap(), cold);
}

#[test]
fn from_bytes_rejects_malformed_input() {
    let bytes = valid_carry().to_bytes().unwrap();
    let malformed =
        |b: &[u8]| matches!(CarryOver::from_bytes(b), Err(CarryOverError::Malformed(_)));

    assert!(malformed(&[]));
    assert!(malformed(&bytes[..bytes.len() - 1]));
    assert!(malformed(&bytes[..10]));
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(malformed(&longer));

    let flag = bytes.len() - 49;
    let mut bad_flag = bytes.clone();
    bad_flag[flag] = 2;
    assert!(malformed(&bad_flag));
    // Puntos pendientes con la bandera apagada: no es la forma canónica.
    let mut stray = bytes.clone();
    stray[flag] = 0;
    assert!(malformed(&stray));
}

#[test]
fn from_bytes_rejects_other_version_before_reading_the_rest() {
    let mut other = vec![5];
    other.extend_from_slice(b"2.0.0");
    other.extend_from_slice(&[1, 2, 3]); // otro formato: no se interpreta
    assert_eq!(
        CarryOver::from_bytes(&other).unwrap_err(),
        CarryOverError::VersionMismatch {
            expected: STATE_VERSION,
            found: "2.0.0".into(),
        }
    );
    let mut long = valid_carry();
    long.state_version = "x".repeat(256);
    assert!(matches!(long.to_bytes(), Err(CarryOverError::Malformed(_))));
}

#[test]
fn from_carry_over_rejects_another_theta() {
    let c = valid_carry();
    let err = Detector::from_carry_over(THETA + 1, &c).unwrap_err();
    assert_eq!(
        err,
        CarryOverError::ThetaMismatch {
            expected: THETA + 1,
            found: THETA,
        }
    );
    assert!(err.to_string().contains("otro theta"));
    assert_eq!(
        Detector::from_carry_over(0, &c).unwrap_err(),
        CarryOverError::Theta(ThetaError(0))
    );
    assert!(Detector::from_carry_over(THETA, &c).is_ok());
}

#[test]
fn from_carry_over_rejects_another_version() {
    let mut c = valid_carry();
    c.state_version = "0.9.0".into();
    let err = Detector::from_carry_over(THETA, &c).unwrap_err();
    assert_eq!(
        err,
        CarryOverError::VersionMismatch {
            expected: STATE_VERSION,
            found: "0.9.0".into(),
        }
    );
    assert!(err.to_string().contains("state_version"));
    // Ni siquiera una versión "más nueva" del mismo mayor se acepta.
    c.state_version = "1.0.1".into();
    assert!(Detector::from_carry_over(THETA, &c).is_err());
}

#[test]
fn from_carry_over_rejects_inconsistent_state() {
    let base = valid_carry();
    let cases: Vec<(&str, CarryOver)> = vec![
        (
            "direction fuera de rango",
            CarryOver {
                direction: 2,
                ..base.clone()
            },
        ),
        (
            "pendiente sin tendencia",
            CarryOver {
                direction: 0,
                ..base.clone()
            },
        ),
        (
            "ext_high por debajo de ext_low",
            CarryOver {
                ext_high: pt(90, 5, 6),
                ..base.clone()
            },
        ),
        (
            "precio cero",
            CarryOver {
                ext_low: pt(0, 2, 2),
                ..base.clone()
            },
        ),
        (
            "precio sobre PRICE_LIMIT",
            CarryOver {
                ext_high: pt(PRICE_LIMIT, 5, 6),
                ..base.clone()
            },
        ),
        (
            "confirmación no posterior a la referencia",
            CarryOver {
                pending: Some(CarryPending {
                    reference: pt(95, 2, 4),
                    confirm: pt(105, 3, 4),
                }),
                ..base.clone()
            },
        ),
        (
            "confirmación del lado contrario a la tendencia",
            CarryOver {
                pending: Some(CarryPending {
                    reference: pt(95, 2, 2),
                    confirm: pt(90, 3, 4),
                }),
                ..base.clone()
            },
        ),
        (
            "extremo vigente antes de la confirmación",
            CarryOver {
                ext_high: pt(112, 5, 3),
                ..base.clone()
            },
        ),
        (
            "extremo vigente por debajo de la confirmación",
            CarryOver {
                ext_high: pt(104, 5, 6),
                ..base.clone()
            },
        ),
    ];
    for (name, carry) in cases {
        let err = Detector::from_carry_over(THETA, &carry).unwrap_err();
        assert!(
            matches!(err, CarryOverError::Inconsistent(_)),
            "{name}: {err:?}"
        );
        assert!(
            err.to_string().starts_with("carry-over inconsistente"),
            "{name}"
        );
    }
}

#[test]
fn from_carry_over_accepts_downturn_state() {
    // Espejo del caso válido: el estado tras el corte 9 de la serie a mano.
    let mut d = Detector::new(THETA).unwrap();
    for t in &hand_series()[..9] {
        d.feed(*t);
    }
    let c = d.carry_over().unwrap();
    assert_eq!(c.direction, -1);
    assert!(Detector::from_carry_over(THETA, &c).is_ok());
    let wrong_side = CarryOver {
        ext_low: pt(101, 7, 9),
        ..c
    };
    assert!(matches!(
        Detector::from_carry_over(THETA, &wrong_side),
        Err(CarryOverError::Inconsistent(_))
    ));
}

#[test]
fn carry_over_needs_ticks_and_a_closed_group() {
    let mut d = Detector::new(THETA).unwrap();
    assert_eq!(d.carry_over().unwrap_err(), CarryOverError::NoTicks);
    for t in &hand_series()[..3] {
        d.feed(*t);
    }
    assert_eq!(d.carry_over().unwrap_err(), CarryOverError::OpenGroup);
    d.finish();
    assert!(d.carry_over().is_ok());
}

#[test]
fn resumed_detector_counts_discarded_from_zero() {
    let c = valid_carry();
    let d = Detector::from_carry_over(THETA, &c).unwrap();
    assert_eq!((d.theta(), d.discarded()), (THETA, 0));
    assert_eq!(d.carry_over().unwrap(), c);
}

#[test]
fn fanout_rejects_wrong_count_theta_and_version() {
    let a = valid_carry();
    let b = CarryOver {
        theta: 5_000_000,
        ..a.clone()
    };

    assert_eq!(
        FanOut::from_carry_over(&[THETA, THETA], &[a.clone()]).unwrap_err(),
        FanOutCarryError::Count {
            expected: 2,
            found: 1
        }
    );
    // El carry-over de la posición 1 es de otro θ que el pedido.
    let err = FanOut::from_carry_over(&[THETA, THETA], &[a.clone(), b]).unwrap_err();
    assert_eq!(
        err,
        FanOutCarryError::Detector {
            index: 1,
            source: CarryOverError::ThetaMismatch {
                expected: THETA,
                found: 5_000_000,
            },
        }
    );
    assert!(err.to_string().starts_with("theta en la posición 1"));
    let stale = CarryOver {
        state_version: "0.1.0".into(),
        ..a.clone()
    };
    assert!(matches!(
        FanOut::from_carry_over(&[THETA], &[stale]),
        Err(FanOutCarryError::Detector {
            index: 0,
            source: CarryOverError::VersionMismatch { .. }
        })
    ));
    assert!(FanOut::from_carry_over(&[], &[])
        .unwrap()
        .detectors()
        .is_empty());
    assert!(FanOut::from_carry_over(&[SCALE], &[a]).is_err());
}

#[test]
fn fanout_carry_overs_require_finish() {
    let ticks = hand_series();
    let (p, t, i): (Vec<i64>, Vec<i64>, Vec<i64>) = ticks[..3]
        .iter()
        .map(|k| (k.price, k.time, k.agg_trade_id))
        .fold((vec![], vec![], vec![]), |mut acc, (p, t, i)| {
            acc.0.push(p);
            acc.1.push(t);
            acc.2.push(i);
            acc
        });
    let mut fan = FanOut::new(&[THETA, THETA]).unwrap();
    fan.feed_batch(&p, &t, &i);
    assert_eq!(
        fan.carry_overs().unwrap_err(),
        FanOutCarryError::Detector {
            index: 0,
            source: CarryOverError::OpenGroup,
        }
    );
    fan.finish();
    let carry = fan.carry_overs().unwrap();
    assert_eq!(carry.len(), 2);
    assert_eq!(carry[0], carry[1]);
}
