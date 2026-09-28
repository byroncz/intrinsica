//! Equivalencia contra la v0 (ITSC-240): el detector Rust contra los eventos
//! que `segment_events_kernel` (tag `v0.2.0-legacy`) calculó sobre un día real
//! de la landing (2017-08-18, BTCUSDT spot). Ver `fixtures/README.md`.
//!
//! Toda discrepancia queda clasificada. Con θ = 2 % no hay ninguna. En los
//! demás θ hay ventanas de discrepancia que abren siempre en la confirmación
//! de un instante con más de un tick: la divergencia declarada de ADR-L2-04
//! (la v0 sigue procesando los ticks del instante de confirmación, L2 no). No
//! hay bug del detector ni tick de frontera por redondeo float en este día; de
//! haberlos, la prueba falla porque no los explica ningún instante.

use dc_core::{Detector, Direction, Point};
use std::collections::HashMap;

const TICKS: &str = include_str!("fixtures/ticks.csv");
const EVENTS: &str = include_str!("fixtures/events_v0.csv");
const FINAL: &str = include_str!("fixtures/final_state_v0.csv");

/// Evento comparable: `extreme` es `None` mientras no confirma el siguiente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ev {
    direction: Direction,
    reference: Point,
    confirm: Point,
    extreme: Option<Point>,
}

fn price(s: &str) -> i64 {
    let (int, frac) = s.split_once('.').expect("precio con punto decimal");
    assert_eq!(frac.len(), 8, "precio con 8 decimales: {s}");
    int.parse::<i64>().unwrap() * 100_000_000 + frac.parse::<i64>().unwrap()
}

fn rows(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines().skip(1).map(|l| l.split(',').collect())
}

fn ticks() -> Vec<Point> {
    rows(TICKS)
        .map(|r| Point {
            agg_trade_id: r[0].parse().unwrap(),
            price: price(r[1]),
            time: r[2].parse().unwrap(),
        })
        .collect()
}

fn point(price_: &str, time: &str, id: &str) -> Point {
    Point {
        price: price(price_),
        time: time.parse().unwrap(),
        agg_trade_id: id.parse().unwrap(),
    }
}

/// Eventos de la v0 por θ, en orden.
fn v0_events() -> HashMap<i64, Vec<Ev>> {
    let mut out: HashMap<i64, Vec<Ev>> = HashMap::new();
    for r in rows(EVENTS) {
        let extreme = (!r[9].is_empty()).then(|| point(r[9], r[10], r[11]));
        out.entry(r[0].parse().unwrap()).or_default().push(Ev {
            direction: if r[2] == "1" {
                Direction::Up
            } else {
                Direction::Down
            },
            reference: point(r[3], r[4], r[5]),
            confirm: point(r[6], r[7], r[8]),
            extreme,
        });
    }
    out
}

/// Eventos del detector Rust: los cerrados más el pendiente, sin extremo.
fn rust_events(theta: i64, ticks: &[Point]) -> (Vec<Ev>, Detector) {
    let mut d = Detector::new(theta).unwrap();
    let mut out = Vec::new();
    for t in ticks {
        out.extend(d.feed(*t));
    }
    out.extend(d.finish());
    let mut evs: Vec<Ev> = out
        .iter()
        .map(|e| Ev {
            direction: e.direction,
            reference: e.reference,
            confirm: e.confirm,
            extreme: Some(e.extreme),
        })
        .collect();
    let p = d.state().pending.expect("el día tiene al menos un evento");
    evs.push(Ev {
        direction: p.direction,
        reference: p.reference,
        confirm: p.confirm,
        extreme: None,
    });
    (evs, d)
}

/// Resultado de alinear los eventos de las dos implementaciones.
#[derive(Debug)]
enum Item {
    Same,
    /// Mismo evento (dirección y `agg_trade_id` de confirmación), campos distintos.
    Fields(Ev),
    /// Evento que solo emite Rust.
    RustOnly(Ev),
    /// Evento que solo emite la v0.
    V0Only(Ev),
}

fn align(rust: &[Ev], v0: &[Ev]) -> Vec<Item> {
    let (mut a, mut b, mut out) = (0, 0, Vec::new());
    while a < rust.len() || b < v0.len() {
        match (rust.get(a), v0.get(b)) {
            (Some(r), Some(v))
                if r.direction == v.direction
                    && r.confirm.agg_trade_id == v.confirm.agg_trade_id =>
            {
                out.push(if r == v { Item::Same } else { Item::Fields(*r) });
                a += 1;
                b += 1;
            }
            (Some(r), Some(v)) if r.confirm.agg_trade_id < v.confirm.agg_trade_id => {
                out.push(Item::RustOnly(*r));
                a += 1;
            }
            (Some(r), None) => {
                out.push(Item::RustOnly(*r));
                a += 1;
            }
            (_, Some(v)) => {
                out.push(Item::V0Only(*v));
                b += 1;
            }
            (None, None) => unreachable!(),
        }
    }
    out
}

/// Ventanas de discrepancia: rachas de ítems distintos de `Same`. Devuelve,
/// por ventana, el evento con el que empieza (el primero que se aparta).
fn windows(items: &[Item]) -> Vec<Ev> {
    let mut out = Vec::new();
    let mut open = false;
    for item in items {
        let first = match item {
            Item::Same => {
                open = false;
                continue;
            }
            Item::Fields(r) | Item::RustOnly(r) => r,
            Item::V0Only(v) => v,
        };
        if !open {
            out.push(*first);
            open = true;
        }
    }
    out
}

/// Ventanas de discrepancia que se esperan, por θ: `agg_trade_id` de la
/// confirmación con la que se abre cada una. Son exactamente las que la
/// política de ADR-L2-04 declara divergencia intencional (instante de
/// confirmación atómico; ver `mecanismo`). Ninguna es un bug del detector ni
/// un tick de frontera por redondeo float.
const VENTANAS: [(i64, &[i64]); 5] = [
    (
        100_000,
        &[
            3259, 3400, 4302, 4858, 5299, 5634, 5922, 5973, 6415, 6564, 6726, 7100, 7160, 7582,
        ],
    ),
    (
        250_000,
        &[3472, 4302, 5299, 5634, 5922, 5973, 6415, 6564, 7100, 7582],
    ),
    (500_000, &[4302, 7100, 7149]),
    (1_000_000, &[3259, 3334, 3397, 4626, 7100, 7149, 7445]),
    (2_000_000, &[]),
];

/// Una ventana abre siempre en la confirmación de un instante con más de un
/// tick, y divergen porque la v0 sigue procesando los ticks de ese instante
/// después de confirmar (`kernel.py` retoma el bucle en `t+1`) y L2 no
/// (ADR-L2-03 (2), ADR-L2-04, §8.1 paso 4). Devuelve el mecanismo, o `None`
/// si el instante no puede explicar la divergencia (entonces es un bug del
/// detector o un tick de frontera y la prueba falla):
///
/// - `extremo`: un tick posterior del instante es más extremo que
///   `confirm_price` (más alto en upturn, más bajo en downturn). La v0 mueve
///   el extremo reiniciado; L2 lo deja en `confirm_price`. ADR-L2-04 (b).
/// - `terna`: ningún tick posterior lo supera, pero el último tick del
///   instante tiene otro precio que `confirm_price`. La v0 publica el precio
///   real de ese tick como referencia/extremo y L2 el `confirm_price`
///   (ADR-L2-03, "Terna del extremo reiniciado"). Incluye la reversión
///   dentro del instante, ADR-L2-04 (a).
fn mecanismo(ticks: &[Point], e: &Ev) -> Option<&'static str> {
    let group: Vec<&Point> = ticks.iter().filter(|t| t.time == e.confirm.time).collect();
    let later = group.get(1..)?;
    let beyond = |t: &&Point| match e.direction {
        Direction::Up => t.price > e.confirm.price,
        Direction::Down => t.price < e.confirm.price,
    };
    if later.iter().any(beyond) {
        Some("extremo")
    } else if later.last().is_some_and(|t| t.price != e.confirm.price) {
        Some("terna")
    } else {
        None
    }
}

#[test]
fn eventos_iguales_salvo_las_divergencias_declaradas() {
    let ticks = ticks();
    let v0 = v0_events();
    assert_eq!(
        v0.len(),
        VENTANAS.len(),
        "el fixture trae los mismos θ que la tabla"
    );
    for (theta, esperadas) in VENTANAS {
        let (rust, _) = rust_events(theta, &ticks);
        let items = align(&rust, &v0[&theta]);
        let abiertas = windows(&items);
        let ids: Vec<i64> = abiertas.iter().map(|w| w.confirm.agg_trade_id).collect();
        assert_eq!(ids, esperadas, "θ={theta}: ventanas de discrepancia");
        for w in &abiertas {
            assert!(
                mecanismo(&ticks, w).is_some(),
                "θ={theta}: la discrepancia que abre en la confirmación {} no la explica el instante de confirmación: {w:?}",
                w.confirm.agg_trade_id
            );
        }
        // Una divergencia se reabsorbe pronto: si una ventana creciera, el
        // detector estaría arrastrando un estado distinto sin explicación.
        let mut run = 0;
        for item in &items {
            run = if matches!(item, Item::Same) {
                0
            } else {
                run + 1
            };
            assert!(run <= 3, "θ={theta}: ventana de más de 3 eventos");
        }
    }
}

/// Con θ = 2 % ningún instante de confirmación tiene ticks que muevan el
/// extremo: los 12 eventos del día, con su pendiente, coinciden campo a campo.
#[test]
fn theta_2_por_ciento_es_identico_evento_a_evento() {
    let ticks = ticks();
    let (rust, _) = rust_events(2_000_000, &ticks);
    assert_eq!(rust, v0_events()[&2_000_000]);
}

/// Estado final: tendencia, ambos extremos, primer huérfano y última
/// confirmación. Coincide con la v0 en los 5 θ. `n_events` no se compara: la
/// cuenta de eventos ya está cubierta arriba y difiere donde hay ventanas.
#[test]
fn estado_final_igual_a_la_v0() {
    let ticks = ticks();
    let position = |id: i64| ticks.iter().position(|t| t.agg_trade_id == id).unwrap();
    let mut seen = 0;
    for r in rows(FINAL) {
        let theta: i64 = r[0].parse().unwrap();
        let (_, d) = rust_events(theta, &ticks);
        let state = d.state();
        let ex = state.extremes.expect("hay ticks");
        let direction = if r[2] == "1" {
            Direction::Up
        } else {
            Direction::Down
        };
        assert_eq!(
            state.direction,
            Some(direction),
            "θ={theta}: tendencia final"
        );
        assert_eq!(ex.high.price, price(r[3]), "θ={theta}: extremo alto");
        assert_eq!(ex.low.price, price(r[4]), "θ={theta}: extremo bajo");
        // `last_os_ref` de la v0 es el precio de la última confirmación.
        assert_eq!(
            state.pending.unwrap().confirm.price,
            price(r[5]),
            "θ={theta}: última confirmación"
        );
        // La v0 informa el primer tick posterior al extremo de la tendencia.
        let extreme = if direction == Direction::Up {
            ex.high
        } else {
            ex.low
        };
        assert_eq!(
            position(extreme.agg_trade_id) + 1,
            r[6].parse::<usize>().unwrap(),
            "θ={theta}: primer huérfano"
        );
        assert_eq!(d.discarded(), 0, "θ={theta}: ningún DC vacío descartado");
        seen += 1;
    }
    assert_eq!(seen, VENTANAS.len());
}
