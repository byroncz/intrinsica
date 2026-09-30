//! Fan-out de N θ (ITSC-241): sus eventos por θ deben ser idénticos a los de
//! un `Detector` de ese θ alimentado tick a tick, sin importar cómo se parta
//! la serie en lotes ni cuántos hilos haya.

mod common;

use common::{fifty_thetas, series};
use dc_core::{Detector, Event, FanOut, Point, ThetaError, SCALE};

/// Referencia: un `Detector` por θ, tick a tick, con `finish` al final.
fn reference(thetas: &[i64], p: &[i64], t: &[i64], i: &[i64]) -> (Vec<Vec<Event>>, Vec<Detector>) {
    let mut detectors: Vec<Detector> = thetas.iter().map(|&t| Detector::new(t).unwrap()).collect();
    let mut events = vec![Vec::new(); thetas.len()];
    for k in 0..p.len() {
        let tick = Point {
            price: p[k],
            time: t[k],
            agg_trade_id: i[k],
        };
        for (d, ev) in detectors.iter_mut().zip(&mut events) {
            ev.extend(d.feed(tick));
        }
    }
    for (d, ev) in detectors.iter_mut().zip(&mut events) {
        ev.extend(d.finish());
    }
    (events, detectors)
}

/// Parte la serie en lotes de los tamaños de `sizes` (cíclicos) y devuelve
/// los eventos acumulados por θ, con `finish` al final.
fn run_fanout(
    fanout: &mut FanOut,
    sizes: &[usize],
    p: &[i64],
    t: &[i64],
    i: &[i64],
) -> Vec<Vec<Event>> {
    let mut events = vec![Vec::new(); fanout.detectors().len()];
    let (mut start, mut k) = (0, 0);
    while start < p.len() {
        let end = (start + sizes[k % sizes.len()]).min(p.len());
        let batch = fanout.feed_batch(&p[start..end], &t[start..end], &i[start..end]);
        for (acc, new) in events.iter_mut().zip(batch) {
            acc.extend(new);
        }
        start = end;
        k += 1;
    }
    for (acc, last) in events.iter_mut().zip(fanout.finish()) {
        acc.extend(last);
    }
    events
}

#[test]
fn fifty_thetas_match_detector_with_variable_batches() {
    let thetas = fifty_thetas();
    let (p, t, i) = series(300_000, 7);
    let (expected, detectors) = reference(&thetas, &p, &t, &i);
    // Hay eventos de sobra para que la comparación diga algo.
    assert!(expected.iter().filter(|e| e.len() > 10).count() >= 40);

    // Lotes de un tick, chicos (caen en la ruta secuencial), medianos y de
    // varios trozos (ruta con hilos), mezclados.
    let sizes = [1, 1, 2, 3, 7, 100, 999, 1, 4_096, 4_097, 20_000, 1, 65_536];
    let mut fanout = FanOut::new(&thetas).unwrap();
    let got = run_fanout(&mut fanout, &sizes, &p, &t, &i);

    assert_eq!(got, expected);
    for (d, r) in fanout.detectors().iter().zip(&detectors) {
        assert_eq!(d.state(), r.state());
        assert_eq!(d.discarded(), r.discarded());
        assert_eq!(d.theta(), r.theta());
    }
}

#[test]
fn result_does_not_depend_on_thread_count() {
    let thetas = fifty_thetas();
    let (p, t, i) = series(120_000, 11);
    let (expected, _) = reference(&thetas, &p, &t, &i);
    // 1 hilo, uno que no divide 50, uno que sí, y más hilos que detectores.
    for threads in [1, 3, 10, 64] {
        let mut fanout = FanOut::with_threads(&thetas, threads).unwrap();
        let got = run_fanout(&mut fanout, &[10_000], &p, &t, &i);
        assert_eq!(got, expected, "con {threads} hilos");
    }
}

#[test]
fn events_follow_the_order_of_the_input_thetas() {
    // Desordenados y con un θ repetido: cada posición conserva su θ.
    let thetas = [2_000_000, 50_000, 1_000_000, 50_000];
    let (p, t, i) = series(50_000, 3);
    let (expected, _) = reference(&thetas, &p, &t, &i);
    let mut fanout = FanOut::new(&thetas).unwrap();
    let got = run_fanout(&mut fanout, &[5_000], &p, &t, &i);
    assert_eq!(got, expected);
    assert_eq!(got[1], got[3]);
    assert_ne!(got[0], got[1]);
    let order: Vec<i64> = fanout.detectors().iter().map(Detector::theta).collect();
    assert_eq!(order, thetas);
}

#[test]
fn new_rejects_invalid_theta() {
    assert_eq!(FanOut::new(&[10_000, 0]).unwrap_err(), ThetaError(0));
    assert_eq!(
        FanOut::with_threads(&[SCALE], 4).unwrap_err(),
        ThetaError(SCALE)
    );
}

#[test]
fn empty_inputs_are_harmless() {
    let mut none = FanOut::new(&[]).unwrap();
    assert!(none.feed_batch(&[100], &[1], &[1]).is_empty());
    assert!(none.finish().is_empty());

    let mut one = FanOut::new(&[10_000_000]).unwrap();
    assert_eq!(one.feed_batch(&[], &[], &[]), vec![Vec::<Event>::new()]);
    assert_eq!(one.finish(), vec![None]);
}

#[test]
#[should_panic(expected = "columnas de distinto largo")]
fn mismatched_columns_panic() {
    let mut fanout = FanOut::new(&[10_000_000]).unwrap();
    fanout.feed_batch(&[100, 101], &[1, 2], &[1]);
}
