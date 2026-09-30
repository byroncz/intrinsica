//! Benchmark de ticks/s por core (ITSC-241), sobre la serie sintética de
//! `tests/common`. Ejecutar en release:
//!
//! ```text
//! cargo run --release -p dc_core --example bench_fanout -- [ticks] [lote] [corridas]
//! ```
//!
//! Por defecto 10 M de ticks, lotes de 65 536 y 3 corridas por caso (se
//! reporta la mejor). La serie se genera lote a lote fuera del cronómetro:
//! la memoria es O(lote) también aquí.

#[path = "../tests/common/mod.rs"]
mod common;

use common::{fifty_thetas, Synthetic};
use dc_core::{Detector, FanOut, Point};
use std::env;
use std::time::{Duration, Instant};

const SEED: u64 = 42;

struct Config {
    ticks: usize,
    batch: usize,
    runs: usize,
}

/// Recorre la serie por lotes; `feed` procesa uno y devuelve los eventos que
/// cerró. Solo cronometra `feed`. Devuelve (tiempo, eventos).
fn timed(cfg: &Config, mut feed: impl FnMut(&[i64], &[i64], &[i64]) -> usize) -> (Duration, usize) {
    let mut series = Synthetic::new(SEED);
    let (mut p, mut t, mut i) = (Vec::new(), Vec::new(), Vec::new());
    let (mut elapsed, mut events, mut left) = (Duration::ZERO, 0, cfg.ticks);
    while left > 0 {
        let n = left.min(cfg.batch);
        series.fill(n, &mut p, &mut t, &mut i);
        let start = Instant::now();
        events += feed(&p, &t, &i);
        elapsed += start.elapsed();
        left -= n;
    }
    (elapsed, events)
}

fn best(cfg: &Config, mut run: impl FnMut() -> (Duration, usize)) -> (Duration, usize) {
    (0..cfg.runs)
        .map(|_| run())
        .min_by_key(|(elapsed, _)| *elapsed)
        .expect("al menos una corrida")
}

fn row(
    label: &str,
    cfg: &Config,
    thetas: usize,
    cores: usize,
    (elapsed, events): (Duration, usize),
) {
    let secs = elapsed.as_secs_f64();
    let ticks_s = cfg.ticks as f64 / secs;
    println!(
        "{label:<28} {thetas:>3} θ {cores:>2} hilos {:>8.2} s {:>7.2} M ticks/s {:>7.2} M ticks/s/core {:>7.2} M θ·ticks/s/core {events:>9} eventos",
        secs,
        ticks_s / 1e6,
        ticks_s / cores as f64 / 1e6,
        ticks_s * thetas as f64 / cores as f64 / 1e6,
    );
}

/// Base: N `Detector` en un solo hilo, tick a tick, sin fan-out.
fn sequential(cfg: &Config, thetas: &[i64]) -> (Duration, usize) {
    let mut detectors: Vec<Detector> = thetas.iter().map(|&t| Detector::new(t).unwrap()).collect();
    timed(cfg, |p, t, i| {
        let mut events = 0;
        for k in 0..p.len() {
            let tick = Point {
                price: p[k],
                time: t[k],
                agg_trade_id: i[k],
            };
            events += detectors.iter_mut().filter_map(|d| d.feed(tick)).count();
        }
        events
    })
}

fn fanout(cfg: &Config, thetas: &[i64], threads: usize) -> (Duration, usize) {
    let mut fanout = FanOut::with_threads(thetas, threads).unwrap();
    timed(cfg, |p, t, i| {
        fanout.feed_batch(p, t, i).iter().map(Vec::len).sum()
    })
}

fn main() {
    let arg = |n: usize, default: usize| {
        env::args()
            .nth(n)
            .map_or(default, |s| s.parse().expect("argumento numérico"))
    };
    let cfg = Config {
        ticks: arg(1, 10_000_000),
        batch: arg(2, 65_536),
        runs: arg(3, 3),
    };
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let thetas = fifty_thetas();
    println!(
        "{} ticks sintéticos, lotes de {}, mejor de {} corridas, {cores} núcleos disponibles\n",
        cfg.ticks, cfg.batch, cfg.runs
    );

    let one = &thetas[..1];
    row(
        "Detector, 1 hilo",
        &cfg,
        1,
        1,
        best(&cfg, || sequential(&cfg, one)),
    );
    row(
        "FanOut, 1 hilo",
        &cfg,
        1,
        1,
        best(&cfg, || fanout(&cfg, one, 1)),
    );
    row(
        "Detector x50, 1 hilo",
        &cfg,
        50,
        1,
        best(&cfg, || sequential(&cfg, &thetas)),
    );
    let mut threads = vec![1, 2, 5, cores];
    threads.retain(|&n| n <= cores);
    threads.dedup();
    for n in threads {
        row(
            "FanOut x50",
            &cfg,
            50,
            n,
            best(&cfg, || fanout(&cfg, &thetas, n)),
        );
    }
}
