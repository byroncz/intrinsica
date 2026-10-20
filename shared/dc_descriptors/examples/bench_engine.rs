//! Mide el motor sobre una serie sintética: `cargo run --release -p dc_descriptors
//! --example bench_engine -- [ticks] [row_group] [thetas] [hilos...]`.
//!
//! Imprime, por número de hilos, la pared, los ns por tick y θ y la fracción de la cola.
#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use dc_descriptors::RowGroupInput;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let number =
        |i: usize, default: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(default);
    let (ticks, row_group, thetas) = (number(0, 4_000_000), number(1, 1 << 20), number(2, 50));
    let threads: Vec<usize> = if args.len() > 3 {
        args[3..].iter().filter_map(|a| a.parse().ok()).collect()
    } else {
        vec![1, 2, 4]
    };

    let mut rng = Rng::new(42);
    let mut series = Series::random(&mut rng, ticks, false);
    let events: Vec<_> = (0..thetas)
        .map(|t| events(&mut rng, &mut series, 1 + (t as u64 % 5), 50))
        .collect();
    let sizes: Vec<_> = events
        .iter()
        .map(|e| skeleton(&mut rng, e.len(), 100_000))
        .collect();
    let columns = Columns::new(&series);
    let bounds: Vec<_> = (0..ticks.div_ceil(row_group))
        .map(|k| (k * row_group, ((k + 1) * row_group).min(ticks)))
        .collect();
    let total: usize = events.iter().map(Vec::len).sum();
    println!(
        "{ticks} ticks, {} row groups, {thetas} θ, {total} eventos",
        bounds.len()
    );

    for threads in threads {
        // Las filas de L2 se arman antes de medir: aquí solo cuenta el motor.
        let engine = engine(threads, &sizes);
        for t in 0..thetas {
            for batch in batches(&series, &events[t], &sizes[t]) {
                engine.push_boundaries(t, batch).expect("push_boundaries");
            }
        }
        let started = Instant::now();
        for (k, &(lo, hi)) in bounds.iter().enumerate() {
            engine
                .push(RowGroupInput {
                    id: k as u64,
                    ticks: columns.row_group(lo, hi),
                    before_month: false,
                })
                .expect("push");
        }
        engine.finish().expect("finish");
        let wall = started.elapsed().as_secs_f64();
        println!(
            "{threads} hilos: {wall:.2} s, {:.2} ns por tick y θ, cola con ítems {:.0} %",
            wall * threads as f64 * 1e9 / (ticks * thetas) as f64,
            engine.busy() * 100.0
        );
    }
}
