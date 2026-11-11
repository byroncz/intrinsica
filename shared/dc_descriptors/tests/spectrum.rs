//! La familia F (`spectrum.parquet`) del motor: valores calculados a mano, el presupuesto de
//! ticks, la fusión ordenada de parciales en todos los cortes y órdenes de `push`, los retornos
//! en vuelo y un oráculo ingenuo (DFT de O(n²) y Haar recursivo, sin `rustfft`) sobre series
//! aleatorias.

mod common;

use common::{batches, skeleton, Columns, Ev, Rng, Series, Shared};
use dc_descriptors::{
    spectrum_col as sc, Config, Engine, EngineError, EventErrorKind, Family, OverBudget,
    RowGroupInput, SealedRowGroup, TickError, SPECTRUM_COLUMNS, SPECTRUM_WIDTH,
};
use std::f64::consts::PI;
use std::time::Duration;

const BUDGET: u64 = 20_000_000;
/// Las bandas de una fase: siete fracciones, nulas donde la banda no tiene bins.
type Fractions = [Option<f64>; 7];

/// Una fila de salida: los 64 bits de cada celda, o `None` si es nula.
type Out = [Option<i64>; SPECTRUM_WIDTH];

fn decode(sealed: &SealedRowGroup) -> Vec<Out> {
    (0..sealed.rows)
        .map(|r| {
            let mut row: Out = [None; SPECTRUM_WIDTH];
            for (c, cell) in row.iter_mut().enumerate() {
                let at = r * 8;
                let value = i64::from_le_bytes(sealed.columns[c][at..at + 8].try_into().unwrap());
                let present = match &sealed.masks[c] {
                    Some(mask) => (mask.bits[r >> 3] >> (r & 7)) & 1 == 1,
                    None => true,
                };
                assert!(
                    present || value == 0,
                    "una celda nula de la columna {c} no vale 0"
                );
                assert!(
                    present || SPECTRUM_COLUMNS[c].nullable,
                    "la columna {c} no admite nulos"
                );
                *cell = present.then_some(value);
            }
            row
        })
        .collect()
}

/// Los siete valores de un registro de una fila que empieza en la columna `base`.
fn register(row: &Out, base: usize) -> Fractions {
    let mut out = [None; 7];
    for (band, slot) in out.iter_mut().enumerate() {
        *slot = row[base + band].map(|bits| f64::from_bits(bits as u64));
    }
    out
}

fn near(got: &Fractions, wanted: &Fractions, tolerance: f64, what: &str) {
    for (band, (g, w)) in got.iter().zip(wanted).enumerate() {
        match (g, w) {
            (None, None) => {}
            (Some(g), Some(w)) => assert!(
                (g - w).abs() <= tolerance,
                "{what}, banda {}: {g} y debería ser {w}",
                band + 1
            ),
            other => panic!("{what}, banda {}: {other:?}", band + 1),
        }
    }
}

/// Una serie de ticks con los precios dados: ids 1, 2, ..., tiempos crecientes.
fn series(prices: &[i64]) -> Series {
    Series {
        ids: (1..=prices.len() as i64).collect(),
        time: (0..prices.len() as i64).map(|i| 1_000 + 10 * i).collect(),
        price: prices.to_vec(),
        qty: vec![1; prices.len()],
        sold: vec![false; prices.len()],
    }
}

/// Un evento: la confirmación son los ticks `[r, c)` y el overshoot `[c, e)`. La dirección sale
/// del precio del tick `R` y el de `C`; el ancla de `c` es el precio del tick `R`.
fn event(s: &Series, r: usize, c: usize, e: usize) -> Ev {
    let confirm_price = s.price[c - 1];
    Ev {
        r,
        c,
        e,
        direction: if confirm_price >= s.price[r - 1] {
            1
        } else {
            -1
        },
        confirm_price,
        extreme_price: s.price[e.max(c) - 1],
    }
}

fn bounds_of(cuts: &[usize], len: usize) -> Vec<(usize, usize)> {
    let mut bounds = Vec::new();
    let mut lo = 0;
    for &cut in cuts.iter().chain([&len]) {
        bounds.push((lo, cut));
        lo = cut;
    }
    bounds
}

fn spectrum_engine(threads: usize, rows: &[usize], budget_ticks: u64) -> Engine<Shared> {
    Engine::new(Config {
        thetas: vec![100_000],
        threads,
        row_groups: vec![rows.to_vec()],
        family: Family::Spectrum,
        silence_us: 0,
        budget_ticks,
    })
    .expect("configuración válida")
}

struct Run {
    rows: Vec<Out>,
    over_budget: Vec<OverBudget>,
}

/// Corre los eventos sobre `series` cortada en `cuts` (cuentas de ticks que inician un row
/// group), con los row groups empujados en `order`, y devuelve sus filas y los avisos.
#[allow(clippy::too_many_arguments)]
fn run_spectrum(
    series: &Series,
    evs: &[Ev],
    cuts: &[usize],
    order: &[usize],
    threads: usize,
    lots: &[usize],
    rows: &[usize],
    budget_ticks: u64,
) -> Result<Run, EngineError> {
    let bounds = bounds_of(cuts, series.len());
    let engine = spectrum_engine(threads, rows, budget_ticks);
    let columns = Columns::new(series);
    for batch in batches(series, evs, lots) {
        engine.push_boundaries(0, batch)?;
    }
    let mut run = Run {
        rows: Vec::new(),
        over_budget: Vec::new(),
    };
    let mut take = |engine: &Engine<Shared>| -> Result<(), EngineError> {
        while let Some(sealed) = engine.pop_sealed()? {
            run.rows.extend(decode(&sealed));
            run.over_budget.extend(sealed.over_budget.iter().copied());
        }
        Ok(())
    };
    for &k in order {
        let (lo, hi) = bounds[k];
        engine.push(RowGroupInput {
            id: k as u64,
            ticks: columns.row_group(lo, hi),
            before_month: false,
        })?;
        take(&engine)?;
    }
    engine.finish()?;
    take(&engine)?;
    assert_eq!(engine.buffered_ticks(), 0, "el búfer se libera al cerrar");
    Ok(run)
}

fn run_one(series: &Series, ev: &Ev, cuts: &[usize], order: &[usize]) -> Out {
    let run = run_spectrum(
        series,
        std::slice::from_ref(ev),
        cuts,
        order,
        2,
        &[1],
        &[1],
        BUDGET,
    )
    .expect("el evento cierra");
    assert_eq!(run.rows.len(), 1);
    run.rows[0]
}

fn whole(series: &Series, ev: &Ev) -> Out {
    run_one(series, ev, &[], &[0])
}

// ---------------------------------------------------------------------------------------------
// El oráculo ingenuo: sin `rustfft`, sin compartir una línea con el motor.
// ---------------------------------------------------------------------------------------------

/// Los retornos de la fase `[from, to)` con el ancla dada, redondeados a `f32` como el búfer.
fn naive_returns(s: &Series, from: usize, to: usize, anchor: i64) -> Vec<f64> {
    let mut previous = anchor;
    (from..to)
        .map(|i| {
            // ln(1 + Δp/p₀): el cociente de precios de 10¹² con pasos de unidades pierde 4 dígitos.
            let r = (((s.price[i] - previous) as f64) / (previous as f64)).ln_1p() as f32;
            previous = s.price[i];
            f64::from(r)
        })
        .collect()
}

fn naive_mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Fracciones por banda con la DFT directa.
fn naive_fourier(x: &[f64]) -> Fractions {
    let n = x.len();
    if x.iter().all(|&v| v == x[0]) {
        return [None; 7];
    }
    let mean = naive_mean(x);
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(j, v)| (v - mean) * (0.5 - 0.5 * (2.0 * PI * j as f64 / n as f64).cos()))
        .collect();
    let mut bands = [0.0f64; 7];
    let mut has = [false; 7];
    for k in 1..=n / 2 {
        let (mut re, mut im) = (0.0, 0.0);
        for (j, v) in y.iter().enumerate() {
            let angle = 2.0 * PI * ((j * k) % n) as f64 / n as f64;
            re += v * angle.cos();
            im -= v * angle.sin();
        }
        // El período n/k cae en [2^b, 2^(b+1)) con la aritmética entera de `n` y `k`.
        let band = (1..=6)
            .find(|b| n >= (1usize << b) * k && n < (1usize << (b + 1)) * k)
            .unwrap_or(7);
        bands[band - 1] += re * re + im * im;
        has[band - 1] = true;
    }
    let total: f64 = bands.iter().sum();
    let mut out = [None; 7];
    for b in 0..7 {
        out[b] = has[b].then(|| bands[b] / total);
    }
    out
}

fn naive_haar(x: &[f64]) -> Fractions {
    let n = x.len();
    let length = 1usize << (usize::BITS - 1 - n.leading_zeros());
    let tail = &x[n - length..];
    if tail.iter().all(|&v| v == tail[0]) {
        return [None; 7];
    }
    let mean = naive_mean(tail);
    let mut approx: Vec<f64> = tail.iter().map(|v| v - mean).collect();
    let mut levels = Vec::new();
    while approx.len() > 1 {
        let detail: f64 = approx
            .chunks(2)
            .map(|p| ((p[0] - p[1]) / 2f64.sqrt()).powi(2))
            .sum();
        approx = approx
            .chunks(2)
            .map(|p| (p[0] + p[1]) / 2f64.sqrt())
            .collect();
        levels.push(detail);
    }
    let mut bands = [0.0f64; 7];
    for (level, energy) in levels.iter().enumerate() {
        bands[level.min(6)] += energy;
    }
    let total: f64 = bands.iter().sum();
    let mut out = [None; 7];
    for b in 0..levels.len().min(7) {
        out[b] = Some(bands[b] / total);
    }
    out
}

/// La fila que dice el oráculo para el evento `ev` sobre `s`: los cuatro registros.
fn naive_row(s: &Series, ev: &Ev) -> [Fractions; 4] {
    let anchor = ev.reference_price(s);
    let mut out = [[None; 7]; 4];
    for (i, (from, to, anchor)) in [(ev.r, ev.c, anchor), (ev.c, ev.e, ev.confirm_price)]
        .into_iter()
        .enumerate()
    {
        if to - from < 16 {
            continue;
        }
        let x = naive_returns(s, from, to, anchor);
        out[i] = naive_fourier(&x);
        out[2 + i] = naive_haar(&x);
    }
    out
}

fn registers(row: &Out) -> [Fractions; 4] {
    [
        register(row, sc::C_FOURIER),
        register(row, sc::OS_FOURIER),
        register(row, sc::C_HAAR),
        register(row, sc::OS_HAAR),
    ]
}

// ---------------------------------------------------------------------------------------------
// Valores a mano
// ---------------------------------------------------------------------------------------------

/// Una fase con retornos `a, −a, a, ...` (64 ticks de 1010 y 1000 tras el tick `R` de 1000).
fn alternating() -> (Series, Ev) {
    let mut prices = vec![1_000i64];
    prices.extend((0..64).map(|k| if k % 2 == 0 { 1_010 } else { 1_000 }));
    let s = series(&prices);
    let mut ev = event(&s, 1, 65, 65);
    // El último tick vuelve a 1000: la dirección y `confirm_price` solo fijan el ancla.
    ev.direction = 1;
    ev.confirm_price = 1_500;
    (s, ev)
}

#[test]
fn alternating_returns_put_all_the_energy_in_the_first_band() {
    let (s, ev) = alternating();
    let row = whole(&s, &ev);
    assert_eq!(row[sc::CONFIRM_AGG_TRADE_ID], Some(65));
    let one_then_zeros = [
        Some(1.0),
        Some(0.0),
        Some(0.0),
        Some(0.0),
        Some(0.0),
        Some(0.0),
        None,
    ];
    near(
        &register(&row, sc::C_FOURIER),
        &one_then_zeros,
        1e-6,
        "c_fourier",
    );
    // Haar: los pares valen a y −a, todo en el nivel 1.
    near(&register(&row, sc::C_HAAR), &one_then_zeros, 1e-6, "c_haar");
    // Overshoot vacío: ninguna de sus 14 columnas existe.
    for base in [sc::OS_FOURIER, sc::OS_HAAR] {
        assert_eq!(register(&row, base), [None; 7]);
    }
}

#[test]
fn a_sine_of_period_8_over_64_returns_leaves_five_sixths_and_one_sixth() {
    // Retornos A·cos(2πk/8): precios que los siguen sobre un precio de 5·10¹² (el redondeo del
    // entero es de 10⁻¹³ relativo, 10⁻¹⁰ de la amplitud).
    let amplitude = 1e-3f64;
    let mut prices = vec![5_000_000_000_000i64];
    let mut level = (prices[0] as f64).ln();
    for k in 1..=64 {
        level += amplitude * (2.0 * PI * k as f64 / 8.0).cos();
        prices.push(level.exp().round() as i64);
    }
    let s = series(&prices);
    let mut ev = event(&s, 1, 65, 65);
    ev.direction = 1;
    ev.confirm_price = 2 * prices[0];
    let row = whole(&s, &ev);
    // Hann deja 2/3 en el bin 8 y 1/6 en cada vecino: banda 3 = 5/6 y banda 2 = 1/6.
    near(
        &register(&row, sc::C_FOURIER),
        &[
            Some(0.0),
            Some(1.0 / 6.0),
            Some(5.0 / 6.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            None,
        ],
        1e-6,
        "c_fourier",
    );
    // Haar: la senoide de período 8 vive en las escalas 4 y 8 (niveles 2 y 3), no en la 2.
    let haar = register(&row, sc::C_HAAR);
    let sum: f64 = haar.iter().flatten().sum();
    assert!((sum - 1.0).abs() < 1e-12, "{sum}");
    assert!(haar[0].unwrap() < 0.2, "{haar:?}");
}

/// Una caminata de `ticks` ticks con pasos de ±1..3 sobre 5·10¹².
fn walk(rng: &mut Rng, ticks: usize) -> Vec<i64> {
    let mut price = 5_000_000_000_000i64;
    (0..ticks)
        .map(|_| {
            price += rng.below(7) as i64 - 3;
            price
        })
        .collect()
}

#[test]
fn a_phase_of_15_returns_is_null_and_one_of_16_has_the_first_four_bands() {
    let mut rng = Rng::new(15);
    let prices = walk(&mut rng, 80);
    let s = series(&prices);
    for (n, os, present) in [
        (15usize, 0usize, 0usize),
        (16, 0, 4),
        (16, 15, 4),
        (15, 16, 0),
    ] {
        let ev = event(&s, 1, 1 + n, 1 + n + os);
        let row = whole(&s, &ev);
        let got = registers(&row);
        let counts: Vec<usize> = got.iter().map(|r| r.iter().flatten().count()).collect();
        let c = if n >= 16 { present } else { 0 };
        let o = if os >= 16 { 4 } else { 0 };
        assert_eq!(counts, [c, o, c, o], "n = {n}, os = {os}");
        let wanted = naive_row(&s, &ev);
        for (g, w) in got.iter().zip(&wanted) {
            near(g, w, 1e-9, &format!("n = {n}, os = {os}"));
        }
    }
}

#[test]
fn the_bands_without_bins_are_null_and_the_others_sum_to_one() {
    let mut rng = Rng::new(7);
    let prices = walk(&mut rng, 400);
    let s = series(&prices);
    for (n, present) in [
        (31, 4),
        (32, 5),
        (63, 5),
        (64, 6),
        (127, 6),
        (128, 7),
        (300, 7),
    ] {
        let ev = event(&s, 1, 1 + n, 1 + n);
        let row = whole(&s, &ev);
        for base in [sc::C_FOURIER, sc::C_HAAR] {
            let got = register(&row, base);
            let nulls: Vec<bool> = got.iter().map(Option::is_none).collect();
            let wanted: Vec<bool> = (0..7).map(|b| b >= present).collect();
            assert_eq!(nulls, wanted, "n = {n}, columna {base}");
            let sum: f64 = got.iter().flatten().sum();
            assert!((sum - 1.0).abs() < 1e-9, "n = {n}, columna {base}: {sum}");
            assert!(got.iter().flatten().all(|f| (0.0..=1.0).contains(f)));
        }
    }
}

#[test]
fn an_empty_overshoot_and_a_flat_phase_leave_their_columns_null() {
    // Confirmación plana (ningún retorno distinto de 0) de 20 ticks y overshoot con movimiento.
    let mut prices = vec![5_000_000_000_000i64; 21];
    let mut rng = Rng::new(3);
    let mut price = prices[20];
    for _ in 0..30 {
        price += rng.below(5) as i64 - 2;
        prices.push(price);
    }
    let s = series(&prices);
    let mut ev = event(&s, 1, 21, 51);
    // El ancla de `c` es el precio del tick R, plano como la fase: `confirm_price` va por encima.
    ev.direction = 1;
    ev.confirm_price = prices[0] + 1_000;
    let row = whole(&s, &ev);
    let got = registers(&row);
    assert_eq!(got[0], [None; 7], "c_fourier de una fase plana");
    assert_eq!(got[2], [None; 7], "c_haar de una fase plana");
    // 30 retornos: las bandas 1 a 4 y, por la cola de 16, los niveles 1 a 4.
    assert!(got[1].iter().flatten().count() == 4 && got[3].iter().flatten().count() == 4);
    let wanted = naive_row(&s, &ev);
    near(&got[1], &wanted[1], 1e-9, "os_fourier");
    near(&got[3], &wanted[3], 1e-9, "os_haar");
    // La confirmación plana tampoco tiene ubicación (Σ r² = 0); el overshoot sí.
    assert_eq!(locations(&row)[0], None);
    assert!(locations(&row)[1].is_some());
}

// ---------------------------------------------------------------------------------------------
// Ubicación de la energía total
// ---------------------------------------------------------------------------------------------

/// `c_energy_loc` y `os_energy_loc` de una fila.
fn locations(row: &Out) -> [Option<f64>; 2] {
    [sc::C_ENERGY_LOC, sc::OS_ENERGY_LOC].map(|c| row[c].map(|bits| f64::from_bits(bits as u64)))
}

/// La ubicación `Σ t_k r_k² / Σ r_k²` con `t_k = (k − 0,5)/n`, en una suma simple.
fn naive_location(x: &[f64]) -> Option<f64> {
    let n = x.len() as f64;
    let energy: f64 = x.iter().map(|r| r * r).sum();
    let moment: f64 = x
        .iter()
        .enumerate()
        .map(|(i, r)| (i as f64 + 0.5) / n * r * r)
        .sum();
    (energy > 0.0).then(|| moment / energy)
}

fn naive_locations(s: &Series, ev: &Ev) -> [Option<f64>; 2] {
    let anchor = ev.reference_price(s);
    [(ev.r, ev.c, anchor), (ev.c, ev.e, ev.confirm_price)].map(|(from, to, anchor)| {
        (to - from >= 16)
            .then(|| naive_location(&naive_returns(s, from, to, anchor)))
            .flatten()
    })
}

#[test]
fn the_energy_location_follows_the_oracle_and_stays_inside_zero_one() {
    let mut rng = Rng::new(21);
    let prices = walk(&mut rng, 900);
    let s = series(&prices);
    // Tamaños de fase de 16 a 300 retornos, con y sin overshoot, y el caso de 15 (nulo).
    for (n, os) in [
        (15usize, 20usize),
        (16, 0),
        (17, 16),
        (31, 33),
        (64, 15),
        (127, 128),
        (300, 300),
    ] {
        let ev = event(&s, 1, 1 + n, 1 + n + os);
        let row = whole(&s, &ev);
        let got = locations(&row);
        let wanted = naive_locations(&s, &ev);
        for (what, g, w) in [("c", got[0], wanted[0]), ("os", got[1], wanted[1])] {
            match (g, w) {
                (None, None) => {}
                (Some(g), Some(w)) => {
                    assert!((g - w).abs() <= 1e-12, "{what}, n = {n}: {g} y {w}");
                    assert!(g > 0.0 && g < 1.0, "{what}, n = {n}: {g}");
                }
                other => panic!("{what}, n = {n}: {other:?}"),
            }
        }
        assert_eq!(got[0].is_some(), n >= 16, "n = {n}");
        assert_eq!(got[1].is_some(), os >= 16, "os = {os}");
    }
}

#[test]
fn constant_nonzero_returns_have_location_one_half_and_no_fourier() {
    // Cada tick duplica el precio: 20 retornos iguales a ln 2. Sin energía tras centrar, con
    // ubicación 0,5.
    let prices: Vec<i64> = (0..=20).map(|k| 1_000i64 << k).collect();
    let s = series(&prices);
    let mut ev = event(&s, 1, 21, 21);
    ev.direction = 1;
    ev.confirm_price = 5_000;
    let row = whole(&s, &ev);
    assert_eq!(register(&row, sc::C_FOURIER), [None; 7]);
    assert_eq!(register(&row, sc::C_HAAR), [None; 7]);
    let [c, os] = locations(&row);
    assert!(
        (c.expect("la ubicación existe") - 0.5).abs() < 1e-15,
        "{c:?}"
    );
    assert_eq!(os, None, "overshoot vacío");
}

#[test]
fn an_event_over_the_budget_has_null_locations() {
    let mut rng = Rng::new(22);
    let s = series(&walk(&mut rng, 70));
    let ev = event(&s, 1, 30, 60);
    let run = run_spectrum(&s, &[ev], &[], &[0], 1, &[1], &[1], 35).unwrap();
    assert_eq!(locations(&run.rows[0]), [None, None]);
    assert_eq!(run.over_budget.len(), 1);
}

#[test]
fn the_locations_are_bit_identical_with_any_threads_cuts_and_lots() {
    let mut rng = Rng::new(23);
    let s = series(&walk(&mut rng, 400));
    let evs = [
        event(&s, 1, 40, 70),
        event(&s, 70, 110, 150),
        event(&s, 150, 190, 230),
        event(&s, 230, 270, 300),
        event(&s, 300, 340, 370),
        event(&s, 370, 390, 400),
    ];
    let reference = run_spectrum(&s, &evs, &[], &[0], 1, &[6], &[6], BUDGET).unwrap();
    assert!(reference.rows.iter().all(|r| locations(r)[0].is_some()));
    // Lotes de eventos de 1 a 100 (el tamaño es solo un tope), cortes de tick y hilos.
    for (threads, cuts, lots) in [
        (4usize, vec![50, 140, 250], vec![1usize; 6]),
        (4, vec![100, 220], vec![6]),
        (2, vec![30, 31, 200], vec![2, 3, 1]),
        (3, vec![120, 121, 122, 300], vec![1, 5]),
    ] {
        let order: Vec<usize> = (0..=cuts.len()).rev().collect();
        let run = run_spectrum(&s, &evs, &cuts, &order, threads, &lots, &[3, 3], BUDGET).unwrap();
        assert_eq!(run.rows, reference.rows, "{threads} hilos, cortes {cuts:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Presupuesto
// ---------------------------------------------------------------------------------------------

#[test]
fn an_event_over_the_budget_is_decided_with_l2_and_leaves_a_null_row_and_a_warning() {
    let mut rng = Rng::new(5);
    let mut prices = walk(&mut rng, 70);
    // Un tick con precio 0 dentro del primer evento: si el motor lo leyera, fallaría.
    prices[10] = 0;
    let s = series(&prices);
    let (first, second) = (event(&s, 1, 30, 40), event(&s, 40, 60, 70));
    let evs = [first.clone(), second.clone()];
    // Con un presupuesto de 20 ticks, el primer evento (ids 1..40 → 39 ticks) lo pasa y el
    // segundo (ids 40..70 → 30) también; con 35 solo el primero.
    let run = run_spectrum(&s, &evs[..1], &[], &[0], 1, &[1], &[1], 35).unwrap();
    assert_eq!(run.rows.len(), 1);
    let row = &run.rows[0];
    assert_eq!(row[sc::CONFIRM_AGG_TRADE_ID], Some(first.confirm_id(&s)));
    assert!(row[1..].iter().all(Option::is_none), "{row:?}");
    assert_eq!(
        run.over_budget,
        [OverBudget {
            confirm_agg_trade_id: first.confirm_id(&s),
            ticks: (first.extreme_id(&s) - first.reference_id(&s)) as u64,
        }]
    );
    // Sin pasar del presupuesto el motor sí lee el tick en 0.
    assert!(matches!(
        run_spectrum(&s, &evs[..1], &[], &[0], 1, &[1], &[1], BUDGET),
        Err(EngineError::Ticks(TickError::ZeroPrice))
    ));
    // En el límite exacto no es sobre presupuesto: `extreme − reference > budget`.
    let span = (second.extreme_id(&s) - second.reference_id(&s)) as u64;
    let two = |budget| {
        let mut prices = walk(&mut Rng::new(5), 70);
        prices[10] = 5_000_000_000_000;
        let s = series(&prices);
        run_spectrum(&s, &[second.clone()], &[], &[0], 1, &[1], &[1], budget).unwrap()
    };
    assert!(two(span).over_budget.is_empty());
    assert_eq!(two(span - 1).over_budget.len(), 1);
}

#[test]
fn an_event_over_the_budget_that_crosses_row_groups_warns_once() {
    let mut rng = Rng::new(9);
    let s = series(&walk(&mut rng, 60));
    let ev = event(&s, 1, 30, 50);
    for order in [[0, 1, 2], [2, 1, 0], [1, 0, 2]] {
        let run = run_spectrum(&s, &[ev.clone()], &[20, 40], &order, 2, &[1], &[1], 10).unwrap();
        assert_eq!(run.over_budget.len(), 1, "orden {order:?}");
        assert!(run.rows[0][1..].iter().all(Option::is_none));
    }
}

// ---------------------------------------------------------------------------------------------
// Fusión ordenada
// ---------------------------------------------------------------------------------------------

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn go(items: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
        if k == items.len() {
            out.push(items.clone());
            return;
        }
        for i in k..items.len() {
            items.swap(k, i);
            go(items, k + 1, out);
            items.swap(k, i);
        }
    }
    let mut out = Vec::new();
    go(&mut (0..n).collect(), 0, &mut out);
    out
}

/// Un evento de 22 ticks de confirmación y 19 de overshoot, tras el tick `R`.
fn long_event() -> (Series, Ev) {
    let mut rng = Rng::new(21);
    let s = series(&walk(&mut rng, 45));
    let ev = event(&s, 1, 23, 42);
    (s, ev)
}

#[test]
fn an_event_in_three_row_groups_merges_to_the_same_bits_in_every_push_order() {
    let (s, ev) = long_event();
    let expected = whole(&s, &ev);
    let wanted = naive_row(&s, &ev);
    for (got, want) in registers(&expected).iter().zip(&wanted) {
        near(got, want, 1e-9, "el evento entero");
    }
    // R es el único tick del primer row group (el ancla viene de L2) y R como último tick
    // de uno; el corte entre fases; tramos de un tick y un row group entero dentro de la
    // confirmación.
    for cuts in [
        vec![1, 23],
        vec![1, 22],
        vec![2, 30],
        vec![10, 23],
        vec![23, 24],
        vec![22, 23],
        vec![1, 2, 3],
        vec![5, 6, 30],
        vec![1, 22, 23, 24],
    ] {
        for order in permutations(cuts.len() + 1) {
            let got = run_one(&s, &ev, &cuts, &order);
            assert_eq!(got, expected, "cortes {cuts:?}, orden {order:?}");
        }
    }
}

#[test]
fn a_middle_row_group_with_ticks_of_one_phase_only() {
    let (s, ev) = long_event();
    let expected = whole(&s, &ev);
    // [R] [c...] [c...] [c os] [os...] [os]: el del medio solo trae confirmación.
    for cuts in [vec![1, 8, 15, 23, 30], vec![1, 8, 15, 22, 35, 40]] {
        for order in [
            (0..=cuts.len()).collect::<Vec<_>>(),
            (0..=cuts.len()).rev().collect::<Vec<_>>(),
        ] {
            assert_eq!(
                run_one(&s, &ev, &cuts, &order),
                expected,
                "{cuts:?} {order:?}"
            );
        }
    }
}

#[test]
fn a_row_group_inside_one_event_and_before_the_next_keeps_both_rows_right() {
    let mut rng = Rng::new(33);
    let s = series(&walk(&mut rng, 120));
    let evs = [event(&s, 1, 30, 50), event(&s, 50, 80, 100)];
    let rows = [2usize];
    let whole_run = run_spectrum(&s, &evs, &[], &[0], 1, &[2], &rows, BUDGET).unwrap();
    for cuts in [vec![20, 60], vec![50, 51], vec![10, 20, 30, 55, 100]] {
        for order in [
            (0..=cuts.len()).collect::<Vec<_>>(),
            (0..=cuts.len()).rev().collect::<Vec<_>>(),
        ] {
            let got = run_spectrum(&s, &evs, &cuts, &order, 3, &[1, 1], &[1, 1], BUDGET).unwrap();
            assert_eq!(got.rows, whole_run.rows, "{cuts:?} {order:?}");
        }
    }
    for (row, ev) in whole_run.rows.iter().zip(&evs) {
        for (got, want) in registers(row).iter().zip(&naive_row(&s, ev)) {
            near(got, want, 1e-9, "dos eventos");
        }
    }
}

#[test]
fn the_store_is_empty_when_the_input_ends() {
    let (s, ev) = long_event();
    let bounds = bounds_of(&[10, 30], s.len());
    let engine = spectrum_engine(2, &[1], BUDGET);
    let columns = Columns::new(&s);
    for batch in batches(&s, &[ev.clone()], &[1]) {
        engine.push_boundaries(0, batch).unwrap();
    }
    for k in [0, 1] {
        engine
            .push(RowGroupInput {
                id: k as u64,
                ticks: columns.row_group(bounds[k].0, bounds[k].1),
                before_month: false,
            })
            .unwrap();
    }
    assert!(matches!(
        engine.finish(),
        Err(EngineError::Incomplete { missing: 1, .. })
    ));
}

// ---------------------------------------------------------------------------------------------
// Retornos en vuelo
// ---------------------------------------------------------------------------------------------

fn settle(engine: &Engine<Shared>) {
    for _ in 0..2_000 {
        if engine.inflight() == 0 {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("los row groups no terminaron");
}

#[test]
fn buffered_ticks_counts_the_returns_held_by_open_events() {
    let (s, ev) = long_event();
    // Ticks 0..10 | 10..30 | 30..45: el evento ocupa los tres (R = tick 0; C = tick 22; E = tick 41).
    let bounds = bounds_of(&[10, 30], s.len());
    let engine = spectrum_engine(2, &[1], BUDGET);
    let columns = Columns::new(&s);
    for batch in batches(&s, &[ev.clone()], &[1]) {
        engine.push_boundaries(0, batch).unwrap();
    }
    assert_eq!(engine.buffered_ticks(), 0);
    let push = |k: usize| {
        engine
            .push(RowGroupInput {
                id: k as u64,
                ticks: columns.row_group(bounds[k].0, bounds[k].1),
                before_month: false,
            })
            .unwrap();
        settle(&engine);
    };
    // El primer row group trae el tick R y 9 de la confirmación (ticks 1 a 9): 8 retornos
    // guardados (el primero, desde el ancla, sale al cerrar).
    push(0);
    assert_eq!(engine.buffered_ticks(), 8);
    // El segundo trae 13 ticks de la confirmación (ticks 10 a 22) y 7 del overshoot (23 a 29):
    // al unirse, la confirmación guarda 21 retornos (22 ticks, 8 + 12 y el que une los dos) y el
    // overshoot 6.
    push(1);
    assert_eq!(engine.buffered_ticks(), 21 + 6);
    // El tercero trae el resto del overshoot (30 a 41): el evento cierra y suelta todo.
    push(2);
    let _ = engine.finish().unwrap();
    assert_eq!(engine.buffered_ticks(), 0);
    assert!(engine.pop_sealed().unwrap().is_some());
}

// ---------------------------------------------------------------------------------------------
// Contra el oráculo, con cualquier hilo, lote y orden
// ---------------------------------------------------------------------------------------------

/// Eventos encadenados de 1 a 150 ticks por fase sobre una caminata de `ticks` ticks.
fn random_events(rng: &mut Rng, s: &Series, count: usize) -> Vec<Ev> {
    let mut evs = Vec::new();
    let mut r = 1 + rng.below(3) as usize;
    for _ in 0..count {
        let c = r + 1 + rng.below(150) as usize;
        let e = c + if rng.percent(25) {
            0
        } else {
            rng.below(150) as usize
        };
        if e > s.len() {
            break;
        }
        evs.push(event(s, r, c, e));
        r = e;
    }
    evs
}

#[test]
fn random_events_match_the_naive_oracle_for_any_cut_order_thread_and_lot() {
    for seed in 1..=4u64 {
        let mut rng = Rng::new(seed);
        let s = series(&walk(&mut rng, 1_500));
        let evs = random_events(&mut rng, &s, 12);
        assert!(evs.len() >= 5);
        let rows = skeleton(&mut rng, evs.len(), 4);
        let lots = skeleton(&mut rng, evs.len(), 5);
        let baseline = run_spectrum(&s, &evs, &[], &[0], 1, &lots, &rows, BUDGET).unwrap();
        assert_eq!(baseline.rows.len(), evs.len());
        for (row, ev) in baseline.rows.iter().zip(&evs) {
            let wanted = naive_row(&s, ev);
            let got = registers(row);
            for (g, w) in got.iter().zip(&wanted) {
                near(g, w, 1e-9, &format!("semilla {seed}, evento {ev:?}"));
            }
            assert_eq!(row[sc::CONFIRM_AGG_TRADE_ID], Some(ev.confirm_id(&s)));
        }
        // Mismos bits con 1 y 4 hilos, cortes aleatorios y cualquier orden de push.
        for trial in 0..6 {
            let mut cut_rng = Rng::new(seed * 100 + trial);
            let cuts: Vec<usize> = common::split(&mut cut_rng, s.len(), 200)
                .iter()
                .skip(1)
                .map(|b| b.0)
                .collect();
            let mut order: Vec<usize> = (0..=cuts.len()).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, cut_rng.below(i as u64 + 1) as usize);
            }
            for threads in [1, 4] {
                let lots = skeleton(&mut cut_rng, evs.len(), 6);
                let got =
                    run_spectrum(&s, &evs, &cuts, &order, threads, &lots, &rows, BUDGET).unwrap();
                assert_eq!(
                    got.rows, baseline.rows,
                    "semilla {seed}, hilos {threads}, cortes {cuts:?}"
                );
            }
        }
    }
}

#[test]
fn bad_events_and_ticks_are_errors_not_values() {
    let mut rng = Rng::new(11);
    let s = series(&walk(&mut rng, 40));
    let ev = event(&s, 1, 20, 30);
    let kind = |series: &Series, ev: &Ev| match run_spectrum(
        series,
        std::slice::from_ref(ev),
        &[],
        &[0],
        1,
        &[1],
        &[1],
        BUDGET,
    ) {
        Err(EngineError::Event(e)) => e.kind,
        other => panic!(
            "se esperaba un error de evento y vino {:?}",
            other.map(|r| r.rows)
        ),
    };
    // Los anclas salen de L2 y no tienen logaritmo en 0: `confirm_price` (ancla del overshoot)...
    let wrong = Ev {
        direction: -1,
        confirm_price: 0,
        ..ev.clone()
    };
    assert!(matches!(
        kind(&s, &wrong),
        EventErrorKind::Invariant {
            field: "confirm_price",
            ..
        }
    ));
    // ... y el precio de referencia (ancla de la confirmación), que sale del tick R.
    let mut zero = s.clone();
    zero.price[0] = 0;
    assert!(matches!(
        kind(&zero, &ev),
        EventErrorKind::Invariant {
            field: "reference_price",
            ..
        }
    ));
    // Un precio en 0 dentro de la fase no se oculta: es un error de tick.
    let mut bad = s.clone();
    bad.price[5] = 0;
    assert!(matches!(
        run_spectrum(&bad, &[ev], &[], &[0], 1, &[1], &[1], BUDGET),
        Err(EngineError::Ticks(TickError::ZeroPrice))
    ));
}
