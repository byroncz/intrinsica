//! La familia T (`path.parquet`) del motor: valores calculados a mano, la fusión ordenada de
//! parciales en todos los cortes y órdenes de `push`, y un oráculo ingenuo en `f64` sobre
//! series aleatorias.

mod common;

use common::{batches, events, skeleton, split, Columns, Ev, Rng, Series, Shared};
use dc_descriptors::{
    path_col as pc, Config, Engine, EngineError, EventErrorKind, Family, RowGroupInput,
    SealedRowGroup, TickError, PATH_COLUMNS, PATH_WIDTH,
};

const SILENCE: i64 = 2_000;

/// Una fila de salida: los 64 bits de cada celda, o `None` si es nula.
type Out = [Option<i64>; PATH_WIDTH];

fn decode(sealed: &SealedRowGroup) -> Vec<Out> {
    (0..sealed.rows)
        .map(|r| {
            let mut row: Out = [None; PATH_WIDTH];
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
                    present || PATH_COLUMNS[c].nullable,
                    "la columna {c} no admite nulos"
                );
                *cell = present.then_some(value);
            }
            row
        })
        .collect()
}

fn float(row: &Out, column: usize) -> Option<f64> {
    row[column].map(|bits| f64::from_bits(bits as u64))
}

fn near(got: Option<f64>, wanted: Option<f64>, what: &str) {
    match (got, wanted) {
        (None, None) => {}
        (Some(g), Some(w)) => assert!(
            (g - w).abs() <= 1e-12 * w.abs().max(f64::MIN_POSITIVE),
            "{what}: {g} y debería ser {w}"
        ),
        other => panic!("{what}: {other:?}"),
    }
}

fn series(ticks: &[(i64, i64, bool)]) -> Series {
    Series {
        ids: (1..=ticks.len() as i64).collect(),
        time: ticks.iter().map(|t| t.1).collect(),
        price: ticks.iter().map(|t| t.0).collect(),
        qty: vec![1; ticks.len()],
        sold: ticks.iter().map(|t| t.2).collect(),
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

fn path_engine(threads: usize, rows: &[usize], silence_us: i64) -> Engine<Shared> {
    Engine::new(Config {
        thetas: vec![100_000],
        threads,
        row_groups: vec![rows.to_vec()],
        family: Family::Path,
        silence_us,
    })
    .expect("configuración válida")
}

/// Corre los eventos sobre `series` cortada en `cuts` (cuentas de ticks que inician un row
/// group), con los row groups empujados en `order`, y devuelve sus filas.
#[allow(clippy::too_many_arguments)]
fn run_path(
    series: &Series,
    evs: &[Ev],
    cuts: &[usize],
    order: &[usize],
    threads: usize,
    lots: &[usize],
    rows: &[usize],
    silence_us: i64,
) -> Result<Vec<Out>, EngineError> {
    let bounds = bounds_of(cuts, series.len());
    let engine = path_engine(threads, rows, silence_us);
    let columns = Columns::new(series);
    for batch in batches(series, evs, lots) {
        engine.push_boundaries(0, batch)?;
    }
    let mut out: Vec<Out> = Vec::new();
    for &k in order {
        let (lo, hi) = bounds[k];
        engine.push(RowGroupInput {
            id: k as u64,
            ticks: columns.row_group(lo, hi),
            before_month: false,
        })?;
        while let Some(sealed) = engine.pop_sealed()? {
            out.extend(decode(&sealed));
        }
    }
    engine.finish()?;
    while let Some(sealed) = engine.pop_sealed()? {
        out.extend(decode(&sealed));
    }
    Ok(out)
}

fn run_one(series: &Series, ev: &Ev, cuts: &[usize], order: &[usize]) -> Out {
    let rows = run_path(
        series,
        std::slice::from_ref(ev),
        cuts,
        order,
        2,
        &[1],
        &[1],
        SILENCE,
    )
    .expect("el evento cierra");
    assert_eq!(rows.len(), 1);
    rows[0]
}

fn whole(series: &Series, ev: &Ev) -> Out {
    run_one(series, ev, &[], &[0])
}

fn ln_sq(a: f64, b: f64) -> f64 {
    (a / b).ln().powi(2)
}

/// Upturn de 6 ticks (R, 3 de confirmación, 2 de overshoot) con todos los registros a mano.
fn upturn() -> (Series, Ev) {
    let s = series(&[
        (100, 900, false),
        (101, 1_000, false),
        (103, 1_500, false),
        (104, 4_000, true),
        (105, 4_500, false),
        (106, 9_000, true),
    ]);
    let ev = Ev {
        r: 1,
        c: 4,
        e: 6,
        direction: 1,
        confirm_price: 104,
        extreme_price: 106,
    };
    (s, ev)
}

#[test]
fn an_upturn_with_overshoot_by_hand() {
    let (s, ev) = upturn();
    let row = whole(&s, &ev);
    assert_eq!(row[pc::CONFIRM_AGG_TRADE_ID], Some(4));
    // Σ(Δln p)² con el paso desde el ancla p_R = 100.
    let c_var = ln_sq(101.0, 100.0) + ln_sq(103.0, 101.0) + ln_sq(104.0, 103.0);
    near(
        float(&row, pc::C_REALIZED_VAR),
        Some(c_var),
        "c_realized_var",
    );
    let os_var = ln_sq(105.0, 104.0) + ln_sq(106.0, 105.0);
    near(
        float(&row, pc::OS_REALIZED_VAR),
        Some(os_var),
        "os_realized_var",
    );
    // Camino 1 + 2 + 1 = 4 sobre un neto de |104 − 100| = 4; overshoot 1 + 1 sobre |106 − 104|.
    assert_eq!(float(&row, pc::C_TORTUOSITY), Some(1.0));
    assert_eq!(float(&row, pc::OS_TORTUOSITY), Some(1.0));
    // Lados + + −: n = 3, S = 1, P = 0 → (−3·(2 − 1 + 1) + 2) / 24 = −1/6. Overshoot + −: −1/2.
    near(
        float(&row, pc::C_SIGN_AUTOCORR),
        Some(-1.0 / 6.0),
        "c_sign_autocorr",
    );
    assert_eq!(float(&row, pc::OS_SIGN_AUTOCORR), Some(-0.5));
    // Huecos de la confirmación: 500 y 2 500; solo el segundo pasa de 2 000.
    assert_eq!(row[pc::C_MAX_GAP], Some(2_500));
    assert_eq!(row[pc::OS_MAX_GAP], Some(4_500));
    near(
        float(&row, pc::C_IDLE_FRACTION),
        Some(2_500.0 / 3_000.0),
        "c_idle",
    );
    assert_eq!(float(&row, pc::OS_IDLE_FRACTION), Some(1.0));
    // 4·(p − 100) ≥ k·4: 101 alcanza el 25 % (t = 1 000), 103 el 50 y el 75 % (t = 1 500).
    assert_eq!(float(&row, pc::C_T25), Some(0.0));
    near(float(&row, pc::C_T50), Some(500.0 / 3_000.0), "c_t50");
    near(float(&row, pc::C_T75), Some(500.0 / 3_000.0), "c_t75");
}

#[test]
fn a_downturn_mirrors_the_levels() {
    let s = series(&[
        (200, 10, false),
        (198, 11, true),
        (196, 12, true),
        (195, 13, true),
    ]);
    let ev = Ev {
        r: 1,
        c: 3,
        e: 4,
        direction: -1,
        confirm_price: 196,
        extreme_price: 195,
    };
    let row = whole(&s, &ev);
    // 4·(200 − p) ≥ k·4: 198 → 8: 25 % y 50 % (t = 11); 196 → 16: 75 % (t = 12).
    assert_eq!(float(&row, pc::C_T25), Some(0.0));
    assert_eq!(float(&row, pc::C_T50), Some(0.0));
    assert_eq!(float(&row, pc::C_T75), Some(1.0));
    assert_eq!(float(&row, pc::C_TORTUOSITY), Some(1.0));
    // Un solo tick de overshoot: sin autocorrelación ni fracción ociosa, con hueco 0.
    assert_eq!(row[pc::OS_MAX_GAP], Some(0));
    assert_eq!(float(&row, pc::OS_SIGN_AUTOCORR), None);
    assert_eq!(float(&row, pc::OS_IDLE_FRACTION), None);
    near(
        float(&row, pc::OS_REALIZED_VAR),
        Some(ln_sq(195.0, 196.0)),
        "os_realized_var",
    );
    assert_eq!(float(&row, pc::OS_TORTUOSITY), Some(1.0));
}

#[test]
fn a_one_tick_phase_with_an_empty_overshoot() {
    let s = series(&[(100, 10, false), (105, 20, true)]);
    let ev = Ev {
        r: 1,
        c: 2,
        e: 2,
        direction: 1,
        confirm_price: 105,
        extreme_price: 105,
    };
    let row = whole(&s, &ev);
    near(
        float(&row, pc::C_REALIZED_VAR),
        Some(ln_sq(105.0, 100.0)),
        "c_realized_var",
    );
    assert_eq!(float(&row, pc::C_TORTUOSITY), Some(1.0));
    assert_eq!(row[pc::C_MAX_GAP], Some(0));
    for column in [
        pc::C_SIGN_AUTOCORR,
        pc::C_IDLE_FRACTION,
        pc::C_T25,
        pc::C_T50,
        pc::C_T75,
        pc::OS_REALIZED_VAR,
        pc::OS_TORTUOSITY,
        pc::OS_SIGN_AUTOCORR,
        pc::OS_MAX_GAP,
        pc::OS_IDLE_FRACTION,
    ] {
        assert_eq!(row[column], None, "la columna {column} debía ser nula");
    }
}

#[test]
fn an_overshoot_of_one_tick_and_one_that_returns_to_the_confirmation_price() {
    let mut ticks = vec![(100, 10, false), (105, 20, true), (107, 30, false)];
    let ev = Ev {
        r: 1,
        c: 2,
        e: 3,
        direction: 1,
        confirm_price: 105,
        extreme_price: 107,
    };
    let row = whole(&series(&ticks), &ev);
    near(
        float(&row, pc::OS_REALIZED_VAR),
        Some(ln_sq(107.0, 105.0)),
        "os_realized_var",
    );
    assert_eq!(float(&row, pc::OS_TORTUOSITY), Some(1.0));
    assert_eq!(row[pc::OS_MAX_GAP], Some(0));
    assert_eq!(float(&row, pc::OS_SIGN_AUTOCORR), None);
    assert_eq!(float(&row, pc::OS_IDLE_FRACTION), None);

    // El overshoot vuelve a p_C: p_E = p_C, así que T-04 es nulo y T-02 vale 0.
    ticks[2] = (105, 30, false);
    let back = Ev {
        extreme_price: 105,
        ..ev
    };
    let row = whole(&series(&ticks), &back);
    assert_eq!(float(&row, pc::OS_REALIZED_VAR), Some(0.0));
    assert_eq!(row[pc::OS_TORTUOSITY], None);
    assert_eq!(row[pc::OS_MAX_GAP], Some(0));
}

#[test]
fn an_atomic_sweep_has_no_duration_dependent_registers() {
    let s = series(&[
        (100, 10, false),
        (101, 20, false),
        (102, 20, true),
        (103, 20, false),
    ]);
    let ev = Ev {
        r: 1,
        c: 4,
        e: 4,
        direction: 1,
        confirm_price: 103,
        extreme_price: 103,
    };
    let row = whole(&s, &ev);
    for column in [pc::C_IDLE_FRACTION, pc::C_T25, pc::C_T50, pc::C_T75] {
        assert_eq!(row[column], None);
    }
    assert_eq!(row[pc::C_MAX_GAP], Some(0));
    // + − +: n = 3, S = 1, P = −2, s₁ = sₙ = +1 → (9·(−2) − 0 + 2) / 24 = −2/3.
    near(
        float(&row, pc::C_SIGN_AUTOCORR),
        Some(-2.0 / 3.0),
        "c_sign_autocorr",
    );
    assert_eq!(float(&row, pc::C_TORTUOSITY), Some(1.0));
}

#[test]
fn the_sign_autocorrelation_cases() {
    let make = |sides: &[bool]| {
        let mut ticks = vec![(100, 10, false)];
        ticks.extend(
            sides
                .iter()
                .enumerate()
                .map(|(i, &sold)| (101 + i as i64, 11 + i as i64, sold)),
        );
        let n = ticks.len();
        let ev = Ev {
            r: 1,
            c: n,
            e: n,
            direction: 1,
            confirm_price: ticks[n - 1].0,
            extreme_price: ticks[n - 1].0,
        };
        float(&whole(&series(&ticks), &ev), pc::C_SIGN_AUTOCORR)
    };
    // Alternancia estricta de 4 ticks: −0,75. Dos rachas de dos: 0,25.
    assert_eq!(make(&[false, true, false, true]), Some(-0.75));
    assert_eq!(make(&[false, false, true, true]), Some(0.25));
    // Todos del mismo lado: |S| = n, nula. Con uno solo, nula.
    assert_eq!(make(&[true, true, true]), None);
    assert_eq!(make(&[false, false, false, false, false]), None);
    assert_eq!(make(&[true]), None);
    // Una racha con un solo tick del otro lado: −0,3.
    assert_eq!(make(&[false, false, true, false, false]), Some(-0.3));
}

#[test]
fn gaps_over_and_under_the_silence_threshold() {
    // Huecos de 100 (no cuenta: no es mayor) y 101 (cuenta) con silencio 100.
    let s = series(&[
        (100, 0, false),
        (101, 5, false),
        (102, 105, true),
        (103, 206, false),
    ]);
    let ev = Ev {
        r: 1,
        c: 4,
        e: 4,
        direction: 1,
        confirm_price: 103,
        extreme_price: 103,
    };
    let rows = run_path(&s, &[ev.clone()], &[], &[0], 1, &[1], &[1], 100).unwrap();
    assert_eq!(rows[0][pc::C_MAX_GAP], Some(101));
    near(
        float(&rows[0], pc::C_IDLE_FRACTION),
        Some(101.0 / 201.0),
        "c_idle_fraction",
    );
    // Con silencio 101 ningún hueco cuenta; con 99 cuentan los dos.
    let rows = run_path(&s, &[ev.clone()], &[], &[0], 1, &[1], &[1], 101).unwrap();
    assert_eq!(float(&rows[0], pc::C_IDLE_FRACTION), Some(0.0));
    let rows = run_path(&s, &[ev], &[], &[0], 1, &[1], &[1], 99).unwrap();
    assert_eq!(float(&rows[0], pc::C_IDLE_FRACTION), Some(201.0 / 201.0));
}

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

#[test]
fn an_event_in_three_row_groups_merges_to_the_same_bits_in_every_push_order() {
    let (s, ev) = upturn();
    let expected = whole(&s, &ev);
    // R es el único tick del primer row group (el ancla viene de L2); el segundo trae la
    // confirmación entera y el tercero el overshoot; y otros cortes con tramos de un tick.
    for cuts in [
        vec![1, 4],
        vec![1, 3],
        vec![2, 4],
        vec![3, 5],
        vec![1, 2, 3],
        vec![2, 3, 4, 5],
    ] {
        for order in permutations(cuts.len() + 1) {
            let got = run_one(&s, &ev, &cuts, &order);
            assert_eq!(got, expected, "cortes {cuts:?}, orden {order:?}");
        }
    }
}

#[test]
fn a_middle_row_group_without_overshoot_ticks_and_one_inside_the_confirmation() {
    let (s, ev) = upturn();
    let expected = whole(&s, &ev);
    // [R] [c c] [c os] [os]: el del medio tiene ticks de las dos fases; y [R c] [c] [c os] [os].
    for cuts in [vec![1, 3, 4, 5], vec![2, 3, 4, 5], vec![1, 2, 3, 4, 5]] {
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
fn the_store_is_empty_when_the_input_ends() {
    let (s, ev) = upturn();
    let bounds = bounds_of(&[2, 4], s.len());
    let engine = path_engine(2, &[1], SILENCE);
    let columns = Columns::new(&s);
    for batch in batches(&s, &[ev.clone()], &[1]) {
        engine.push_boundaries(0, batch).unwrap();
    }
    // Falta el último row group: el evento queda abierto y finish lo dice.
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

#[test]
fn bad_events_and_ticks_are_errors_not_values() {
    let (s, ev) = upturn();
    let kind = |series: &Series, ev: &Ev| match run_path(
        series,
        std::slice::from_ref(ev),
        &[],
        &[0],
        1,
        &[1],
        &[1],
        SILENCE,
    ) {
        Err(EngineError::Event(e)) => e.kind,
        other => panic!("se esperaba un error de evento y vino {other:?}"),
    };
    // Una dirección que no es 1 ni -1.
    let wrong = Ev {
        direction: 0,
        ..ev.clone()
    };
    assert!(matches!(
        kind(&s, &wrong),
        EventErrorKind::Direction { got: 0 }
    ));
    // Un precio de referencia en 0.
    let mut zero = s.clone();
    zero.price[0] = 0;
    assert!(matches!(
        kind(&zero, &ev),
        EventErrorKind::Invariant {
            field: "reference_price",
            ..
        }
    ));
    // Un tick de la fase con price en 0 no se acepta: ln(0) sería -inf.
    let mut tick = s.clone();
    tick.price[2] = 0;
    match run_path(&tick, &[ev], &[], &[0], 1, &[1], &[1], SILENCE) {
        Err(EngineError::Ticks(TickError::ZeroPrice)) => {}
        other => panic!("{other:?}"),
    }
}

/// Lo que el oráculo ingenuo dice de una fase.
struct Naive {
    var: f64,
    length: i128,
    net: i128,
    autocorr: Option<f64>,
    max_gap: i64,
    idle: i64,
    duration: i64,
}

fn naive_phase(s: &Series, from: usize, to: usize, anchor: i64, net: i128, silence: i64) -> Naive {
    let mut var = 0.0;
    let mut length = 0i128;
    let mut max_gap = 0;
    let mut idle = 0;
    let mut previous = anchor;
    for i in from..to {
        let delta = s.price[i] - previous;
        length += i128::from(delta.abs());
        var += (delta as f64 / previous as f64).ln_1p().powi(2);
        if i > from {
            let gap = s.time[i] - s.time[i - 1];
            max_gap = max_gap.max(gap);
            if gap > silence {
                idle += gap;
            }
        }
        previous = s.price[i];
    }
    let sides: Vec<f64> = (from..to)
        .map(|i| if s.sold[i] { -1.0 } else { 1.0 })
        .collect();
    let n = sides.len() as f64;
    let mean = sides.iter().sum::<f64>() / n;
    let den: f64 = sides.iter().map(|x| (x - mean).powi(2)).sum();
    let autocorr = (sides.len() >= 2 && den > 1e-9).then(|| {
        let num: f64 = sides
            .windows(2)
            .map(|w| (w[1] - mean) * (w[0] - mean))
            .sum();
        num / den
    });
    Naive {
        var,
        length,
        net,
        autocorr,
        max_gap,
        idle,
        duration: s.time[to - 1] - s.time[from],
    }
}

fn check_against_naive(s: &Series, ev: &Ev, row: &Out) {
    let reference = ev.reference_price(s);
    let net_c = i128::from((ev.confirm_price - reference).abs());
    let c = naive_phase(s, ev.r, ev.c, reference, net_c, SILENCE);
    let relative = |got: Option<f64>, wanted: Option<f64>, what: &str| match (got, wanted) {
        (None, None) => {}
        (Some(g), Some(w)) => assert!(
            (g - w).abs() <= 1e-9 * w.abs() + 1e-12,
            "{what}: {g} y el oráculo dice {w}"
        ),
        other => panic!("{what}: {other:?}"),
    };
    relative(
        float(row, pc::C_REALIZED_VAR),
        Some(c.var),
        "c_realized_var",
    );
    relative(
        float(row, pc::C_TORTUOSITY),
        Some(c.length as f64 / c.net as f64),
        "c_tortuosity",
    );
    relative(
        float(row, pc::C_SIGN_AUTOCORR),
        c.autocorr,
        "c_sign_autocorr",
    );
    assert_eq!(row[pc::C_MAX_GAP], Some(c.max_gap));
    relative(
        float(row, pc::C_IDLE_FRACTION),
        (c.duration > 0).then(|| c.idle as f64 / c.duration as f64),
        "c_idle_fraction",
    );
    // Los niveles, a mano sobre los ticks de la confirmación.
    let need = i128::from(ev.confirm_price - reference) * i128::from(ev.direction);
    for (k, column) in [pc::C_T25, pc::C_T50, pc::C_T75].into_iter().enumerate() {
        let wanted = (c.duration > 0).then(|| {
            let hit = (ev.r..ev.c)
                .find(|&i| {
                    4 * i128::from(s.price[i] - reference) * i128::from(ev.direction)
                        >= (k as i128 + 1) * need
                })
                .expect("el tick C alcanza el 100 %");
            (s.time[hit] - s.time[ev.r]) as f64 / c.duration as f64
        });
        relative(float(row, column), wanted, "c_t");
    }
    if ev.e == ev.c {
        for column in [
            pc::OS_REALIZED_VAR,
            pc::OS_TORTUOSITY,
            pc::OS_SIGN_AUTOCORR,
            pc::OS_MAX_GAP,
            pc::OS_IDLE_FRACTION,
        ] {
            assert_eq!(row[column], None);
        }
        return;
    }
    let net_o = i128::from((ev.extreme_price - ev.confirm_price).abs());
    let o = naive_phase(s, ev.c, ev.e, ev.confirm_price, net_o, SILENCE);
    relative(
        float(row, pc::OS_REALIZED_VAR),
        Some(o.var),
        "os_realized_var",
    );
    relative(
        float(row, pc::OS_TORTUOSITY),
        (o.net > 0).then(|| o.length as f64 / o.net as f64),
        "os_tortuosity",
    );
    relative(
        float(row, pc::OS_SIGN_AUTOCORR),
        o.autocorr,
        "os_sign_autocorr",
    );
    assert_eq!(row[pc::OS_MAX_GAP], Some(o.max_gap));
    relative(
        float(row, pc::OS_IDLE_FRACTION),
        (o.duration > 0).then(|| o.idle as f64 / o.duration as f64),
        "os_idle_fraction",
    );
}

fn shuffled(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.below(i as u64 + 1) as usize);
    }
    order
}

#[test]
fn random_series_match_the_naive_oracle_and_do_not_depend_on_cuts_orders_or_threads() {
    for seed in 1..=12u64 {
        let mut rng = Rng::new(seed * 7919);
        let mut s = Series::random(&mut rng, 4_000, seed % 2 == 0);
        let evs = events(&mut rng, &mut s, 6, 60);
        assert!(
            evs.len() >= 2,
            "pocos eventos con la semilla {seed}: {}",
            evs.len()
        );
        let rows = skeleton(&mut rng, evs.len(), 40);
        let mut first: Option<Vec<Out>> = None;
        for (variant, threads) in [1usize, 4, 4, 1].into_iter().enumerate() {
            let cuts: Vec<usize> = split(&mut rng, s.len(), 1 + 40 * (variant as u64 + 1))
                .iter()
                .skip(1)
                .map(|b| b.0)
                .collect();
            let groups = cuts.len() + 1;
            let order = shuffled(&mut rng, groups);
            let lots = skeleton(&mut rng, evs.len(), 1 + 30 * variant as u64);
            let got = run_path(&s, &evs, &cuts, &order, threads, &lots, &rows, SILENCE)
                .unwrap_or_else(|e| panic!("semilla {seed}: {e}"));
            assert_eq!(got.len(), evs.len());
            match &first {
                None => {
                    for (ev, row) in evs.iter().zip(&got) {
                        check_against_naive(&s, ev, row);
                    }
                    first = Some(got);
                }
                Some(expected) => {
                    assert_eq!(&got, expected, "semilla {seed}, variante {variant}")
                }
            }
        }
    }
}
