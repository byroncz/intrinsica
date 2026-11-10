//! La familia A (`fit.parquet`) del motor: valores calculados a mano, la fusión asociativa de
//! parciales en todos los cortes y órdenes de `push`, y un oráculo ingenuo en `f64` sobre
//! series aleatorias.

mod common;

use common::{batches, events, skeleton, split, Columns, Ev, Rng, Series, Shared};
use dc_descriptors::{
    fit_col as fc, Config, Engine, EngineError, EventErrorKind, Family, PhaseName, RowGroupInput,
    SealedRowGroup, FIT_COLUMNS, FIT_WIDTH,
};

/// Una fila de salida: los 64 bits de cada celda, o `None` si es nula.
type Out = [Option<i64>; FIT_WIDTH];

fn decode(sealed: &SealedRowGroup) -> Vec<Out> {
    (0..sealed.rows)
        .map(|r| {
            let mut row: Out = [None; FIT_WIDTH];
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
                    present || FIT_COLUMNS[c].nullable,
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

/// `got` y `wanted` iguales con tolerancia relativa de 1e-12 (absoluta si el valor es menor
/// que 1), o los dos nulos.
fn near(got: Option<f64>, wanted: Option<f64>, what: &str) {
    match (got, wanted) {
        (None, None) => {}
        (Some(g), Some(w)) => assert!(
            (g - w).abs() <= 1e-12 * w.abs().max(1.0),
            "{what}: {g} y debería ser {w}"
        ),
        other => panic!("{what}: {other:?}"),
    }
}

/// Los nueve registros de una fila, en el orden de la tabla.
fn registers(row: &Out) -> [Option<f64>; 9] {
    [
        fc::C_SLOPE,
        fc::OS_SLOPE,
        fc::EV_SLOPE,
        fc::C_ACCEL,
        fc::OS_ACCEL,
        fc::EV_ACCEL,
        fc::C_JUMP_SHARE,
        fc::OS_JUMP_SHARE,
        fc::EV_JUMP_SHARE,
    ]
    .map(|column| float(row, column))
}

fn expect(row: &Out, wanted: [Option<f64>; 9]) {
    let names = [
        "c_slope",
        "os_slope",
        "ev_slope",
        "c_accel",
        "os_accel",
        "ev_accel",
        "c_jump_share",
        "os_jump_share",
        "ev_jump_share",
    ];
    for (k, (got, wanted)) in registers(row).into_iter().zip(wanted).enumerate() {
        near(got, wanted, names[k]);
    }
}

fn series(ticks: &[(i64, i64)]) -> Series {
    Series {
        ids: (1..=ticks.len() as i64).collect(),
        time: ticks.iter().map(|t| t.1).collect(),
        price: ticks.iter().map(|t| t.0).collect(),
        qty: vec![1; ticks.len()],
        sold: vec![false; ticks.len()],
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

fn fit_engine(threads: usize, rows: &[usize]) -> Engine<Shared> {
    Engine::new(Config {
        thetas: vec![100_000],
        threads,
        row_groups: vec![rows.to_vec()],
        family: Family::Fit,
        silence_us: 0,
        budget_ticks: 0,
    })
    .expect("configuración válida")
}

/// Corre los eventos sobre `series` cortada en `cuts` (cuentas de ticks que inician un row
/// group), con los row groups empujados en `order`, y devuelve sus filas.
fn run_fit(
    series: &Series,
    evs: &[Ev],
    cuts: &[usize],
    order: &[usize],
    threads: usize,
    (lots, rows): (&[usize], &[usize]),
) -> Result<Vec<Out>, EngineError> {
    let bounds = bounds_of(cuts, series.len());
    let engine = fit_engine(threads, rows);
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
    let rows = run_fit(
        series,
        std::slice::from_ref(ev),
        cuts,
        order,
        2,
        (&[1], &[1]),
    )
    .expect("el evento cierra");
    assert_eq!(rows.len(), 1);
    rows[0]
}

fn whole(series: &Series, ev: &Ev) -> Out {
    run_one(series, ev, &[], &[0])
}

fn ev(r: usize, c: usize, e: usize, direction: i64, confirm_price: i64, extreme_price: i64) -> Ev {
    Ev {
        r,
        c,
        e,
        direction,
        confirm_price,
        extreme_price,
    }
}

/// Upturn de 5 ticks tras R: `(t, p)` = (0, 1 000)=R, (100, 1 010), (200, 1 020), (400, 1 040)=C,
/// (500, 1 050), (800, 1 070)=E.
fn upturn() -> (Series, Ev) {
    let s = series(&[
        (1_000, 0),
        (1_010, 100),
        (1_020, 200),
        (1_040, 400),
        (1_050, 500),
        (1_070, 800),
    ]);
    (s, ev(1, 4, 6, 1, 1_040, 1_070))
}

/// Los nueve valores de `upturn`, a mano.
fn upturn_wanted() -> [Option<f64>; 9] {
    [
        // c: y = τ exacta: pendiente (40 · 10⁶) / (1 000 · 400) = 100 por segundo, aceleración 0
        // y p_prev = 1 020: (1 040 − 1 020) / 40.
        Some(100.0),
        // os: dos ticks, τ = ¼ y 1, y = ⅓ y 1: b = (8/9), pendiente (8/9)·30·10⁶/(1 000·400).
        Some(200.0 / 3.0),
        // ev: mínimos cuadrados de p sobre t: 26 000 / 300 000 por µs = 86,666… por segundo.
        Some(260.0 / 3.0),
        Some(0.0),
        // os: la parábola por (¼, ⅓) y (1, 1) tiene a = −4/9.
        Some(-4.0 / 9.0),
        // ev: a = −2 176 / 7 917 (momentos sobre k/8 con k = 1, 2, 4, 5, 8).
        Some(-2_176.0 / 7_917.0),
        Some(0.5),
        // os: p_prev = 1 050: (1 070 − 1 050) / 30.
        Some(2.0 / 3.0),
        // ev: p_prev = 1 050: (1 070 − 1 050) / 70.
        Some(2.0 / 7.0),
    ]
}

#[test]
fn an_upturn_with_overshoot_by_hand() {
    let (s, e) = upturn();
    let row = whole(&s, &e);
    assert_eq!(row[fc::CONFIRM_AGG_TRADE_ID], Some(4));
    expect(&row, upturn_wanted());
}

#[test]
fn a_downturn_with_an_empty_overshoot_has_null_os_and_the_event_equals_the_confirmation() {
    // R = (10, 2 000); confirmación (20, 1 990), (30, 1 980), (40, 1 960)=C; sin overshoot.
    let s = series(&[(2_000, 10), (1_990, 20), (1_980, 30), (1_960, 40)]);
    let e = ev(1, 4, 4, -1, 1_960, 1_960);
    let row = whole(&s, &e);
    // Mínimos cuadrados de p sobre t: −300 / 200 = −1,5 por µs → −750 por segundo (p_R = 2 000).
    // y = ¼, ½, 1 en τ = ⅓, ⅔, 1: a = 81 / 152. p_prev = 1 980: (1 960 − 1 980) / (−40) = ½.
    let c = [Some(-750.0), Some(81.0 / 152.0), Some(0.5)];
    expect(&row, [c[0], None, c[0], c[1], None, c[1], c[2], None, c[2]]);
}

#[test]
fn one_tick_phases_have_no_slope_or_acceleration_and_a_jump_share_of_one() {
    // R = (10, 100); C = (20, 105) es el único tick de la confirmación; E = (30, 108) el único
    // del overshoot.
    let s = series(&[(100, 10), (105, 20), (108, 30)]);
    let e = ev(1, 2, 3, 1, 105, 108);
    let row = whole(&s, &e);
    // c y os: nada es anterior a su cierre, así que p_prev = p₀ y la participación vale 1. ev:
    // dos ticks, τ = ½ y 1, y = 5/8 y 1 → pendiente (3/10)·10⁶/100 = 3 000, a = −½, y
    // p_prev = 105: (108 − 105) / 8.
    expect(
        &row,
        [
            None,
            None,
            Some(3_000.0),
            None,
            None,
            Some(-0.5),
            Some(1.0),
            Some(1.0),
            Some(0.375),
        ],
    );
}

#[test]
fn a_confirmation_in_one_instant_has_no_coordinates_but_the_event_does() {
    // t_fin = t₀ en la confirmación: R, y los tres ticks, en t = 10. El overshoot, un tick.
    let s = series(&[(100, 10), (103, 10), (105, 10), (110, 25)]);
    let e = ev(1, 3, 4, 1, 105, 110);
    let row = whole(&s, &e);
    // ev: t = 10, 10, 25 y p = 103, 105, 110: pendiente 60 / 150 = 0,4 por µs → 4 000 por
    // segundo; un solo τ positivo distinto (el del tick de t = 25): aceleración nula; p_prev =
    // 105 (mayor id con t < 25): (110 − 105) / 10.
    expect(
        &row,
        [
            None,
            None,
            Some(4_000.0),
            None,
            None,
            None,
            Some(1.0),
            Some(1.0),
            Some(0.5),
        ],
    );
}

#[test]
fn two_ticks_at_the_same_time_have_no_slope_and_p_prev_takes_the_greater_id() {
    // Los dos ticks de la confirmación en t = 20 < t_C = 30: C es un tercero. El tick de mayor
    // id con t < 30 es el segundo de ellos.
    let s = series(&[(100, 10), (102, 20), (104, 20), (106, 30)]);
    let e = ev(1, 4, 4, 1, 106, 106);
    let row = whole(&s, &e);
    // y = 2/6, 4/6, 1 en τ = ½, ½, 1: tres ticks, no todos en el mismo t. p_prev = 104.
    assert!(float(&row, fc::C_SLOPE).is_some());
    near(
        float(&row, fc::C_JUMP_SHARE),
        Some(2.0 / 6.0),
        "c_jump_share",
    );
    // Todos los ticks en el mismo t: sin pendiente.
    let s = series(&[(100, 10), (102, 20), (104, 20)]);
    let e = ev(1, 3, 3, 1, 104, 104);
    let row = whole(&s, &e);
    assert_eq!(float(&row, fc::C_SLOPE), None);
    assert_eq!(float(&row, fc::C_ACCEL), None);
    // El tick de cierre es de t = 20 = t_C: nada es anterior, p_prev = p₀ y la participación es 1.
    near(float(&row, fc::C_JUMP_SHARE), Some(1.0), "c_jump_share");
}

#[test]
fn an_overshoot_that_ends_where_it_began_keeps_only_the_slope() {
    // Upturn cuyo overshoot (tiene ticks) no pasa del precio de confirmación: p_E = p_C.
    let s = series(&[
        (1_000, 0),
        (1_020, 200),
        (1_040, 400),
        (1_030, 500),
        (1_040, 600),
    ]);
    let e = ev(1, 3, 5, 1, 1_040, 1_040);
    let row = whole(&s, &e);
    // os: p = 1 030 en t = 500 y 1 040 en t = 600: 0,1 por µs / 1 000 → 100 por segundo; la
    // aceleración y la participación del salto son nulas porque p_fin = p₀.
    near(float(&row, fc::OS_SLOPE), Some(100.0), "os_slope");
    assert_eq!(float(&row, fc::OS_ACCEL), None);
    assert_eq!(float(&row, fc::OS_JUMP_SHARE), None);
    // c: y = τ en τ = ½ y 1.
    near(float(&row, fc::C_SLOPE), Some(100.0), "c_slope");
    near(float(&row, fc::C_ACCEL), Some(0.0), "c_accel");
    near(float(&row, fc::C_JUMP_SHARE), Some(0.5), "c_jump_share");
    // ev: p_fin = 1 040 ≠ p₀ = 1 000: sí tiene aceleración y participación del salto.
    assert!(float(&row, fc::EV_ACCEL).is_some());
    near(float(&row, fc::EV_JUMP_SHARE), Some(0.25), "ev_jump_share");
}

#[test]
fn a_parabolic_confirmation_has_the_exact_coefficient() {
    // y = ¾τ² + ¼τ sobre τ = k/4 con p = 1 000 + 1 600·y: 1 000, 1 175, 1 500, 1 975, 2 600.
    let s = series(&[
        (1_000, 0),
        (1_175, 100),
        (1_500, 200),
        (1_975, 300),
        (2_600, 400),
    ]);
    let e = ev(1, 5, 5, 1, 2_600, 2_600);
    let row = whole(&s, &e);
    // b = 95/80 por mínimos cuadrados con intercepto: pendiente (19/16)·1 600·10⁶ / (1 000 · 400).
    let c = [Some(4_750.0), Some(0.75), Some(625.0 / 1_600.0)];
    expect(&row, [c[0], None, c[0], c[1], None, c[1], c[2], None, c[2]]);
}

/// Todos los subconjuntos de cortes de una serie de `len` ticks.
fn all_cuts(len: usize) -> Vec<Vec<usize>> {
    (0..1u32 << (len - 1))
        .map(|mask| (1..len).filter(|k| mask >> (k - 1) & 1 == 1).collect())
        .collect()
}

#[test]
fn an_event_over_row_groups_gives_the_same_bits_in_every_cut_and_push_order() {
    let (s, e) = upturn();
    let expected = whole(&s, &e);
    expect(&expected, upturn_wanted());
    let mut rng = Rng::new(7);
    for cuts in all_cuts(s.len()) {
        let groups = cuts.len() + 1;
        let mut orders: Vec<Vec<usize>> = vec![
            (0..groups).collect(),
            (0..groups).rev().collect(),
            (0..groups).cycle().skip(1).take(groups).collect(),
        ];
        for _ in 0..3 {
            let mut order: Vec<usize> = (0..groups).collect();
            for i in (1..groups).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            orders.push(order);
        }
        for order in orders {
            let row = run_one(&s, &e, &cuts, &order);
            assert_eq!(row, expected, "cortes {cuts:?}, orden {order:?}");
        }
    }
}

#[test]
fn three_row_groups_with_every_push_order_give_the_same_bits() {
    let (s, e) = upturn();
    let expected = whole(&s, &e);
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        assert_eq!(run_one(&s, &e, &[2, 4], &order), expected, "{order:?}");
    }
}

#[test]
fn a_middle_row_group_without_ticks_of_a_phase_and_r_as_the_last_tick_of_one() {
    let (s, e) = upturn();
    let expected = whole(&s, &e);
    // El row group 1 (el tick 4) solo trae overshoot: su parcial de la confirmación está vacío.
    assert_eq!(run_one(&s, &e, &[4, 5], &[0, 1, 2]), expected);
    assert_eq!(run_one(&s, &e, &[4, 5], &[2, 0, 1]), expected);
    // R es el último tick del row group 0: el evento no aporta ningún tick a ese row group.
    assert_eq!(run_one(&s, &e, &[1], &[0, 1]), expected);
    assert_eq!(run_one(&s, &e, &[1], &[1, 0]), expected);
    // E es el último tick de un row group y el siguiente no trae nada del evento.
    assert_eq!(run_one(&s, &e, &[6], &[0]), expected);
}

#[test]
fn a_sum_that_does_not_fit_fails_closed() {
    // Un tick de la confirmación a 10¹⁴ con p_C − p_R = 1: |y| = 10¹⁴ pasa de 2⁴⁷.
    let s = series(&[(100, 10), (100_000_000_000_000, 15), (101, 20)]);
    let e = ev(1, 3, 3, 1, 101, 101);
    let error = run_fit(&s, &[e], &[], &[0], 1, (&[1], &[1])).unwrap_err();
    match error {
        EngineError::Event(event) => {
            assert_eq!(
                event.kind,
                EventErrorKind::FitOverflow {
                    phase: PhaseName::Confirmation
                }
            );
            assert_eq!(event.confirm_agg_trade_id, 3);
        }
        other => panic!("se esperaba un error de evento y vino {other:?}"),
    }
}

#[test]
fn an_event_whose_times_go_backwards_is_an_invariant_error() {
    // confirm_time = 10 < reference_time = 20.
    let s = series(&[(100, 20), (105, 10)]);
    let e = ev(1, 2, 2, 1, 105, 105);
    match run_fit(&s, &[e], &[], &[0], 1, (&[1], &[1])).unwrap_err() {
        EngineError::Event(event) => assert!(
            matches!(
                event.kind,
                EventErrorKind::Invariant {
                    field: "confirm_time",
                    ..
                }
            ),
            "{:?}",
            event.kind
        ),
        other => panic!("se esperaba un error de evento y vino {other:?}"),
    }
}

/// La fila que sale de calcular los nueve registros directo en `f64` sobre la serie: las
/// ecuaciones normales con sumas ingenuas, para contrastar el motor con otra cuenta.
fn oracle(s: &Series, e: &Ev) -> [Option<f64>; 9] {
    struct Frame {
        t0: f64,
        p0: f64,
        tf: f64,
        pf: f64,
    }
    let t = |k: usize| s.time[k] as f64;
    let p = |k: usize| s.price[k] as f64;
    let (reference_price, confirm_price, extreme_price) = (
        e.reference_price(s) as f64,
        e.confirm_price as f64,
        e.extreme_price as f64,
    );
    let (reference_time, confirm_time, extreme_time) = (
        e.reference_time(s) as f64,
        e.confirm_time(s) as f64,
        e.extreme_time(s) as f64,
    );
    let system = |frame: Frame, ticks: std::ops::Range<usize>| -> [Option<f64>; 3] {
        let n = ticks.len();
        if n == 0 {
            return [None; 3];
        }
        let span = frame.tf - frame.t0;
        let flat = frame.pf == frame.p0;
        let scale = if flat { frame.p0 } else { frame.pf - frame.p0 };
        let mut sums = [0.0f64; 7];
        let (mut low, mut high) = (f64::MAX, f64::MIN);
        let (mut later_low, mut later_high) = (f64::MAX, f64::MIN);
        let mut before: Option<usize> = None;
        for k in ticks {
            let tau = (t(k) - frame.t0) / span;
            let y = (p(k) - frame.p0) / scale;
            let terms = [
                tau,
                tau * tau,
                tau.powi(3),
                tau.powi(4),
                y,
                tau * y,
                tau * tau * y,
            ];
            for (sum, term) in sums.iter_mut().zip(terms) {
                *sum += term;
            }
            low = low.min(t(k));
            high = high.max(t(k));
            if t(k) > frame.t0 {
                later_low = later_low.min(t(k));
                later_high = later_high.max(t(k));
            }
            if t(k) < frame.tf {
                before = Some(k);
            }
        }
        let [s1, s2, s3, s4, sy, sty, s2y] = sums;
        let nf = n as f64;
        let slope = (n >= 2 && low < high && span > 0.0).then(|| {
            let b = (nf * sty - s1 * sy) / (nf * s2 - s1 * s1);
            b * scale * 1e6 / (reference_price * span)
        });
        let accel = (slope.is_some() && !flat && later_low < later_high)
            .then(|| (s2y * s2 - sty * s3) / (s4 * s2 - s3 * s3));
        let jump = (!flat).then(|| {
            let prev = before.map_or(frame.p0, p);
            (frame.pf - prev) / (frame.pf - frame.p0)
        });
        [slope, accel, jump]
    };
    let c = system(
        Frame {
            t0: reference_time,
            p0: reference_price,
            tf: confirm_time,
            pf: confirm_price,
        },
        e.r..e.c,
    );
    let o = system(
        Frame {
            t0: confirm_time,
            p0: confirm_price,
            tf: extreme_time,
            pf: extreme_price,
        },
        e.c..e.e,
    );
    let v = system(
        Frame {
            t0: reference_time,
            p0: reference_price,
            tf: extreme_time,
            pf: extreme_price,
        },
        e.r..e.e,
    );
    [c[0], o[0], v[0], c[1], o[1], v[1], c[2], o[2], v[2]]
}

/// Compara el motor con el oráculo ingenuo: la misma nulidad y, donde hay valor, un acuerdo
/// de 1e-6 relativo (las sumas ingenuas en `f64` pierden dígitos en las diferencias).
fn assert_agrees(got: &Out, wanted: [Option<f64>; 9], context: &str) {
    let names = [
        "c_slope",
        "os_slope",
        "ev_slope",
        "c_accel",
        "os_accel",
        "ev_accel",
        "c_jump_share",
        "os_jump_share",
        "ev_jump_share",
    ];
    for (k, (got, wanted)) in registers(got).into_iter().zip(wanted).enumerate() {
        match (got, wanted) {
            (None, None) => {}
            (Some(g), Some(w)) => {
                assert!(g.is_finite(), "{context}: {} no es finito", names[k]);
                assert!(
                    (g - w).abs() <= 1e-6 * w.abs().max(1e-3),
                    "{context}: {} {g} contra {w}",
                    names[k]
                );
            }
            other => panic!("{context}: {} {other:?}", names[k]),
        }
    }
}

#[test]
fn random_events_agree_with_the_naive_oracle_and_do_not_depend_on_threads_or_cuts() {
    for seed in 1..=6u64 {
        let mut rng = Rng::new(seed * 7919);
        let mut series = Series::random(&mut rng, 3_000, true);
        let evs = events(&mut rng, &mut series, 12, 70);
        assert!(evs.len() > 10, "pocos eventos con la semilla {seed}");
        let lots = skeleton(&mut rng, evs.len(), 9);
        let rows = skeleton(&mut rng, evs.len(), 13);
        let bounds = split(&mut rng, series.len(), 400);
        let cuts: Vec<usize> = bounds.iter().skip(1).map(|&(lo, _)| lo).collect();
        let groups = bounds.len();
        let in_order: Vec<usize> = (0..groups).collect();
        let mut shuffled = in_order.clone();
        for i in (1..groups).rev() {
            shuffled.swap(i, rng.below(i as u64 + 1) as usize);
        }

        let baseline = run_fit(&series, &evs, &cuts, &in_order, 1, (&lots, &rows)).unwrap();
        assert_eq!(baseline.len(), evs.len());
        for (k, (row, ev)) in baseline.iter().zip(&evs).enumerate() {
            assert_agrees(
                row,
                oracle(&series, ev),
                &format!("semilla {seed}, evento {k}"),
            );
            // Sin ±inf ni NaN, y la clave es el tick de confirmación.
            assert_eq!(row[fc::CONFIRM_AGG_TRADE_ID], Some(ev.confirm_id(&series)));
        }
        for (threads, order) in [(4, &in_order), (4, &shuffled), (2, &shuffled)] {
            let again = run_fit(&series, &evs, &cuts, order, threads, (&lots, &rows)).unwrap();
            assert_eq!(again, baseline, "semilla {seed}, {threads} hilos");
        }
        // Otros cortes y otros tamaños de lote: las mismas filas, bit a bit.
        let other_bounds = split(&mut rng, series.len(), 90);
        let other_cuts: Vec<usize> = other_bounds.iter().skip(1).map(|&(lo, _)| lo).collect();
        let order: Vec<usize> = (0..other_bounds.len()).rev().collect();
        let other_lots = skeleton(&mut rng, evs.len(), 3);
        let again = run_fit(&series, &evs, &other_cuts, &order, 3, (&other_lots, &rows)).unwrap();
        assert_eq!(again, baseline, "semilla {seed}, otros cortes");
    }
}

#[test]
fn the_three_events_of_the_bridge_probe_by_hand() {
    // Los tres eventos a mano de `l3_fit_bridge_probe` (ITSC-398), con sus ticks: la
    // aceleración y el salto son los que calculó la sonda; la pendiente se calcula aquí.
    // Evento 1: R = (0; 100), ticks (2; 102) (5; 104) (8; 109) (10; 110) = C.
    // Evento 2: R = (100; 200), (110; 198) (110; 195) (130; 192) (140; 190) (140; 191) = C, y
    //   L2 cierra en 190 aunque el último tick vale 191 (grupo de empate).
    // Evento 3: R = (0; 1 000), (5; 1 012) (10; 1 004) (20; 1 010) = C.
    let cases = [
        (
            series(&[(100, 0), (102, 2), (104, 5), (109, 8), (110, 10)]),
            ev(1, 5, 5, 1, 110, 110),
            // t = 2, 5, 8, 10 con p = 102, 104, 109, 110: 39,75 / 36,75 = 53/49 por µs.
            [530_000.0 / 49.0, 1_080.0 / 5_759.0, 0.1],
        ),
        (
            series(&[
                (200, 100),
                (198, 110),
                (195, 110),
                (192, 130),
                (190, 140),
                (191, 140),
            ]),
            ev(1, 6, 6, -1, 190, 190),
            // t = 10, 10, 30, 40, 40 con p = 198, 195, 192, 190, 191: −186 / 920 por µs.
            [-23_250.0 / 23.0, -22.0 / 39.0, 0.2],
        ),
        (
            series(&[(1_000, 0), (1_012, 5), (1_004, 10), (1_010, 20)]),
            ev(1, 4, 4, 1, 1_010, 1_010),
            // t = 5, 10, 20 con p = 1 012, 1 004, 1 010: −3,333 / 116,67 por µs.
            [-200.0 / 7.0, -864.0 / 505.0, 0.6],
        ),
    ];
    for (s, e, [slope, accel, jump]) in cases {
        let row = whole(&s, &e);
        // Sin overshoot, el evento entero es la confirmación.
        let wanted = [
            Some(slope),
            None,
            Some(slope),
            Some(accel),
            None,
            Some(accel),
            Some(jump),
            None,
            Some(jump),
        ];
        expect(&row, wanted);
    }
}
