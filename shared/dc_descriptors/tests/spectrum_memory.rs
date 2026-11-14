//! El pico de memoria del cierre de una fase de la familia F, medido y no estimado (ADR-L3-19).
//!
//! Un asignador global cuenta los bytes vivos y su máximo. Cierra una fase de `n` retornos con
//! `close_spectrum` y compara el pico con la cota `closing_bound(n)`, la fórmula que cita el
//! ADR-L3-19, §7.1 de TRD-L3 y el README del crate. Esta prueba es la única de su binario: otro
//! hilo asignando falsearía el contador.

use dc_descriptors::spectrum::accumulate;
use dc_descriptors::{close_spectrum, EventIn, Region, SpectrumAcc, TickColumns};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[path = "common/closing.rs"]
mod closing;
use closing::{bluestein_inner, closing_bound};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: delega en `System` y solo suma y resta contadores atómicos.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    // `realloc` queda en el valor por defecto (asignar, copiar, soltar): cuenta los dos bloques a
    // la vez, que es lo que pasa en el peor caso.
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// El mayor factor primo de `n`.
fn largest_prime_factor(mut n: usize) -> usize {
    let mut largest = 1;
    let mut p = 2;
    while p * p <= n {
        while n % p == 0 {
            largest = p;
            n /= p;
        }
        p += 1;
    }
    if n > 1 {
        largest = n;
    }
    largest
}

/// El primer primo `≥ from`.
fn next_prime(from: usize) -> usize {
    (from..)
        .find(|&n| n > 1 && largest_prime_factor(n) == n)
        .unwrap()
}

fn measure(n: usize) -> (usize, usize) {
    let base = LIVE.load(Relaxed);
    let rows = n;
    let wide = |value: &dyn Fn(usize) -> i64| -> Region<Vec<u8>> {
        let mut bytes = Vec::with_capacity(rows * 16);
        for i in 0..rows {
            bytes.extend_from_slice(&value(i).to_le_bytes());
            bytes.extend_from_slice(&[0u8; 8]);
        }
        Region {
            buffer: bytes,
            offset: 0,
        }
    };
    let narrow = |value: &dyn Fn(usize) -> i64| -> Region<Vec<u8>> {
        let mut bytes = Vec::with_capacity(rows * 8);
        for i in 0..rows {
            bytes.extend_from_slice(&value(i).to_le_bytes());
        }
        Region {
            buffer: bytes,
            offset: 0,
        }
    };
    // Precios que suben y bajan sin ser constantes: un paso pseudoaleatorio de ±3 sobre 1 000 000.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut price = 1_000_000i64;
    let prices: Vec<i64> = (0..rows)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            price += (state >> 61) as i64 - 3;
            price
        })
        .collect();
    let ticks = TickColumns::new(
        rows,
        narrow(&|i| i as i64),
        narrow(&|i| i as i64),
        wide(&|i| prices[i]),
        wide(&|_| 1),
        Region {
            buffer: vec![0u8; rows.div_ceil(8)],
            offset: 0,
        },
    )
    .expect("buffers completos");
    drop(prices);
    let acc = SpectrumAcc {
        confirmation: accumulate(&ticks, 0, rows).expect("precios válidos"),
        ..SpectrumAcc::default()
    };
    drop(ticks);
    let held = LIVE.load(Relaxed) - base;
    PEAK.store(LIVE.load(Relaxed), Relaxed);
    let event = EventIn {
        reference: 0,
        confirm: 1,
        confirm_time: 0,
        extreme: rows as i64,
        direction: 1,
        confirm_price: 1_000_000,
        extreme_price: 1_000_000,
        reference_price: 1_000_000,
        reference_time: 0,
        extreme_time: 0,
    };
    let row = close_spectrum(acc, &event).expect("cierre válido");
    assert_eq!(row.ticks, rows as u64);
    let peak = PEAK.load(Relaxed) - base;
    assert_eq!(LIVE.load(Relaxed), base, "el cierre no deja nada vivo");
    (held, peak)
}

/// Bytes de margen para las asignaciones chicas (planes, mariposas, `Arc`): lo medido sobrepasa
/// `closing_bound` en 32 B como mucho.
const SLACK: usize = 64 << 10;

fn check(n: usize, what: &str) -> usize {
    let (held, peak) = measure(n);
    eprintln!(
        "{what}: n = {n}, m = {}: búfer {held} B, pico {peak} B, cota {} B ({:.1} B por retorno)",
        bluestein_inner(n),
        closing_bound(n),
        peak as f64 / n as f64
    );
    assert!(
        peak <= closing_bound(n) + SLACK,
        "{what}: con n = {n} el pico fue {peak} B y la cota {} B",
        closing_bound(n)
    );
    peak
}

#[test]
fn the_closing_peak_stays_under_the_documented_bound() {
    // Bluestein (n primo con un factor primo grande en n − 1), con las dos longitudes internas
    // que elige `rustfft`: 3·2^20 (n cerca de 2^20) y 2^22 (n cerca de 1,7 M, donde 3·2^20 ya no
    // alcanza `2n − 1`).
    for from in [1usize << 20, 1_700_000] {
        let n = next_prime(from);
        assert!(
            largest_prime_factor(n - 1) > 23,
            "n = {n} debe ser Bluestein"
        );
        check(n, "Bluestein");
    }
    // El mayor primo por debajo de `SERIAL_CLOSE_MIN`: el peor cierre que corre en paralelo.
    let below = (2..dc_descriptors::spectrum::SERIAL_CLOSE_MIN)
        .rev()
        .find(|&n| largest_prime_factor(n) == n)
        .unwrap();
    let peak = check(below, "bajo el umbral");
    assert!(
        peak <= 160 << 20,
        "un cierre bajo el umbral pasó de 160 MiB: {peak} B"
    );
    // Los otros planes tienen menos tablas de longitud `n` y quedan bajo la misma cota: Rader
    // (n primo con n − 1 de factores pequeños: 786 433 = 3·2^18 + 1), una potencia de 2 y un
    // compuesto.
    assert!(largest_prime_factor(786_432) <= 23);
    check(786_433, "Rader");
    check(1 << 20, "potencia de 2");
    check(1_000_000, "compuesto");
}
