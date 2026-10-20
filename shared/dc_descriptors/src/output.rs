//! Las casillas de salida: un buffer por columna para cada row group de salida de cada θ.
//!
//! El esqueleto fija cuántas filas tiene cada row group de salida de un θ (son los de
//! `events.parquet` de L2): el evento `i` de un θ es la fila `i` del esqueleto, así que su
//! casilla es fija y no depende de qué hilo lo cierre ni en qué orden. Los buffers se
//! reservan al llegar la primera fila de su row group y se entregan, con su dueño, al
//! llenarse la última: el motor no conserva nada de un row group sellado.

use crate::layout::{col, Row, COLUMNS, WIDTH};
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Mutex, PoisonError};

/// Un row group de salida lleno: `columns[c]` es la columna `c` en el layout de Arrow
/// (`decimal128` de 16 bytes o `int64` de 8, little-endian). Si algún `os_ticks` es 0,
/// `validity` es el mapa de bits de las filas con overshoot (las columnas `os_*`
/// nulables son nulas en el resto).
#[derive(Debug)]
pub struct SealedRowGroup {
    pub theta: usize,
    pub index: usize,
    pub rows: usize,
    pub columns: Vec<Vec<u8>>,
    pub validity: Option<Vec<u8>>,
    pub null_count: usize,
}

/// Por qué no se pudo escribir una fila.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputError {
    /// El evento cae más allá de la última fila del esqueleto.
    Beyond { event: u64, rows: u64 },
    /// Un row group de salida recibió más filas que las suyas.
    Overfull { theta: usize, index: usize },
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Beyond { event, rows } => {
                write!(
                    f,
                    "el evento {event} no cabe en un esqueleto de {rows} filas"
                )
            }
            Self::Overfull { theta, index } => write!(
                f,
                "el row group de salida {index} del θ {theta} recibió más filas que las suyas"
            ),
        }
    }
}

impl std::error::Error for OutputError {}

enum State {
    Empty,
    Open {
        columns: Vec<Vec<u8>>,
        filled: usize,
    },
    Sealed,
}

struct Slot {
    size: usize,
    state: Mutex<State>,
}

/// Todas las casillas de la unidad, por θ y row group de salida.
pub struct OutTable {
    slots: Vec<Vec<Slot>>,
    /// `starts[t][j]` es la fila (evento) donde empieza el row group `j` del θ `t`; el
    /// último valor es el total de filas.
    starts: Vec<Vec<u64>>,
}

impl OutTable {
    /// `sizes[t]` son las filas de cada row group de salida del θ `t`, en orden.
    pub fn new(sizes: &[Vec<usize>]) -> Self {
        let slots = sizes
            .iter()
            .map(|per_theta| {
                per_theta
                    .iter()
                    .map(|&size| Slot {
                        size,
                        state: Mutex::new(State::Empty),
                    })
                    .collect()
            })
            .collect();
        let starts = sizes
            .iter()
            .map(|per_theta| {
                let mut edges = vec![0u64];
                for &size in per_theta {
                    edges.push(edges.last().copied().unwrap_or(0) + size as u64);
                }
                edges
            })
            .collect();
        Self { slots, starts }
    }

    /// Filas del esqueleto del θ.
    pub fn rows(&self, theta: usize) -> u64 {
        self.starts[theta].last().copied().unwrap_or(0)
    }

    /// Filas que aún no se escribieron en el θ.
    pub fn missing(&self, theta: usize) -> u64 {
        self.slots[theta]
            .iter()
            .map(
                |slot| match &*slot.state.lock().unwrap_or_else(PoisonError::into_inner) {
                    State::Empty => slot.size as u64,
                    State::Open { filled, .. } => (slot.size - filled) as u64,
                    State::Sealed => 0,
                },
            )
            .sum()
    }

    /// El row group de salida y la fila de la casilla del evento `event` del θ.
    fn locate(&self, theta: usize, event: u64) -> Option<(usize, usize)> {
        let edges = &self.starts[theta];
        if event >= self.rows(theta) {
            return None;
        }
        let j = edges.partition_point(|&start| start <= event) - 1;
        Some((j, (event - edges[j]) as usize))
    }

    /// Escribe las filas de los eventos `rows` (índice de evento y fila) del θ, en una
    /// casilla cada una. Devuelve los row groups de salida que quedaron llenos.
    ///
    /// Toma el lock de cada row group de salida una vez por tramo de filas contiguas, no
    /// por fila. Las filas de `rows` deben venir ordenadas por evento.
    pub fn write(
        &self,
        theta: usize,
        rows: &[(u64, Row)],
    ) -> Result<Vec<SealedRowGroup>, OutputError> {
        let mut sealed = Vec::new();
        let mut i = 0;
        while i < rows.len() {
            let (j, first) = self.locate(theta, rows[i].0).ok_or(OutputError::Beyond {
                event: rows[i].0,
                rows: self.rows(theta),
            })?;
            let end = self.starts[theta][j + 1];
            let run = rows[i..]
                .iter()
                .take_while(|(event, _)| *event < end)
                .count();
            if let Some(full) = self.fill(theta, j, first, &rows[i..i + run])? {
                sealed.push(full);
            }
            i += run;
        }
        Ok(sealed)
    }

    fn fill(
        &self,
        theta: usize,
        j: usize,
        first: usize,
        run: &[(u64, Row)],
    ) -> Result<Option<SealedRowGroup>, OutputError> {
        let slot = &self.slots[theta][j];
        let base = self.starts[theta][j];
        let mut state = slot.state.lock().unwrap_or_else(PoisonError::into_inner);
        if matches!(*state, State::Empty) {
            let columns = COLUMNS
                .iter()
                .map(|spec| vec![0u8; slot.size * spec.bytes()])
                .collect();
            *state = State::Open { columns, filled: 0 };
        }
        let State::Open { columns, filled } = &mut *state else {
            return Err(OutputError::Overfull { theta, index: j });
        };
        debug_assert!(first as u64 == run[0].0 - base);
        for (event, row) in run {
            put(columns, (event - base) as usize, row);
        }
        *filled += run.len();
        if *filled > slot.size {
            return Err(OutputError::Overfull { theta, index: j });
        }
        if *filled < slot.size {
            return Ok(None);
        }
        let State::Open { columns, .. } = std::mem::replace(&mut *state, State::Sealed) else {
            unreachable!("el estado era Open");
        };
        drop(state);
        Ok(Some(seal(theta, j, slot.size, columns)))
    }
}

fn put(columns: &mut [Vec<u8>], r: usize, row: &Row) {
    for (c, spec) in COLUMNS.iter().enumerate() {
        let value = row[c];
        if spec.wide {
            // Extensión de signo a 128 bits: el entero sin escalar del decimal.
            columns[c][r * 16..r * 16 + 16].copy_from_slice(&i128::from(value).to_le_bytes());
        } else {
            columns[c][r * 8..r * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
    }
}

fn seal(theta: usize, index: usize, rows: usize, columns: Vec<Vec<u8>>) -> SealedRowGroup {
    debug_assert_eq!(columns.len(), WIDTH);
    let mut validity = vec![0u8; rows.div_ceil(8)];
    let mut valid = 0;
    for (r, ticks) in columns[col::OS_TICKS].chunks_exact(8).enumerate() {
        if i64::from_le_bytes(ticks.try_into().expect("8 bytes")) > 0 {
            validity[r >> 3] |= 1 << (r & 7);
            valid += 1;
        }
    }
    let null_count = rows - valid;
    SealedRowGroup {
        theta,
        index,
        rows,
        columns,
        validity: (null_count > 0).then_some(validity),
        null_count,
    }
}

/// Entrega los row groups sellados de cada θ en orden de row group de salida, sin
/// importar en qué orden se llenaron: el escritor de familias exige ese orden.
pub struct Delivery {
    next: Vec<usize>,
    held: Vec<BTreeMap<usize, SealedRowGroup>>,
    ready: VecDeque<SealedRowGroup>,
}

impl Delivery {
    pub fn new(thetas: usize) -> Self {
        Self {
            next: vec![0; thetas],
            held: (0..thetas).map(|_| BTreeMap::new()).collect(),
            ready: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, sealed: SealedRowGroup) {
        let theta = sealed.theta;
        self.held[theta].insert(sealed.index, sealed);
        while let Some(next) = self.held[theta].remove(&self.next[theta]) {
            self.next[theta] += 1;
            self.ready.push_back(next);
        }
    }

    pub fn pop(&mut self) -> Option<SealedRowGroup> {
        self.ready.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(ticks: i64, overshoot: i64) -> Row {
        let mut row = [0i64; WIDTH];
        row[col::C_TICKS] = ticks;
        row[col::OS_TICKS] = overshoot;
        row[col::C_PRICE_MAX] = -5;
        row
    }

    #[test]
    fn rows_land_in_their_slot_whatever_the_order_and_seal_when_full() {
        let table = OutTable::new(&[vec![3, 2]]);
        assert_eq!(table.rows(0), 5);
        // Las filas 4 y 0 primero; el row group 1 (filas 3 y 4) sigue abierto.
        assert!(table.write(0, &[(4, row(40, 0))]).unwrap().is_empty());
        assert!(table.write(0, &[(0, row(10, 1))]).unwrap().is_empty());
        assert_eq!(table.missing(0), 3);
        let sealed = table
            .write(0, &[(1, row(20, 0)), (2, row(30, 2)), (3, row(35, 0))])
            .unwrap();
        // Se llenan los dos, el 0 y el 1.
        let by_index: Vec<_> = sealed.iter().map(|s| (s.index, s.rows)).collect();
        assert_eq!(by_index, [(0, 3), (1, 2)]);
        let first = &sealed[0];
        let ticks: Vec<i64> = first.columns[col::C_TICKS]
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(ticks, [10, 20, 30]);
        // Overshoot: filas 0 y 2 con overshoot, la 1 nula.
        assert_eq!(first.validity.as_deref(), Some(&[0b101u8][..]));
        assert_eq!(first.null_count, 1);
        // Un decimal negativo lleva la extensión de signo en la palabra alta.
        let wide = &first.columns[col::C_PRICE_MAX][..16];
        assert_eq!(i128::from_le_bytes(wide.try_into().unwrap()), -5);
        assert_eq!(table.missing(0), 0);
    }

    #[test]
    fn an_event_beyond_the_skeleton_and_a_repeated_row_are_errors() {
        let table = OutTable::new(&[vec![2]]);
        assert_eq!(
            table.write(0, &[(2, row(1, 1))]).unwrap_err(),
            OutputError::Beyond { event: 2, rows: 2 }
        );
        table.write(0, &[(0, row(1, 1)), (1, row(1, 1))]).unwrap();
        assert_eq!(
            table.write(0, &[(0, row(1, 1))]).unwrap_err(),
            OutputError::Overfull { theta: 0, index: 0 }
        );
    }

    #[test]
    fn all_rows_with_overshoot_have_no_validity_buffer() {
        let table = OutTable::new(&[vec![2]]);
        let sealed = table.write(0, &[(0, row(1, 1)), (1, row(1, 3))]).unwrap();
        assert!(sealed[0].validity.is_none());
        assert_eq!(sealed[0].null_count, 0);
    }

    #[test]
    fn delivery_is_in_output_row_group_order_per_theta() {
        let mut delivery = Delivery::new(2);
        let sealed = |theta, index| SealedRowGroup {
            theta,
            index,
            rows: 0,
            columns: vec![],
            validity: None,
            null_count: 0,
        };
        delivery.insert(sealed(0, 1));
        delivery.insert(sealed(1, 0));
        assert_eq!(delivery.pop().map(|s| (s.theta, s.index)), Some((1, 0)));
        assert!(delivery.pop().is_none());
        delivery.insert(sealed(0, 0));
        assert_eq!(delivery.pop().map(|s| (s.theta, s.index)), Some((0, 0)));
        assert_eq!(delivery.pop().map(|s| (s.theta, s.index)), Some((0, 1)));
    }
}
