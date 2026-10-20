//! Lo que el motor recibe por row group: las columnas de ticks (buffers de Arrow, sin
//! copia) y, por θ, las filas de L2 de los eventos que tocan el row group.

use std::fmt;

/// Un buffer de Arrow con su desfase: `offset` es el índice del primer elemento (en
/// bits para `is_buyer_maker`), como `Array.offset` de Arrow.
///
/// `B` es quien posee la memoria: el motor la lee con `AsRef<[u8]>` y la suelta cuando
/// el row group se procesó entero. No copia nunca los ticks.
#[derive(Debug)]
pub struct Region<B> {
    pub buffer: B,
    pub offset: usize,
}

/// Los cuatro buffers de un row group de L1 que el motor lee, todos en el layout de
/// Arrow y little-endian: `transact_time` `int64`, `price` y `quantity` `decimal128`
/// (16 bytes, el entero sin escalar) e `is_buyer_maker` booleano de un bit por tick.
#[derive(Debug)]
pub struct TickColumns<B> {
    rows: usize,
    time: Region<B>,
    price: Region<B>,
    quantity: Region<B>,
    sold: Region<B>,
}

impl<B: AsRef<[u8]>> TickColumns<B> {
    /// Falla si algún buffer no alcanza para `rows` ticks desde su desfase.
    pub fn new(
        rows: usize,
        time: Region<B>,
        price: Region<B>,
        quantity: Region<B>,
        sold: Region<B>,
    ) -> Result<Self, InputError> {
        let columns = Self {
            rows,
            time,
            price,
            quantity,
            sold,
        };
        columns.check("time", &columns.time, rows * 8, columns.time.offset * 8)?;
        columns.check(
            "price",
            &columns.price,
            rows * 16,
            columns.price.offset * 16,
        )?;
        columns.check(
            "quantity",
            &columns.quantity,
            rows * 16,
            columns.quantity.offset * 16,
        )?;
        let bits = columns.sold.offset + rows;
        columns.check("is_buyer_maker", &columns.sold, bits.div_ceil(8), 0)?;
        Ok(columns)
    }

    fn check(
        &self,
        name: &str,
        region: &Region<B>,
        len: usize,
        skip: usize,
    ) -> Result<(), InputError> {
        let have = region.buffer.as_ref().len();
        if have < skip + len {
            return Err(InputError(format!(
                "el buffer de {name} mide {have} bytes y {} ticks desde el desfase {} piden {}",
                self.rows,
                region.offset,
                skip + len
            )));
        }
        Ok(())
    }

    /// Ticks del row group.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Los bytes de `transact_time` de los ticks `[start, end)`.
    pub(crate) fn time_bytes(&self, start: usize, end: usize) -> &[u8] {
        let from = (self.time.offset + start) * 8;
        &self.time.buffer.as_ref()[from..from + (end - start) * 8]
    }

    /// Los bytes de `price` (16 por tick) de los ticks `[start, end)`.
    pub(crate) fn price_bytes(&self, start: usize, end: usize) -> &[u8] {
        let from = (self.price.offset + start) * 16;
        &self.price.buffer.as_ref()[from..from + (end - start) * 16]
    }

    /// Los bytes de `quantity` (16 por tick) de los ticks `[start, end)`.
    pub(crate) fn quantity_bytes(&self, start: usize, end: usize) -> &[u8] {
        let from = (self.quantity.offset + start) * 16;
        &self.quantity.buffer.as_ref()[from..from + (end - start) * 16]
    }

    /// El mapa de bits de `is_buyer_maker` y el índice de bit del tick `start`.
    pub(crate) fn sold_bits(&self, start: usize) -> (&[u8], usize) {
        (self.sold.buffer.as_ref(), self.sold.offset + start)
    }
}

/// Una fila de L2 de un evento con sus posiciones en el row group (TRD-L3 §7.4).
///
/// `positions` son `R`, `C` y `E` como cuenta de ticks del row group con id ≤ la
/// frontera: la confirmación son los ticks `[R, C)` y el overshoot `[C, E)`. Una
/// frontera fuera del row group queda en 0 (anterior) o en su largo (posterior).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventIn {
    pub positions: [usize; 3],
    pub starts_before: bool,
    pub ends_after: bool,
    pub ticks_before_month: i64,
    pub reference: i64,
    pub confirm: i64,
    pub confirm_time: i64,
    pub extreme: i64,
    pub direction: i64,
    pub confirm_price: i64,
    pub extreme_price: i64,
}

/// Los eventos de un θ que tocan el row group, en orden. `first_event` es el índice de
/// la primera fila en `events.parquet` de L2 de ese θ: el índice de cada evento es su
/// casilla, y la fila de salida del evento `i` es la fila `i` del esqueleto.
#[derive(Debug, Clone)]
pub struct ThetaPiece {
    pub theta: usize,
    pub first_event: u64,
    pub events: Vec<EventIn>,
}

/// Un row group de L1 con los eventos de todos los θ que lo tocan.
///
/// `id` es el ordinal del row group entre los que entrega el lector (0, 1, 2, ...,
/// sin huecos): un evento que cruza row groups tiene un parcial en cada uno de los que
/// van del que contiene su `R` al que contiene su `E`, y con los ordinales se sabe
/// cuántos son. Los row groups pueden empujarse en cualquier orden.
#[derive(Debug)]
pub struct RowGroupInput<B> {
    pub id: u64,
    pub ticks: TickColumns<B>,
    pub pieces: Vec<ThetaPiece>,
}

/// Un `push` que rompe el contrato de entrada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputError(pub String);

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InputError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(len: usize, offset: usize) -> Region<Vec<u8>> {
        Region {
            buffer: vec![0; len],
            offset,
        }
    }

    #[test]
    fn a_short_buffer_is_rejected_with_the_column_name() {
        let error = TickColumns::new(
            10,
            region(80, 0),
            region(160, 0),
            region(159, 0),
            region(2, 0),
        )
        .unwrap_err();
        assert!(error.0.contains("quantity"), "{error}");
    }

    #[test]
    fn the_offsets_count_in_elements_and_the_bitmap_in_bits() {
        // 10 ticks desde el elemento 3: 13 elementos de 8 o 16 bytes y 13 bits.
        assert!(TickColumns::new(
            10,
            region(13 * 8, 3),
            region(13 * 16, 3),
            region(13 * 16, 3),
            region(2, 3)
        )
        .is_ok());
        assert!(TickColumns::new(
            10,
            region(13 * 8, 3),
            region(13 * 16, 3),
            region(13 * 16, 3),
            region(1, 3)
        )
        .is_err());
    }
}
