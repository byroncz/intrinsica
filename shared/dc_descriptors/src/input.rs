//! Lo que el motor recibe: por θ, las columnas de L2 de sus eventos en lotes, en orden, y,
//! por row group, las columnas de L1 (buffers de Arrow, sin copia).
//!
//! El motor mismo calcula dónde cae cada frontera de un evento en un row group: nadie le
//! entrega posiciones (ADR-L3-16).

use std::collections::VecDeque;
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

fn check<B: AsRef<[u8]>>(
    name: &str,
    region: &Region<B>,
    rows: usize,
    skip: usize,
    len: usize,
) -> Result<(), InputError> {
    let have = region.buffer.as_ref().len();
    if have < skip + len {
        return Err(InputError(format!(
            "el buffer de {name} mide {have} bytes y {rows} filas desde el desfase {} piden {}",
            region.offset,
            skip + len
        )));
    }
    Ok(())
}

/// Las dos columnas de un row group de L1 con las que se ubican las fronteras de un evento:
/// `agg_trade_id` y `transact_time`, ambas `int64` en el layout de Arrow y little-endian.
/// Los ids crecen estrictamente (contrato de L1).
#[derive(Debug)]
pub struct KeyColumns<B> {
    rows: usize,
    ids: Region<B>,
    time: Region<B>,
}

impl<B: AsRef<[u8]>> KeyColumns<B> {
    /// Falla si el row group está vacío o algún buffer no alcanza para `rows` ticks desde
    /// su desfase.
    pub fn new(rows: usize, ids: Region<B>, time: Region<B>) -> Result<Self, InputError> {
        if rows == 0 {
            return Err(InputError("un row group de L1 no puede estar vacío".into()));
        }
        check("agg_trade_id", &ids, rows, ids.offset * 8, rows * 8)?;
        check("time", &time, rows, time.offset * 8, rows * 8)?;
        Ok(Self { rows, ids, time })
    }

    /// Ticks del row group.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// El `agg_trade_id` del tick `i`.
    pub fn id(&self, i: usize) -> i64 {
        read_i64(&self.ids, i)
    }

    /// El `transact_time` del tick `i`.
    pub fn time_at(&self, i: usize) -> i64 {
        read_i64(&self.time, i)
    }

    /// Cuántos ticks tienen id ≤ `x`: `searchsorted(ids, x, "right")`. Así 0 si `x` cae
    /// antes del row group y `rows` si cae después.
    pub fn rank(&self, x: i64) -> usize {
        let (mut lo, mut hi) = (0, self.rows);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.id(mid) <= x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Los cinco buffers de un row group de L1 que el motor lee, todos en el layout de Arrow y
/// little-endian: `agg_trade_id` y `transact_time` `int64` ([`KeyColumns`]), `price` y
/// `quantity` `decimal128` (16 bytes, el entero sin escalar) e `is_buyer_maker` booleano de
/// un bit por tick.
#[derive(Debug)]
pub struct TickColumns<B> {
    keys: KeyColumns<B>,
    price: Region<B>,
    quantity: Region<B>,
    sold: Region<B>,
}

impl<B: AsRef<[u8]>> TickColumns<B> {
    /// Falla si el row group está vacío o algún buffer no alcanza para `rows` ticks desde
    /// su desfase.
    pub fn new(
        rows: usize,
        ids: Region<B>,
        time: Region<B>,
        price: Region<B>,
        quantity: Region<B>,
        sold: Region<B>,
    ) -> Result<Self, InputError> {
        let keys = KeyColumns::new(rows, ids, time)?;
        check("price", &price, rows, price.offset * 16, rows * 16)?;
        check("quantity", &quantity, rows, quantity.offset * 16, rows * 16)?;
        check(
            "is_buyer_maker",
            &sold,
            rows,
            0,
            (sold.offset + rows).div_ceil(8),
        )?;
        Ok(Self {
            keys,
            price,
            quantity,
            sold,
        })
    }

    /// Las columnas con las que se ubican las fronteras.
    pub fn keys(&self) -> &KeyColumns<B> {
        &self.keys
    }

    /// Ticks del row group.
    pub fn rows(&self) -> usize {
        self.keys.rows
    }

    /// Los bytes de `transact_time` de los ticks `[start, end)`.
    pub(crate) fn time_bytes(&self, start: usize, end: usize) -> &[u8] {
        let from = (self.keys.time.offset + start) * 8;
        &self.keys.time.buffer.as_ref()[from..from + (end - start) * 8]
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

fn read_i64<B: AsRef<[u8]>>(region: &Region<B>, i: usize) -> i64 {
    let at = (region.offset + i) * 8;
    i64::from_le_bytes(
        region.buffer.as_ref()[at..at + 8]
            .try_into()
            .expect("8 bytes"),
    )
}

/// Una fila de L2 de un evento (TRD-L3 §7.1): las ocho columnas que usa L3. `reference_price`
/// es el ancla de la fase de confirmación de la familia T; S no la lee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventIn {
    pub reference: i64,
    pub confirm: i64,
    pub confirm_time: i64,
    pub extreme: i64,
    pub direction: i64,
    pub confirm_price: i64,
    pub extreme_price: i64,
    pub reference_price: i64,
}

/// Un lote de filas consecutivas de `events.parquet` de L2 de un θ (un row group o un tramo
/// de uno), en el layout de Arrow: `reference`,
/// `confirm`, `confirm_time` y `extreme` `int64`, `direction` `int8` y los tres precios
/// (`confirm_price`, `extreme_price` y `reference_price`) `decimal128` (16 bytes, el entero sin
/// escalar, que cabe en `i64` por contrato de L2).
#[derive(Debug)]
pub struct EventBatch<B> {
    rows: usize,
    reference: Region<B>,
    confirm: Region<B>,
    confirm_time: Region<B>,
    extreme: Region<B>,
    direction: Region<B>,
    confirm_price: Region<B>,
    extreme_price: Region<B>,
    reference_price: Region<B>,
}

impl<B: AsRef<[u8]>> EventBatch<B> {
    /// Falla si algún buffer no alcanza para `rows` eventos desde su desfase.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: usize,
        reference: Region<B>,
        confirm: Region<B>,
        confirm_time: Region<B>,
        extreme: Region<B>,
        direction: Region<B>,
        confirm_price: Region<B>,
        extreme_price: Region<B>,
        reference_price: Region<B>,
    ) -> Result<Self, InputError> {
        for (name, region, width) in [
            ("reference_agg_trade_id", &reference, 8),
            ("confirm_agg_trade_id", &confirm, 8),
            ("confirm_time", &confirm_time, 8),
            ("extreme_agg_trade_id", &extreme, 8),
            ("direction", &direction, 1),
            ("confirm_price", &confirm_price, 16),
            ("extreme_price", &extreme_price, 16),
            ("reference_price", &reference_price, 16),
        ] {
            check(name, region, rows, region.offset * width, rows * width)?;
        }
        Ok(Self {
            rows,
            reference,
            confirm,
            confirm_time,
            extreme,
            direction,
            confirm_price,
            extreme_price,
            reference_price,
        })
    }

    /// Eventos del lote.
    pub fn rows(&self) -> usize {
        self.rows
    }

    fn price(region: &Region<B>, i: usize, what: &str) -> Result<i64, InputError> {
        let at = (region.offset + i) * 16;
        let raw = &region.buffer.as_ref()[at..at + 16];
        i64::try_from(i128::from_le_bytes(raw.try_into().expect("16 bytes")))
            .map_err(|_| InputError(format!("{what} de la fila {i} no cabe en i64")))
    }

    fn event(&self, i: usize) -> Result<EventIn, InputError> {
        let direction = self.direction.buffer.as_ref()[self.direction.offset + i] as i8;
        Ok(EventIn {
            reference: read_i64(&self.reference, i),
            confirm: read_i64(&self.confirm, i),
            confirm_time: read_i64(&self.confirm_time, i),
            extreme: read_i64(&self.extreme, i),
            direction: i64::from(direction),
            confirm_price: Self::price(&self.confirm_price, i, "confirm_price")?,
            extreme_price: Self::price(&self.extreme_price, i, "extreme_price")?,
            reference_price: Self::price(&self.reference_price, i, "reference_price")?,
        })
    }
}

/// Los eventos de un θ que el motor tiene en mano: los lotes de L2 que llegaron con
/// `push_boundaries`, en el orden de `events.parquet`, menos los que ya soltó. El evento
/// `i` es la fila `i` del esqueleto, sea cual sea el lote que lo trajo.
///
/// Cumplen `R < C ≤ E` y `E_i ≤ R_{i+1}` (lo comprueba [`Boundaries::push`], también entre
/// un lote y el anterior), así que `R` y `E` crecen y las búsquedas binarias valen: qué
/// eventos toca un row group se decide con los ids, sin estado entre row groups.
///
/// Solo se sueltan lotes del frente ([`Boundaries::release`]) y solo cuando sus eventos y
/// los de los lotes anteriores cerraron. Un row group sin procesar no toca ningún evento
/// cerrado (el evento cierra cuando llegaron los parciales de todos los row groups que
/// toca), así que buscar en `[base, len)` da lo mismo que buscar en todos.
#[derive(Debug)]
pub struct Boundaries<B> {
    lots: VecDeque<EventBatch<B>>,
    /// `starts[k]` es el índice global de la primera fila de `lots[k]`; el último es el
    /// total recibido.
    starts: VecDeque<usize>,
    first_reference: Option<i64>,
    last_reference: Option<i64>,
    last_extreme: Option<i64>,
}

impl<B> Default for Boundaries<B> {
    fn default() -> Self {
        Self {
            lots: VecDeque::new(),
            starts: VecDeque::from([0]),
            first_reference: None,
            last_reference: None,
            last_extreme: None,
        }
    }
}

impl<B: AsRef<[u8]>> Boundaries<B> {
    /// Agrega un lote detrás de los anteriores. Valida el orden de sus eventos, también
    /// contra el último recibido, y que los precios quepan en `i64`; si falla, no agrega
    /// nada. Un lote vacío no cambia nada.
    pub fn push(&mut self, batch: EventBatch<B>) -> Result<(), InputError> {
        if batch.rows == 0 {
            return Ok(());
        }
        let base = self.len();
        let mut previous = self.last_extreme;
        for i in 0..batch.rows {
            let event = batch.event(i)?;
            let index = base + i;
            if !(event.reference < event.confirm && event.confirm <= event.extreme) {
                return Err(InputError(format!(
                    "el evento {index} no cumple R < C <= E ({}, {}, {})",
                    event.reference, event.confirm, event.extreme
                )));
            }
            if previous.is_some_and(|extreme| event.reference < extreme) {
                return Err(InputError(format!(
                    "el evento {index} empieza en la referencia {}, antes del extremo {} \
                     del anterior",
                    event.reference,
                    previous.unwrap_or_default()
                )));
            }
            previous = Some(event.extreme);
        }
        self.first_reference
            .get_or_insert(read_i64(&batch.reference, 0));
        self.last_reference = Some(read_i64(&batch.reference, batch.rows - 1));
        self.last_extreme = previous;
        self.starts.push_back(base + batch.rows);
        self.lots.push_back(batch);
        Ok(())
    }

    /// Saca los lotes del frente cuyos eventos son todos anteriores a `closed` y los
    /// devuelve, para que el llamador suelte sus buffers fuera de su lock.
    pub fn release(&mut self, closed: usize) -> Vec<EventBatch<B>> {
        let mut released = Vec::new();
        while self.starts.len() > 1 && self.starts[1] <= closed {
            self.starts.pop_front();
            released.extend(self.lots.pop_front());
        }
        released
    }

    /// Eventos recibidos del θ, en mano o ya soltados.
    pub fn len(&self) -> usize {
        self.starts[self.starts.len() - 1]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// El primer evento en mano: los anteriores ya se soltaron.
    pub fn base(&self) -> usize {
        self.starts[0]
    }

    /// Lotes en mano.
    pub fn lots(&self) -> usize {
        self.lots.len()
    }

    /// La `reference_agg_trade_id` del primer evento recibido.
    pub fn first_reference(&self) -> Option<i64> {
        self.first_reference
    }

    /// La `reference_agg_trade_id` del último evento recibido.
    pub fn last_reference(&self) -> Option<i64> {
        self.last_reference
    }

    /// El `extreme_agg_trade_id` del último evento recibido.
    pub fn last_extreme(&self) -> Option<i64> {
        self.last_extreme
    }

    fn batch_of(&self, index: usize) -> (&EventBatch<B>, usize) {
        debug_assert!(self.base() <= index && index < self.len());
        let b = self.starts.partition_point(|&s| s <= index) - 1;
        (&self.lots[b], index - self.starts[b])
    }

    /// La fila `index` (base ≤ index < len).
    pub(crate) fn event(&self, index: usize) -> EventIn {
        let (batch, i) = self.batch_of(index);
        batch.event(i).expect("validado en Boundaries::push")
    }

    /// La `reference_agg_trade_id` del evento `index`.
    pub(crate) fn reference(&self, index: usize) -> i64 {
        let (batch, i) = self.batch_of(index);
        read_i64(&batch.reference, i)
    }

    /// El `extreme_agg_trade_id` del evento `index`.
    pub(crate) fn extreme(&self, index: usize) -> i64 {
        let (batch, i) = self.batch_of(index);
        read_i64(&batch.extreme, i)
    }

    /// Cuántos eventos tienen `E ≤ x`, que son los primeros: los que ya cerraron en un
    /// row group cuyo último id es `x`. Busca entre los que están en mano: nunca da menos
    /// que `base`.
    pub(crate) fn extremes_up_to(&self, x: i64) -> usize {
        self.search(|i| self.extreme(i) <= x)
    }

    /// Cuántos eventos tienen `R ≤ x`: los que empiezan en un row group cuyo último id es
    /// `x`, o antes. Como `extremes_up_to`, nunca da menos que `base`.
    pub(crate) fn references_up_to(&self, x: i64) -> usize {
        self.search(|i| self.reference(i) <= x)
    }

    /// El primer índice de `[base, len)` donde `holds` deja de cumplirse (cumple en un
    /// prefijo).
    fn search(&self, holds: impl Fn(usize) -> bool) -> usize {
        let (mut lo, mut hi) = (self.base(), self.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if holds(mid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Un row group de L1 y si es de un mes anterior al de los eventos del motor.
///
/// `id` es el ordinal del row group entre los que entrega el lector (0, 1, 2, ..., sin
/// huecos): un evento que cruza row groups tiene un parcial en cada uno de los que van del
/// que contiene su `R` al que contiene su `E`, y con los ordinales se sabe cuántos son y
/// cuál es el anterior a cada uno. Los row groups pueden empujarse en cualquier orden.
///
/// `before_month` dice que el row group es de un mes anterior al de los eventos: sus
/// ticks de `(R, E]` cuentan como `ticks_before_month` (TRD-L3 §7.4).
#[derive(Debug)]
pub struct RowGroupInput<B> {
    pub id: u64,
    pub ticks: TickColumns<B>,
    pub before_month: bool,
}

/// Una entrada que rompe el contrato.
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
            region(13 * 8, 3),
            region(13 * 16, 3),
            region(13 * 16, 3),
            region(2, 3)
        )
        .is_ok());
        assert!(TickColumns::new(
            10,
            region(13 * 8, 3),
            region(13 * 8, 3),
            region(13 * 16, 3),
            region(13 * 16, 3),
            region(1, 3)
        )
        .is_err());
    }

    #[test]
    fn rank_counts_the_ids_up_to_the_value() {
        let ids: Vec<u8> = [10i64, 11, 15, 20]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let ticks = TickColumns::new(
            4,
            Region {
                buffer: ids,
                offset: 0,
            },
            region(32, 0),
            region(64, 0),
            region(64, 0),
            region(1, 0),
        )
        .unwrap();
        let ranks: Vec<_> = [9, 10, 12, 15, 20, 21]
            .map(|x| ticks.keys().rank(x))
            .to_vec();
        assert_eq!(ranks, [0, 1, 2, 3, 4, 4]);
    }
}
