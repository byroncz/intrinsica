# dc_frames

Lector de tramas: entrega los ticks de cada evento DC, por fase, recortando L1
con las fronteras de L2. No guarda nada: las tramas se leen, no se
materializan. Contrato en [TRD-L3 §7.4](../../docs/TRD/l3.md#74-contrato-del-lector-de-tramas)
y decisión en ADR-L3-06. Es también el oráculo de las pruebas de L3.

## Qué contiene

- `read_frames(thetas, l1_root, l2_root, start, end, provider="binance",
  market="spot", asset="BTCUSDT")`: iterador de `EventFrames`, uno por evento
  cerrado de los meses `start`..`end` (tuplas `(año, mes)`, inclusivas, de las
  particiones de L2). `thetas` es un θ (`Decimal` o `str`) o varios; con varios,
  cada row group de L1 se decodifica una sola vez. Dentro de un θ salen por
  `confirm_agg_trade_id`; entre θ no hay orden.
- `frames_of(event, l1_root, ...)`: las tramas de un evento cuya fila de L2 ya
  se tiene (un `RecordBatch` de una fila o un mapa con sus 11 columnas).
  Decodifica solo los row groups que el evento toca.
- `EventFrames(theta, event, confirmation, overshoot)`: `event` es la fila de L2
  sin transformar; `confirmation` son los ticks de `(R, C]` y `overshoot` los de
  `(C, E]`, vacío si `E = C`.
- `Frame(transact_time, price, quantity, is_buyer_maker)`: cuatro arrays de
  Arrow con los tipos de L1. No trae `agg_trade_id` ni `is_best_match`.
- Errores: `FrameBoundaryError` (una frontera de L2 no es un tick de L1) y
  `FramesInputError` (falta un archivo, el esquema o el orden no cumplen el
  contrato).

```python
from dc_frames import read_frames

for event in read_frames("0.00100000", l1_root, l2_root, (2023, 3), (2023, 3)):
    prices = event.confirmation.price  # pa.Array decimal128(18, 8)
```

## Cómo funciona

- **Pertenencia por valor.** Cada frontera (`reference_`, `confirm_` y
  `extreme_agg_trade_id`) se busca con `np.searchsorted` sobre los ids del row
  group, para todos sus eventos a la vez. Nunca se resta ni se compara por
  tiempo, porque el proveedor deja huecos de ids. Una frontera que no es un tick
  de L1 (un hueco, o L1 y L2 que no cuadran) falla con `FrameBoundaryError`.
- **Slices sin copia.** Un evento que cabe en un row group sale como slices de
  sus buffers. Si el consumidor los conserva, ancla ese row group.
- **Eventos que cruzan.** Si un evento sigue en el row group siguiente (o en el
  mes siguiente), el lector copia solo los ticks que ya leyó y suelta el row
  group. Al entregarlo concatena una columna a la vez.
- **Memoria.** Un row group de L1, un row group de eventos por θ y el evento en
  curso de cada θ. El pico por evento es O(evento).
- **Saltos.** Los row groups de L1 que ningún θ necesita se saltan con las
  estadísticas min/max de `agg_trade_id`; el primer evento puede empezar en
  meses anteriores a `start`, que el lector retrocede a abrir.

## Pruebas

```bash
uv run pytest shared/dc_frames
```

Usan `ticks.csv` y `events_v0.csv` de `shared/dc_core/tests/fixtures`, con la
cantidad de cada tick igual a su `agg_trade_id` para leer los ids de una trama
sin que el lector los entregue.
