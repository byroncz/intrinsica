# pyutils

Utilidades Python comunes de las capas: el escritor Parquet atómico y el hash
de contenido. Salen de `l1_ingest` (ITSC-245) para que L2 no mantenga una
copia; las capas no se importan entre sí (RNF-14 del maestro), así que lo
común vive aquí.

## Qué contiene

- `pyutils.PartitionWriter(path, schema, sort_order=None, row_group_size=None,
  compact_encoding=False)`:
  escribe un Parquet por lotes (ZSTD-3, estadísticas, `sorting_columns` si hay
  `sort_order`) a un temporal y `commit` lo renombra sobre el destino. Sin
  `row_group_size`, cada lote o tabla es su propio row group. Con
  `compact_encoding` no usa diccionario y codifica los enteros en
  `DELTA_BINARY_PACKED` (L2, ITSC-290): −40 % de bytes y 5× menos tiempo de
  codificación sobre eventos reales, con el mismo `content_hash`. Detalle de la
  sobrescritura atómica y de GCS en
  [`docs/data-contracts.md`](../../docs/data-contracts.md).
- `pyutils.ContentHasher(schema)` y `pyutils.content_hash(table)`: SHA-256 del
  contenido lógico (esquema más filas en orden), sin importar el chunking ni
  los bytes del Parquet. Los tres caminos del hasher están en
  [Cómo se hashea](#cómo-se-hashea).
- `pyutils.resolve_fs(path)`: sistema de archivos Arrow y ruta, local o `gs://`.

## Cómo lo instala una capa

```bash
uv add --package <capa> pyutils
```

Lo propio de cada capa (esquema, orden, tamaño de row group, rutas) se pasa
por parámetro; ver `l1_ingest.write.output_writer`.

## Cómo se hashea

`ContentHasher` elige el camino de cada columna por su esquema, no por el lote,
para que el hash no dependa de cómo se parta la serie:

1. **Entero, decimal, booleano o float64 no nulable:** los bytes de sus valores
   (en float64, 8 bytes little-endian por valor, vista del buffer sin copia).
2. **Entero, decimal o booleano nulable:** el `repr` de Python de cada celda más
   `\0` (`Decimal('12.50000000')`, `-3`, `True`, `None`), porque el buffer de un
   valor nulo no está definido. Se construye por columna completa con Arrow, sin
   un objeto de Python por celda: `cast` a texto, prefijo y sufijo con
   `binary_replace_slice` y `None\0` en los nulos. El resultado es un solo buffer
   que se entrega a SHA-256 de una vez y se libera al terminar el `update`
   (ITSC-359).
3. **float64 nulable:** una celda de 9 bytes por fila: `\x01` más los 8 bytes del
   valor, o nueve ceros si es nulo. El ancho fijo separa un nulo de cualquier
   valor, `0.0` incluido. Se construye por columna completa con Arrow (vista
   `binary(8)` del buffer, prefijo con `binary_replace_slice` y ceros en los
   nulos), sin pasar por el texto, que no es un contrato de Arrow para los
   flotantes.
4. **Texto y cualquier otro tipo (float32 incluido):** valor a valor, con `repr`
   de Python.

float64 se hashea por sus bits: NaN e infinitos no fallan y dan sus bytes crudos,
sin canonizar (`0.0` y `-0.0` dan digests distintos, y dos NaN con payload
distinto también). L3 nunca los produce. `HASHER_VERSION` no sube por este camino:
ninguna huella publicada tiene columnas float64 y el digest de los demás tipos no
cambia.

**Por qué el digest no cambió.** El camino 2 produce los mismos bytes que el
bucle `digest.update(repr(value).encode() + b"\0")` que lo precedía, y SHA-256
incremental da el mismo resultado partido en celdas o en un solo buffer. El único
riesgo es que el texto de Arrow no coincida con `repr(Decimal)`: pasa en el cero
con escala (`Decimal('0E-8')`) y con exponente ajustado menor que −6
(`Decimal('1.0E-7')`). Hoy Arrow lo escribe igual, pero no es un contrato suyo, así
que las celdas con |valor| < 10⁻⁶ se rehacen con `repr` de Python (son pocas). Con
escala entre 0 y 6 no hay tales celdas y se omite la máscara; con escala mayor el
umbral se construye en el tipo de la columna, para que la comparación no suba la
precisión sobre 38 (`decimal128(38, 0)` fallaba); con escala negativa se rehacen
todas.

**Cómo se probó.** `tests/test_hashing_equivalence.py` conserva el hasher anterior
(`ReferenceHasher`) y compara los digests sobre lotes aleatorios con semilla
(`decimal128(18, 8)`, `(9, 8)`, `(38, 10)`, `(9, 0)`, `(38, 0)`, `(9, 6)`, `(9, -2)`,
`decimal256(50, 2)` y `(60, 10)`, `int64` y `bool`, nulables y no
nulables, con 0 %, 30 % y 100 % de nulos, ceros, negativos, valores con exponente
ajustado menor que −6, máximos de la precisión, lotes vacíos y partidos). Un digest
fijo calculado con el hasher anterior detecta cualquier cambio futuro.

**Rendimiento.** Cuatro columnas nulables como las de `summaries.parquet` (tres
`decimal128(18, 8)` y un `int64`), 32 768 filas, 30 % de nulos, mínimo de varias
repeticiones: `update` pasa de ≈ 74 ms a ≈ 9 ms (≈ 8×, 12,5 % del anterior). En
2021-01 son 1 335 `update` por mes: el hash baja de ≈ 99 s a ≈ 12 s. La meta
inicial de la card era 10×; el humano aceptó 8× como criterio de rendimiento: los
≈ 3 s que separan 8× de 10× no justifican una dependencia nueva ni tocar
`uv.lock`. El `cast` de decimal a texto de Arrow es ≈ 54 % de lo que queda (1,3 ms
por columna) y no tiene alternativa más barata. La prueba exige al menos 5× para
atrapar una regresión al bucle en Python.
