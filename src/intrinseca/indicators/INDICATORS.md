# Taxonomía de Indicadores de Cambio Direccional

## Referencia Técnica para el Motor Intrinseca

---

## 1. Introducción

Este documento constituye la **referencia canónica** para todos los indicadores implementados, en desarrollo y proyectados dentro del motor de análisis Intrinseca. Proporciona especificaciones técnicas precisas para cada métrica: fórmulas matemáticas, unidades, dependencias, estado de implementación y referencias bibliográficas.

### 1.1 Propósito de Este Documento

Los indicadores son **funciones derivadas** que transforman los eventos de Cambio Direccional (DC) en métricas cuantificables para análisis, trading algorítmico y aprendizaje automático. Este documento:

1. **Especifica** cada indicador con rigor matemático
2. **Documenta** el estado de implementación actual
3. **Identifica** discrepancias entre la literatura y la implementación
4. **Prioriza** indicadores pendientes de desarrollo

### 1.2 Fundamentos Teóricos

Para comprender los indicadores, es necesario dominar los conceptos fundamentales del paradigma de Cambio Direccional:

- **Umbral (θ):** Parámetro de sensibilidad
- **Punto Extremo (EXT):** Máximo/mínimo local
- **Punto de Confirmación (DCC):** Validación de reversión
- **Eventos DC y OS:** Fases de la tendencia

> **Referencia obligatoria:** Para las definiciones formales de estos conceptos primitivos —tanto en tiempo continuo (teórico) como en tiempo discreto (práctico)— y las salvedades de implementación, consulte el documento **`core/DC_FRAMEWORK.md`**.

### 1.3 Estructura de los Indicadores

Los indicadores de Intrinseca se organizan en dos niveles:

| Nivel             | Descripción                               | Ejemplo                         |
| ----------------- | ----------------------------------------- | ------------------------------- |
| **Event-Level**   | Calculados para cada evento DC individual | `overshoot`, `velocity`         |
| **Summary-Level** | Agregaciones sobre conjuntos de eventos   | `avg_duration`, `volatility_dc` |

Los indicadores event-level se computan mediante `with_columns()` en Polars; los summary-level mediante `select()`.

### 1.4 Dependencias Entre Indicadores

Los indicadores forman un **grafo acíclico dirigido (DAG)** de dependencias. El `IndicatorRegistry` resuelve estas dependencias topológicamente para garantizar el orden correcto de cómputo.

```
dc_magnitude ──────────────────────┬─→ dc_return ─────────┬─→ tmv
                                   │                      ├─→ avg_return
                                   │                      └─→ volatility_dc
                                   ├─→ dc_velocity
                                   └─→ event_magnitude ───┬─→ event_velocity

os_magnitude ──────────────────────┬─→ os_return
                                   ├─→ avg_os_magnitude
                                   ├─→ os_velocity
                                   └─→ event_magnitude ───┘

dc_time ───────────────────────────┬─→ dc_velocity
                                   ├─→ avg_dc_time
                                   └─→ event_time ────────→ event_velocity

os_time ───────────────────────────┬─→ os_velocity
                                   └─→ event_time
```

---

## 2. Indicadores Implementados

Los siguientes indicadores están completamente implementados y disponibles para uso en producción.

### 2.1 Indicadores a Nivel de Evento (Event-Level)

Estos indicadores se calculan para cada evento DC individual.

---

#### 2.1.1 DC Magnitude (A1)

| Atributo           | Valor                               |
| ------------------ | ----------------------------------- |
| **Nombre interno** | `dc_magnitude`                      |
| **Módulo**         | `indicators/metrics/event/price.py` |
| **Estado**         | ✅ Implementado                     |
| **Categoría**      | `event/price`                       |

##### Definición Teórica

El DC Magnitude mide el cambio de precio absoluto durante la fase DC, desde el punto de referencia hasta el punto de confirmación (DCC). Es equivalente al atributo A1 en Adegboye et al. (2017).

**Fórmula canónica (A1):**

$$\text{DC Magnitude}_N = P_{DCC,N} - P_{REF,N}$$

Donde:

- $P_{DCC,N}$ es el precio de confirmación del evento $N$ (`confirm_price`)
- $P_{REF,N}$ es el precio de referencia del evento $N$ (`reference_price`)

**Unidades:** Unidades de precio del activo subyacente.

**Interpretación:** El signo indica la dirección del movimiento:

- Positivo para upturns
- Negativo para downturns

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("confirm_price") - pl.col("reference_price")
```

**Columnas Silver utilizadas:** `confirm_price`, `reference_price`

**Relación:** `dc_magnitude / reference_price = dc_return`

**Referencias:** Adegboye et al. (2017) - Atributo A1.

---

#### 2.1.2 OS Magnitude

| Atributo           | Valor                               |
| ------------------ | ----------------------------------- |
| **Nombre interno** | `os_magnitude`                      |
| **Módulo**         | `indicators/metrics/event/price.py` |
| **Estado**         | ✅ Implementado                     |
| **Categoría**      | `event/price`                       |

##### Definición Teórica

El OS Magnitude mide la magnitud absoluta del movimiento de precio durante la fase OS, es decir, desde el punto de confirmación (DCC) hasta el punto extremo del mismo evento (Glattfelder et al., 2011).

**Fórmula canónica:**

$$\text{OS Magnitude}_N = P_{EXT,N} - P_{DCC,N}$$

Donde:

- $P_{EXT,N}$ es el precio extremo del evento $N$ (fin de la fase OS, último tick de `price_os`)
- $P_{DCC,N}$ es el precio de confirmación del evento $N$ (fin de la fase DC, último tick de `price_dc`)

**Estructura temporal del evento N:**

```
reference_price[N] → DC phase → confirm_price[N] → OS phase → extreme_price[N]
                                     (DCC)                          ↓
                                                         = reference_price[N+1]
```

**Unidades:** Unidades de precio del activo subyacente.

**Interpretación:** Un OS magnitude positivo indica que el precio continuó moviéndose en la dirección de la tendencia confirmada. La magnitud indica la "rentabilidad potencial" de seguir la tendencia después de la confirmación.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    # Ambos valores están en la misma fila del evento N
    return pl.col("extreme_price") - pl.col("confirm_price")
```

**Columnas Silver utilizadas:**

- `extreme_price`: Precio en el punto extremo (fin del OS)
- `confirm_price`: Precio en el punto de confirmación (DCC)

##### Salvedades

| Caso                      | Comportamiento                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| Último evento de la serie | `extreme_price = -1.0` (provisional); OS Magnitude inválido                                |
| OS Magnitude cero         | Ocurre cuando $P_{EXT,N} = P_{DCC,N}$; indica reversión inmediata sin movimiento adicional |

**Referencias:** Glattfelder et al. (2011), Tsang et al. (2015).

---

#### 2.1.3 Event Magnitude

| Atributo           | Valor                               |
| ------------------ | ----------------------------------- |
| **Nombre interno** | `event_magnitude`                   |
| **Módulo**         | `indicators/metrics/event/price.py` |
| **Estado**         | ✅ Implementado                     |
| **Categoría**      | `event/price`                       |
| **Dependencias**   | `dc_magnitude`, `os_magnitude`      |

##### Definición Teórica

El Event Magnitude mide el cambio de precio total absoluto a lo largo del evento completo (fases DC + OS), desde el punto de referencia hasta el punto extremo.

**Fórmula canónica:**

$$\text{Event Magnitude}_N = P_{EXT,N} - P_{REF,N}$$

Equivalentemente, por la estructura aditiva de las fases:

$$\text{Event Magnitude}_N = \text{DC Magnitude}_N + \text{OS Magnitude}_N$$

Donde:

- $P_{EXT,N}$ es el precio extremo del evento $N$ (`extreme_price`)
- $P_{REF,N}$ es el precio de referencia del evento $N$ (`reference_price`)

**Unidades:** Unidades de precio del activo subyacente.

**Interpretación:** El signo indica la dirección del movimiento total:

- Positivo para upturns
- Negativo para downturns

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("dc_magnitude") + pl.col("os_magnitude")
```

**Dependencias:** Requiere que `dc_magnitude` y `os_magnitude` estén calculados previamente.

**Relación con otros indicadores:**

- `event_magnitude / reference_price` = retorno total del evento
- `event_magnitude / event_time` = `event_velocity`

##### Salvedades

| Caso          | Comportamiento                                                 |
| ------------- | -------------------------------------------------------------- |
| Último evento | `os_magnitude` puede ser inválido → `event_magnitude` inválido |
| Evento sin OS | `event_magnitude = dc_magnitude` exactamente                   |

**Referencias:** Extensión Intrinseca basada en Glattfelder et al. (2011).

---

#### 2.1.4 DC Return

| Atributo           | Valor                               |
| ------------------ | ----------------------------------- |
| **Nombre interno** | `dc_return`                         |
| **Módulo**         | `indicators/metrics/event/price.py` |
| **Estado**         | ✅ Implementado                     |
| **Categoría**      | `event/price`                       |
| **Dependencias**   | `dc_magnitude`                      |

##### Definición Teórica

El DC Return cuantifica el retorno relativo (porcentual) del movimiento de precio durante la fase DC (Guillaume et al., 1997).

**Fórmula canónica:**

$$\text{DC Return}_N = \frac{\text{DC Magnitude}_N}{P_{REF,N}} = \frac{P_{DCC,N} - P_{REF,N}}{P_{REF,N}}$$

**Unidades:** Adimensional (proporción).

**Propiedad teórica:** Por construcción del algoritmo DC, $|\text{DC Return}| \geq \theta$.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("dc_magnitude") / pl.col("reference_price")
```

**Dependencias:** Requiere que `dc_magnitude` esté calculado previamente.

##### Salvedades

| Aspecto           | Comportamiento                                |
| ----------------- | --------------------------------------------- |
| Magnitud mínima   | ≥ θ debido a slippage                         |
| División por cero | No ocurre: `reference_price` siempre positivo |

**Referencias:** Guillaume et al. (1997), Tsang (2010).

---

#### 2.1.5 OS Return

| Atributo           | Valor                               |
| ------------------ | ----------------------------------- |
| **Nombre interno** | `os_return`                         |
| **Módulo**         | `indicators/metrics/event/price.py` |
| **Estado**         | ✅ Implementado                     |
| **Categoría**      | `event/price`                       |
| **Dependencias**   | `os_magnitude`                      |

##### Definición Teórica

El OS Return cuantifica el retorno relativo durante la fase de Overshoot, normalizado por el precio de confirmación (Tsang et al., 2015).

**Fórmula canónica:**

$$\text{OS Return}_N = \frac{\text{OS Magnitude}_N}{P_{DCC,N}} = \frac{P_{EXT,N} - P_{DCC,N}}{P_{DCC,N}}$$

**Unidades:** Adimensional (proporción).

**Interpretación:** Mide la "ganancia" relativa obtenible por un trader que entra en la posición exactamente en el punto de confirmación (DCC) y sale en el punto extremo del mismo evento.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("os_magnitude") / pl.col("confirm_price")
```

**Dependencia:** Requiere que `os_magnitude` esté calculado previamente.

**Columnas utilizadas:**

- `os_magnitude`: Indicador calculado
- `confirm_price`: Columna Silver

##### Salvedades

| Caso              | Comportamiento                      |
| ----------------- | ----------------------------------- |
| Último evento     | `null` (heredado de `os_magnitude`) |
| OS Magnitude cero | OS Return = 0 exactamente           |

**Referencias:** Tsang et al. (2015).

---

#### 2.1.6 DC Slippage (Facial)

| Atributo           | Valor                               |
| ------------------ | ----------------------------------- |
| **Nombre interno** | `dc_slippage`                       |
| **Módulo**         | `indicators/metrics/event/price.py` |
| **Estado**         | ✅ Implementado                     |
| **Categoría**      | `event/price`                       |
| **Parámetros**     | `theta` (default: 0.005)            |

##### Definición Teórica

El DC Slippage (Facial) cuantifica la diferencia entre el precio de confirmación real observado y el precio de confirmación teórico (exactamente en el umbral θ).

**Fórmula canónica:**

Para un upturn (+1):
$$\text{Slippage}_N = P_{DCC,N} - P_{REF,N} \times (1 + \theta)$$

Para un downturn (-1):
$$\text{Slippage}_N = P_{DCC,N} - P_{REF,N} \times (1 - \theta)$$

**Fórmula combinada:**
$$\text{Slippage}_N = P_{DCC,N} - P_{REF,N} \times (1 + \text{event\_type} \times \theta)$$

**Unidades:** Unidades de precio del activo subyacente.

**Interpretación:**

- Slippage positivo → El precio "saltó" más allá del umbral teórico
- Slippage ≈ 0 → Mercado continuo con alta liquidez
- Slippage alto → Gaps, flash events, o baja liquidez

##### Implementación Práctica

```python
def __init__(self, theta: float = 0.005):
    self.theta = theta

def get_expression(self) -> pl.Expr:
    theoretical_confirm = pl.col("reference_price") * (
        1.0 + pl.col("event_type").cast(pl.Float64) * self.theta
    )
    return pl.col("confirm_price") - theoretical_confirm
```

**Columnas Silver utilizadas:**

- `confirm_price`: Precio real de confirmación (conservador)
- `reference_price`: Precio extremo del evento anterior
- `event_type`: Dirección del evento (+1 upturn, -1 downturn)

##### Salvedades

| Aspecto              | Comportamiento                                                      |
| -------------------- | ------------------------------------------------------------------- |
| Slippage siempre ≥ 0 | Por construcción (política conservadora selecciona el mejor precio) |
| Dependencia de θ     | Debe coincidir con el θ usado en el procesamiento                   |

**Referencias:** Extensión Intrinseca.

---

#### 2.1.7 DC Slippage (Real)

| Atributo           | Valor                               |
| ------------------ | ----------------------------------- |
| **Nombre interno** | `dc_slippage_real`                  |
| **Módulo**         | `indicators/metrics/event/price.py` |
| **Estado**         | ✅ Implementado                     |
| **Categoría**      | `event/price`                       |
| **Parámetros**     | `theta` (default: 0.005)            |

##### Definición Teórica

El DC Slippage (Real) cuantifica el **peor caso** de slippage: la diferencia entre el precio más lejano del umbral teórico (entre todos los ticks del instante de confirmación) y el precio teórico.

A diferencia del Slippage Facial que usa el precio conservador (`confirm_price`), este indicador busca el precio que maximiza la desviación del umbral.

**Fórmula:**

Para un upturn (+1):
$$P_{worst} = \max\{P_i : T_i = T_{DCC}\}$$
$$\text{Slippage Real}_N = P_{worst} - P_{REF,N} \times (1 + \theta)$$

Para un downturn (-1):
$$P_{worst} = \min\{P_i : T_i = T_{DCC}\}$$
$$\text{Slippage Real}_N = P_{worst} - P_{REF,N} \times (1 - \theta)$$

**Unidades:** Unidades de precio del activo subyacente.

**Interpretación:**

- Mide el máximo slippage posible que un trader pudo haber experimentado
- La diferencia `(Slippage Real - Slippage Facial)` indica la **dispersión de precios** en el instante de confirmación (ruido de microestructura)
- Si `Real == Facial`, había un solo precio en el instante de confirmación

##### Implementación Práctica

```python
def _compute_worst_confirm_price(price_dc, time_dc, confirm_time, event_type):
    prices_at_confirm = [p for p, t in zip(price_dc, time_dc) if t == confirm_time]
    if not prices_at_confirm:
        return None
    return max(prices_at_confirm) if event_type == 1 else min(prices_at_confirm)

def get_expression(self) -> pl.Expr:
    worst_price = pl.struct(["price_dc", "time_dc", "confirm_time", "event_type"]).map_elements(
        lambda row: _compute_worst_confirm_price(
            row["price_dc"], row["time_dc"], row["confirm_time"], row["event_type"]
        ),
        return_dtype=pl.Float64,
    )
    theoretical_confirm = pl.col("reference_price") * (
        1.0 + pl.col("event_type").cast(pl.Float64) * self.theta
    )
    return worst_price - theoretical_confirm
```

**Columnas Silver utilizadas:**

- `price_dc`: Lista de precios durante la fase DC
- `time_dc`: Lista de timestamps durante la fase DC
- `confirm_time`: Timestamp de confirmación
- `reference_price`: Precio extremo del evento anterior
- `event_type`: Dirección del evento (+1 upturn, -1 downturn)

##### Salvedades

| Aspecto                    | Comportamiento                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Requiere corrección kernel | El kernel debe incluir TODOS los ticks del instante de confirmación en `price_dc` (corregido via `last_same_ts_idx`) |
| Rendimiento                | Usa `map_elements` (Python puro), más lento que expresiones nativas                                                  |
| Dependencia de θ           | Debe coincidir con el θ usado en el procesamiento                                                                    |

##### Relación con Slippage Facial

| Escenario                         | Slippage Facial | Slippage Real |
| --------------------------------- | --------------- | ------------- |
| Un solo tick en confirmación      | X               | X (iguales)   |
| Múltiples ticks, mismo precio     | X               | X (iguales)   |
| Múltiples ticks, precios diversos | Conservador     | Peor caso     |

**Referencias:** Extensión Intrinseca.

---

#### 2.1.8 DC Time

| Atributo           | Valor                              |
| ------------------ | ---------------------------------- |
| **Nombre interno** | `dc_time`                          |
| **Módulo**         | `indicators/metrics/event/time.py` |
| **Estado**         | ✅ Implementado                    |
| **Categoría**      | `event/time`                       |

##### Definición Teórica

DC Time mide el intervalo de tiempo físico transcurrido durante la fase DC, desde el momento del punto de referencia (inicio del DC) hasta el momento de la confirmación (Glattfelder et al., 2011).

**Fórmula canónica:**

$$\text{DC Time}_N = T_{DCC,N} - T_{REF,N} = T_{DCC,N} - T_{EXT,N-1}$$

Donde:

- $T_{REF,N}$ es el timestamp del punto de referencia (inicio del DC) = `reference_time[N]` = $T_{EXT,N-1}$
- $T_{DCC,N}$ es el timestamp de confirmación (fin del DC) = `confirm_time[N]`

**Unidades:** Tiempo (segundos en literatura; nanosegundos en implementación).

**Equivalencia en literatura:** Corresponde al atributo **A2 (DCtime)** en la taxonomía de Adegboye et al. (2017).

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("confirm_time") - pl.col("reference_time")
```

**Columnas Silver utilizadas:**

- `confirm_time`: Timestamp de confirmación (DCC, nanosegundos desde epoch, Int64)
- `reference_time`: Timestamp del punto de referencia (inicio del DC, Int64)

**Unidades de implementación:** Nanosegundos (Int64). Para convertir a segundos: dividir por $10^9$.

##### Salvedades

| Caso            | Teoría                | Práctica                                |
| --------------- | --------------------- | --------------------------------------- |
| Duración mínima | > 0 (tiempo continuo) | ≥ 0 (puede ser 0 en gaps/flash events)  |
| Flash event     | No definido           | DC Time = 0 cuando $T_{DCC} = T_{REF}$  |
| Overflow        | No aplica             | Int64 soporta ~292 años en nanosegundos |

**Referencias:** Glattfelder et al. (2011), Adegboye et al. (2017).

---

#### 2.1.9 OS Time

| Atributo           | Valor                              |
| ------------------ | ---------------------------------- |
| **Nombre interno** | `os_time`                          |
| **Módulo**         | `indicators/metrics/event/time.py` |
| **Estado**         | ✅ Implementado                    |
| **Categoría**      | `event/time`                       |

##### Definición Teórica

OS Time mide el intervalo de tiempo físico transcurrido durante la fase Overshoot, desde el momento de la confirmación (DCC) hasta el momento del punto extremo.

**Fórmula canónica:**

$$\text{OS Time}_N = T_{EXT,N} - T_{DCC,N}$$

Donde:

- $T_{DCC,N}$ es el timestamp de confirmación (fin del DC / inicio del OS) = `confirm_time[N]`
- $T_{EXT,N}$ es el timestamp del punto extremo (fin del OS) = `extreme_time[N]`

**Unidades:** Nanosegundos (Int64).

> [!NOTE]
> Este indicador **no tiene equivalente directo en la literatura Q1**. Es una extensión de Intrinseca para completitud funcional.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("extreme_time") - pl.col("confirm_time")
```

**Columnas Silver utilizadas:**

- `extreme_time`: Timestamp del punto extremo (fin del OS, Int64)
- `confirm_time`: Timestamp de confirmación (DCC, Int64)

##### Salvedades

| Caso           | Comportamiento                                            |
| -------------- | --------------------------------------------------------- |
| Último evento  | `extreme_time = -1` (provisional) → OS Time inválido (<0) |
| Overshoot cero | OS Time = 0 exactamente                                   |

**Referencias:** N/A (extensión Intrinseca).

---

#### 2.1.10 Event Time

| Atributo           | Valor                              |
| ------------------ | ---------------------------------- |
| **Nombre interno** | `event_time`                       |
| **Módulo**         | `indicators/metrics/event/time.py` |
| **Estado**         | ✅ Implementado                    |
| **Categoría**      | `event/time`                       |
| **Dependencias**   | `dc_time`, `os_time`               |

##### Definición Teórica

Event Time mide la duración total del evento DC completo (fases DC + OS), desde el punto de referencia hasta el punto extremo.

**Fórmula canónica:**

$$\text{Event Time}_N = \text{DC Time}_N + \text{OS Time}_N = T_{EXT,N} - T_{REF,N}$$

**Unidades:** Nanosegundos (Int64).

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("dc_time") + pl.col("os_time")
```

**Dependencias:** Requiere que `dc_time` y `os_time` estén calculados previamente.

##### Salvedades

| Caso          | Comportamiento                        |
| ------------- | ------------------------------------- |
| Último evento | Heredado de `os_time`: valor inválido |

**Referencias:** N/A (extensión Intrinseca).

---

#### 2.1.11 DC Velocity (A3)

| Atributo           | Valor                              |
| ------------------ | ---------------------------------- |
| **Nombre interno** | `dc_velocity`                      |
| **Módulo**         | `indicators/metrics/event/time.py` |
| **Estado**         | ✅ Implementado                    |
| **Categoría**      | `event/time`                       |
| **Dependencias**   | `dc_time`, `dc_magnitude`          |

##### Definición Teórica

DC Velocity mide la tasa de cambio de precio por unidad de tiempo durante la fase DC. Representa el "impulso" o "momentum" de la reversión inicial (Adegboye et al., 2017).

**Fórmula canónica (A3 / σ₀):**

$$\text{DC Velocity}_N = \frac{A1_N}{A2_N} = \frac{P_{DCC,N} - P_{REF,N}}{T_{DCC,N} - T_{REF,N}} = \frac{\text{dc\_magnitude}}{\text{dc\_time}}$$

**Unidades:** Unidades de precio por segundo.

**Interpretación:** Velocidades altas se correlacionan estadísticamente con fases OS más cortas.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    dc_time_sec = pl.col("dc_time") / 1_000_000_000.0

    return pl.when(dc_time_sec > 0).then(
        pl.col("dc_magnitude") / dc_time_sec
    ).otherwise(0.0)
```

**Dependencias:** Requiere `dc_time` y `dc_magnitude` calculados previamente.

##### Salvedades

| Aspecto           | Comportamiento |
| ----------------- | -------------- |
| División por cero | Retorna 0.0    |
| Flash events      | Velocity = 0   |

**Referencias:** Adegboye et al. (2017) - Atributo A3.

---

#### 2.1.12 OS Velocity

| Atributo           | Valor                              |
| ------------------ | ---------------------------------- |
| **Nombre interno** | `os_velocity`                      |
| **Módulo**         | `indicators/metrics/event/time.py` |
| **Estado**         | ✅ Implementado                    |
| **Categoría**      | `event/time`                       |
| **Dependencias**   | `os_time`, `os_magnitude`          |

##### Definición Teórica

OS Velocity mide la tasa de cambio de precio por unidad de tiempo durante la fase Overshoot.

**Fórmula:**

$$\text{OS Velocity}_N = \frac{\text{OS Magnitude}_N}{\text{OS Time}_N} = \frac{P_{EXT,N} - P_{DCC,N}}{T_{EXT,N} - T_{DCC,N}}$$

**Unidades:** Unidades de precio por segundo.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    os_time_sec = pl.col("os_time") / 1_000_000_000.0

    return pl.when(os_time_sec > 0).then(
        pl.col("os_magnitude") / os_time_sec
    ).otherwise(0.0)
```

**Dependencias:** Requiere `os_time` y `os_magnitude` calculados previamente.

##### Salvedades

| Aspecto           | Comportamiento      |
| ----------------- | ------------------- |
| División por cero | Retorna 0.0         |
| Último evento     | Heredado de os_time |

**Referencias:** N/A (extensión Intrinseca).

---

#### 2.1.13 Event Velocity

| Atributo           | Valor                              |
| ------------------ | ---------------------------------- |
| **Nombre interno** | `event_velocity`                   |
| **Módulo**         | `indicators/metrics/event/time.py` |
| **Estado**         | ✅ Implementado                    |
| **Categoría**      | `event/time`                       |
| **Dependencias**   | `event_time`                       |

##### Definición Teórica

Event Velocity mide la tasa de cambio de precio total del evento completo.

**Fórmula:**

$$\text{Event Velocity}_N = \frac{P_{EXT,N} - P_{REF,N}}{\text{Event Time}_N}$$

**Unidades:** Unidades de precio por segundo.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    event_time_sec = pl.col("event_time") / 1_000_000_000.0
    total_magnitude = pl.col("extreme_price") - pl.col("reference_price")

    return pl.when(event_time_sec > 0).then(
        total_magnitude / event_time_sec
    ).otherwise(0.0)
```

**Dependencias:** Requiere `event_time` calculado previamente.

##### Salvedades

| Aspecto           | Comportamiento         |
| ----------------- | ---------------------- |
| División por cero | Retorna 0.0            |
| Último evento     | Heredado de event_time |

**Referencias:** N/A (extensión Intrinseca).

**Discrepancia:** La definición canónica de A3 usa valor absoluto. La implementación actual preserva el signo del cambio de precio, lo que permite distinguir upturns (positivo) de downturns (negativo).

**Referencias:** Adegboye et al. (2017).

---

#### 2.1.14 Runs Count

| Atributo           | Valor                              |
| ------------------ | ---------------------------------- |
| **Nombre interno** | `runs_count`                       |
| **Módulo**         | `indicators/metrics/event/tick.py` |
| **Estado**         | ✅ Implementado                    |
| **Categoría**      | `event/tick`                       |

##### Definición Teórica

Runs Count cuantifica el número de "cruces de grilla direccional" durante la fase OS de un evento. Un cruce ocurre cuando el precio se mueve al menos θ en la dirección de la tendencia desde el último punto de referencia.

**Nota:** Este indicador **no aparece en la literatura canónica de DC**. Es una extensión propietaria de Intrinseca para capturar la microestructura del evento.

**Interpretación:** Un alto número de runs indica un movimiento sostenido y direccional; un bajo número sugiere un movimiento abrupto seguido de consolidación.

##### Implementación Práctica

```python
def _count_runs(prices: list, event_type: int, theta: float = 0.005) -> int:
    if prices is None or len(prices) < 2:
        return 0

    ref = prices[0]
    mult = (1.0 + theta) if event_type == 1 else (1.0 - theta)
    count = 0

    for p in prices[1:]:
        threshold = ref * mult
        if (event_type == 1 and p >= threshold) or (event_type == -1 and p <= threshold):
            count += 1
            ref = p

    return count

def get_expression(self) -> pl.Expr:
    return pl.struct(["price_os", "event_type"]).map_elements(
        lambda row: _count_runs(row["price_os"], row["event_type"]),
        return_dtype=pl.Int64
    )
```

**Columnas Silver utilizadas:**

- `price_os`: Lista de precios durante la fase OS (List[Float64])
- `event_type`: Tipo de evento (1 = upturn, -1 = downturn)

**Unidades:** Número entero no negativo.

##### Salvedades

| Caso                         | Comportamiento                                    |
| ---------------------------- | ------------------------------------------------- |
| `price_os` es `null` o vacío | Retorna 0                                         |
| Lista con un solo precio     | Retorna 0                                         |
| θ hardcodeado                | Actualmente usa θ = 0.005; debería parametrizarse |

**Limitación de rendimiento:** Usa `map_elements` (Python puro), lo cual es más lento que expresiones nativas de Polars. Considerar vectorización futura.

---

### 2.2 Indicadores Agregados (Summary-Level)

Estos indicadores colapsan el DataFrame de eventos en estadísticas resumidas. Se calculan mediante `select()` en lugar de `with_columns()`.

---

#### 2.2.1 TMV (Total Movement Value)

| Atributo           | Valor                                   |
| ------------------ | --------------------------------------- |
| **Nombre interno** | `tmv`                                   |
| **Módulo**         | `indicators/metrics/summary/stats.py`   |
| **Estado**         | ⚠️ Implementado (variante simplificada) |
| **Categoría**      | `summary/stats`                         |
| **Dependencias**   | `dc_return`                             |

##### Definición Teórica

El Total Movement Value canónico es la magnitud del movimiento total de una tendencia (de extremo a extremo), **normalizada por el umbral θ** (Tsang et al., 2015).

**Fórmula canónica (por evento):**

$$\text{TMV}_i = \frac{1}{\theta} \left| \frac{P_{EXT,i+1} - P_{EXT,i}}{P_{EXT,i}} \right|$$

**Interpretación canónica:**

- TMV = 1.0 → No hubo overshoot (movimiento mínimo = θ)
- TMV = 2.0 → El precio se movió el doble del umbral

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("dc_return").abs().sum()
```

**Dependencia:** Requiere que `dc_return` esté calculado previamente.

##### Salvedades

| Aspecto             | Definición Canónica    | Implementación Actual     |
| ------------------- | ---------------------- | ------------------------- |
| Nivel               | Por evento             | Agregado (suma total)     |
| Normalización por θ | Sí                     | **No**                    |
| Incluye OS          | Sí (extremo a extremo) | **No** (solo fase DC)     |
| Unidades            | Adimensional           | Adimensional (proporción) |

**⚠️ Discrepancia significativa:** La implementación actual calcula una **métrica agregada de volatilidad** (suma de retornos DC absolutos), no el TMV canónico. Para obtener el TMV por evento según la literatura, se requiere implementar el indicador descrito en la sección 4.1.2.

**Referencias:** Tsang et al. (2015), Tsang & Ma (2021).

---

#### 2.2.2 Average Duration

| Atributo           | Valor                                 |
| ------------------ | ------------------------------------- |
| **Nombre interno** | `avg_duration`                        |
| **Módulo**         | `indicators/metrics/summary/stats.py` |
| **Estado**         | ✅ Implementado                       |
| **Categoría**      | `summary/stats`                       |
| **Dependencias**   | `duration_ns`                         |

##### Definición Teórica

Promedio aritmético de las duraciones de todos los eventos DC en el conjunto de datos.

**Fórmula:**

$$\overline{\text{Duration}} = \frac{1}{n} \sum_{i=1}^{n} \text{Duration}_i$$

**Interpretación:** Proporciona una medida de la "velocidad típica" del mercado para confirmar cambios de tendencia bajo el umbral θ especificado.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("duration_ns").mean()
```

**Dependencia:** Requiere que `duration_ns` esté calculado previamente.

**Unidades:** Nanosegundos (Float64 por ser promedio).

##### Salvedades

| Aspecto                  | Comportamiento                                       |
| ------------------------ | ---------------------------------------------------- |
| Eventos con Duration = 0 | Incluidos en el promedio (pueden sesgar hacia abajo) |
| Valores null             | Excluidos automáticamente por Polars `.mean()`       |

---

#### 2.2.3 Average Return

| Atributo           | Valor                                 |
| ------------------ | ------------------------------------- |
| **Nombre interno** | `avg_return`                          |
| **Módulo**         | `indicators/metrics/summary/stats.py` |
| **Estado**         | ✅ Implementado                       |
| **Categoría**      | `summary/stats`                       |
| **Dependencias**   | `dc_return`                           |

##### Definición Teórica

Promedio aritmético de los retornos DC de todos los eventos.

**Fórmula:**

$$\overline{\text{DC Return}} = \frac{1}{n} \sum_{i=1}^{n} \text{DC Return}_i$$

**Interpretación:** Un valor cercano a cero indica simetría entre upturns y downturns. Valores positivos o negativos persistentes sugieren un sesgo direccional en el período analizado.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("dc_return").mean()
```

**Dependencia:** Requiere que `dc_return` esté calculado previamente.

**Unidades:** Adimensional (Float64).

##### Salvedades

| Aspecto               | Comportamiento                                                               |
| --------------------- | ---------------------------------------------------------------------------- |
| Cancelación de signos | Upturns (+) y downturns (-) se cancelan; usar `abs()` para magnitud promedio |
| Valores null          | Excluidos automáticamente                                                    |

---

#### 2.2.4 Average Overshoot

| Atributo           | Valor                                 |
| ------------------ | ------------------------------------- |
| **Nombre interno** | `avg_overshoot`                       |
| **Módulo**         | `indicators/metrics/summary/stats.py` |
| **Estado**         | ✅ Implementado                       |
| **Categoría**      | `summary/stats`                       |
| **Dependencias**   | `overshoot`                           |

##### Definición Teórica

Promedio aritmético de los overshoots de todos los eventos.

**Fórmula:**

$$\overline{\text{Overshoot}} = \frac{1}{n} \sum_{i=1}^{n} \text{Overshoot}_i$$

**Ley de escala (Glattfelder et al., 2011):** En mercados eficientes, $\langle \text{Overshoot} \rangle \approx \theta \times P_{promedio}$.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("overshoot").mean()
```

**Dependencia:** Requiere que `overshoot` esté calculado previamente.

**Unidades:** Unidades de precio (Float64).

##### Salvedades

| Aspecto              | Comportamiento                                                               |
| -------------------- | ---------------------------------------------------------------------------- |
| Último evento (null) | Excluido del promedio                                                        |
| Overshoots cero      | Incluidos; pueden indicar régimen de reversión a la media                    |
| Interpretación       | Valores bajos → reversiones rápidas; valores altos → tendencias persistentes |

---

#### 2.2.5 Volatility DC

| Atributo           | Valor                                 |
| ------------------ | ------------------------------------- |
| **Nombre interno** | `volatility_dc`                       |
| **Módulo**         | `indicators/metrics/summary/stats.py` |
| **Estado**         | ✅ Implementado                       |
| **Categoría**      | `summary/stats`                       |
| **Dependencias**   | `dc_return`                           |

##### Definición Teórica

Desviación estándar de los retornos DC, utilizada como proxy de volatilidad en el espacio de tiempo intrínseco (Guillaume et al., 1997).

**Fórmula (desviación estándar muestral):**

$$\sigma_{DC} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (\text{DC Return}_i - \overline{\text{DC Return}})^2}$$

**Interpretación:** A diferencia de la volatilidad tradicional (calculada sobre retornos en tiempo físico), esta métrica captura la dispersión de magnitudes de los eventos DC, proporcionando una medida de volatilidad **agnóstica a la escala temporal**.

##### Implementación Práctica

```python
def get_expression(self) -> pl.Expr:
    return pl.col("dc_return").std()
```

**Dependencia:** Requiere que `dc_return` esté calculado previamente.

**Unidades:** Adimensional (Float64).

##### Salvedades

| Aspecto           | Comportamiento                           |
| ----------------- | ---------------------------------------- |
| Tipo de std       | Polars usa ddof=1 por defecto (muestral) |
| Mínimo de eventos | Requiere n ≥ 2 para resultado válido     |
| Valores null      | Excluidos automáticamente                |

**Nota:** Por construcción, $|\text{DC Return}| \geq \theta$, por lo que la volatilidad tiene un piso implícito relacionado con el umbral.

**Referencias:** Guillaume et al. (1997).

---

## 3. Indicadores en Desarrollo

Los siguientes indicadores están parcialmente implementados o tienen placeholders en el código.

### 3.1 Indicadores de Series Temporales Intra-Evento

Ubicación: `indicators/metrics/event/series.py`

Estos indicadores operan sobre las columnas de listas anidadas (`price_dc`, `time_dc`, `price_os`, `time_os`) para extraer características de la microestructura del evento.

| Indicador             | Descripción                                           | Estado         |
| --------------------- | ----------------------------------------------------- | -------------- |
| `FourierDominantFreq` | Frecuencia dominante vía FFT de la serie intra-evento | 📋 Planificado |
| `WaveletEnergy`       | Energía por escala wavelet                            | 📋 Planificado |
| `AutoCorrelation`     | Autocorrelación lag-1 de retornos intra-evento        | 📋 Planificado |
| `SeriesEntropy`       | Entropía de Shannon de los retornos intra-evento      | 📋 Planificado |

---

## 4. Indicadores por Implementar

Los siguientes indicadores están documentados en la literatura pero no tienen implementación actual.

### 4.1 Indicadores de Magnitud Normalizada

---

#### 4.1.1 Total Move (TM)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Alta               |

**Definición:**

El Total Move es la magnitud absoluta del desplazamiento de precio desde un punto extremo hasta el siguiente punto extremo. Representa la "vida útil completa" de una tendencia en el marco DC (Tsang et al., 2015).

**Fórmula:**

$$\text{TM}_i = |P_{EXT,i+1} - P_{EXT,i}|$$

Equivalentemente:
$$\text{TM}_i = |\text{DC}_i| + |\text{OS}_i|$$

**Unidades:** Unidades de precio.

**Referencias:** Tsang et al. (2015).

---

#### 4.1.2 TMV por Evento (Canónico)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Alta               |

**Definición:**

El Total Movement Value canónico es el Total Move normalizado por el umbral θ, expresando la magnitud en "unidades de umbral" (Tsang et al., 2015).

**Fórmula:**

$$\text{TMV}_i = \frac{1}{\theta} \left| \frac{P_{EXT,i+1} - P_{EXT,i}}{P_{EXT,i}} \right|$$

**Unidades:** Adimensional.

**Interpretación:** TMV = 1.0 implica movimiento mínimo (sin overshoot). Valores mayores indican tendencias que exceden el umbral de confirmación.

**Referencias:** Tsang et al. (2015).

---

#### 4.1.3 OSV (Overshoot Value)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Alta               |

**Definición:**

El Overshoot Value es la magnitud del overshoot normalizada por el umbral θ (Tsang et al., 2015).

**Fórmula:**

$$\text{OSV}_i = \frac{1}{\theta} \left| \frac{P_{EXT,i+1} - P_{DCC,i}}{P_{DCC,i}} \right|$$

**Unidades:** Adimensional.

**Interpretación:** Mide cuántas "unidades de umbral" recorrió el precio después de la confirmación. Un OSV promedio de 1.0 a través de miles de eventos es consistente con la **ley de escala del factor 2** (Glattfelder et al., 2011).

**Referencias:** Glattfelder et al. (2011), Tsang et al. (2015).

---

#### 4.1.4 aTMV (Active TMV)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Media              |

**Definición:**

El Active TMV es una variante dinámica del TMV calculada en tiempo real con el precio actual, sin esperar a que la tendencia termine. Es esencial para gestión de riesgo en vivo (Tsang & Ma, 2021).

**Fórmula:**

$$\text{aTMV}(t) = \frac{1}{\theta} \left| \frac{P(t) - P_{EXT}}{P_{EXT}} \right|$$

Donde $P(t)$ es el precio actual y $P_{EXT}$ es el último extremo confirmado.

**Unidades:** Adimensional.

**Interpretación:** Funciona como un "termómetro" de la tendencia activa. Estudios empíricos muestran que la probabilidad de reversión aumenta exponencialmente cuando aTMV cruza ciertos umbrales (e.g., 1.7, 2.5) (Tsang & Ma, 2021).

**Referencias:** Tsang & Ma (2021).

---

### 4.2 Indicadores de Tiempo y Frecuencia

---

#### 4.2.1 NDC (Number of Directional Changes)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Alta               |

**Definición:**

NDC cuantifica el número de eventos DC observados en un período de tiempo físico determinado. Es la medida fundamental de volatilidad en tiempo intrínseco (Guillaume et al., 1997; Aloud et al., 2012).

**Fórmula:**

$$\text{NDC}_{[t_1, t_2]} = |\{i : T_{DCC,i} \in [t_1, t_2]\}|$$

**Unidades:** Número entero.

**Interpretación:** NDC alto indica un mercado "nervioso" con reversiones frecuentes. NDC bajo sugiere tendencias persistentes. La relación entre NDC y θ sigue leyes de escala bien documentadas.

**Referencias:** Guillaume et al. (1997), Aloud et al. (2012).

---

#### 4.2.2 AT (Accumulated Time)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Media              |

**Definición:**

AT mide la asimetría temporal entre el tiempo que el mercado pasa en tendencias alcistas versus bajistas (Kampouridis, 2025).

**Fórmula:**

$$\text{AT}_{[t_1, t_2]} = \sum_{i \in \text{upturns}} \text{Duration}_i - \sum_{j \in \text{downturns}} \text{Duration}_j$$

**Unidades:** Nanosegundos (o unidad temporal elegida).

**Interpretación:** AT positivo indica que las subidas son más lentas que las bajadas (o viceversa). Útil para detectar asimetrías en la dinámica del mercado.

**Referencias:** Kampouridis (2025).

---

### 4.3 Indicadores de Microestructura para Machine Learning

Estos indicadores, definidos por Adegboye et al. (2017), están diseñados para construir vectores de características para modelos de clasificación y regresión.

---

#### 4.3.1 A1 (DC Price)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Alta               |

**Definición:**

A1 es la diferencia absoluta de precio entre el punto extremo y el punto de confirmación.

**Fórmula:**

$$A1_i = |P_{DCC,i} - P_{EXT,i}|$$

**Unidades:** Unidades de precio.

**Diferencia con DC Return:** A1 es absoluto (no relativo) y captura gaps de liquidez donde el precio "salta" más allá del umbral teórico.

**Referencias:** Adegboye et al. (2017).

---

#### 4.3.2 A4 (DC t-1 Price)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Media              |

**Definición:**

A4 registra el precio de confirmación del evento inmediatamente anterior.

**Fórmula:**

$$A4_i = P_{DCC,i-1}$$

**Unidades:** Unidades de precio.

**Interpretación:** Permite detectar patrones de "higher highs / lower lows" en el espacio DC.

**Referencias:** Adegboye et al. (2017).

---

#### 4.3.3 A5 (DC t-1 OS Flag)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Media              |

**Definición:**

A5 es un indicador binario que señala si el evento anterior tuvo un overshoot significativo.

**Fórmula:**

$$A5_i = \begin{cases} 1 & \text{si } |\text{Overshoot}_{i-1}| > 0 \\ 0 & \text{en caso contrario} \end{cases}$$

**Unidades:** Binario {0, 1}.

**Interpretación:** Captura patrones de alternancia entre eventos con y sin overshoot.

**Referencias:** Adegboye et al. (2017).

---

#### 4.3.4 A6 (Flash Event Flag)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Media              |

**Definición:**

A6 se activa cuando el tiempo de extremo y confirmación son idénticos (Duration = 0), indicando un "flash crash" o gap de apertura.

**Fórmula:**

$$A6_i = \begin{cases} 1 & \text{si } T_{DCC,i} = T_{EXT,i} \\ 0 & \text{en caso contrario} \end{cases}$$

**Unidades:** Binario {0, 1}.

**Interpretación:** Estos eventos representan rupturas de la continuidad estadística y requieren tratamiento especial.

**Referencias:** Adegboye et al. (2017).

---

### 4.4 Indicadores de Régimen de Mercado

---

#### 4.4.1 CDC (Coastline)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Alta               |

**Definición:**

La métrica Coastline, inspirada en la geometría fractal de Mandelbrot, suma los valores absolutos de todos los movimientos totales en un período. Representa el "camino total" recorrido por el precio y el máximo retorno teórico posible (Glattfelder et al., 2011).

**Fórmula:**

$$\text{CDC}(\theta) = \sum_{i=1}^{\text{NDC}} |\text{TMV}_i|$$

**Unidades:** Adimensional (si usa TMV) o unidades de precio (si usa TM).

**Interpretación:** CDC cuantifica la "energía total" disipada por el mercado. Es independiente de si el precio neto subió o bajó.

**Referencias:** Glattfelder et al. (2011).

---

#### 4.4.2 mRV (Micro-market Relative Volatility)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Baja               |

**Definición:**

mRV evalúa la volatilidad relativa entre dos mercados diferentes usando exclusivamente la frecuencia y magnitud de sus eventos DC, eliminando la necesidad de sincronización temporal (Li, 2022).

**Fórmula:**

$$\text{mRV}_{A,B} = \frac{\sum |\text{TMV}_A|}{\sum |\text{TMV}_B|}$$

**Unidades:** Adimensional (ratio).

**Interpretación:** Permite comparar la "actividad intrínseca" de mercados con diferentes horarios de operación.

**Referencias:** Li (2022).

---

#### 4.4.3 SMQ (Scale of Market Quakes)

| Atributo      | Valor              |
| ------------- | ------------------ |
| **Estado**    | ❌ No implementado |
| **Prioridad** | Baja               |

**Definición:**

SMQ es un indicador inspirado en la escala de Richter sismológica, diseñado para cuantificar el impacto de eventos noticiosos (Bisig et al., 2009).

**Fórmula:**

$$\text{SMQ} = \frac{|\text{OS}|}{|\text{DC}|}$$

**Unidades:** Adimensional (ratio).

**Interpretación:** Valores muy superiores a 1.0 indican un "terremoto de mercado" donde el overshoot es desproporcionado respecto al movimiento de confirmación.

**Referencias:** Bisig et al. (2009).

---

## 5. Matriz de Cobertura

| Indicador        | Literatura | Implementado | Prioridad |
| ---------------- | ---------- | ------------ | --------- |
| DC Magnitude     | ✅         | ✅           | -         |
| OS Magnitude     | ✅         | ✅           | -         |
| Event Magnitude  | ❌         | ✅           | -         |
| DC Return        | ✅         | ✅           | -         |
| OS Return        | ✅         | ✅           | -         |
| DC Slippage      | ❌         | ✅           | -         |
| DC Slippage Real | ❌         | ✅           | -         |
| DC Time (A2)     | ✅         | ✅           | -         |
| OS Time          | ❌         | ✅           | -         |
| Event Time       | ❌         | ✅           | -         |
| DC Velocity (A3) | ✅         | ✅           | -         |
| OS Velocity      | ❌         | ✅           | -         |
| Event Velocity   | ❌         | ✅           | -         |
| Runs Count       | ❌         | ✅           | -         |
| TMV (agregado)   | ⚠️         | ⚠️           | Alta      |
| Avg Duration     | ❌         | ✅           | -         |
| Avg Return       | ❌         | ✅           | -         |
| Avg Overshoot    | ❌         | ✅           | -         |
| Volatility DC    | ✅         | ✅           | -         |
| Upturn Ratio     | ❌         | ✅           | -         |
| TM (Total Move)  | ✅         | ✅           | -         |
| TMV (por evento) | ✅         | ✅           | -         |
| OSV              | ✅         | ✅           | -         |
| aTMV             | ✅         | ❌           | Media     |
| NDC              | ✅         | ✅           | -         |
| AT               | ✅         | ✅           | -         |
| A1 (DC Price)    | ✅         | ✅           | -         |
| A4               | ✅         | ✅           | -         |
| A5               | ✅         | ✅           | -         |
| A6               | ✅         | ✅           | -         |
| CDC              | ✅         | ✅           | -         |
| mRV              | ✅         | ❌           | Baja      |
| SMQ              | ✅         | ❌           | Baja      |

**Leyenda:**

- ✅ Completo
- ⚠️ Parcial o discrepante con literatura
- ❌ Ausente

---

## 6. Referencias Bibliográficas

Adegboye, A., Kampouridis, M., & Tsang, E. (2017). _Machine learning classification of price extrema based on directional change indicators_. In Proceedings of the 9th International Conference on Agents and Artificial Intelligence (ICAART), pp. 378-385.

Aloud, M., Tsang, E., Olsen, R., & Dupuis, A. (2012). _A directional-change event approach for studying financial time series_. Economics: The Open-Access, Open-Assessment E-Journal, 6(2012-36), 1-17.

Bisig, T., Dupuis, A., Impagliazzo, V., & Olsen, R. (2009). _The scale of market quakes_. Technical Report, Olsen Ltd.

Glattfelder, J. B., Dupuis, A., & Olsen, R. B. (2011). _Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws_. Quantitative Finance, 11(4), 599-614.

Guillaume, D. M., Dacorogna, M. M., Davé, R. D., Müller, U. A., Olsen, R. B., & Pictet, O. V. (1997). _From the bird's eye to the microscope: A survey of new stylized facts of the intra-daily foreign exchange markets_. Finance and Stochastics, 1(2), 95-129.

Kampouridis, M. (2025). _Multi-objective genetic programming-based algorithmic trading using directional changes_. [En preparación].

Li, X. (2022). _Relating volatility and jumps between two markets under Directional Change_. Working Paper, University of Essex.

Tsang, E. P. K. (2010). _Directional changes, definitions_. Technical Report, Centre for Computational Finance and Economic Agents (CCFEA), University of Essex.

Tsang, E. P. K., Tao, R., Serguieva, A., & Ma, S. (2015). _Profiling high-frequency equity price movements in directional changes_. Quantitative Finance, 17(2), 217-225.

Tsang, E. P. K., & Ma, S. (2021). _Distribution of aTMV, an empirical study_. Working Paper, University of Essex.

---

## 7. Historial de Revisiones

| Versión | Fecha      | Autor       | Descripción                                                                                                                                                                                                     |
| ------- | ---------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0.0   | 2026-01-31 | Claude Code | Documento inicial                                                                                                                                                                                               |
| 1.1.0   | 2026-01-31 | Claude Code | Refactorización: primitivas movidas a `core/DC_FRAMEWORK.md`; nueva introducción orientada a indicadores                                                                                                        |
| 1.2.0   | 2026-01-31 | Claude Code | Agregada estructura Teoría/Práctica/Salvedades a cada indicador implementado; código de implementación incluido                                                                                                 |
| 1.3.0   | 2026-02-01 | Claude Code | Agregado Event Magnitude (§2.1.3); actualizado DAG de dependencias y matriz de cobertura; renumeración de secciones                                                                                             |
| 1.4.0   | 2026-02-01 | Claude Code | Agregado DC Slippage Facial (§2.1.6); documentación de viabilidad de Slippage Real                                                                                                                              |
| 1.5.0   | 2026-02-01 | Claude Code | Agregado DC Slippage Real (§2.1.7) tras corrección de kernel para incluir todos los ticks del instante de confirmación                                                                                          |
| 1.6.0   | 2026-02-05 | Claude Code | Implementados 10 indicadores: TotalMove, TmvEvent, OsvEvent, A1DcPriceAbs, A4PrevDccPrice, A5PrevOsFlag, A6FlashEvent, Ndc, Cdc, AccumulatedTime. Corregido θ en RunsCount. Cobertura: 29/32 indicadores (91%). |

---

_Este documento fue generado como parte del proyecto Intrinseca y debe mantenerse actualizado conforme se implementen nuevos indicadores._
