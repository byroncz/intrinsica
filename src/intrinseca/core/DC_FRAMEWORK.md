# Marco Teórico del Análisis de Cambio Direccional

## Fundamentos, Definiciones y Consideraciones de Implementación

---

## 1. Introducción al Paradigma del Tiempo Intrínseco

### 1.1 Motivación: Las Limitaciones del Tiempo Físico

El análisis tradicional de series temporales financieras se fundamenta en el muestreo de precios a intervalos cronológicos fijos: segundos, minutos, horas o días. Esta aproximación, heredada de las ciencias físicas y la ingeniería de señales, asume implícitamente que el flujo de información y la actividad del mercado son **homogéneos** a lo largo del tiempo cronológico.

Sin embargo, la evidencia empírica demuestra que los mercados financieros operan bajo un **tiempo relativo a la actividad**. Períodos de alta volatilidad comprimen una enorme cantidad de "eventos" de mercado en pocos segundos físicos, mientras que períodos de baja liquidez pueden transcurrir horas sin cambios significativos. El uso de tiempo físico introduce ruido estadístico considerable y puede oscurecer patrones críticos de cambio de régimen (Guillaume et al., 1997).

> "The tick frequency is itself a stochastic variable that depends on market activity. Therefore, the usual physical time scale is not appropriate for studying price dynamics."
> — Guillaume et al. (1997, p. 4)

### 1.2 El Paradigma del Cambio Direccional (Directional Change, DC)

Como respuesta a estas limitaciones, el marco de **Cambio Direccional (DC)** propone una forma alternativa de muestreo. En lugar de dejar que el tiempo dicte cuándo registrar un precio, el marco DC **deja que el precio dicte cuándo registrar un evento de tiempo**.

El componente central de este sistema es el **umbral de cambio direccional (θ)**, un parámetro definido por el observador que representa la magnitud del movimiento de precio considerado "significativo". Bajo este prisma, el tiempo solo "avanza" cuando el mercado cambia de dirección o extiende una tendencia en una magnitud de θ.

Este enfoque actúa como un **filtro de ruido natural**. Las fluctuaciones de precios menores que θ se descartan como irrelevantes, permitiendo concentrar el análisis en los puntos de inflexión donde realmente se manifiesta el comportamiento colectivo de los agentes (Glattfelder et al., 2011).

### 1.3 Definición Formal del Tiempo Intrínseco

El **Tiempo Intrínseco** es una escala temporal alternativa donde la distancia entre puntos representa la magnitud de la evolución del precio (δ) en lugar del paso de minutos u horas. Formalmente:

Sea $\{P(t_i)\}_{i=1}^{N}$ una serie de precios en tiempo físico. El tiempo intrínseco $\tau$ se define como una función de conteo de eventos:

$$\tau(t) = \sum_{i=1}^{k(t)} \mathbb{1}[\text{Evento DC en } t_i]$$

Donde $k(t)$ es el número de ticks hasta el tiempo $t$, y $\mathbb{1}[\cdot]$ es la función indicadora.

**Interpretación:** El reloj intrínseco solo avanza cuando ocurre un evento significativo. Un día con 100 eventos DC tiene más "tiempo intrínseco" que un día con 10 eventos, independientemente de su duración física.

---

## 2. Reseña de los Trabajos Fundacionales

El paradigma de Cambio Direccional tiene sus raíces en la investigación de alta frecuencia desarrollada principalmente por investigadores de Olsen Ltd y la Universidad de Essex. A continuación se presenta una reseña cronológica de las contribuciones seminales.

### 2.1 Guillaume et al. (1997) — El Origen

**Referencia:** Guillaume, D. M., Dacorogna, M. M., Davé, R. D., Müller, U. A., Olsen, R. B., & Pictet, O. V. (1997). _From the bird's eye to the microscope: A survey of new stylized facts of the intra-daily foreign exchange markets_. Finance and Stochastics, 1(2), 95-129.

**Contribución:** Este trabajo seminal introdujo por primera vez la idea de que los eventos de mercado no ocurren de manera uniforme en el tiempo físico. Los autores documentaron los "hechos estilizados" de los mercados de divisas intradía, estableciendo que la frecuencia de ticks es una variable estocástica dependiente de la actividad del mercado.

**Impacto:** Sentó las bases conceptuales para el análisis basado en eventos y el concepto de tiempo intrínseco.

### 2.2 Tsang (2010) — Formalización de Definiciones

**Referencia:** Tsang, E. P. K. (2010). _Directional changes, definitions_. Technical Report, Centre for Computational Finance and Economic Agents (CCFEA), University of Essex.

**Contribución:** Formalizó las definiciones operativas del marco DC: umbral θ, punto extremo (EXT), punto de confirmación (DCC), evento DC y evento Overshoot. Estableció la distinción entre identificación retrospectiva y en tiempo real.

**Impacto:** Proporcionó el vocabulario técnico estándar utilizado por toda la literatura posterior.

### 2.3 Glattfelder et al. (2011) — Las 12 Leyes de Escala

**Referencia:** Glattfelder, J. B., Dupuis, A., & Olsen, R. B. (2011). _Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws_. Quantitative Finance, 11(4), 599-614.

**Contribución:** Descubrimiento de 12 leyes de escala empíricas que relacionan la volatilidad, la duración y la magnitud de los eventos DC. Demostró que, en promedio, la longitud del Overshoot es aproximadamente igual al umbral θ (la "ley del factor 2").

**Impacto:** Estableció el DC como un marco riguroso con propiedades estadísticas predecibles y universales.

### 2.4 Aloud et al. (2012) — Enfoque Basado en Eventos

**Referencia:** Aloud, M., Tsang, E., Olsen, R., & Dupuis, A. (2012). _A directional-change event approach for studying financial time series_. Economics: The Open-Access, Open-Assessment E-Journal, 6(2012-36), 1-17.

**Contribución:** Formuló leyes de escala que relacionan la volatilidad con el tamaño del umbral. Introdujo el uso del NDC (Number of Directional Changes) como medida fundamental de volatilidad intrínseca.

**Impacto:** Consolidó el uso del DC para perfilado de regímenes de mercado.

### 2.5 Tsang et al. (2015) — Perfilado de Movimientos de Precios

**Referencia:** Tsang, E. P. K., Tao, R., Serguieva, A., & Ma, S. (2015). _Profiling high-frequency equity price movements in directional changes_. Quantitative Finance, 17(2), 217-225.

**Contribución:** Sistematizó el TMV (Total Movement Value) como métrica normalizada para comparar activos. Introdujo el concepto de OSV (Overshoot Value) para cuantificar la rentabilidad post-confirmación.

**Impacto:** Proporcionó las métricas canónicas para análisis cuantitativo.

### 2.6 Adegboye et al. (2017) — Machine Learning y DC

**Referencia:** Adegboye, A., Kampouridis, M., & Tsang, E. (2017). _Machine learning classification of price extrema based on directional change indicators_. In Proceedings of the 9th International Conference on Agents and Artificial Intelligence (ICAART), pp. 378-385.

**Contribución:** Definió los atributos A1-A6 para construir vectores de características para modelos de clasificación. Proporcionó pseudocódigo detallado para la implementación algorítmica del DC.

**Impacto:** Estableció el puente entre el análisis DC y el aprendizaje automático.

### 2.7 Tsang & Ma (2021) — Distribución del aTMV

**Referencia:** Tsang, E. P. K., & Ma, S. (2021). _Distribution of aTMV, an empirical study_. Working Paper, University of Essex.

**Contribución:** Demostró que el aTMV (active TMV) sigue una distribución de ley de potencia. Estableció umbrales empíricos para la probabilidad de reversión.

**Impacto:** Proporcionó fundamentos para la gestión de riesgo en tiempo real.

---

## 3. Definiciones Primitivas: Teoría y Práctica

### 3.0 Visión General: Anatomía del Cambio Direccional

El siguiente diagrama ilustra dos eventos DC consecutivos (un **upturn** seguido de un **downturn**), mostrando todas las primitivas fundamentales del marco: umbral (θ), punto extremo (EXT), punto de confirmación (DCC), fase DC y fase Overshoot (OS).

```
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                         ANATOMÍA DEL CAMBIO DIRECCIONAL
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                                                        LEYENDA
  PRECIO                                                                                                ────────
     │                                                                                                  ★ Extremo
     │                                                             ★ EXT₁                                 (EXT)
     │                                                           ╱      ╲            Máximo local       ◆ Confirm.
 112 ┤                                                         ╱          ╲          (retrospectivo)      (DCC)
     │                                                       ╱              ╲                           ┄ Umbral θ
 110 ┤                                                     ╱                  ╲
     │                                                   ╱                      ╲
 108 ┤┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╱┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄  θ₁⁻ = EXT₁ × (1 - θ)
     │                                               ╱                            ╲
 106 ┤                                             ╱                                ◆ DCC₂ ──── Confirma DOWNTURN
     │                                           ╱                                    ╲         (precio cruza θ₁⁻)
 104 ┤                                         ╱                                        ╲
     │                                       ╱                                            ╲
 102 ┤                                   ◆ DCC₁ ──── Confirma UPTURN                        ╲
     │                                 ╱             (precio cruza θ₀⁺)                       ╲
 100 ┤┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╱┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄  θ₀⁺ = EXT₀ × (1 + θ)                      ╲
     │                           ╱                                                              ╲
  98 ┤                         ╱                                                                  ╲
     │                       ╱                                                                      ╲
  96 ┤          ╲          ╱                                                                          ╲
     │            ╲      ╱                                                                              ╲
  94 ┤              ╲  ╱                                                                                  ╲
     │                ★ EXT₀                                                                                ★ EXT₂
  92 ┤                          Mínimo local                                                      Mínimo local
     │                          (reference_price Evento 1)                EXT₁ = reference_price del Evento 2
     └──────────────────────────────────────────────────────────────────────────────────────────────────────────────► t
                     t₀                  t₁                           t₂           t₃                  t₄
                     │                   │                            │             │                   │
                     │◄─── FASE DC₁ ────►│◄──────── FASE OS₁ ────────►│◄ FASE DC₂ ─►│◄──── FASE OS₂ ───►│
                     │  (confirmación)   │       (continuación)       │  (confirm)  │   (continuación)  │
                     │                   │                            │             │                   │
                     ╰─────────── EVENTO 1 (Upturn) ──────────────────┴──── EVENTO 2 (Downturn) ────────╯

════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

Las siguientes definiciones constituyen los **atributos fundamentales** del sistema DC. Para cada primitiva se presenta:

1. La **definición teórica** en tiempo continuo
2. La **definición práctica** para implementación en tiempo discreto
3. Las **salvedades** relevantes

### 3.1 Umbral de Cambio Direccional (θ)

#### Definición Teórica (Tiempo Continuo)

El umbral θ ∈ ℝ⁺ es un parámetro exógeno que define la sensibilidad del análisis. Representa la variación porcentual mínima requerida para que el sistema reconozca un cambio en la dirección de la tendencia predominante (Guillaume et al., 1997; Glattfelder et al., 2011).

$$\theta \in (0, 1) \quad \text{típicamente } \theta \in [0.001, 0.05]$$

#### Definición Práctica (Tiempo Discreto)

En la implementación, θ se aplica como un factor multiplicativo sobre el precio de referencia:

```
umbral_upturn   = P_ref × (1 + θ)
umbral_downturn = P_ref × (1 - θ)
```

#### Salvedades de Implementación

**Granularidad vs. Umbral:** Cuando θ es muy pequeño relativo a la granularidad de los datos (e.g., θ = 0.001 con datos diarios), el cambio de precio real durante un evento DC puede ser significativamente mayor que θ debido a los saltos discretos entre observaciones. Esto viola la ley de escala $\langle OS \rangle \approx \theta$ (Glattfelder et al., 2011).

**Recomendación:** El umbral debe calibrarse según la frecuencia de muestreo. Para datos diarios, θ ≥ 0.01; para datos de tick, θ puede ser tan pequeño como 0.0001.

---

### 3.2 Punto Extremo (Extreme Point, EXT)

#### Definición Teórica (Tiempo Continuo)

El punto extremo $P_{EXT}$ es el precio máximo (en una tendencia alcista) o mínimo (en una tendencia bajista) alcanzado antes de que se confirme una reversión. En tiempo continuo, es el punto donde la derivada primera del precio es cero:

$$\frac{dP}{dt}\bigg|_{t_{EXT}} = 0$$

Para una tendencia alcista (upturn):
$$P_{EXT} = \max\{P(t') : t' \in [t_{inicio}, t_{confirmación}]\}$$

Para una tendencia bajista (downturn):
$$P_{EXT} = \min\{P(t') : t' \in [t_{inicio}, t_{confirmación}]\}$$

**Propiedad fundamental:** La identificación del EXT es **retrospectiva**. Solo se conoce con certeza una vez que el precio ha revertido al menos θ desde dicho punto (Tsang, 2010).

#### Definición Práctica (Tiempo Discreto)

En series discretas, el EXT es un tick específico con índice, timestamp y precio definidos:

```python
EXT = {
    "index": idx_extremo,
    "time": T_extremo,
    "price": P_extremo
}
```

El algoritmo mantiene variables dinámicas `P_high` y `P_low` que se actualizan tick a tick:

```
Si estado = UPTURN y P(t) > P_high:
    P_high ← P(t)
    T_high ← t
```

#### Salvedades de Implementación

**El Dilema de la Frontera Compartida:** El punto extremo cumple una **doble función ontológica** (Adegboye et al., 2017):

1. **Función Terminal:** Es el punto final del Overshoot del evento actual. Es el máximo/mínimo absoluto del conjunto de ticks que componen el evento.

2. **Función Inicial:** Es el punto de referencia base para calcular el Cambio Direccional del siguiente evento.

**Resolución:** En la literatura Q1, el extremo se comparte **paramétricamente**. El tick se asigna al final del Overshoot para cerrar la "historia" de la tendencia, y el siguiente evento comienza inmediatamente después pero **referenciando el precio del extremo**.

**Modelo de datos recomendado:**

```python
Evento_N = {
    "os_end_index": idx_ext,        # Incluye el extremo
    "os_end_price": P(idx_ext)
}

Evento_N+1 = {
    "dc_start_index": idx_ext,      # Referencia el extremo
    "dc_reference_price": P(idx_ext),
    "ticks_start": idx_ext + 1      # Ticks propios empiezan después
}
```

---

### 3.3 Punto de Confirmación (Directional Change Confirmation, DCC)

#### Definición Teórica (Tiempo Continuo)

El punto de confirmación $P_{DCC}$ es el primer precio observado que satisface la condición de reversión respecto al punto extremo (Glattfelder et al., 2011).

Para confirmar un downturn (después de un upturn):
$$P_{DCC} = \text{primer } P(t) \text{ tal que } P(t) \leq P_{EXT} \times (1 - \theta)$$

Para confirmar un upturn (después de un downturn):
$$P_{DCC} = \text{primer } P(t) \text{ tal que } P(t) \geq P_{EXT} \times (1 + \theta)$$

En tiempo continuo, existe un **DCC teórico** $P_{DCC}^*$ exactamente en el umbral:
$$P_{DCC}^* = P_{EXT} \times (1 \pm \theta)$$

#### Definición Práctica (Tiempo Discreto)

En series discretas, el DCC observado generalmente **difiere** del DCC teórico debido al muestreo discreto:

$$P_{DCC} = P(t) \quad \text{donde } t = \min\{t' : |P(t') - P_{EXT}| \geq \theta \times P_{EXT}\}$$

La diferencia $|P_{DCC} - P_{DCC}^*|$ se denomina **slippage** o deslizamiento.

#### Salvedades de Implementación

**Pertenencia del DCC:** El tick de confirmación pertenece **inequívocamente** al evento DC, no al Overshoot (Adegboye et al., 2017). Esto se deriva de la lógica algorítmica:

```
t_dc_end   ← t       # El DCC cierra el DC
t_os_start ← t + 1   # El OS comienza DESPUÉS del DCC
```

**El Escenario de Salto (Gap):** Cuando el primer tick después del extremo ya supera el umbral θ (común en gaps de apertura o flash crashes):

| Parámetro   | Valor                            |
| ----------- | -------------------------------- |
| Duración DC | $T_1 - T_0$ (positiva, no cero)  |
| Magnitud DC | $\geq \theta$ (incluye el salto) |
| Magnitud OS | Movimiento post-$T_1$ únicamente |

**El evento DC nunca es vacío.** El slippage se absorbe en el DC, no en el OS. Esto preserva la pureza estadística del Overshoot como medida de la continuación post-confirmación.

---

### 3.4 Evento de Cambio Direccional (DC Event)

#### Definición Teórica (Tiempo Continuo)

El evento DC comprende el segmento temporal y de precios entre $P_{EXT}$ y $P_{DCC}$. Representa la fase de **validación** de un cambio de tendencia (Guillaume et al., 1997).

Por definición, la magnitud de este movimiento es exactamente θ (normalizado):
$$\Delta_{DC} = |P_{DCC} - P_{EXT}| = P_{EXT} \times \theta$$

#### Definición Práctica (Tiempo Discreto)

En implementación:

```python
DC_Event = {
    "start_index": idx_extremo,
    "end_index": idx_dcc,           # El DCC cierra el DC
    "start_price": P_extremo,
    "end_price": P_dcc,
    "magnitude": abs(P_dcc - P_extremo)
}
```

**Contenido de ticks:** El evento DC contiene todos los ticks desde el extremo (inclusive) hasta el DCC (inclusive).

#### Salvedades de Implementación

**Magnitud Real vs. Teórica:** En tiempo discreto, $\Delta_{DC} \geq \theta \times P_{EXT}$ debido al slippage. Esta discrepancia es inherente y no representa un error de implementación.

**Regla Conservadora para Ticks Simultáneos:** Cuando múltiples ticks tienen el mismo timestamp, se aplica la regla conservadora (Intrinseca):

- En upturn: seleccionar el precio **mínimo** que cruza θ
- En downturn: seleccionar el precio **máximo** que cruza θ

Esto garantiza determinismo en la segmentación.

**Inclusión Completa del Instante de Confirmación:** Todos los ticks con el mismo timestamp que el tick de confirmación deben pertenecer a la fase DC, no al OS. Esto evita:

- Ticks del mismo instante en fases diferentes (violación semántica)
- `os_time = 0` espurio cuando hay ticks contemporáneos al DCC
- Velocidades infinitas artificiales (`os_magnitude / 0`)
- Overshoots "fantasma" con ticks que no son continuación post-confirmación

**Implementación:** El kernel usa `last_same_ts_idx` para incluir todos los ticks del grupo temporal, independientemente de cuál tenga el precio conservador.

---

### 3.5 Evento de Excedente (Overshoot Event, OS)

#### Definición Teórica (Tiempo Continuo)

El evento OS comprende el segmento que inicia en $P_{DCC}$ y termina en el siguiente $P_{EXT}$. Representa el **impulso** del mercado una vez confirmada la nueva dirección (Glattfelder et al., 2011).

$$\Delta_{OS} = |P_{EXT,siguiente} - P_{DCC}|$$

**Ley de escala fundamental:** $\langle \Delta_{OS} \rangle \approx \theta$ (Glattfelder et al., 2011).

#### Definición Práctica (Tiempo Discreto)

```python
OS_Event = {
    "start_index": idx_dcc + 1,     # Comienza DESPUÉS del DCC
    "end_index": idx_next_extremo,
    "reference_price": P_dcc,       # Magnitud medida desde DCC
    "end_price": P_next_extremo
}
```

#### Salvedades de Implementación

**Overshoot Vacío (Zero OS):** Ocurre cuando la tendencia se revierte inmediatamente después de la confirmación, sin que el precio haga un nuevo extremo más allá del DCC.

$$\omega = |P_{EXT,siguiente} - P_{DCC}| = 0 \implies P_{EXT,siguiente} = P_{DCC}$$

**Implicaciones del Zero OS:**

- El TMV = θ (movimiento mínimo posible)
- Duración intrínseca del OS = 0
- Señal de resistencia microestructural o agotamiento de liquidez

**Interpretación:** Los eventos de overshoot cero no son ruido estadístico; representan **puntos de agotamiento crítico** donde el impulso se disipa instantáneamente en el momento de la confirmación.

##### Caso Crítico: Cálculo del Umbral del Siguiente Evento en Overshoot Cero

Cuando ocurre un overshoot cero, el **DCC del evento N se convierte en el precio de referencia** para calcular el umbral del evento N+1. Esto es una consecuencia directa de la definición:

**Definición formal:**

Sea el evento $N$ un upturn confirmado en $P_{DCC,N}$. Si no hay movimiento adicional antes de la reversión:

$$P_{EXT,N} = P_{DCC,N}$$

El umbral para confirmar el evento $N+1$ (downturn) se calcula:

$$\text{threshold}_{N+1} = P_{EXT,N} \times (1 - \theta) = P_{DCC,N} \times (1 - \theta)$$

**Diagrama temporal:**

```
Evento N (Upturn)                    Evento N+1 (Downturn)
─────────────────────────────────────────────────────────────────
     EXT_N-1        DCC_N = EXT_N           DCC_N+1
        │               │                      │
   P_low ──────────> P_dcc ──────────────> P_dcc'
        │               │     (sin OS)         │
        └── DC_N ───────┘                      │
                        │                      │
                        └────── DC_N+1 ────────┘
```

**Implementación en kernel (líneas 328-334, kernel.py):**

```python
# Al confirmar evento N (upturn):
if new_trend == 1:
    ext_high_price = conf_price   # DCC se convierte en referencia
    ext_high_idx = conf_idx

# Al evaluar evento N+1 (downturn):
if p <= ext_high_price * theta_down:  # Usa P_DCC_N como referencia
    new_trend = int8(-1)
```

**Consecuencia para los datos Silver:**

En caso de overshoot cero, el evento N tendrá:

- `extreme_price = confirm_price`
- `extreme_time = confirm_time`
- La lista `price_os` estará vacía o contendrá cero elementos

**Referencia:** Ver documento "Overshoots vacíos" en `docs/references/` para análisis detallado de este fenómeno y sus implicaciones para trading algorítmico.

---

## 4. Identidades Matemáticas Fundamentales

### 4.1 Identidad del Movimiento Total (TMV)

$$TMV = \Delta_{DC} + \Delta_{OS}$$

En tiempo discreto, dado que $\Delta_{DC} \geq \theta$:

$$TMV \geq \theta$$

El caso $TMV = \theta$ corresponde a un Overshoot cero.

### 4.2 Ley de Escala del Factor 2

Empíricamente, en mercados líquidos y con umbrales apropiados (Glattfelder et al., 2011):

$$\langle TMV \rangle \approx 2\theta$$

Equivalentemente:
$$\langle \Delta_{OS} \rangle \approx \theta$$

**Interpretación:** Una vez confirmada una tendencia, es estadísticamente probable que continúe por otra distancia θ antes de revertirse.

### 4.3 Longitud de Costa (Coastline)

La suma de todos los movimientos totales en un período:

$$CDC(\theta) = \sum_{i=1}^{NDC} |TMV_i|$$

Representa el "camino total" recorrido por el precio, inspirado en la paradoja de Mandelbrot sobre la longitud de las costas.

---

## 5. Consideraciones para la Implementación

### 5.1 Arquitectura de Datos Recomendada

Basado en la evidencia de literatura Q1, se recomienda un modelo de **superposición referencial**:

1. **Serie Temporal Bruta (Inmutable):** Array de ticks indexados.

2. **Lista de Eventos (Metadatos):** Cada evento referencia índices de la serie bruta.

3. **Parámetros Compartidos:** El precio extremo se pasa explícitamente como referencia entre eventos adyacentes.

### 5.2 Validación de Determinismo

Para garantizar resultados reproducibles:

- Aplicar regla conservadora para ticks simultáneos
- Usar aritmética de punto fijo para comparaciones de umbral cuando sea posible
- Documentar el tratamiento de casos límite (gaps, zero OS)

### 5.3 Manejo de Errores Numéricos

- División protegida cuando el overshoot anterior es cero
- Límites de duración para evitar overflow en nanosegundos
- Validación de que $\Delta_{DC} \geq \theta \times P_{EXT}$ (puede fallar por errores de redondeo)

---

## 6. Referencias Bibliográficas

Adegboye, A., Kampouridis, M., & Tsang, E. (2017). _Machine learning classification of price extrema based on directional change indicators_. In Proceedings of the 9th International Conference on Agents and Artificial Intelligence (ICAART), pp. 378-385.

Aloud, M., Tsang, E., Olsen, R., & Dupuis, A. (2012). _A directional-change event approach for studying financial time series_. Economics: The Open-Access, Open-Assessment E-Journal, 6(2012-36), 1-17.

Bisig, T., Dupuis, A., Impagliazzo, V., & Olsen, R. (2009). _The scale of market quakes_. Technical Report, Olsen Ltd.

Glattfelder, J. B., Dupuis, A., & Olsen, R. B. (2011). _Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws_. Quantitative Finance, 11(4), 599-614.

Guillaume, D. M., Dacorogna, M. M., Davé, R. D., Müller, U. A., Olsen, R. B., & Pictet, O. V. (1997). _From the bird's eye to the microscope: A survey of new stylized facts of the intra-daily foreign exchange markets_. Finance and Stochastics, 1(2), 95-129.

Kampouridis, M. (2025). _Multi-objective genetic programming-based algorithmic trading using directional changes_. [En preparación].

Tsang, E. P. K. (2010). _Directional changes, definitions_. Technical Report, Centre for Computational Finance and Economic Agents (CCFEA), University of Essex.

Tsang, E. P. K., Tao, R., Serguieva, A., & Ma, S. (2015). _Profiling high-frequency equity price movements in directional changes_. Quantitative Finance, 17(2), 217-225.

Tsang, E. P. K., & Ma, S. (2021). _Distribution of aTMV, an empirical study_. Working Paper, University of Essex.

---

## 8. Capacidades de Implementación del Core Intrinseca

Esta sección documenta las capacidades técnicas desarrolladas en el Core de Intrinseca para manejar las particularidades del análisis DC en mercados financieros reales. Cada capacidad representa una solución a un problema específico encontrado durante la implementación y validación con datos de alta frecuencia.

### 8.1 Arquitectura General

El Core de Intrinseca implementa una arquitectura de tres capas optimizada para HFT:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE INTRINSECA                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Engine    │───▶│   Kernel    │───▶│    State    │───▶│   Silver    │  │
│  │ (Orquesta)  │    │ (Numba JIT) │    │  (Arrow)    │    │  (Parquet)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │Reconcilia-  │    │Convergence  │    │ Validation  │    │  ListArray  │  │
│  │   tion      │    │  Analysis   │    │   Layer     │    │  Zero-copy  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Módulos del Core:**

| Módulo         | Archivo             | Responsabilidad                                 |
| -------------- | ------------------- | ----------------------------------------------- |
| Engine         | `engine.py`         | Orquestación Bronze→Silver, gestión de pipeline |
| Kernel         | `kernel.py`         | Segmentación DC con Numba JIT, O(n)             |
| State          | `state.py`          | Persistencia de huérfanos y estado algorítmico  |
| Reconciliation | `reconciliation.py` | Corrección retroactiva de eventos               |
| Convergence    | `convergence.py`    | Validación de determinismo                      |

---

### 8.2 Capacidades del Kernel de Segmentación

El kernel Numba JIT es el corazón del sistema. Las siguientes capacidades han sido desarrolladas para garantizar correctitud y rendimiento:

#### 8.2.1 Regla Conservadora para Ticks Simultáneos

**Problema:** En datos de alta frecuencia, múltiples ticks pueden tener el mismo timestamp (resolución de milisegundos o microsegundos). Sin una regla determinista, diferentes ejecuciones podrían producir resultados diferentes.

**Solución implementada:**

- En **upturn**: seleccionar el precio **mínimo** que cruza θ (más conservador)
- En **downturn**: seleccionar el precio **máximo** que cruza θ (más conservador)

```python
# kernel.py líneas 258-278
if new_trend == 1:
    # Upturn: precio MÍNIMO >= threshold
    if pj >= threshold and pj < best_price:
        best_price = pj
else:
    # Downturn: precio MÁXIMO <= threshold
    if pj <= threshold and pj > best_price:
        best_price = pj
```

**Beneficio:** Determinismo garantizado en la segmentación.

#### 8.2.2 Inclusión Completa del Instante de Confirmación

**Problema:** Ticks con el mismo timestamp que el DCC podrían quedar en fases diferentes (DC vs OS), causando:

- Violación semántica (ticks contemporáneos en eventos distintos)
- `os_time = 0` espurio (velocidades infinitas)
- Overshoots "fantasma"

**Solución implementada:**

```python
# Se trackean dos índices:
# - best_idx: tick con precio más conservador (para confirm_price)
# - last_same_ts_idx: último tick del grupo (para conf_idx)
conf_idx = last_same_ts_idx  # Todos van al DC
```

**Beneficio:** Todos los ticks del instante de confirmación pertenecen a la fase DC.

#### 8.2.3 Resize Dinámico de Búferes

**Problema:** La cantidad de eventos DC es impredecible y depende de la volatilidad del mercado.

**Solución implementada:**

- Estimación inicial con factor de seguridad 3x
- Resize dinámico duplicando capacidad cuando se agota
- Orden correcto: cerrar OS anterior **antes** del resize

```python
# Estimación inicial
estimated_events = max(n // _EVENT_RATIO_ESTIMATE, _MIN_EVENT_SLOTS) * 3

# Resize cuando se necesita
if n_events >= estimated_events:
    new_size = estimated_events * 2
    # Resize todos los búferes...
```

**Beneficio:** Manejo de cualquier volumen de datos sin overflow.

#### 8.2.4 Llenado Retrospectivo de Extremos

**Problema:** El precio extremo de un evento solo se conoce cuando se confirma el evento **siguiente**.

**Solución implementada:**

```python
# Al confirmar evento N+1, llenar extreme_price del evento N
extreme_prices[n_events - 1] = prices[prev_ext_idx]
extreme_times[n_events - 1] = timestamps[prev_ext_idx]
```

El último evento queda con `extreme_price = -1.0` (provisional) hasta la reconciliación.

#### 8.2.5 Operaciones Vectorizadas

**Problema:** Loops tick-a-tick son lentos incluso con Numba.

**Solución implementada:**

```python
# Antes (loop):
for i in range(prev_os_start, prev_ext_idx + 1):
    os_prices[os_ptr] = prices[i]
    os_ptr += 1

# Después (vectorizado):
os_length = prev_ext_idx + 1 - prev_os_start
os_prices[os_ptr : os_ptr + os_length] = prices[prev_os_start : prev_ext_idx + 1]
os_ptr += os_length
```

**Beneficio:** ~50M ticks/segundo en hardware típico.

#### 8.2.6 Validación de Longitudes No Negativas

**Problema:** Edge cases pueden producir longitudes negativas si `prev_os_start > prev_ext_idx`.

**Solución implementada:**

```python
os_length = prev_ext_idx + 1 - prev_os_start
if os_length > 0:
    # Copiar datos...
# Siempre escribir os_offsets para mantener consistencia
os_offsets[n_events] = os_ptr
```

**Beneficio:** Prevención de comportamiento indefinido en casos límite.

---

### 8.3 Sistema de Continuidad Cross-Día (Stitching)

#### 8.3.1 Gestión de Ticks Huérfanos

**Problema:** Un día puede terminar en medio de un evento DC no confirmado.

**Solución implementada:**

```
Día N                          Día N+1
─────────────────────────────  ─────────────────────────
[eventos confirmados][huérfanos][nuevos ticks...]
                     └─────────────────┘
                          stitching
```

Los ticks huérfanos se persisten en Arrow IPC con metadata algorítmica:

- `current_trend`: Tendencia activa (1/-1/0)
- `last_os_ref`: Precio de referencia
- `orphan_prices/times/quantities/directions`: Ticks pendientes

#### 8.3.2 Formato de Persistencia Arrow IPC

**Decisión de diseño:** Arrow IPC (Feather v2) sin compresión para máxima velocidad de lectura.

```python
# state.py - Estructura del archivo state.arrow
table = pa.table({
    "price": pa.array(orphan_prices, type=pa.float64()),
    "time": pa.array(orphan_times, type=pa.int64()),
    "quantity": pa.array(orphan_quantities, type=pa.float64()),
    "direction": pa.array(orphan_directions, type=pa.int8()),
})
# Metadata en schema
metadata = {
    b"current_trend": str(trend).encode(),
    b"last_os_ref": f"{ref:.15g}".encode(),
    b"last_processed_date": date.isoformat().encode(),
}
```

#### 8.3.3 Búsqueda de Estado Anterior

**Capacidad:** Lookback de hasta 7 días para encontrar el estado más reciente.

```python
for days_back in range(1, MAX_LOOKBACK_DAYS + 1):
    prev_date = current_date - timedelta(days=days_back)
    state = load_state(build_state_path(..., prev_date))
    if state is not None:
        return state
```

**Beneficio:** Tolerancia a días sin datos (fines de semana, festivos).

---

### 8.4 Sistema de Reconciliación Retroactiva

#### 8.4.1 Tipos de Reconciliación

**Problema:** Al final del día, un evento puede parecer confirmado pero al día siguiente se determina que fue:

- Una reversión real (CONFIRM_REVERSAL)
- Una falsa alarma (EXTEND_OS)

**Solución implementada:**

```python
class ReconciliationType(Enum):
    NONE = "none"
    CONFIRM_REVERSAL = "confirm"  # Actualizar extreme_price
    EXTEND_OS = "extend_os"       # Extender OS con nuevo extremo
```

**Lógica de detección:**

```python
if trend == 1:  # Alcista
    if first_price <= ext_high * (1 - theta):
        return CONFIRM_REVERSAL  # Confirmó reversión bajista
    elif first_price > ext_high:
        return EXTEND_OS         # Siguió subiendo
```

#### 8.4.2 Escritura Atómica con Backup

**Mitigación de corrupción:**

1. Crear backup del archivo original
2. Escribir a archivo temporal
3. Rename atómico
4. Si falla, restaurar desde backup

```python
backup_path = silver_path.with_suffix(".parquet.bak")
shutil.copy2(silver_path, backup_path)
# Modificar DataFrame...
temp_path = silver_path.with_suffix(".parquet.tmp")
df.write_parquet(temp_path)
temp_path.rename(silver_path)  # Atómico en POSIX
```

---

### 8.5 Validación de Integridad

#### 8.5.1 Validación Post-Kernel

**Problema:** Bugs en el kernel pueden producir estructuras inconsistentes.

**Solución implementada:**

```python
# engine.py - Validación antes de escribir Parquet
expected_offset_len = n_events + 1
if len(dc_offsets) != expected_offset_len:
    raise ValueError("Integridad fallida: dc_offsets...")
if len(os_offsets) != expected_offset_len:
    raise ValueError("Integridad fallida: os_offsets...")
# Validar todos los atributos de evento
for name, arr in [("reference_prices", reference_prices), ...]:
    if len(arr) != n_events:
        raise ValueError(f"Integridad fallida: {name}...")
```

**Beneficio:** Detección temprana de bugs, nunca se escriben datos corruptos.

#### 8.5.2 Validación de Estado

```python
def validate(self) -> tuple[bool, list[str]]:
    errors = []
    # Verificar longitudes consistentes
    if len(self.orphan_times) != n:
        errors.append("orphan_times length mismatch")
    # Verificar tipos
    if self.orphan_prices.dtype != np.float64:
        errors.append("orphan_prices dtype mismatch")
    # Verificar trend válido
    if self.current_trend not in (-1, 0, 1):
        errors.append("current_trend invalid")
    # Verificar timestamps ordenados
    if not np.all(np.diff(self.orphan_times) >= 0):
        errors.append("orphan_times not monotonic")
    return (len(errors) == 0, errors)
```

#### 8.5.3 Warning por Procesamiento Incompleto

**Problema:** El procesamiento puede detenerse prematuramente sin advertencia.

**Solución implementada:**

```python
if len(results) < n_days:
    warnings.warn(
        f"⚠️ ADVERTENCIA: Solo se procesaron {len(results)}/{n_days} días.",
        UserWarning,
    )
```

---

### 8.6 Análisis de Convergencia

#### 8.6.1 Comparación Vectorizada

**Capacidad:** Comparar resultados de reprocesamiento para validar determinismo.

```python
# Comparación con tolerancia configurable
time_diffs = np.abs(prev_times[:min_len] - new_times[:min_len])
time_matches = time_diffs <= tolerance_ns
type_matches = prev_types[:min_len] == new_types[:min_len]
events_equal = time_matches & type_matches
```

#### 8.6.2 Detección de Punto de Convergencia

**Capacidad:** Identificar dónde dos series divergentes vuelven a coincidir.

```python
# Buscar primer True después del primer False
remaining = events_equal[first_discrepancy_idx + 1:]
if np.any(remaining):
    convergence_idx = first_discrepancy_idx + 1 + np.argmax(remaining)
```

---

### 8.7 Optimizaciones de Rendimiento

| Optimización    | Técnica                       | Impacto                          |
| --------------- | ----------------------------- | -------------------------------- |
| Compilación JIT | Numba `@njit(cache=True)`     | ~100x vs Python puro             |
| Sin GIL         | `nogil=True`                  | Paralelismo potencial            |
| Fast math       | `fastmath=True`               | ~10% más rápido                  |
| Zero-copy       | Arrow ListArray desde offsets | Sin copia de datos               |
| Memory mapping  | `memory_map=True` en lectura  | Reduce uso de RAM                |
| Pre-cálculo     | `theta_up = 1.0 + theta`      | Evita multiplicaciones en loop   |
| Warmup          | `warmup_kernel()`             | Evita latencia JIT en producción |

---

### 8.8 Manejo de Fenómenos del Mercado

#### 8.8.1 Gaps de Apertura

**Fenómeno:** El precio de apertura puede estar muy alejado del cierre anterior.

**Manejo:** El gap se absorbe en la fase DC (como slippage), preservando la pureza estadística del OS.

#### 8.8.2 Flash Crashes

**Fenómeno:** Caídas/subidas súbitas con recuperación rápida.

**Manejo:**

- Múltiples eventos DC en rápida sucesión
- Overshoots potencialmente muy pequeños o cero
- El sistema captura la microestructura completa

#### 8.8.3 Baja Liquidez / Gaps Temporales

**Fenómeno:** Períodos sin ticks (noches, fines de semana).

**Manejo:**

- Lookback de 7 días para encontrar estado
- Stitching preserva continuidad algorítmica
- Reconciliación corrige eventos al cruzar gaps

#### 8.8.4 Alta Frecuencia de Ticks

**Fenómeno:** Miles de ticks por segundo con mismo timestamp.

**Manejo:**

- Regla conservadora determinista
- Inclusión completa del instante de confirmación
- Vectorización para mantener rendimiento

---

### 8.9 Resumen de Bugs Corregidos

Esta sección documenta los bugs más significativos encontrados y corregidos durante el desarrollo:

| Bug                                      | Impacto                                       | Solución                                |
| ---------------------------------------- | --------------------------------------------- | --------------------------------------- |
| Resize antes de cerrar OS                | Offsets con valores no inicializados          | Reordenar: cerrar OS → resize           |
| conf_idx usando best_idx                 | Ticks del mismo timestamp en fases diferentes | Usar last_same_ts_idx                   |
| Terminación prematura por convergencia   | Solo 13/30 días procesados                    | Flag `stop_on_convergence=False`        |
| Falta de validación post-kernel          | Datos corruptos escritos                      | Validación exhaustiva antes de write    |
| os_length negativo potencial             | Comportamiento indefinido                     | Validación `if os_length > 0`           |
| Sin warning por procesamiento incompleto | Fallo silencioso                              | `warnings.warn()` si `results < n_days` |

---

### 8.10 Estrategias de Particionado

El Engine soporta dos estrategias de particionado para la salida Silver:

#### 8.10.1 Particionado Diario (por defecto)

**Método:** `process_day()` / `process_date_range()`

```
{base}/{ticker}/theta={θ}/year={YYYY}/month={MM}/day={DD}/data.parquet
```

**Características:**

- Un archivo Parquet por día
- Huérfanos persistidos entre días (`state.arrow` por día)
- Menor uso de RAM (~50MB por día)
- Soporta análisis de convergencia intra-procesamiento

**Caso de uso:** Procesamiento streaming, recursos limitados, análisis diario.

#### 8.10.2 Particionado Mensual

**Método:** `process_month()`

```
{base}/{ticker}/theta={θ}/year={YYYY}/month={MM}/data.parquet
```

**Características:**

- Un archivo Parquet por mes (sin subdirectorio `day=`)
- Huérfanos solo entre meses
- Una sola invocación del kernel Numba
- Mayor uso de RAM (~1.5GB para BTCUSDT)

**Caso de uso:** Batch processing, reprocesamiento masivo.

#### 8.10.3 Comparación

| Aspecto      | Diario | Mensual |
| ------------ | ------ | ------- |
| Archivos/mes | ~30    | 1       |
| Estados/mes  | ~30    | 1       |
| Overhead I/O | Alto   | Bajo    |
| RAM          | ~50MB  | ~1.5GB  |
| Convergencia | Sí     | No      |
| Kernel calls | N      | 1       |

#### 8.10.4 Lectura Transparente

`read_dc_events()` detecta automáticamente el esquema:

```python
# Funciona con ambos esquemas
dataset = read_dc_events(base_path, ticker, year, month)
# Si existe month/MM/data.parquet → lee directo
# Si existen month/MM/day=DD/data.parquet → concatena
```

**Nota:** Si se especifica `day_start` o `day_end`, siempre busca archivos diarios.

---

## 9. Historial de Revisiones

| Versión | Fecha      | Autor       | Descripción                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------- | ---------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0.0   | 2026-01-31 | Claude Code | Documento inicial                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 1.1.0   | 2026-01-31 | Claude Code | Agregada documentación del caso crítico: cálculo del umbral en overshoot cero (DCC como referencia)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 1.2.0   | 2026-02-01 | Claude Code | Agregado diagrama ASCII pedagógico "Anatomía del Cambio Direccional" (§3.0) con visualización de primitivas, fases y relaciones fundamentales                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 1.3.0   | 2026-02-01 | Claude Code | Documentada política de inclusión completa del instante de confirmación en §3.3; corrección de kernel para evitar ticks del mismo timestamp en fases diferentes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2.0.0   | 2026-02-02 | Claude Code | **Nueva sección §8: Capacidades de Implementación del Core.** Documentación exhaustiva de: arquitectura del core, capacidades del kernel (regla conservadora, inclusión completa de instante de confirmación, resize dinámico, llenado retrospectivo, operaciones vectorizadas, validación de longitudes), sistema de continuidad cross-día (stitching, huérfanos, Arrow IPC), reconciliación retroactiva (tipos, escritura atómica), validación de integridad (post-kernel, estado, warnings), análisis de convergencia, optimizaciones de rendimiento, manejo de fenómenos del mercado (gaps, flash crashes, baja liquidez, alta frecuencia), y resumen de bugs corregidos. |
| 2.1.0   | 2026-02-06 | Claude Code | **Nueva sección §8.10: Estrategias de Particionado.** Documentación de particionado diario vs mensual (`process_month()`), tabla comparativa, y lectura transparente en `read_dc_events()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

---

## 10. Anexo: Estructura de Archivos del Core

```
src/intrinseca/core/
├── __init__.py           # Exports públicos del módulo
├── engine.py             # Motor de orquestación Bronze → Silver
├── kernel.py             # Kernel Numba JIT para segmentación DC
├── state.py              # Gestión de estado y ticks huérfanos (Arrow IPC)
├── reconciliation.py     # Sistema de reconciliación retroactiva
├── convergence.py        # Análisis de convergencia y determinismo
└── DC_FRAMEWORK.md       # Este documento
```

### Dependencias entre módulos

```
                    ┌──────────────┐
                    │   engine.py  │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌─────────────────┐
    │kernel.py │    │ state.py │    │reconciliation.py│
    └──────────┘    └──────────┘    └─────────────────┘
                           │               │
                           └───────┬───────┘
                                   ▼
                           ┌──────────────┐
                           │convergence.py│
                           └──────────────┘
```

---

_Este documento forma parte de la documentación técnica del motor Intrinseca. Para la taxonomía de indicadores derivados, consulte `indicators/INDICATORS.md`._
