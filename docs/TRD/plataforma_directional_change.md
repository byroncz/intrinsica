# Documento de Requerimientos Técnicos (TRD)
## Plataforma de Análisis de Directional Change sobre datos de mercado de Binance (BTC/USDT)
 
> **Arquitectura Medallion de 4 capas · Single-Node Big Data · Google Cloud Platform**
 
| Campo | Valor |
|---|---|
| Documento | TRD maestro — Plataforma DC |
| Versión | **2.1** |
| Estado | Línea base ampliada (operaciones + stack + decisiones de L1) |
| Fecha | Junio de 2026 |
| Alcance | Capa Batch (Arquitectura Lambda) |
| Clasificación | Académico / Uso personal |
| Contexto | Trabajo de grado — Maestría en Finanzas · Universidad EAFIT (Medellín, Colombia) |
 
---
 
## Cómo leer este documento
 
Este es el **TRD maestro**: fija el alcance global, la arquitectura, las decisiones transversales y los requerimientos de alto nivel. Es un documento vivo. Cada capa del pipeline derivará en su propio **TRD particular** que profundiza algoritmos, esquemas y criterios de aceptación cuando esa capa se desarrolle. Rige el principio de que **toda afirmación polémica se reanaliza y confirma, no se hereda como dada**.
 
- Si buscas **el qué y el porqué de cada decisión** → secciones 6 (ADR) y 8 (Operaciones).
- Si buscas **qué hay que construir** → secciones 4 y 5 (requerimientos).
- Si buscas **cuánto cuesta y en qué máquina corre** → sección 7.
- Si buscas **cómo agregar una capa nueva** → sección 8.6 (playbook).
---
 
## Tabla de contenido
 
1. [Introducción y propósito](#1-introducción-y-propósito)
2. [Alcance](#2-alcance)
3. [Arquitectura general del sistema](#3-arquitectura-general-del-sistema)
4. [Requerimientos funcionales](#4-requerimientos-funcionales)
5. [Requerimientos no funcionales](#5-requerimientos-no-funcionales)
6. [Decisiones arquitectónicas (ADR)](#6-decisiones-arquitectónicas-adr)
7. [Dimensionamiento y modelo de costos](#7-dimensionamiento-y-modelo-de-costos)
8. [Operaciones (DevOps / Platform / MLOps)](#8-operaciones-devops--platform--mlops)
9. [Restricciones, supuestos y consideraciones transversales](#9-restricciones-supuestos-y-consideraciones-transversales)
10. [Riesgos y mitigaciones](#10-riesgos-y-mitigaciones)
11. [Roadmap de implementación](#11-roadmap-de-implementación)
12. [Anexos](#12-anexos)
### Documentos derivados (TRD por capa)
 
Este TRD maestro fija las **invariantes** (contratos entre capas, formatos, particionado, presupuesto, región). Cada TRD de capa las hereda y detalla.
 
| TRD derivado | Cubre | Estado previsto |
|---|---|---|
| TRD-L1 | Ingesta, ordenamiento y materialización a Parquet de aggTrades crudos | A redactar al iniciar Capa 1 |
| TRD-L2 | Detección de eventos DC (extremos, confirmaciones, overshoots) por θ | A redactar al iniciar Capa 2 |
| TRD-L3 | Tramas de series de tiempo cortadas por evento DC | A redactar al iniciar Capa 3 |
| TRD-L4 | Indicadores inter e intra-evento por θ | A redactar al iniciar Capa 4 |
| TRD-ML | Backtesting walk-forward, ML y evaluación de estrategias | A redactar tras completar L1–L4 |
 
### Historial de revisiones
 
| Versión | Fecha | Descripción |
|---|---|---|
| 1.0 | Jun 2026 | Línea base inicial. Consolida los análisis de créditos GCP, free tier, dimensionamiento de cómputo, ventana de procesamiento, arquitectura de Capa 2 y selección de lenguaje. |
| 2.0 | Jun 2026 | Ampliación de operaciones y stack. Añade estrategia de repositorios, IaC modular, registro de imágenes, CI/CD, observabilidad a costo cero, secretos y orquestación; valida el stack Polars + DuckDB + Arrow con catálogo DuckLake. Incorpora ADR 05–08, nuevos requerimientos y supuestos. |
| 2.1 | Jun 2026 | Incorpora la discusión de diseño de la **Capa 1 (L1)** y sus implicaciones transversales: corrige la terminología *environment*→*stack*; fija la estrategia de un único GCP project; reasigna el cómputo de L1 a **Cloud Run Jobs** (difiriendo Cloud Batch a TRD-L2); precisa el contrato de datos de L1 (orden por tiempo, timestamp µs por magnitud, DECIMAL exacto, ZSTD-3, partición mensual inmutable); añade el **lago de hallazgos de Data Quality** y la emisión por librería compartida; acota Secret Manager a etapas posteriores. Marca como **abiertos** la política de fallo de DQ y el dimensionamiento del backfill (a validar con sonda). |
 
---
 
## 1. Introducción y propósito
 
Este documento especifica los requerimientos técnicos de una plataforma de datos para el análisis de **Directional Change (DC)** sobre series de tiempo de alta frecuencia del mercado de criptomonedas, construida bajo el paradigma **single-node big data** sobre Google Cloud Platform (GCP). En su primera etapa (objeto del trabajo de grado), la plataforma soporta el procesamiento histórico masivo y el backtesting walk-forward de estrategias basadas en DC.
 
### 1.1 Contexto y motivación
 
El enfoque de Directional Change (Tsang, Glattfelder, Dupuis y Olsen) reemplaza el muestreo por tiempo físico por un muestreo basado en **eventos intrínsecos**: se registra un evento cuando el precio revierte un umbral porcentual θ desde un extremo local. Esto permite estudiar la dinámica del mercado en "tiempo intrínseco" y construir indicadores robustos a la microestructura. El proyecto aplica este marco a los **aggTrades de SPOT BTC/USDT de Binance**, con ~8 años de historia, para alimentar análisis de Machine Learning y, fuera del alcance del grado, un bot de trading algorítmico.
 
### 1.2 Objetivos
 
1. Construir un pipeline reproducible (infraestructura como código) que ingiera, conforme, enriquezca e indexe los aggTrades de BTC/USDT en una arquitectura medallion de 4 capas.
2. Detectar eventos DC para 50 umbrales θ distintos de forma eficiente, correcta y reusable hacia tiempo real.
3. Habilitar backtesting walk-forward y análisis ML sobre los indicadores derivados.
4. Mantener el costo operativo dentro de un presupuesto de créditos GCP de ~100 USD/mes, idealmente muy por debajo.
### 1.3 Audiencia
 
Autor del trabajo de grado (desarrollador e investigador), asesor académico y revisores técnicos. Los TRD por capa servirán además como guía de implementación detallada.
 
---
 
## 2. Alcance
 
### 2.1 Dentro de alcance (trabajo de grado — capa Batch)
 
- Ingesta histórica y conformación de aggTrades SPOT BTC/USDT desde `data.binance.vision`.
- Arquitectura medallion de 4 capas en Parquet sobre Cloud Storage (us-east1).
- Detección de eventos DC para 50 valores de θ.
- Cálculo de tramas (Capa 3) e indicadores (Capa 4) por evento.
- Procesamiento por ventanas mensuales con carry-over de puntos huérfanos.
- Backtesting walk-forward (Walk-Through) y análisis ML sobre los indicadores.
- Despliegue con Terraform (local) y, posteriormente, GitHub Actions.
### 2.2 Fuera de alcance (etapas posteriores / uso personal)
 
- **Capa Kappa**: bot de trading en tiempo real consumiendo el websocket de Binance. El diseño debe *permitir* su reúso, pero su implementación no es parte del grado.
- Ejecución de órdenes reales, gestión de riesgo en vivo y conectividad con exchange en producción.
- Activos distintos a BTC/USDT (el diseño debe ser extensible a más activos, pero la primera etapa es mono-activo).
> **Arquitectura Lambda → Kappa.** La plataforma se concibe como arquitectura Lambda: una capa Batch (este TRD) para el histórico y, a mediano plazo, una capa Kappa para tiempo real. Las decisiones de la Capa 2 priorizan la reusabilidad batch↔streaming para no mantener dos implementaciones del detector DC.
 
---
 
## 3. Arquitectura general del sistema
 
Patrón medallion de 4 capas, todas materializadas en **Parquet** sobre Cloud Storage en **us-east1**. El cómputo se asigna **por capa según su perfil** (ver §3.2), no por reflejo a un único servicio. Flujo: `data.binance.vision` → Capa 1 (raw) → Capa 2 (eventos DC) → Capa 3 (tramas) → Capa 4 (indicadores) → backtesting/ML.
 
### 3.1 Vista de capas (flujo de datos)
 
| Capa | Nombre | Contenido | Naturaleza de cómputo |
|---|---|---|---|
| 1 | Raw / Landing | Parquet con aggTrades crudos (8 columnas), ordenados temporalmente | Exigente en memoria (carga + sort). **Meses independientes** (sin carry-over en L1). Paralelizable por ventana histórica |
| 2 | DC Events | Por θ: intervalos extremo→confirmación y confirmación→fin de overshoot. Una fila por evento | **Estrictamente secuencial** dentro de θ (O(n), una pasada). Paralelizable entre θ. **El carry-over de huérfanos aparece aquí**, no en L1 |
| 3 | Tramas | Series (precio, volumen, etc.) cortadas por evento DC. Una fila por evento; pares de columnas (confirmación, overshoot) por serie | Altamente paralelizable por θ y dentro de θ |
| 4 | Indicadores | Métricas por evento DC. Tantas como se definan | Inter-evento altamente paralelizable; intra-evento por θ |
 
> **Nota.** En L1 los meses son **independientes**: el sort y la materialización de un mes no dependen de otro. La dependencia secuencial por θ y el carry-over de puntos huérfanos son propiedad de la Capa 2, no de la Capa 1. Esto es lo que habilita ejecutar L1 como un *task array* de tareas independientes (ver §3.2 y ADR-04).
 
### 3.2 Componentes de cómputo y servicios GCP
 
| Componente | Servicio GCP | Rol |
|---|---|---|
| Ingesta incremental diaria | **Cloud Run Job** (modo `daily`) | Descarga el ZIP/CSV diario, ordena y materializa Parquet provisional del mes en curso |
| Cómputo de Capa 1 (backfill + cierre mensual) | **Cloud Run Jobs** (task array, 1 tarea por mes) | Ejecuta la imagen de L1 en modos `backfill` / `monthly-close`; meses independientes |
| Cómputo pesado de Capas 2–4 | **Cloud Batch + Spot VMs (c2d/c3)** — *a confirmar por capa* | Fan-out de 50 θ y joins/agregaciones multinúcleo. **La elección de Cloud Batch se decide y justifica en el TRD-L2** |
| Motor de datos (L1/L3/L4) | DuckDB + Polars + Apache Arrow (embebidos) | Sort out-of-core, joins, agregaciones; interoperabilidad zero-copy vía Arrow |
| Almacenamiento medallion | Cloud Storage (Standard/Nearline/Coldline) | Buckets de las 4 capas + carry-over de huérfanos (L2) + lago de hallazgos de DQ |
| Metacatálogo / catálogo | Parquet hive-partitioned plano + DuckLake | Descubrimiento de particiones (provider × market × asset × θ × fecha); linaje barato sin Iceberg/Delta |
| IaC y CI/CD | Terraform (local) → GitHub Actions | Aprovisionamiento reproducible de toda la infraestructura |
| Registro de imágenes | Artifact Registry | **Una imagen Docker delgada por capa, parametrizada por modo de ejecución** |
| Seguridad y acceso | IAM, Service Accounts (Secret Manager solo en etapas posteriores) | Permisos mínimos, cuentas de servicio por job. **L1 no requiere Secret Manager** (datos públicos sin auth) |
 
> **Nota — corrige a v2.0.** En v2.0 todo el cómputo pesado se asignaba por reflejo a Cloud Batch + Spot, y la ingesta incremental a una Cloud Run Function. Reanalizado: el backfill de L1 es **liviano y memory-bound** (estimado ~40 core-horas) y cabe en o cerca del free tier de **Cloud Run Jobs** usando *task arrays* (una tarea por mes, meses independientes). Por tanto:
> - **L1 corre sobre Cloud Run Jobs** como cómputo único, en una **sola imagen** con modos `backfill`, `daily`, `monthly-close` (no imágenes separadas por modo).
> - La **ingesta incremental** usa un **Cloud Run Job** (modo `daily`), no una Cloud Run Function.
> - **Cloud Batch se difiere** y se reevalúa en el TRD-L2, donde el fan-out de 50 θ sobre la serie compartida sí puede justificar VMs Spot multinúcleo.
> - El dimensionamiento de L1 (GiB-segundos y tiempo de pared) queda **a validar con una sonda de un mes representativo** (ver §7.4 y §9.2).
 
### 3.3 Particionado y contrato de datos
 
- **Particionado hive (Capa 1, landing):** `provider=binance/market=spot/asset=BTCUSDT/year=YYYY/month=MM` dentro del bucket `landing`. El particionado de Capas 2–4 añade la dimensión θ (activo × θ × fecha).
- **Formato y compresión:** Parquet con **ZSTD nivel 3** en toda la capa medallion. Ficheros consolidados por mes (y por θ en L2–L4) para minimizar operaciones Clase B de GCS.
- **Tipos del contrato de datos (L1):**
  - **Precio y cantidad como DECIMAL exacto** (precisión ≤ 18 para que Parquet lo respalde sobre INT64); **se evita float**. El contrato entre capas es el **tipo lógico DECIMAL de Parquet**, agnóstico del motor (DuckDB/Polars/Arrow/Rust lo leen igual).
  - **Timestamp normalizado a microsegundos** venga como venga, mediante **detección por magnitud** (no por fecha), para cubrir el cambio ms→µs de Binance (SPOT, 1-ene-2025) y cualquier inconsistencia futura.
  - **Detección de header por archivo** (parsear el primer campo como entero), no asumida globalmente.
  - Se conservan las **8 columnas crudas** (el set exacto de columnas de salida se fija en el TRD-L1; `isBestMatch` está deprecado/constante).
- **Orden:** **tiempo como clave de orden primaria**; `aggTradeId` se usa **solo para deduplicar, desempatar empates de timestamp y verificar faltantes**, no como clave de orden.
- **Región única:** todos los buckets y el cómputo en us-east1 → egress intra-región nulo.
- **Mensual = verdad; diarios provisionales.** Las particiones diarias del mes en curso son provisionales; al cierre de mes se reemplazan por el consolidado mensual de Binance. **Una partición mensual cerrada queda inmutable (congelada).**
- **Carry-over (L2):** objeto diminuto por θ con los puntos huérfanos al cierre de cada mes (estado del detector: extremo, dirección, acumulador). Es propiedad de la Capa 2.
> **Nota.** El orden por tiempo (y no por aggTradeId) es lo que garantiza la pertinencia del análisis DC; el aggTradeId, al ser ≈ monotónico con el tiempo, sigue siendo útil para integridad (deduplicación y detección de huecos) pero no define el orden. La normalización de timestamp por magnitud evita depender de la fecha del archivo para decidir la unidad.
 
### 3.4 Lago de hallazgos de Data Quality
 
Se introduce un **lago de hallazgos de calidad de datos (Data Quality findings)**: un dataset **append-only en Parquet sobre GCS**, **particionado por fecha de detección**, con **esquema unificado** para todos los chequeos de todas las capas.
 
**Esquema del hallazgo (unificado):**
 
| Campo | Descripción |
|---|---|
| `finding_id` | Identificador único del hallazgo |
| `detected_at` | Timestamp de detección en **microsegundos** |
| `layer` | Capa que emite el hallazgo (l1, l2, …) |
| `check_type` | Tipo de chequeo (p. ej. gap, dup, schema, range) |
| `severity` | Severidad (p. ej. info, warning, error) |
| `status` | Estado del hallazgo |
| `provider` / `market` / `asset` / `year` / `month` | Coordenadas del dato examinado |
| `metric_value` | Valor numérico asociado al chequeo |
| `details` | **JSON libre** con el detalle específico del chequeo |
| `run_id` | Identificador de la ejecución |
| `image_version` / `git_sha` | Trazabilidad a la imagen exacta que generó el hallazgo |
 
**Propiedades:**
 
- Es **distinto del lago de meta-métricas operativas (RF-19)**: aquel mide **operación** (duración, costo, preempciones, eventos por θ); este mide **calidad de datos**.
- Cada chequeo deja **doble señal**: un **log de consola** (señal inmediata para depuración) **y** una **fila persistida** (registro consumible y auditable).
- La **capa de consumo** (tabla externa de BigQuery sobre el Parquet + dashboards en Looker) se **difiere**; el dataset se diseña desde ya para que esa capa se conecte sin migración.
---
 
## 4. Requerimientos funcionales
 
*Prioridad MoSCoW: M (Must), S (Should), C (Could).*
 
| ID | Nombre | Descripción | Prio. |
|---|---|---|---|
| RF-01 | Ingesta histórica | Descargar los archivos mensuales de aggTrades SPOT BTC/USDT desde `data.binance.vision` para todo el histórico (desde 2017-08-17). | M |
| RF-02 | Ingesta incremental | Ingerir los archivos diarios del mes en curso mediante un **Cloud Run Job** (modo `daily`) y, al cierre de mes, reemplazar las particiones diarias provisionales por el consolidado mensual de Binance. | M |
| RF-03 | Ordenamiento | Garantizar el orden temporal estricto de la serie antes del análisis DC, usando el **tiempo como clave de orden primaria**. `aggTradeId` se usa solo para **deduplicar, desempatar timestamps iguales y verificar faltantes**, no como clave de orden. | M |
| RF-04 | Materialización Parquet | Convertir los datos crudos a Parquet en la Capa 1 con: **tipos DECIMAL exactos** (precisión ≤ 18) para precio y cantidad; **timestamp normalizado a microsegundos por detección de magnitud**; **detección de header por archivo**; **compresión ZSTD nivel 3**. | M |
| RF-05 | Detección DC multi-θ | Detectar eventos DC (extremo, confirmación, overshoot) para los 50 valores de θ. | M |
| RF-06 | Carry-over de huérfanos | El análisis de un mes M consume los huérfanos de M-1 y persiste los huérfanos resultantes para M+1, por cada θ. | M |
| RF-07 | Tramas por evento | Materializar, por evento DC, las series de tiempo disponibles cortadas en (confirmación, overshoot). | M |
| RF-08 | Indicadores por evento | Calcular indicadores inter e intra-evento por θ. | M |
| RF-09 | Metacatálogo | Catalogar las particiones (activo × θ × fecha) para descubrimiento y consulta. | S |
| RF-10 | Backtesting walk-forward | Soportar backtesting walk-forward (Walk-Through) sobre los indicadores. | M |
| RF-11 | Reusabilidad streaming | El núcleo del detector DC debe ser reusable sin reescritura para consumo por-tick en tiempo real (Capa Kappa). | S |
| RF-12 | Idempotencia | Todo job debe ser idempotente: re-ejecutar una ventana produce el mismo resultado (escritura por rutas únicas por ventana/θ). | M |
| RF-13 | Reanudabilidad | El pipeline debe reanudarse desde el último mes completado por θ tras una interrupción. | M |
| RF-14 | Extensibilidad multi-activo | Permitir añadir activos adicionales sin cambios estructurales. | C |
| RF-15 | Imágenes delgadas por capa | Empacar cada capa en **una sola imagen Docker delgada, parametrizada por modo de ejecución** (p. ej. L1: `backfill` / `daily` / `monthly-close`), con dependencias aisladas, para reprocesar particiones sin cargar lógica de otras capas. No se crean imágenes separadas por modo. | M |
| RF-16 | Build selectivo | El CI reconstruye y publica solo la imagen de la(s) capa(s) cuyo código (o el código compartido del que dependen) cambió. | S |
| RF-17 | Catálogo de datos | Catalogar mediante Parquet hive-partitioned plano y, opcionalmente, DuckLake para linaje/versionado de bajo costo. | S |
| RF-18 | Scaffolding de capas | Plantilla (cookie-cutter) que estandarice la creación de una capa (estructura, Dockerfile, módulo Terraform, CI, contrato de datos). | S |
| RF-19 | Meta-métricas | Cada job emite métricas del pipeline (eventos por θ, duración, costo, preempciones, versión de imagen) a un data lake de meta-métricas operativas. | S |
| RF-20 | Lago de hallazgos de Data Quality | Persistir los hallazgos de calidad de datos en un dataset **append-only en Parquet sobre GCS, particionado por fecha de detección**, con esquema unificado (ver §3.4). Distinto del lago de meta-métricas operativas (RF-19). | S |
| RF-21 | Emisión de hallazgos por librería compartida | Emitir los hallazgos mediante una función compartida `emit_findings()` cuyo contrato vive en `/shared` y se **hornea en cada imagen en build-time** (no es un servicio en ejecución). Escritura a **rutas únicas por tarea**; **sin cola de mensajería** a esta escala. | S |
 
---
 
## 5. Requerimientos no funcionales
 
| ID | Atributo | Requerimiento | Prio. |
|---|---|---|---|
| RNF-01 | Costo | Costo operativo total ≤ 100 USD/mes; objetivo de diseño < 5 USD/mes en estado estacionario. | M |
| RNF-02 | Rendimiento (backfill) | El histórico completo (50 θ × ~3·10⁹ trades) se procesa en el orden de pocas horas en un único nodo Spot. | S |
| RNF-03 | Eficiencia de memoria | Memoria pico dimensionada a una ventana mensual, no a la serie completa. El **dimensionamiento concreto de L1 (GiB-segundos y tiempo de pared) se valida con una sonda de un mes representativo** antes de fijar la configuración de Cloud Run Jobs. | M |
| RNF-04 | Correctitud | Detección DC determinista y verificable; el carry-over no introduce discontinuidades entre meses (L2). En L1, la correctitud incluye la **normalización de timestamp a µs**, los **tipos DECIMAL exactos** y la **inmutabilidad de la partición mensual cerrada**. | M |
| RNF-05 | Reproducibilidad | Infraestructura desplegable vía IaC; resultados reproducibles a partir de código + datos. | M |
| RNF-06 | Resiliencia a preempción | Tolerar preempción de Spot con pérdida máxima de una ventana-θ y reintento automático. | M |
| RNF-07 | Observabilidad | Emitir métricas de costo, duración por ventana, conteo de eventos por θ y tasa de preempción. | S |
| RNF-08 | Seguridad | Mínimo privilegio (IAM), cuentas de servicio por job, sin credenciales embebidas (Secret Manager). | M |
| RNF-09 | Portabilidad / reúso | Núcleo de cómputo agnóstico de la fuente (Parquet batch o websocket streaming). | S |
| RNF-10 | Latencia (futura Kappa) | Detector DC con latencia predecible por-tick (sin pausas de GC) para habilitar tiempo real. | C |
| RNF-11 | Costo cero de servicios cross | Todo el stack transversal (CI/CD, registro, observabilidad, secretos, orquestación) opera a costo cero o mínimo. | M |
| RNF-12 | Sin lock-in / portabilidad | La observabilidad permite cambiar de backend (GCP → Prometheus/Grafana) sin reescritura (OpenTelemetry). | C |
| RNF-13 | Reproducibilidad de imágenes | Cada partición procesada es trazable a la imagen exacta (git SHA + versión semántica) que la generó. | S |
| RNF-14 | Acoplamiento débil entre capas | Las capas se comunican por contratos de datos Parquet documentados, no por acoplamiento de código. | M |
 
---
 
## 6. Decisiones arquitectónicas (ADR)
 
*Cada decisión es una invariante heredada por los TRD de capa.*
 
### 6.1 ADR-01 — Ventana de procesamiento mensual
 
| Campo | Contenido |
|---|---|
| **Decisión** | Procesar por ventanas mensuales (~96 ventanas para 8 años), alineadas con el delivery mensual de Binance. |
| **Alternativas** | Diaria, semanal, trimestral, anual, ventana única. |
| **Justificación** | El mes es la mayor ventana que mantiene el sort de Capa 1 acotado (~6–12 GB) y coincide con la frontera natural del carry-over de huérfanos, proveyendo 96 checkpoints por cadena de θ. Ventanas menores no reducen memoria (la Capa 2 ya es streaming) y multiplican el overhead; ventanas mayores disparan la memoria de Capa 1 sin reducir el costo de cómputo (O(n), invariante al tamaño de ventana). |
| **Benchmark** | En la evaluación multicriterio (8 criterios), la ventana mensual obtuvo la mayor puntuación. |
| **Implicación** | Dependencia secuencial por θ entre meses (M+1 requiere huérfanos de M); los 50 θ son cadenas independientes y paralelas. |
 
### 6.2 ADR-02 — Detección DC por "fan-out por tick" (memoria compartida)
 
| Campo | Contenido |
|---|---|
| **Decisión** | Cargar la serie de la ventana UNA vez en memoria read-only y evaluar los 50 θ por cada tick (fan-out interno), compartiendo la serie entre los θ. |
| **Alternativa rechazada** | Una instancia/proceso por θ leyendo la serie completa de forma independiente (50× lectura). |
| **Justificación** | El detector DC mantiene estado escalar O(1) por θ (extremo, dirección, acumulador); el estado de los 50 θ (~2–3 KB) cabe en caché L1. El fan-out es aritmética sobre datos en caché (~1 ns) frente al acceso a RAM (~70–100 ns), casi gratis. El esquema por-θ desperdicia 50× el ancho de banda de memoria y el trabajo de decodificación de Parquet sin acelerar nada. |
| **Memoria** | Pico de Capa 2 ≈ 1×serie de la ventana (~0,5–0,6 GB) + 50×estado diminuto, **no** 50×serie. Desacopla el número de θ del tamaño de RAM. |
| **Reusabilidad** | El patrón "un tick → 50 detectores → emite eventos" es idéntico al de streaming en tiempo real (Kappa): el mismo núcleo se alimenta desde Parquet (batch) o websocket (streaming). |
 
### 6.3 ADR-03 — Lenguaje del núcleo: Rust (con orquestación Python)
 
| Campo | Contenido |
|---|---|
| **Decisión** | Núcleo del detector DC en Rust, orquestado desde Python vía PyO3. Para el grado se admite Python+Numba como puente si apremia el tiempo, con plan de portar a Rust antes de la Kappa. |
| **Alternativas** | C++ (máximo rendimiento, más propenso a errores), Python+Numba (rápido de desarrollar, JIT + GC), Julia (tiene GC, ecosistema menor), Python puro/pandas (inadecuado: el hot loop DC es secuencial con dependencia de estado, no vectorizable columnar). |
| **Justificación** | Rust ofrece rendimiento near-C++ sin garbage collector → latencia predecible (clave para la Kappa), seguridad de memoria en compilación, ecosistema fuerte (Polars/arrow-rs, tokio/tungstenite). Máxima reusabilidad batch↔streaming. |
| **Hardware** | Workload memory-bound, no CPU-bound; AVX2 (c2d/EPYC Milan) basta. AVX-512 (C3/C3D) solo se justificaría al escalar a miles de θ. |
 
### 6.4 ADR-04 — Cómputo por capa: Cloud Run Jobs para L1, Cloud Batch diferido a L2
 
| Campo | Contenido |
|---|---|
| **Decisión** | Asignar el cómputo **por capa según su perfil**. **Capa 1: Cloud Run Jobs** (task array, una tarea por mes) como cómputo único, en una sola imagen con modos `backfill`/`daily`/`monthly-close`. **Cloud Batch + Spot se difiere** y se decide/justifica en el **TRD-L2**, donde el fan-out de 50 θ sobre la serie compartida puede requerir VMs multinúcleo. |
| **Reanálisis (corrige v2.0)** | v2.0 asignaba Cloud Batch + Spot a todo el cómputo pesado por reflejo. Hallazgo: el backfill de L1 es **liviano y memory-bound** (~40 core-horas estimadas), con **meses independientes**, y cabe en o cerca del free tier de Cloud Run Jobs mediante *task arrays*. No hay justificación para provisionar VMs Spot en L1. |
| **Justificación** | Cloud Run Jobs ofrece *task arrays* (una tarea por índice = por mes), reintentos, y un **free tier perpetuo propio** que probablemente absorbe L1; elimina la gestión de VMs y la exposición a preempción para una carga que no la necesita. Cloud Batch no tiene costo de servicio (solo paga el Compute Engine subyacente), por lo que diferirlo no cuesta nada y evita complejidad prematura. |
| **Dimensionamiento** | **A validar con una sonda de un mes representativo** (medir GiB-segundos y tiempo de pared reales) antes de fijar memoria/CPU por tarea y confirmar el encaje en el free tier de Cloud Run. |
| **Free tier** | Cada servicio tiene su **propio cupo perpetuo por billing account** (ver §7). El e2-micro del free tier de Compute Engine ya no es necesario para la ingesta incremental, que ahora corre como Cloud Run Job. |
 
### 6.5 ADR-05 — Stack de procesamiento: DuckDB + Polars + Apache Arrow
 
| Campo | Contenido |
|---|---|
| **Decisión** | DuckDB + Polars + Apache Arrow como stack transversal, embebido en las imágenes de capa. DuckDB es el motor por defecto; Polars la alternativa para transformaciones tipo dataframe; Arrow la capa de interoperabilidad zero-copy. |
| **Alternativas** | DataFusion (Rust; relevante solo para I/O de Capa 2), chDB/ClickHouse (sin ventaja clara a esta escala), Pandas/Dask/Vaex (1–2 órdenes de magnitud más lentos), BigQuery serverless (no más barato para reproceso por ventanas). |
| **Evidencia** | PDS-H (mayo 2025, SF-100 ~100 GB): DuckDB 19,65 s y Polars streaming 23,94 s, frente a PySpark 312 s y Dask 548 s. En estrés de 140 GB, DuckDB usó ~1,3 GB de RAM pico vs ~17 GB de Polars. |
| **Mapeo por capa** | L1 (sort masivo out-of-core) y L3/L4 (joins, agregaciones, window functions): DuckDB por su spilling a disco maduro. L2: Rust para el hot loop; DataFusion/Arrow para I/O. Cada capa revalidará su óptimo al desarrollarse. |
| **Costo / GCP** | Lectura de Parquet desde GCS intra-región (us-east1) sin egress. Imágenes delgadas instalan solo las extensiones DuckDB necesarias por capa (`httpfs`, `ducklake`). |
 
### 6.6 ADR-06 — Catálogo: Parquet hive-partitioned plano + DuckLake (no Iceberg/Delta)
 
| Campo | Contenido |
|---|---|
| **Decisión** | Catalogar con Parquet hive-partitioned plano como baseline y, si se desea linaje/versionado, DuckLake con catálogo SQL en un archivo DuckDB/SQLite sobre GCS. **No** usar Iceberg ni Delta Lake. |
| **Justificación** | Para un pipeline que reprocesa/reemplaza particiones mensualmente, el time-travel de Iceberg/Delta no justifica su costo: sus metadatos (manifests, snapshots) y la retención de versiones viejas inflan el storage y exigen jobs de expiración/compactación. DuckLake da linaje barato con un catálogo de un solo archivo y evita Cloud SQL. |
| **Costo** | Catálogo DuckLake en archivo SQLite/DuckDB sobre GCS = centavos de storage. Evita instancias siempre encendidas. |
| **Limitación** | DuckLake con SQLite/DuckDB es single-writer (suficiente para batch single-node). Migrar a Postgres solo si se requiriera concurrencia de escritura multi-cliente. Iceberg + BigLake metastore queda como vía futura si surge interoperabilidad multi-motor. |
 
### 6.7 ADR-07 — Repositorio: monorepo políglota, stacks (no environments), imagen por capa por modo
 
| Campo | Contenido |
|---|---|
| **Decisión** | Un único monorepo políglota (Rust + Python) con **una imagen Docker delgada por capa, parametrizada por modo de ejecución**, código compartido en `/shared`, infraestructura en `/infra` y plantilla de capa en `/scaffold`. La IaC separa **stacks** (`stacks/batch`, `stacks/kappa`), no *environments*. |
| **Corrección terminológica (v2.1)** | Lo que v2.0 llamaba `environments/batch` y `environments/kappa` **no son entornos de despliegue**: son **stacks/arquitecturas distintas** (todo el pipeline batch vs. el sistema Kappa futuro). Se renombran a `stacks/batch` y `stacks/kappa`. El término *environment* se **reserva** para entornos de despliegue (dev/qa/uat/prod), que hoy no se instancian. |
| **Imagen por capa** | Una sola imagen por capa con modos de ejecución (p. ej. L1: `backfill`/`daily`/`monthly-close`), **no** imágenes separadas por modo. Multi-stage build + base slim/distroless + `.dockerignore` por capa. |
| **Versionado** | Tags Git por capa (`l<N>_<nombre>/vX.Y.Z`) e imágenes etiquetadas con versión semántica + git SHA para trazabilidad inmutable y reproceso con la imagen exacta. |
| **Extensibilidad** | Plantilla cookie-cutter en `/scaffold` + contratos de datos Parquet documentados (incluido el contrato de hallazgos de DQ) → agregar una capa es un proceso repetible. El módulo Terraform `layer` se instancia una vez por capa. |
 
### 6.8 ADR-08 — Observabilidad a costo cero, nativa de GCP
 
| Campo | Contenido |
|---|---|
| **Decisión** | Stack de operaciones a ~0 USD/mes: GitHub (monorepo) + GitHub Actions (free tier) + Artifact Registry + Terraform con estado en GCS + Cloud Logging/Monitoring (free tier) + meta-métricas en BigQuery (free tier) + dashboards en Looker Studio + Secret Manager + Cloud Workflows + Budget alerts. |
| **Justificación** | La prioridad de gasto es el pipeline de datos; los servicios cross deben ser gratis o casi. Los free tiers nativos cubren el volumen de un solo dev. Se evita Cloud Composer (Airflow gestionado, cientos de USD/mes) y self-hosted Prometheus/Grafana (overhead de operación). |
| **Anti lock-in** | Instrumentar con OpenTelemetry (exportador OTLP) para migrar de backend hacia la Kappa (dashboards de baja latencia / P&L) sin reescribir. |
| **Alerta de pricing** | Cloud Monitoring comenzará a cobrar alerting no antes de ~sep-2026 (0,35 USD/metric reference); mantener pocas políticas y apoyarse en alertas exentas (Billing, Quota, Uptime). |
 
---
 
## 7. Dimensionamiento y modelo de costos
 
*Cifras de referencia a 2026 para us-east1, sujetas a la variabilidad diaria de los precios Spot. El presupuesto no es la restricción dominante; lo son la cuota de vCPU, la memoria pico y la simplicidad operativa.*
 
### 7.1 Volumen de datos (estimado)
 
| Métrica | Valor | Nota |
|---|---|---|
| Inicio de la serie | 2017-08-17 | ~8 años, no 10 |
| aggTrades totales (8 años) | ~2,5–3,5 mil millones | Estimado |
| CSV crudo histórico | ~185 GB | Descomprimido |
| Parquet comprimido histórico | ~30 GB | Snappy/ZSTD |
| Medallion completo (4 capas) | ~45–75 GB | Capas 2–4 muy livianas |
| Ventana mensual reciente | ~30–40 M filas / ~2,5–5 GB CSV | ~0,5–0,6 GB precio+ts en memoria |
 
### 7.2 Tarifas de cómputo (us-east1, 2026)
 
Se mantienen las tarifas de Cloud Batch + Spot como referencia para Capas 2–4 (a confirmar en TRD-L2). Para **L1 se usa Cloud Run Jobs**, cuyo costo se modela por **vCPU-segundo y GiB-segundo de ejecución de tarea**, con un **cupo perpetuo gratuito propio**. El costo efectivo de L1 **queda a validar con la sonda** (§7.4); la expectativa es que el backfill caiga en o cerca del free tier.
 
| Instancia (referencia L2–L4) | vCPU/RAM | On-demand $/h | Spot $/h | Descuento |
|---|---|---|---|---|
| **c2d-standard-8 (base)** | 8 / 32 GB | 0,3632 | ≈0,083 | ≈77 % |
| **c2d-standard-16** | 16 / 64 GB | ≈0,726 | ≈0,175 | ≈76 % |
| **c3-standard-88** | 88 / 352 GB | ≈4,5 | ≈0,69 | ≈84 % |
| **e2-micro (free)** | 2 / 1 GB | ≈0,0084 | (no aplica free a Spot) | — |
 
### 7.3 Costo en estado estacionario (mensual)
 
La estructura de costos se mantiene, con un matiz: el cómputo de **L1 migra de Spot a Cloud Run Jobs** (probablemente ~0 USD en estado estacionario por el free tier propio de Cloud Run). El resto (almacenamiento medallion, operaciones GCS, lago de meta-métricas y lago de hallazgos de DQ) no cambia materialmente.
 
| Concepto | Conservador | Intensivo |
|---|---|---|
| Cómputo (Cloud Run Jobs L1 + Spot/Batch L2–L4) | ≈ 0,50–1,00 USD | ≈ 3–13 USD |
| Almacenamiento (GCS, medallion) | ≈ 0,75–1,50 USD | ≈ 1,50 USD |
| Operaciones GCS (Clase A/B) | ≈ 0,20 USD | ≈ 4,00 USD |
| Disco / Artifact Registry / otros | ≈ 0,40 USD | ≈ 1,85 USD |
| Egress (intra-región) | ≈ 0,00 USD | ≈ 0,00 USD |
| **TOTAL ESTIMADO** | **≈ 2–4 USD/mes** | **≈ 18–22 USD/mes** |
 
> **Holgura presupuestal.** Incluso el escenario intensivo (~18–22 USD/mes) deja > 75 % de margen frente a los 100 USD/mes. El backfill inicial (pico único, ~5–15 USD) lo absorbe el crédito de prueba de 300 USD válido por 90 días.
 
> **Nota — modelo de free tier.** No existe una única bolsa *always-free* compartida entre servicios: **cada servicio tiene su propio cupo perpetuo** (Cloud Run, Compute Engine, GCS, BigQuery, …), contabilizado **por billing account** (compartido entre projects del mismo billing account, **no** multiplicado por project). La **única bolsa compartida** entre servicios es el **crédito de prueba de 300 USD** (90 días). **Cloud Batch no cobra por el servicio de orquestación**: solo se pagan los recursos de Compute Engine que provisiona.
 
### 7.4 Configuración de cómputo (referencia)
 
**Capa 1 — Cloud Run Jobs:**
 
- **Task array** con una tarea por mes (índice de tarea = mes); meses independientes, sin dependencia secuencial.
- Una sola imagen, seleccionando el modo (`backfill`/`daily`/`monthly-close`) por variable de entorno/argumento.
- Reintentos de tarea nativos de Cloud Run Jobs; idempotencia por escritura a rutas únicas (partición mensual).
- **Memoria/CPU por tarea: a fijar tras la sonda** de un mes representativo (medir GiB-segundos y tiempo de pared).
**Capas 2–4 — Cloud Batch + Spot (referencia futura, se confirma en TRD-L2):**
 
- `provisioningModel: SPOT`; `instanceTerminationAction: STOP`.
- `maxRetryCount: 3`; `lifecyclePolicies: RETRY_TASK` con `actionCondition exitCodes=[50001]` (reintento solo ante preempción).
- Paralelismo de θ **intra-proceso** (multihilo sobre la serie compartida), para preservar la memoria compartida del fan-out.
- `allowedLocations`: múltiples zonas de us-east1 (b/c/d) para diluir preempciones correlacionadas.
---
 
## 8. Operaciones (DevOps / Platform / MLOps)
 
*El stack de operaciones se diseña para un único desarrollador con presupuesto mínimo: todo el plano transversal opera a costo cero o casi nulo, reservando el gasto para el pipeline de datos. El diseño es extensible: agregar una capa nueva es un proceso repetible y administrable.*
 
### 8.1 Estrategia de repositorio (monorepo políglota)
 
Un único monorepo aloja la lógica de todas las capas, el código compartido (núcleo Rust de DC + bindings PyO3 + utilidades Python + **`emit_findings()` y el contrato del hallazgo de DQ**), la infraestructura como código y la plantilla de capas. Cada capa produce **una imagen Docker delgada, parametrizada por modo de ejecución**, mediante multi-stage build, base slim/distroless y `.dockerignore` por capa, de modo que reprocesar una partición de una capa no arrastra dependencias de otras.
 
```
dc-platform/                      # raíz del monorepo
├── .github/workflows/            # ci.yml, _build-layer.yml, terraform.yml
├── shared/                       # CÓDIGO COMPARTIDO
│   ├── dc_core/                  # núcleo Rust de Directional Change (crate)
│   ├── dc_pyo3/                  # bindings PyO3 sobre dc_core
│   ├── pyutils/                  # utilidades Python comunes
│   └── dq/                       # emit_findings() + contrato del hallazgo (DQ)
├── layers/                       # UNA CARPETA POR CAPA (una imagen por capa, por modo)
│   ├── l1_ingest/                # Dockerfile, .dockerignore, deps, src/, config/
│   │                             #   modos: backfill | daily | monthly-close
│   ├── l2_dc_events/             #   config/thetas.yaml (los 50 θ versionados)
│   ├── l3_frames/
│   └── l4_indicators/
├── infra/                        # INFRAESTRUCTURA COMO CÓDIGO
│   ├── modules/                  # layer (parametrizable), registry, observability
│   └── stacks/                   # batch/ (actual), kappa/ (futuro)   ← antes "environments/"
├── scaffold/layer_template/      # plantilla cookie-cutter para nuevas capas
└── docs/data-contracts.md        # contratos de interfaz Parquet entre capas (incl. DQ)
```
 
**Versionado:** tags Git por capa (`l<N>_<nombre>/vX.Y.Z`) e imágenes etiquetadas con versión semántica + git SHA, para reproceso trazable con la imagen exacta.
 
### 8.2 Infraestructura como código
 
- **Módulo "layer" parametrizable:** un módulo Terraform reutilizable (bucket/prefijo GCS, ruta de imagen, IAM mínimo, plantilla de job) que se instancia una vez por capa.
- **Estado remoto en GCS:** backend `gcs` con versioning + UBLA (costo casi nulo, locking nativo). Se evita Terraform Cloud y Cloud SQL.
- **Stacks separados (no environments):** `stacks/batch` (actual) y `stacks/kappa` (futuro), cada uno con su propio estado. Son **arquitecturas distintas**, no entornos de despliegue del mismo sistema.
- **Reserva del término *environment*:** se reserva para entornos de despliegue (dev/qa/uat/prod), que **no se instancian hoy**; su introducción se difiere a la etapa del bot de trading.
- **Convivencia:** la IaC vive en el mismo monorepo, versionada junto a la lógica; los módulos child no declaran provider.
> **Nota — corrige v2.0.** v2.0 usaba `environments/batch` y `environments/kappa`. Eran un nombre incorrecto: representan **stacks/arquitecturas** (todo el batch vs. la Kappa), no entornos de despliegue del mismo sistema. Renombrados a `stacks/`.
 
### 8.3 Registro de imágenes y CI/CD
 
- **Artifact Registry (us-east1):** elegido sobre ghcr.io por integración nativa con Cloud Run/Batch y egress cero intra-región. Free tier 0,5 GB + cleanup policies (borrar untagged y conservar N versiones) mantienen el costo en ~0.
- **GitHub Actions (free tier):** 2.000 min/mes en repos privados; build selectivo con `dorny/paths-filter` para reconstruir solo la imagen de la capa cambiada (incluyendo rebuild si cambia `shared/`).
- **Autenticación:** Workload Identity Federation (sin claves de service account de larga vida). Runners GitHub-hosted (no self-hosted).
### 8.4 Observabilidad y meta-métricas (costo cero)
 
- **Logs y métricas nativos:** Cloud Logging (50 GiB/proyecto/mes gratis) + Cloud Monitoring (métricas de sistema gratis + 150 MiB custom/mes). Jobs (Cloud Run Jobs en L1; Cloud Batch en L2–L4) con logging nativo.
- **Lago de meta-métricas operativas (RF-19):** cada job emite una fila con eventos por θ, duración, costo estimado, tasa de preempción y versión de imagen. Mide **operación**.
- **Lago de hallazgos de Data Quality (RF-20):** dataset append-only en Parquet sobre GCS, particionado por fecha de detección, con esquema unificado (§3.4). Mide **calidad de datos**. **Es un lago distinto** del de meta-métricas operativas.
- **Emisión de hallazgos (RF-21):** vía `emit_findings()` en `/shared`, horneada en cada imagen en build-time; escritura a rutas únicas por tarea; **sin Pub/Sub** a esta escala. Cada chequeo deja **log de consola + fila persistida**.
- **Consumo:** la capa de consumo de ambos lagos (tablas externas de BigQuery + dashboards en Looker Studio) se **difiere**, pero los datasets se diseñan para conectarse sin migración.
- **Portabilidad:** instrumentar con OpenTelemetry (OTLP) para migrar de backend hacia la Kappa sin reescritura; reservar Grafana Cloud free tier para dashboards de tiempo real futuros.
> **Nota — sobre no usar cola.** A esta escala no se introduce Pub/Sub para los hallazgos: object storage maneja escrituras concurrentes a **claves distintas** sin contención. Se reevaluaría una cola solo ante contención real de escritura o para el stream de la Kappa (donde sería una suscripción nativa Pub/Sub→BigQuery).
 
### 8.5 Secretos, configuración y orquestación
 
- **Secret Manager — alcance acotado:** **L1 no requiere Secret Manager.** Los datos de `data.binance.vision` son **públicos y sin autenticación**, por lo que la ingesta histórica e incremental no maneja credenciales. Secret Manager aplica a **etapas posteriores** (p. ej. la **API autenticada de Binance en la Capa Kappa**, service accounts con secretos). Free tier de 6 versiones activas; destruir versiones obsoletas (las deshabilitadas siguen contando).
- **Configuración:** parámetros no sensibles (los 50 θ, ventanas, rutas, modo de ejecución) en archivos YAML versionados por capa; solo secretos (cuando existan) en Secret Manager.
- **Orquestación:** Cloud Workflows (5.000 pasos/mes gratis) + Cloud Scheduler para encadenar L1→L2→L3→L4 con dependencias y carry-over. Se descarta Cloud Composer (Airflow) por costo.
### 8.6 Playbook: agregar una nueva capa
 
*Procedimiento repetible para incorporar capas futuras (identificación de regímenes, análisis genético de estrategias, backtesting walk-forward, data lake de meta-métricas, y eventualmente Kappa):*
 
1. Copiar la plantilla `scaffold/layer_template` a `layers/l<N>_<nombre>`.
2. Definir el contrato de datos Parquet (entrada/salida) en `docs/data-contracts.md`.
3. Implementar la lógica con dependencias aisladas y escribir el Dockerfile delgado (parametrizado por modo si aplica).
4. Instanciar el módulo Terraform `layer` en `stacks/batch`.
5. Añadir el filtro de CI (incluyendo `shared/` si aplica) y la cleanup policy de la imagen.
6. Emitir meta-métricas (RF-19) y hallazgos de DQ (RF-20/21) y encadenar la capa en el workflow de orquestación.
7. `terraform plan/apply`, push y verificación del build selectivo y primer run.
### 8.7 Resumen de herramientas y costos transversales
 
| Función | Herramienta | Costo esperado |
|---|---|---|
| Repositorio / CI-CD | GitHub + GitHub Actions (free tier) | 0 USD |
| Registro de imágenes | Artifact Registry (us-east1) + cleanup | ≈ 0 USD |
| IaC / estado | Terraform + backend GCS | ≈ 0 USD |
| Cómputo L1 | Cloud Run Jobs (free tier propio) | ≈ 0 USD (a validar con sonda) |
| Logs / métricas | Cloud Logging + Cloud Monitoring | 0 USD (free tier) |
| Meta-métricas / hallazgos DQ / dashboards | BigQuery + Looker Studio (consumo diferido) | 0 USD (free tier) |
| Alertas | Budget alerts + Pub/Sub + log-based | ≈ 0 USD |
| Secretos | Secret Manager (etapas posteriores) | ≈ 0 USD |
| Orquestación | Cloud Workflows + Scheduler | ≈ 0 USD |
| Motor de datos | DuckDB + Polars + Arrow (embebidos) | Solo cómputo |
 
> **Principio rector.** El único costo real del sistema es el pipeline de cómputo (Cloud Run Jobs / Cloud Batch + Cloud Storage), con objetivo < 5 USD/mes. Todo el plano de operaciones se mantiene en ~0 USD/mes mediante free tiers nativos.
 
---
 
## 9. Restricciones, supuestos y consideraciones transversales
 
### 9.1 Restricciones de plataforma
 
- **Estrategia de GCP projects:** se arranca con **un único project** que aloja **toda la plataforma**. Todas las capas viven en el mismo project, **separadas por buckets + service accounts + IAM**. **Una capa NO es un project.** La separación por **entorno** (p. ej. un project de pruebas dev/qa/prod) se **difiere a la etapa del bot de trading**.
- **Mapeo de jerarquía GCP ↔ AWS (contexto):**
  | GCP | AWS | Rol |
  |---|---|---|
  | Organization | Organization | Raíz de gobernanza |
  | Folder | OU (Organizational Unit) | Agrupación intermedia |
  | **Project** | **Account** | Frontera de aislamiento e IAM |
  | Billing Account | (parte de facturación de la Account) | Frontera de facturación; agrega varios projects |
  En GCP el **project** es la frontera de aislamiento e IAM (equivalente a una *Account* de AWS), mientras que el **billing account** es la frontera de facturación que puede agregar varios projects.
- **Cuota de vCPU por defecto:** 8 vCPU/región en proyectos nuevos. Relevante para Cloud Batch en L2–L4 (no para Cloud Run Jobs de L1); requiere solicitar aumento (autoaprobado en incrementos modestos).
- **Always Free de Cloud Storage:** solo 5 GB-mes Standard en us-east1/us-west1/us-central1. El medallion (~45–75 GB) lo excede; el costo aun así es de ~1–2 USD/mes.
- **Free trial:** 300 USD por 90 días; prohíbe minería de cripto (el análisis ML/DC sobre datos de cripto SÍ está permitido), GPUs y VMs Windows durante el trial.
- **Región única:** us-east1 para todo (cómputo, buckets, Artifact Registry) → egress intra-región nulo.
- **Límite de paralelismo de Batch (referencia L2–L4):** máx. 1.000 tareas en paralelo por job; hasta ~100.000 tareas por task group.
### 9.2 Supuestos
 
- **Backfill de L1 a validar con sonda:** el estimado de ~40 core-horas y el encaje en el free tier de Cloud Run Jobs **se confirman midiendo un mes representativo** (GiB-segundos y tiempo de pared) antes de fijar el dimensionamiento.
- **Timestamp por magnitud:** la unidad (ms/µs) se decide por la **magnitud del valor**, no por la fecha del archivo; se normaliza siempre a microsegundos.
- **Tipos exactos:** precio y cantidad se representan como **DECIMAL exacto** (≤ 18 de precisión); no se usa float en el contrato.
- **Meses independientes en L1:** el procesamiento de un mes no depende de otro (la dependencia secuencial es de L2).
- Estado del detector DC estrictamente O(1) por θ (sin buffers ni materialización intra-evento en Capa 2).
- Para θ pequeños, es muy improbable que un evento cruce varios meses; el carry-over son pocos puntos huérfanos, no eventos enteros.
- Los archivos mensuales de Binance llegan casi ordenados (por aggTradeId ≈ orden temporal), abaratando el sort intra-mes.
- Throughput efectivo del detector ~10–20 M ticks/s/core (estimación de ingeniería; medir con datos reales).
- DuckDB resuelve el sort larger-than-memory con spilling a disco; el `temp_directory` apunta a SSD/NVMe local. Ciertas operaciones holísticas (`list`, `string_agg`, PIVOT) no spillean y deben evitarse o acotarse.
- El catálogo DuckLake con SQLite/DuckDB es single-writer, suficiente para el pipeline batch single-node.
### 9.3 Seguridad y acceso (transversal)
 
- IAM por mínimo privilegio: una cuenta de servicio por tipo de job, con permisos restringidos a sus buckets.
- Sin credenciales embebidas; secretos en Secret Manager (cuando existan; L1 no maneja credenciales).
- Buckets privados; acceso uniforme a nivel de bucket; versionado activado en la Capa 1 raw.
- IaC versionada (Terraform); estados remotos protegidos; revisión de cambios vía pull request al pasar a GitHub Actions.
### 9.4 Ciclo de vida de datos y memoria
 
- **Mensual = verdad, diarios provisionales:** los diarios del mes en curso son provisionales; al cierre se reemplazan por el consolidado mensual y **la partición mensual cerrada queda inmutable (congelada)**.
- Lifecycle de GCS: datos del mes en curso en Standard/Nearline; consolidado mensual estable migrable a Coldline/Archive (cuidando los mínimos de permanencia para evitar cargos por borrado temprano).
- Al cierre de mes: eliminar particiones diarias provisionales (DELETE gratis en GCS) y reemplazar por el consolidado mensual inmutable.
- Entre ventanas: limpiar la serie en memoria y cargar la siguiente; no mantener más de una ventana raw viva por tarea.
---
 
## 10. Riesgos y mitigaciones
 
| ID | Riesgo | Impacto | Mitigación |
|---|---|---|---|
| R-01 | Preempción de Spot a mitad de una ventana (L2–L4). | Medio | Idempotencia + checkpoint por mes/θ + reintento Batch (exitCode 50001) + multi-zona. |
| R-02 | Sort de Capa 1 excede la RAM en meses muy pesados. | Medio | Sort externo/out-of-core (DuckDB) sin agrandar la ventana; ajustar memoria de la tarea Cloud Run tras la sonda. |
| R-03 | Discontinuidad de eventos en el borde mensual (carry-over mal manejado, L2). | Alto | Contrato estricto de huérfanos por θ; pruebas de correctitud contra procesamiento sin particionar en una muestra. |
| R-04 | Cuota de vCPU insuficiente para el dimensionamiento de L2–L4. | Bajo | Solicitar aumento con antelación; la cuota Spot global se fija en 10× la estándar. |
| R-05 | Variabilidad de precios Spot / indisponibilidad de capacidad (L2–L4). | Bajo | Fallback Spot→Standard en el MIG; el costo on-demand del backfill sigue siendo < 15 USD. |
| R-06 | Huecos conocidos en archivos históricos de Binance. | Medio | Validación de integridad por mes (conteos, continuidad de aggTradeId) antes de promover a Capa 2; hallazgo de DQ. |
| R-07 | Deuda de reescritura si se prototipa en Numba y luego se porta a Rust. | Medio | Aislar el núcleo DC tras una interfaz estable; pruebas de equivalencia entre implementaciones. |
| R-08 | Madurez de DuckLake (formato emergente, catálogo single-writer). | Bajo | Datos siguen siendo Parquet portable; baseline en Parquet hive plano; migrar a Postgres/Iceberg solo si surge concurrencia o interoperabilidad multi-motor. |
| R-09 | Inicio del cobro de alerting de Cloud Monitoring (~sep-2026). | Bajo | Mantener pocas políticas; apoyarse en alertas exentas (Billing, Quota, Uptime); usar log-based metrics. |
| R-10 | Lectura de GCS por httpfs más lenta que gcsfs según patrón de acceso. | Bajo | Medir en el entorno real; considerar la extensión nativa de GCS o fsspec/gcsfs; co-localizar en us-east1. |
| R-11 | **Política de fallo de calidad de datos no definida**: ante un chequeo fallido, no está decidido si se hace *fail-closed* (cuarentena, no promover a L2) o continuar-con-hallazgo; ni cómo distinguir un hueco legítimo de Binance (aceptar con hallazgo) de una descarga corrupta (abortar y reintentar). | Alto | **Abierto — a resolver en TRD-L1.** Mientras tanto, todo chequeo deja hallazgo persistido; la decisión de promoción se trata como invariante pendiente, no se hereda. |
| R-12 | **Costura inter-mensual de aggTradeId**: discontinuidad o solapamiento de aggTradeId en el borde entre meses podría indicar faltantes o duplicados no detectados. | Medio | Verificación de continuidad de aggTradeId en el borde entre meses (a especificar en TRD-L1); hallazgo de DQ por discontinuidad. |
 
---
 
## 11. Roadmap de implementación
 
*Cada fase abre el TRD de capa correspondiente, que profundiza algoritmos, esquemas y criterios de aceptación.*
 
| Fase | Capa / hito | Entregables | TRD |
|---|---|---|---|
| F0 | Fundaciones de operaciones | Monorepo, módulo Terraform `layer` + estado GCS, Artifact Registry + cleanup, CI con build selectivo, WIF, un único GCP project. | (este TRD) |
| F1 | Capa 1 — Raw | Ingesta histórica + incremental sobre **Cloud Run Jobs** (task array por mes, una imagen por modo); sort por tiempo (DuckDB); Parquet DECIMAL/ZSTD-3; timestamp µs por magnitud; lifecycle e inmutabilidad mensual; **sonda de dimensionamiento**; **lago de hallazgos de DQ** + `emit_findings()`. **Abiertos:** política de fallo de DQ y set exacto de columnas/costura inter-mensual. | TRD-L1 |
| F2 | Capa 2 — DC Events | Núcleo DC (Rust/Numba), fan-out por tick, carry-over por θ, particionado. **Aquí se decide y justifica el uso de Cloud Batch + Spot** (el fan-out de 50 θ puede requerir VMs multinúcleo). | TRD-L2 |
| F3 | Capa 3 — Tramas | Cortes de series por evento (DuckDB/Polars); paralelización por θ y dentro de θ. | TRD-L3 |
| F4 | Capa 4 — Indicadores | Indicadores inter/intra-evento (DuckDB/Polars); catálogo de métricas. | TRD-L4 |
| F5 | Observabilidad | Meta-métricas y hallazgos de DQ a BigQuery, dashboards Looker Studio, budget y log-based alerts. | (este TRD) |
| F6 | Backtesting/ML | Walk-forward (Walk-Through), features, evaluación de estrategias. | TRD-ML |
| F7 | Capas futuras | Identificación de regímenes, análisis genético de estrategias, etc. (vía scaffolding). | TRD-Lx |
 
### 11.1 Criterios de aceptación de la línea base (capa Batch)
 
1. El backfill histórico completo (8 años, 50 θ) se ejecuta de extremo a extremo de forma reproducible vía IaC.
2. La detección DC es determinista y pasa pruebas de correctitud, incluidas las del borde mensual (carry-over).
3. El costo en estado estacionario se mantiene por debajo de 100 USD/mes (objetivo < 5 USD/mes).
4. El pipeline se reanuda correctamente tras una preempción simulada, sin duplicar ni perder eventos.
5. El backtesting walk-forward produce resultados sobre los indicadores de la Capa 4.
---
 
## 12. Anexos
 
### 12.1 Glosario
 
| Término | Definición |
|---|---|
| Directional Change (DC) | Evento que se registra cuando el precio revierte un umbral θ desde un extremo local; base del muestreo por tiempo intrínseco. |
| θ (theta) | Umbral porcentual de reversión que define un evento DC. El proyecto usa 50 valores distintos. |
| Overshoot | Tramo del precio que sigue a la confirmación de un evento DC hasta el siguiente DC opuesto. |
| Huérfanos (carry-over) | Puntos finales de una ventana que no confirmaron un nuevo evento DC y se arrastran a la ventana siguiente. |
| Medallion | Patrón de capas de datos (raw → conformado → enriquecido) de calidad creciente. |
| Walk-forward / Walk-Through | Backtesting incremental que avanza en el tiempo reentrenando/evaluando sobre ventanas sucesivas. |
| Arquitectura Lambda/Kappa | Lambda: capa batch + capa tiempo real. Kappa: un solo motor de streaming que trata el batch como caso acotado. |
| Spot VM | Instancia de Compute Engine con 60–91 % de descuento, sujeta a preempción. |
| Fan-out por tick | Patrón en que cada tick se evalúa contra los 50 θ en una sola pasada con la serie compartida en memoria. |
| DuckDB | Base de datos analítica embebida (OLAP) con SQL y ejecución out-of-core (spilling a disco); motor por defecto del pipeline. |
| Polars | Librería de DataFrame en Rust con lazy evaluation y streaming engine; alternativa para transformaciones tipo dataframe. |
| Apache Arrow | Formato de memoria columnar que permite interoperabilidad zero-copy entre DuckDB, Polars y Rust. |
| DuckLake | Formato de lakehouse de DuckDB que usa una base SQL (archivo DuckDB/SQLite) como catálogo en vez de árboles de metadatos; da linaje barato. |
| Monorepo políglota | Un único repositorio que aloja capas en distintos lenguajes (Rust, Python) con imágenes Docker delgadas e independientes por capa. |
| Contrato de datos | Esquema Parquet documentado de entrada/salida de cada capa que permite acoplamiento débil entre capas. |
| OpenTelemetry (OTLP) | Estándar de instrumentación que abstrae el backend de observabilidad para evitar lock-in. |
| Stack (vs. environment) | **Stack:** arquitectura/sistema completo distinto (p. ej. el pipeline batch vs. el sistema Kappa). **Environment:** entorno de despliegue de un mismo sistema (dev/qa/uat/prod). En este proyecto se usan *stacks* (`stacks/batch`, `stacks/kappa`); los *environments* se difieren al bot de trading. |
| Cloud Run Jobs | Servicio serverless de GCP para ejecutar cargas a término (no servidores HTTP) con *task arrays* (una tarea por índice), reintentos y free tier propio; cómputo de la Capa 1. |
| Task array | Conjunto de tareas indexadas de un mismo job (en L1, una tarea por mes), ejecutables en paralelo e independientes. |
| Lago de hallazgos de Data Quality | Dataset append-only en Parquet sobre GCS, particionado por fecha de detección, con esquema unificado de hallazgos de calidad de datos; distinto del lago de meta-métricas operativas. |
| `emit_findings()` | Función de la librería compartida (`/shared/dq`) que persiste hallazgos de DQ a rutas únicas; horneada en cada imagen en build-time; sin cola de mensajería. |
| Fail-closed (DQ) | Política (abierta, a resolver en TRD-L1) por la cual un fallo de calidad de datos pone la partición en cuarentena y no la promueve a la capa siguiente. |
| GCP project | Frontera de aislamiento e IAM en GCP (equivalente a una *Account* de AWS). En este proyecto, un único project aloja toda la plataforma; una capa no es un project. |
 
### 12.2 Referencias
 
- Glattfelder, J. B., Dupuis, A., & Olsen, R. B. (2011). *Patterns in high-frequency FX data: discovery of 12 empirical scaling laws.* Quantitative Finance, 11(4), 599–614.
- Tsang, E. P. K., y trabajos relacionados sobre Directional Changes e intrinsic time (Aloud, Tsang, Olsen, Dupuis).
- Google Cloud — Documentación oficial: Cloud Run Jobs, Compute Engine (all-pricing, machine-resource, Spot VMs), Cloud Batch, Cloud Storage pricing, Free Tier, Artifact Registry, Cloud Monitoring/Logging pricing, BigQuery, Secret Manager, Cloud Workflows, BigLake metastore (cloud.google.com).
- Binance — Public data (`data.binance.vision`; github.com/binance/binance-public-data).
- Benchmarks de procesamiento single-node: Polars PDS-H (mayo 2025), DuckDB "Benchmarking over time" (jun 2024), H2O.ai db-benchmark, codecentric (estrés Parquet).
- DuckDB / DuckLake (duckdb.org, ducklake.select), Polars (pola.rs), Apache Arrow (arrow.apache.org), Apache DataFusion (datafusion.apache.org).
- HashiCorp Terraform (módulos, estado remoto en GCS), GitHub Actions y Container Registry (docs.github.com), OpenTelemetry (opentelemetry.io), Grafana Cloud (grafana.com).
---
 
> **Nota de cierre.** Este es el TRD maestro v2.1, que fusiona la v2.0 con la discusión de diseño de la Capa 1 (cómputo en Cloud Run Jobs, contrato de datos de L1, lago de hallazgos de Data Quality, terminología de stacks y estrategia de projects). Las cifras de volumen y precios son estimaciones de referencia a 2026 y deben verificarse contra los datos y tarifas vigentes antes de la implementación de cada fase. Los ítems marcados como **abiertos** (política de fallo de DQ, dimensionamiento del backfill por sonda, set de columnas y costura inter-mensual) se resuelven en el TRD-L1. Cada capa profundizará sus requerimientos en su propio TRD derivado, revalidando el stack y el motor óptimos para su perfil.
 
