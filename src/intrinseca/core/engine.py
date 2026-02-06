"""Motor Silver Layer para Intrinsica.

Orquesta el pipeline completo de transformación Bronze → Silver:
1. Carga de estado anterior (stitching)
2. Ejecución del kernel Numba
3. Construcción de DataFrame Arrow nested
4. Persistencia en Parquet con codificaciones optimizadas

Optimizado para HFT:
- Zero-copy Arrow construction
- Particionado eficiente (sin filtrado O(n) repetido)
- Lazy initialization de recursos
- Métricas de diagnóstico integradas
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from rich import box
from rich.panel import Panel

# Rich (observabilidad)
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .convergence import (
    ConvergenceReport,
    ConvergenceResult,
    compare_dc_events,
)
from .kernel import segment_events_kernel, warmup_kernel
from .reconciliation import (
    ReconciliationType,
    check_reconciliation_needed,
    cleanup_backup,
    reconcile_previous_day,
)
from .state import (
    DCState,
    build_state_path,
    create_empty_state,
    find_previous_state,
    save_state,
)

# =============================================================================
# TYPE ALIASES
# =============================================================================

ArrayF64 = NDArray[np.float64]
ArrayI64 = NDArray[np.int64]
ArrayI8 = NDArray[np.int8]


# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# Parquet
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3

# Columnas Silver
SILVER_COLUMNS = (
    "event_type",
    # Atributos de evento (escalares)
    "reference_price",
    "reference_time",
    "extreme_price",
    "extreme_time",
    "confirm_price",
    "confirm_time",
    # Arrays anidados (microestructura)
    "price_dc",
    "price_os",
    "time_dc",
    "time_os",
    "qty_dc",
    "qty_os",
    "dir_dc",
    "dir_os",
)

# Codificaciones óptimas por tipo de columna
# - BYTE_STREAM_SPLIT: Para floats/ints de alta entropía (precios, timestamps)
# - RLE_DICTIONARY: Para columnas de baja cardinalidad (qty, dir, event_type)
ENCODING_HIGH_ENTROPY = "BYTE_STREAM_SPLIT"
ENCODING_LOW_CARDINALITY = "RLE_DICTIONARY"


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================


class DayResult(NamedTuple):
    """Resultado del procesamiento de un día."""

    df_silver: pl.DataFrame | None
    n_events: int
    n_ticks: int
    convergence: ConvergenceResult | None
    elapsed_ms: float


class MonthResult(NamedTuple):
    """Resultado del procesamiento de un mes completo.

    Similar a DayResult pero sin convergencia (no aplica para mes completo).
    """

    n_events: int
    n_ticks: int
    elapsed_ms: float
    output_path: Path


@dataclass
class EngineStats:
    """Estadísticas acumuladas del Engine."""

    days_processed: int = 0
    total_events: int = 0
    total_ticks: int = 0
    total_time_ms: float = 0.0
    kernel_calls: int = 0

    @property
    def avg_time_per_day_ms(self) -> float:
        """Tiempo promedio por día en ms."""
        return self.total_time_ms / self.days_processed if self.days_processed > 0 else 0.0

    @property
    def throughput_ticks_per_sec(self) -> float:
        """Throughput en ticks/segundo."""
        if self.total_time_ms == 0:
            return 0.0
        return self.total_ticks / (self.total_time_ms / 1000)

    @property
    def events_per_day(self) -> float:
        """Promedio de eventos por día."""
        return self.total_events / self.days_processed if self.days_processed > 0 else 0.0

    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            "days_processed": self.days_processed,
            "total_events": self.total_events,
            "total_ticks": self.total_ticks,
            "total_time_ms": self.total_time_ms,
            "kernel_calls": self.kernel_calls,
            "avg_time_per_day_ms": self.avg_time_per_day_ms,
            "throughput_ticks_per_sec": self.throughput_ticks_per_sec,
            "events_per_day": self.events_per_day,
        }


@dataclass
class EngineConfig:
    """Configuración del Engine (inmutable después de creación)."""

    theta: float
    silver_base_path: Path
    keep_state_files: bool = True
    verbose: bool = False
    collect_stats: bool = True

    # Cache de theta_str (calculado una vez)
    _theta_str: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize configuration after initialization."""
        self.theta = float(self.theta)
        self.silver_base_path = Path(self.silver_base_path)
        # Pre-calcular theta_str (evita recálculo en cada llamada)
        self._theta_str = f"{self.theta:.6f}".rstrip("0").rstrip(".")

    @property
    def theta_str(self) -> str:
        """String formateado de theta (cached)."""
        return self._theta_str


# =============================================================================
# ENGINE PRINCIPAL
# =============================================================================


class Engine:
    """Motor de transformación Bronze → Silver para eventos DC.

    Procesa datos tick-a-tick (Bronze) y los transforma en eventos
    DC anidados (Silver Layer) con estructura:

    ```
    ┌────────────┬────────────┬────────────┬────────────┬─────┐
    │ event_type │ price_dc   │ price_os   │ time_dc    │ ... │
    │ i8         │ list[f64]  │ list[f64]  │ list[i64]  │     │
    ├────────────┼────────────┼────────────┼────────────┼─────┤
    │ 1          │ [90652.5,  │ [91106.0,  │ [17642970, │     │
    │            │  ...]      │  ...]      │  ...]      │     │
    └────────────┴────────────┴────────────┴────────────┴─────┘
    ```

    Attributes:
    ----------
        config: Configuración inmutable del engine
        stats: Estadísticas de procesamiento (si collect_stats=True)

    Example:
    -------
        >>> engine = Engine(theta=0.005, silver_base_path=Path("./silver"))
        >>> result = engine.process_day(df_bronze, "BTCUSDT", date(2025, 11, 28))
        >>> print(f"Eventos: {result.n_events}, Tiempo: {result.elapsed_ms:.2f}ms")

    """

    __slots__ = ("config", "stats", "_compiled", "_console")

    def __init__(
        self,
        theta: float,
        silver_base_path: Path,
        keep_state_files: bool = True,
        verbose: bool = False,
        collect_stats: bool = True,
        warmup: bool = False,
    ) -> None:
        """Inicializa el motor Silver.

        Args:
        ----
            theta: Umbral del algoritmo DC (e.g., 0.005 para 0.5%)
            silver_base_path: Ruta base donde se almacenarán los datos Silver
            keep_state_files: Si True, conserva archivos .arrow históricos.
                              Si False, elimina el .arrow del día anterior.
            verbose: Si True, imprime logs detallados. Si False, barra de progreso.
            collect_stats: Si True, recolecta estadísticas de rendimiento.
            warmup: Si True, pre-compila el kernel inmediatamente.

        Raises:
        ------
            ValueError: Si theta <= 0 o theta >= 1

        """
        if not (0 < theta < 1):
            raise ValueError(f"theta debe estar en (0, 1), recibido: {theta}")

        self.config = EngineConfig(
            theta=theta,
            silver_base_path=silver_base_path,
            keep_state_files=keep_state_files,
            verbose=verbose,
            collect_stats=collect_stats,
        )

        self.stats = EngineStats() if collect_stats else None
        self._compiled = False
        self._console = None  # Lazy initialization

        if warmup:
            self._warmup()

    # -------------------------------------------------------------------------
    # PROPIEDADES (compatibilidad hacia atrás)
    # -------------------------------------------------------------------------

    @property
    def theta(self) -> float:
        """Umbral DC."""
        return self.config.theta

    @property
    def silver_base_path(self) -> Path:
        """Ruta base Silver."""
        return self.config.silver_base_path

    @property
    def keep_state_files(self) -> bool:
        """Conservar archivos de estado."""
        return self.config.keep_state_files

    @property
    def verbose(self) -> bool:
        """Modo verbose."""
        return self.config.verbose

    # -------------------------------------------------------------------------
    # MÉTODOS PRIVADOS
    # -------------------------------------------------------------------------

    def _get_console(self):
        """Lazy initialization de Rich Console."""
        if self._console is None:
            from rich.console import Console

            self._console = Console()
        return self._console

    def _warmup(self) -> None:
        """Pre-compila el kernel Numba si aún no se ha hecho."""
        if not self._compiled:
            warmup_kernel(self.config.theta)
            self._compiled = True

    def _build_data_path(self, ticker: str, year: int, month: int, day: int) -> Path:
        """Construye la ruta al archivo data.parquet (particionado diario).

        Formato: {base}/{ticker}/theta={theta}/year={YYYY}/month={MM}/day={DD}/data.parquet
        """
        return (
            self.config.silver_base_path
            / ticker
            / f"theta={self.config.theta_str}"
            / f"year={year}"
            / f"month={month:02d}"
            / f"day={day:02d}"
            / "data.parquet"
        )

    def _build_month_data_path(self, ticker: str, year: int, month: int) -> Path:
        """Construye la ruta al archivo data.parquet (particionado mensual).

        Formato: {base}/{ticker}/theta={theta}/year={YYYY}/month={MM}/data.parquet
        """
        return (
            self.config.silver_base_path
            / ticker
            / f"theta={self.config.theta_str}"
            / f"year={year}"
            / f"month={month:02d}"
            / "data.parquet"
        )

    def _stitch_data(
        self,
        state: DCState,
        new_prices: ArrayF64,
        new_times: ArrayI64,
        new_quantities: ArrayF64,
        new_directions: ArrayI8,
    ) -> tuple[ArrayF64, ArrayI64, ArrayF64, ArrayI8]:
        """Realiza el stitching de huérfanos anteriores con datos nuevos.

        Los huérfanos del día anterior se anteponen a los datos del día actual
        para garantizar continuidad matemática del algoritmo DC.

        Complejidad: O(n_orphans + n_new)
        """
        if not state.has_orphans:
            return new_prices, new_times, new_quantities, new_directions

        # np.concatenate es O(n) pero inevitable para continuidad
        return (
            np.concatenate([state.orphan_prices, new_prices]),
            np.concatenate([state.orphan_times, new_times]),
            np.concatenate([state.orphan_quantities, new_quantities]),
            np.concatenate([state.orphan_directions, new_directions]),
        )

    def _build_arrow_lists(
        self,
        dc_prices: ArrayF64,
        dc_times: ArrayI64,
        dc_quantities: ArrayF64,
        dc_directions: ArrayI8,
        os_prices: ArrayF64,
        os_times: ArrayI64,
        os_quantities: ArrayF64,
        os_directions: ArrayI8,
        dc_offsets: ArrayI64,
        os_offsets: ArrayI64,
    ) -> dict[str, pa.Array]:
        """Construye columnas Arrow ListArray desde búferes + offsets.

        Implementa isomorfismo de memoria (zero-copy): los offsets del kernel
        Numba se usan directamente para crear ListArrays sin copiar datos.

        Args:
        ----
            dc_prices: Precios de la fase DC.
            dc_times: Timestamps de la fase DC.
            dc_quantities: Cantidades de la fase DC.
            dc_directions: Direcciones de la fase DC.
            os_prices: Precios de la fase OS.
            os_times: Timestamps de la fase OS.
            os_quantities: Cantidades de la fase OS.
            os_directions: Direcciones de la fase OS.
            dc_offsets: Offsets para segmentar búferes DC.
            os_offsets: Offsets para segmentar búferes OS.

        Returns:
        -------
            Dict con 8 columnas ListArray.

        """
        n_events = len(dc_offsets) - 1

        if n_events == 0:
            # Sin eventos: retornar columnas vacías tipadas
            empty_f64 = pa.array([], type=pa.list_(pa.float64()))
            empty_i64 = pa.array([], type=pa.list_(pa.int64()))
            empty_i8 = pa.array([], type=pa.list_(pa.int8()))

            return {
                "price_dc": empty_f64,
                "price_os": empty_f64,
                "time_dc": empty_i64,
                "time_os": empty_i64,
                "qty_dc": empty_f64,
                "qty_os": empty_f64,
                "dir_dc": empty_i8,
                "dir_os": empty_i8,
            }

        # Helper inline para crear ListArray (zero-copy desde numpy)
        def make_list(values: np.ndarray, offsets: np.ndarray, dtype: pa.DataType) -> pa.Array:
            return pa.ListArray.from_arrays(
                pa.array(offsets, type=pa.int64()), pa.array(values, type=dtype)
            )

        return {
            "price_dc": make_list(dc_prices, dc_offsets, pa.float64()),
            "price_os": make_list(os_prices, os_offsets, pa.float64()),
            "time_dc": make_list(dc_times, dc_offsets, pa.int64()),
            "time_os": make_list(os_times, os_offsets, pa.int64()),
            "qty_dc": make_list(dc_quantities, dc_offsets, pa.float64()),
            "qty_os": make_list(os_quantities, os_offsets, pa.float64()),
            "dir_dc": make_list(dc_directions, dc_offsets, pa.int8()),
            "dir_os": make_list(os_directions, os_offsets, pa.int8()),
        }

    def _write_parquet(self, table: pa.Table, path: Path) -> None:
        """Escribe tabla Arrow a Parquet con codificaciones optimizadas.

        Configuración:
        - Compresión: ZSTD nivel 3 (balance velocidad/ratio)
        - use_dictionary: True (para columnas de baja cardinalidad)
        - write_statistics: True (para pruning en lectura)
        - Metadata: theta incluido para trazabilidad
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Agregar theta como metadata del schema para trazabilidad
        # Los indicadores que requieren theta lo leerán de aquí
        existing_metadata = table.schema.metadata or {}
        new_metadata = {
            **existing_metadata,
            b"theta": str(self.config.theta).encode(),
        }
        table = table.replace_schema_metadata(new_metadata)

        pq.write_table(
            table,
            path,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            use_dictionary=True,
            write_statistics=True,
        )

    def _extract_arrays(
        self,
        df: pl.DataFrame,
        price_col: str,
        time_col: str,
        quantity_col: str,
        direction_col: str,
    ) -> tuple[ArrayF64, ArrayI64, ArrayF64, ArrayI8]:
        """Extrae arrays numpy desde DataFrame Polars.

        Maneja conversión de timestamps (Datetime → Int64 ns).
        Intenta zero-copy; fallback a copia si el layout no es contiguo.
        """

        # Helper para extracción con zero-copy preferente
        def safe_to_numpy(series: pl.Series, target_dtype: type) -> np.ndarray:
            try:
                return series.to_numpy(zero_copy_only=True, writable=False)
            except Exception:
                # Fallback: copia si layout no permite zero-copy
                return series.to_numpy()

        prices = safe_to_numpy(df.get_column(price_col).cast(pl.Float64), np.float64)

        # Timestamps: pueden ser Datetime o Int64
        time_dtype = df.schema[time_col]
        if isinstance(time_dtype, pl.Datetime):
            times = safe_to_numpy(df.get_column(time_col).cast(pl.Int64), np.int64)
        else:
            times = safe_to_numpy(df.get_column(time_col), np.int64)

        quantities = safe_to_numpy(df.get_column(quantity_col).cast(pl.Float64), np.float64)
        directions = safe_to_numpy(df.get_column(direction_col).cast(pl.Int8), np.int8)

        return prices, times, quantities, directions

    def _log(self, message: str) -> None:
        """Log condicional según verbose."""
        if self.config.verbose:
            print(message)

    # -------------------------------------------------------------------------
    # MÉTODOS PÚBLICOS
    # -------------------------------------------------------------------------

    def process_day(
        self,
        df_bronze: pl.DataFrame,
        ticker: str,
        process_date: date,
        price_col: str = "price",
        time_col: str = "time",
        quantity_col: str = "quantity",
        direction_col: str = "direction",
        analyze_convergence: bool = False,
        strict_comparison: bool = True,
        tolerance_ns: int = 0,
    ) -> DayResult:
        """Procesa un día de datos Bronze y genera salida Silver.

        Pipeline:
        ```
        Bronze DataFrame
              │
              ▼
        ┌─────────────────┐
        │ Extract Arrays  │ ← Polars → NumPy
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Load Prev State │ ← Arrow file
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Stitch Orphans  │ ← np.concatenate
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Numba Kernel    │ ← JIT compiled
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Build Arrow     │ ← Zero-copy ListArrays
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Write Parquet   │ ← ZSTD compressed
        └─────────────────┘
        ```

        Args:
        ----
            df_bronze: DataFrame Polars con datos Bronze del día
            ticker: Símbolo del instrumento (e.g., "BTCUSDT")
            process_date: Fecha siendo procesada
            price_col: Nombre de la columna de precios
            time_col: Nombre de la columna de timestamps
            quantity_col: Nombre de la columna de cantidades
            direction_col: Nombre de la columna de direcciones
            analyze_convergence: Si True, compara con datos Silver previos
            strict_comparison: Si True, comparación exacta (0 ns tolerancia)
            tolerance_ns: Tolerancia en nanosegundos si strict_comparison=False

        Returns:
        -------
            DayResult con (df_silver, n_events, n_ticks, convergence, elapsed_ms)

        Raises:
        ------
            ValueError: Si df_bronze está vacío o faltan columnas requeridas

        """
        t_start = time.perf_counter()

        self._warmup()

        # 0. Validación
        required_cols = {price_col, time_col, quantity_col, direction_col}
        missing = required_cols - set(df_bronze.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en df_bronze: {missing}")

        # 1. Calcular ruta y cargar datos previos (para convergencia)
        data_path = self._build_data_path(
            ticker, process_date.year, process_date.month, process_date.day
        )
        df_prev: pl.DataFrame | None = None

        if analyze_convergence and data_path.exists():
            try:
                df_prev = pl.read_parquet(data_path, memory_map=True)
                self._log(f"  📊 Datos previos: {len(df_prev)} eventos")
            except Exception as e:
                self._log(f"  ⚠️ Error cargando previos: {e}")

        # 2. Extraer arrays de Bronze
        prices, times, quantities, directions = self._extract_arrays(
            df_bronze, price_col, time_col, quantity_col, direction_col
        )

        # 3. Cargar estado anterior
        prev_state_result = find_previous_state(
            self.config.silver_base_path, ticker, self.config.theta, process_date
        )

        prev_state_path: Path | None = None

        if prev_state_result is not None:
            prev_state_path, prev_state = prev_state_result
            self._log(f"  📂 Estado anterior: {prev_state.n_orphans} huérfanos")
        else:
            prev_state = create_empty_state(process_date)
            self._log("  🆕 Sin estado anterior")

        # 4. Reconciliación retroactiva (si es necesario)
        recon_result = None
        if prev_state.n_orphans > 0 and len(prices) > 0:
            recon_type, recon_context = check_reconciliation_needed(
                prev_state, prices[0], self.config.theta
            )

            if recon_type != ReconciliationType.NONE:
                # Encontrar ruta del Parquet del día anterior
                prev_data_path = self._build_data_path(
                    ticker,
                    prev_state.last_processed_date.year,
                    prev_state.last_processed_date.month,
                    prev_state.last_processed_date.day,
                )

                # Determinar el nuevo precio extremo
                if recon_type == ReconciliationType.CONFIRM_REVERSAL:
                    # El extremo es el último tick antes de la confirmación
                    ext_high, ext_low = prev_state.get_extreme_prices()
                    new_ext_price = ext_high if prev_state.current_trend == 1 else ext_low
                    new_ext_time = (
                        prev_state.orphan_times[-1] if len(prev_state.orphan_times) > 0 else 0
                    )
                else:
                    # EXTEND_OS: usar el precio actual como nuevo extremo
                    new_ext_price = prices[0]
                    new_ext_time = times[0]

                recon_result = reconcile_previous_day(
                    silver_path=prev_data_path,
                    reconciliation_type=recon_type,
                    new_extreme_price=new_ext_price,
                    new_extreme_time=new_ext_time,
                )

                if recon_result.success:
                    self._log(f"  ↩️ Reconciliación {recon_type.value}: {prev_data_path.name}")
                    if recon_result.backup_path:
                        cleanup_backup(recon_result.backup_path)
                else:
                    self._log(f"  ⚠️ Reconciliación fallida: {recon_result.error}")

        # 5. Stitch huérfanos
        stitched = self._stitch_data(prev_state, prices, times, quantities, directions)
        stitched_prices, stitched_times, stitched_quantities, stitched_directions = stitched

        n_ticks = len(stitched_prices)
        self._log(f"  🔗 Ticks combinados: {n_ticks:,}")

        # 5. Ejecutar kernel
        ext_high, ext_low = prev_state.get_extreme_prices()

        kernel_result = segment_events_kernel(
            stitched_prices,
            stitched_times,
            stitched_quantities,
            stitched_directions,
            self.config.theta,
            prev_state.current_trend,
            np.float64(ext_high),
            np.float64(ext_low),
            prev_state.last_os_ref,
        )

        # Desempaquetar resultado (23 elementos)
        (
            dc_prices,
            dc_times,
            dc_quantities,
            dc_directions,
            os_prices,
            os_times,
            os_quantities,
            os_directions,
            event_types,
            dc_offsets,
            os_offsets,
            # Atributos de evento (escalares, zero indirection)
            reference_prices,
            reference_times,
            extreme_prices,
            extreme_times,
            confirm_prices,
            confirm_times,
            # Estado final
            n_events,
            final_trend,
            final_ext_high,
            final_ext_low,
            final_last_os_ref,
            orphan_start_idx,
        ) = kernel_result

        self._log(f"  ✅ Eventos: {n_events}")

        # 6. Extraer huérfanos
        orphan_prices = stitched_prices[orphan_start_idx:]
        orphan_times = stitched_times[orphan_start_idx:]
        orphan_quantities = stitched_quantities[orphan_start_idx:]
        orphan_directions = stitched_directions[orphan_start_idx:]

        self._log(f"  🔚 Huérfanos: {len(orphan_prices)}")

        # 7. Guardar estado
        new_state = DCState(
            orphan_prices=orphan_prices,
            orphan_times=orphan_times,
            orphan_quantities=orphan_quantities,
            orphan_directions=orphan_directions,
            current_trend=np.int8(final_trend),
            last_os_ref=np.float64(final_last_os_ref),
            last_processed_date=process_date,
        )

        state_path = build_state_path(
            self.config.silver_base_path,
            ticker,
            self.config.theta,
            process_date.year,
            process_date.month,
            process_date.day,
        )
        save_state(new_state, state_path)
        self._log(f"  💾 Estado: {state_path.name}")

        # 8. Construir y guardar Silver
        df_silver: pl.DataFrame | None = None

        if n_events > 0:
            # --- Validación de integridad post-kernel ---
            # CRÍTICO: Verificar que todas las estructuras tienen dimensiones consistentes
            # antes de construir el Arrow table. Detecta bugs en el kernel tempranamente.
            expected_offset_len = n_events + 1
            if len(dc_offsets) != expected_offset_len:
                raise ValueError(
                    f"Integridad fallida: dc_offsets tiene {len(dc_offsets)} elementos, "
                    f"esperados {expected_offset_len} (n_events={n_events})"
                )
            if len(os_offsets) != expected_offset_len:
                raise ValueError(
                    f"Integridad fallida: os_offsets tiene {len(os_offsets)} elementos, "
                    f"esperados {expected_offset_len} (n_events={n_events}). "
                    "Posible bug en kernel: os_offsets no se cerró correctamente."
                )
            if len(event_types) != n_events:
                raise ValueError(
                    f"Integridad fallida: event_types tiene {len(event_types)} elementos, "
                    f"esperados {n_events}"
                )
            # Validar atributos de evento
            for name, arr in [
                ("reference_prices", reference_prices),
                ("reference_times", reference_times),
                ("extreme_prices", extreme_prices),
                ("extreme_times", extreme_times),
                ("confirm_prices", confirm_prices),
                ("confirm_times", confirm_times),
            ]:
                if len(arr) != n_events:
                    raise ValueError(
                        f"Integridad fallida: {name} tiene {len(arr)} elementos, "
                        f"esperados {n_events}"
                    )

            list_columns = self._build_arrow_lists(
                dc_prices,
                dc_times,
                dc_quantities,
                dc_directions,
                os_prices,
                os_times,
                os_quantities,
                os_directions,
                dc_offsets,
                os_offsets,
            )

            arrow_table = pa.table(
                {
                    "event_type": pa.array(event_types, type=pa.int8()),
                    # Atributos de evento (escalares, zero indirection)
                    "reference_price": pa.array(reference_prices, type=pa.float64()),
                    "reference_time": pa.array(reference_times, type=pa.int64()),
                    "extreme_price": pa.array(extreme_prices, type=pa.float64()),
                    "extreme_time": pa.array(extreme_times, type=pa.int64()),
                    "confirm_price": pa.array(confirm_prices, type=pa.float64()),
                    "confirm_time": pa.array(confirm_times, type=pa.int64()),
                    # Arrays anidados (microestructura)
                    **list_columns,
                }
            )

            self._write_parquet(arrow_table, data_path)
            self._log(f"  📁 Silver: {data_path}")

            df_silver = pl.from_arrow(arrow_table)
        else:
            self._log("  ⚠️ Sin eventos confirmados")

        # 9. Limpiar estado anterior
        if not self.config.keep_state_files and prev_state_path and prev_state_path.exists():
            prev_state_path.unlink()
            self._log(f"  🧹 Eliminado: {prev_state_path.name}")

        # 10. Análisis de convergencia
        convergence_result: ConvergenceResult | None = None

        if analyze_convergence:
            if df_prev is not None and df_silver is not None:
                convergence_result = compare_dc_events(
                    df_prev=df_prev,
                    df_new=df_silver,
                    ticker=ticker,
                    theta=self.config.theta,
                    day=process_date,
                    strict_comparison=strict_comparison,
                    tolerance_ns=tolerance_ns,
                )
                status = "✅" if convergence_result.converged else "⚠️"
                self._log(f"  📊 Convergencia: {status}")
            else:
                convergence_result = ConvergenceResult(
                    ticker=ticker,
                    theta=self.config.theta,
                    day=process_date,
                    n_events_prev=0,
                    n_events_new=n_events,
                    n_discrepant_events=0,
                    first_discrepancy_idx=-1,
                    convergence_idx=None,
                    converged=False,
                    requires_forward_processing=False,
                    analysis_applicable=False,
                )
                self._log("  🆕 Convergencia N/A (sin previos)")

        # 11. Calcular tiempo y actualizar stats
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if self.stats is not None:
            self.stats.days_processed += 1
            self.stats.total_events += n_events
            self.stats.total_ticks += n_ticks
            self.stats.total_time_ms += elapsed_ms
            self.stats.kernel_calls += 1

        return DayResult(
            df_silver=df_silver,
            n_events=n_events,
            n_ticks=n_ticks,
            convergence=convergence_result,
            elapsed_ms=elapsed_ms,
        )

    def process_date_range(
        self,
        df_bronze: pl.DataFrame,
        ticker: str,
        time_col: str = "time",
        analyze_convergence: bool = False,
        strict_comparison: bool = True,
        tolerance_ns: int = 0,
        stop_on_convergence: bool = False,
    ) -> tuple[dict[date, pl.DataFrame | None], ConvergenceReport | None]:
        """Procesa un rango de datos Bronze particionándolos por día.

        Usa `partition_by` de Polars para evitar filtrado O(n) repetido.
        Itera secuencialmente garantizando continuidad de estado.

        Args:
        ----
            df_bronze: DataFrame con datos de múltiples días
            ticker: Símbolo del instrumento
            time_col: Columna de timestamps
            analyze_convergence: Si True, compara con datos Silver previos
            strict_comparison: Si True, comparación exacta
            tolerance_ns: Tolerancia si strict_comparison=False
            stop_on_convergence: Si True, detiene el procesamiento cuando se
                detecta convergencia. ADVERTENCIA: Usar con precaución, puede
                causar pérdida de datos si hay días posteriores sin procesar.
                Por defecto es False para garantizar procesamiento completo.

        Returns:
        -------
            Tuple (results_dict, convergence_report)
            - results_dict: {date: DataFrame Silver}
            - convergence_report: Reporte si analyze_convergence=True

        """
        results: dict[date, pl.DataFrame | None] = {}
        convergence_report: ConvergenceReport | None = None

        if analyze_convergence:
            convergence_report = ConvergenceReport(ticker=ticker, theta=self.config.theta)

        # Preparar datos: convertir timestamps y extraer fecha
        if df_bronze.schema[time_col] == pl.Int64:
            df_bronze = df_bronze.with_columns(
                pl.from_epoch(pl.col(time_col), time_unit="ns").alias("_datetime")
            )
            time_col_for_date = "_datetime"
        else:
            time_col_for_date = time_col

        df_bronze = df_bronze.with_columns(pl.col(time_col_for_date).dt.date().alias("_date"))

        # Obtener fechas ordenadas
        unique_dates = df_bronze.get_column("_date").unique().sort().to_list()
        n_days = len(unique_dates)

        # Contadores
        total_events = 0
        total_ticks = 0

        if self.config.verbose:
            # Modo verbose: logs tradicionales
            # Usar partition_by para evitar O(n) por cada día
            partitions = df_bronze.partition_by("_date", as_dict=True, maintain_order=True)

            print(f"📅 Procesando {n_days} días...")
            if analyze_convergence:
                print("   📊 Convergencia activada")

            for d in unique_dates:
                print(f"\n🗓️ {d}")
                # partition_by retorna tuplas como claves
                df_day = partitions.get((d,))
                if df_day is None:
                    continue
                df_day = df_day.drop("_date")
                if "_datetime" in df_day.columns:
                    df_day = df_day.drop("_datetime")

                result = self.process_day(
                    df_day,
                    ticker,
                    d,
                    time_col=time_col,
                    analyze_convergence=analyze_convergence,
                    strict_comparison=strict_comparison,
                    tolerance_ns=tolerance_ns,
                )

                results[d] = result.df_silver
                total_events += result.n_events
                total_ticks += result.n_ticks

                if result.convergence and convergence_report:
                    convergence_report.add_result(result.convergence)
                    if (
                        stop_on_convergence
                        and result.convergence.analysis_applicable
                        and result.convergence.converged
                    ):
                        print(f"  🛑 Convergencia en {d} - deteniendo procesamiento")
                        break
        else:
            # Modo silencioso: barra de progreso
            # Usar partition_by para evitar O(n) por cada día
            partitions = df_bronze.partition_by("_date", as_dict=True, maintain_order=True)

            console = self._get_console()

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Bronze → Silver"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total} días"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("", total=n_days)

                for d in unique_dates:
                    # partition_by retorna tuplas como claves
                    df_day = partitions.get((d,))
                    if df_day is None:
                        progress.advance(task)
                        continue
                    df_day = df_day.drop("_date")
                    if "_datetime" in df_day.columns:
                        df_day = df_day.drop("_datetime")

                    result = self.process_day(
                        df_day,
                        ticker,
                        d,
                        time_col=time_col,
                        analyze_convergence=analyze_convergence,
                        strict_comparison=strict_comparison,
                        tolerance_ns=tolerance_ns,
                    )

                    results[d] = result.df_silver
                    total_events += result.n_events
                    total_ticks += result.n_ticks
                    progress.advance(task)

                    if result.convergence and convergence_report:
                        convergence_report.add_result(result.convergence)
                        if (
                            stop_on_convergence
                            and result.convergence.analysis_applicable
                            and result.convergence.converged
                        ):
                            break

            # Resumen compacto
            days_with_events = sum(1 for r in results.values() if r is not None)

            if (
                analyze_convergence
                and convergence_report
                and convergence_report.n_days_analyzed > 0
            ):
                conv_status = "✅" if convergence_report.converged else "⚠️"
                summary = (
                    f"[green]✓ {len(results)}/{n_days} días[/green] • "
                    f"[cyan]{days_with_events} con eventos[/cyan] • "
                    f"[dim]{total_events:,} eventos | {total_ticks:,} ticks[/dim]\n"
                    f"[yellow]📊 {conv_status} Discrepantes: "
                    f"{convergence_report.total_discrepant_events}[/yellow]"
                )
            else:
                summary = (
                    f"[green]✓ {n_days} días[/green] • "
                    f"[cyan]{days_with_events} con eventos[/cyan] • "
                    f"[dim]{total_events:,} eventos | {total_ticks:,} ticks[/dim]"
                )

            console.print(
                Panel(
                    summary,
                    title=f"Engine θ={self.config.theta_str}",
                    border_style="blue",
                    box=box.ROUNDED,
                )
            )

        # ADVERTENCIA: Verificar que todos los días fueron procesados
        if len(results) < n_days:
            missing_count = n_days - len(results)
            import warnings

            warnings.warn(
                f"⚠️ ADVERTENCIA: Solo se procesaron {len(results)}/{n_days} días. "
                f"{missing_count} días no fueron procesados. "
                "Esto puede indicar terminación prematura por convergencia o un error.",
                UserWarning,
                stacklevel=2,
            )
            if self.config.verbose:
                print(
                    f"\n⚠️ ADVERTENCIA: {missing_count} días no procesados. "
                    "Verifique los parámetros o logs para más detalles."
                )

        # Guardar reporte de convergencia
        if convergence_report is not None:
            report_path = (
                self.config.silver_base_path
                / ticker
                / f"theta={self.config.theta_str}"
                / "convergence_report.json"
            )
            convergence_report.save(report_path)

            if self.config.verbose:
                print(f"\n💾 Reporte: {report_path}")
                print(convergence_report.generate_summary())

        return results, convergence_report

    # -------------------------------------------------------------------------
    # PROCESAMIENTO MENSUAL
    # -------------------------------------------------------------------------

    def process_month(
        self,
        df_bronze: pl.DataFrame,
        ticker: str,
        year: int,
        month: int,
        price_col: str = "price",
        time_col: str = "time",
        quantity_col: str = "quantity",
        direction_col: str = "direction",
    ) -> MonthResult:
        """Procesa un mes completo sin particionado diario.

        A diferencia de process_date_range(), este método ejecuta el kernel
        Numba una sola vez para todo el mes, eliminando el overhead de
        huérfanos entre días.

        Ventajas:
        - Sin overhead de persistencia/carga de estado entre días
        - Una sola invocación del kernel Numba
        - Archivo Parquet único por mes

        Tradeoffs:
        - Requiere más RAM (todo el mes en memoria)
        - No genera análisis de convergencia intra-mes

        Args:
        ----
            df_bronze: DataFrame Polars con datos Bronze del mes completo
            ticker: Símbolo del instrumento (e.g., "BTCUSDT")
            year: Año del mes a procesar
            month: Mes a procesar (1-12)
            price_col: Nombre de la columna de precios
            time_col: Nombre de la columna de timestamps
            quantity_col: Nombre de la columna de cantidades
            direction_col: Nombre de la columna de direcciones

        Returns:
        -------
            MonthResult con estadísticas del procesamiento

        Example:
        -------
            >>> engine = Engine(theta=0.005, silver_base_path=Path("./silver"))
            >>> result = engine.process_month(df_january, "BTCUSDT", 2025, 1)
            >>> print(f"Procesados {result.n_events} eventos en {result.elapsed_ms:.0f}ms")

        """
        import time

        start_time = time.perf_counter()

        # 1. Warmup si necesario
        self._warmup()

        # 2. Extraer arrays
        prices, times, quantities, directions = self._extract_arrays(
            df_bronze, price_col, time_col, quantity_col, direction_col
        )
        n_ticks = len(prices)

        if n_ticks == 0:
            output_path = self._build_month_data_path(ticker, year, month)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return MonthResult(
                n_events=0,
                n_ticks=0,
                elapsed_ms=elapsed_ms,
                output_path=output_path,
            )

        self._log(f"📦 Procesando {ticker} {year}-{month:02d}: {n_ticks:,} ticks")

        # 3. Buscar estado del mes anterior (si existe)
        from calendar import monthrange

        # Último día del mes anterior
        if month == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, month - 1

        _, prev_last_day = monthrange(prev_year, prev_month)
        prev_date = date(prev_year, prev_month, prev_last_day)

        prev_result = find_previous_state(
            self.config.silver_base_path,
            ticker,
            self.config.theta,
            prev_date,
            max_lookback=31,  # Buscar hasta un mes atrás
        )

        if prev_result:
            _, prev_state = prev_result
            self._log(f"  📂 Estado anterior: {prev_state.n_orphans} huérfanos")
        else:
            prev_state = create_empty_state(prev_date)
            self._log("  📂 Sin estado anterior (inicio fresco)")

        # 4. Stitch con huérfanos del mes anterior
        stitched_prices, stitched_times, stitched_quantities, stitched_directions = (
            self._stitch_data(prev_state, prices, times, quantities, directions)
        )

        # 5. Ejecutar kernel
        ext_high, ext_low = prev_state.get_extreme_prices()

        kernel_result = segment_events_kernel(
            stitched_prices,
            stitched_times,
            stitched_quantities,
            stitched_directions,
            self.config.theta,
            prev_state.current_trend,
            np.float64(ext_high),
            np.float64(ext_low),
            prev_state.last_os_ref,
        )

        (
            event_types,
            dc_indices,
            os_indices,
            os_start_indices,
            dcc_indices,
            extreme_indices,
            confirm_indices,
            reference_indices,
            final_trend,
            final_os_ref,
            orphan_start_idx,
        ) = kernel_result

        n_events = len(event_types)
        self._log(f"  ⚡ Kernel: {n_events} eventos detectados")

        if n_events == 0:
            # Sin eventos, solo guardar estado para el siguiente mes
            orphan_prices = stitched_prices[orphan_start_idx:]
            orphan_times = stitched_times[orphan_start_idx:]
            orphan_quantities = stitched_quantities[orphan_start_idx:]
            orphan_directions = stitched_directions[orphan_start_idx:]

            new_state = DCState(
                orphan_prices=orphan_prices,
                orphan_times=orphan_times,
                orphan_quantities=orphan_quantities,
                orphan_directions=orphan_directions,
                current_trend=final_trend,
                last_os_ref=final_os_ref,
                last_processed_date=date(year, month, monthrange(year, month)[1]),
            )

            # Guardar estado en el directorio del mes
            state_path = (
                self.config.silver_base_path
                / ticker
                / f"theta={self.config.theta_str}"
                / f"year={year}"
                / f"month={month:02d}"
                / f"state_{ticker}_{self.config.theta_str}.arrow"
            )
            save_state(new_state, state_path)

            output_path = self._build_month_data_path(ticker, year, month)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return MonthResult(
                n_events=0,
                n_ticks=n_ticks,
                elapsed_ms=elapsed_ms,
                output_path=output_path,
            )

        # 6. Construir listas Arrow
        arrow_lists = self._build_arrow_lists(
            stitched_prices,
            stitched_times,
            stitched_quantities,
            stitched_directions,
            dc_indices,
            os_indices,
            os_start_indices,
            dcc_indices,
            extreme_indices,
            confirm_indices,
            reference_indices,
        )

        # 7. Construir tabla Arrow con estructura anidada
        arrow_table = self._build_arrow_table(
            n_events,
            event_types,
            stitched_prices,
            stitched_times,
            stitched_quantities,
            extreme_indices,
            confirm_indices,
            reference_indices,
            dcc_indices,
            arrow_lists,
        )

        # 8. Escribir Parquet
        output_path = self._build_month_data_path(ticker, year, month)
        self._write_parquet(arrow_table, output_path)
        self._log(f"  💾 Guardado: {output_path}")

        # 9. Guardar estado para el siguiente mes
        orphan_prices = stitched_prices[orphan_start_idx:]
        orphan_times = stitched_times[orphan_start_idx:]
        orphan_quantities = stitched_quantities[orphan_start_idx:]
        orphan_directions = stitched_directions[orphan_start_idx:]

        self._log(f"  🔚 Huérfanos: {len(orphan_prices)}")

        new_state = DCState(
            orphan_prices=orphan_prices,
            orphan_times=orphan_times,
            orphan_quantities=orphan_quantities,
            orphan_directions=orphan_directions,
            current_trend=final_trend,
            last_os_ref=final_os_ref,
            last_processed_date=date(year, month, monthrange(year, month)[1]),
        )

        state_path = (
            self.config.silver_base_path
            / ticker
            / f"theta={self.config.theta_str}"
            / f"year={year}"
            / f"month={month:02d}"
            / f"state_{ticker}_{self.config.theta_str}.arrow"
        )
        save_state(new_state, state_path)

        # 10. Actualizar estadísticas
        if self.stats:
            self.stats.days_processed += 1  # Contamos como 1 "unidad" procesada
            self.stats.total_events += n_events
            self.stats.total_ticks += n_ticks
            self.stats.kernel_calls += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_time_ms += elapsed_ms if self.stats else 0

        self._log(f"  ✅ Completado en {elapsed_ms:.0f}ms")

        return MonthResult(
            n_events=n_events,
            n_ticks=n_ticks,
            elapsed_ms=elapsed_ms,
            output_path=output_path,
        )

    # -------------------------------------------------------------------------
    # DIAGNÓSTICO
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict | None:
        """Retorna estadísticas de rendimiento del engine.

        Returns:
        -------
            Dict con métricas o None si collect_stats=False

        """
        return self.stats.to_dict() if self.stats else None

    def reset_stats(self) -> None:
        """Reinicia las estadísticas acumuladas."""
        if self.stats:
            self.stats = EngineStats()

    def diagnose(self) -> dict:
        """Ejecuta diagnóstico del engine.

        Returns:
        -------
            Dict con información de configuración y estado

        """
        from .kernel import verify_nopython_mode

        kernel_info = verify_nopython_mode() if self._compiled else {"compiled": False}

        return {
            "config": {
                "theta": self.config.theta,
                "theta_str": self.config.theta_str,
                "silver_base_path": str(self.config.silver_base_path),
                "keep_state_files": self.config.keep_state_files,
                "verbose": self.config.verbose,
                "collect_stats": self.config.collect_stats,
            },
            "state": {
                "compiled": self._compiled,
                "console_initialized": self._console is not None,
            },
            "kernel": kernel_info,
            "stats": self.get_stats(),
        }

    def __repr__(self) -> str:
        """Return string representation of Engine."""
        return (
            f"Engine(theta={self.config.theta}, "
            f"silver_base_path={self.config.silver_base_path}, "
            f"compiled={self._compiled})"
        )
