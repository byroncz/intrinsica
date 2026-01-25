"""
Motor Silver Layer para Intrinsica.

Orquesta el pipeline completo de transformación Bronze → Silver:
1. Carga de estado anterior (stitching)
2. Ejecución del kernel Numba
3. Construcción de DataFrame Arrow nested
4. Persistencia en Parquet con codificaciones optimizadas

El motor implementa el principio de "divide y vencerás": solo segmenta y organiza.
No realiza cálculos de indicadores ni estimaciones estadísticas.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box

from .kernel import segment_events_kernel, warmup_kernel
from .state import (
    DCState,
    build_state_path,
    create_empty_state,
    find_previous_state,
    save_state,
)
from .convergence import (
    ConvergenceResult,
    ConvergenceReport,
    compare_dc_events,
)


class Engine:
    """
    Motor de transformación Bronze → Silver para eventos DC.
    
    Procesa datos tick-a-tick (Bronze) y los transforma en eventos
    DC anidados (Silver) con 9 columnas:
    - event_type: Tipo de evento (1=upturn, -1=downturn)
    - price_dc, price_os: Precios en fases DC y OS
    - time_dc, time_os: Timestamps en fases DC y OS
    - qty_dc, qty_os: Cantidades en fases DC y OS
    - dir_dc, dir_os: Direcciones en fases DC y OS
    
    Attributes:
        theta: Umbral del algoritmo DC
        silver_base_path: Ruta base para almacenamiento Silver
        _compiled: Indica si el kernel Numba ya fue pre-compilado
    """
    
    # Configuración de codificaciones Parquet por columna
    PARQUET_ENCODING_CONFIG = {
        # Columnas de alta entropía: BYTE_STREAM_SPLIT para floats/ints
        "price_dc": "BYTE_STREAM_SPLIT",
        "price_os": "BYTE_STREAM_SPLIT",
        "time_dc": "BYTE_STREAM_SPLIT",
        "time_os": "BYTE_STREAM_SPLIT",
        # Columnas de baja cardinalidad: RLE/Dictionary
        "qty_dc": "RLE_DICTIONARY",
        "qty_os": "RLE_DICTIONARY",
        "dir_dc": "RLE_DICTIONARY",
        "dir_os": "RLE_DICTIONARY",
        "event_type": "RLE_DICTIONARY",
    }
    
    def __init__(
        self,
        theta: float,
        silver_base_path: Path,
        keep_state_files: bool = True,
        verbose: bool = False,
    ):
        """
        Inicializa el motor Silver.
        
        Args:
            theta: Umbral del algoritmo DC (e.g., 0.005 para 0.5%)
            silver_base_path: Ruta base donde se almacenarán los datos Silver
            keep_state_files: Si True, conserva archivos .arrow históricos (dev/debug).
                              Si False, elimina el .arrow del día anterior después de
                              procesar exitosamente (producción).
            verbose: Si True, imprime logs detallados por cada día.
                     Si False (default), usa barra de progreso compacta.
        """
        self.theta = float(theta)
        self.silver_base_path = Path(silver_base_path)
        self.keep_state_files = keep_state_files
        self.verbose = verbose
        self._compiled = False
        self._console = Console()
    
    def _warmup(self) -> None:
        """Pre-compila el kernel Numba si aún no se ha hecho."""
        if not self._compiled:
            warmup_kernel(self.theta)
            self._compiled = True
    
    def _build_data_path(
        self,
        ticker: str,
        year: int,
        month: int,
        day: int
    ) -> Path:
        """Construye la ruta al archivo data.parquet."""
        theta_str = f"{self.theta:.6f}".rstrip('0').rstrip('.')
        
        return (
            self.silver_base_path
            / ticker
            / f"theta={theta_str}"
            / f"year={year}"
            / f"month={month:02d}"
            / f"day={day:02d}"
            / "data.parquet"
        )
    
    def _stitch_data(
        self,
        state: DCState,
        new_prices: NDArray[np.float64],
        new_times: NDArray[np.int64],
        new_quantities: NDArray[np.float64],
        new_directions: NDArray[np.int8],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.int8],
    ]:
        """
        Realiza el stitching de huérfanos anteriores con datos nuevos.
        
        Los huérfanos del día anterior se anteponen a los datos del día actual
        para garantizar continuidad matemática del algoritmo DC.
        """
        if not state.has_orphans:
            return new_prices, new_times, new_quantities, new_directions
        
        return (
            np.concatenate([state.orphan_prices, new_prices]),
            np.concatenate([state.orphan_times, new_times]),
            np.concatenate([state.orphan_quantities, new_quantities]),
            np.concatenate([state.orphan_directions, new_directions]),
        )
    
    def _build_arrow_lists(
        self,
        dc_prices: NDArray[np.float64],
        dc_times: NDArray[np.int64],
        dc_quantities: NDArray[np.float64],
        dc_directions: NDArray[np.int8],
        os_prices: NDArray[np.float64],
        os_times: NDArray[np.int64],
        os_quantities: NDArray[np.float64],
        os_directions: NDArray[np.int8],
        dc_offsets: NDArray[np.int64],
        os_offsets: NDArray[np.int64],
    ) -> dict[str, pa.Array]:
        """
        Construye columnas Arrow ListArray desde búferes separados + offsets.
        
        Este método implementa el isomorfismo de memoria (zero-copy):
        los offsets del kernel Numba se usan directamente para crear
        ListArrays de Arrow sin copiar datos.
        """
        n_events = len(dc_offsets) - 1
        
        if n_events == 0:
            # Sin eventos: retornar columnas vacías
            empty_list_f64 = pa.array([], type=pa.list_(pa.float64()))
            empty_list_i64 = pa.array([], type=pa.list_(pa.int64()))
            empty_list_i8 = pa.array([], type=pa.list_(pa.int8()))
            
            return {
                "price_dc": empty_list_f64,
                "price_os": empty_list_f64,
                "time_dc": empty_list_i64,
                "time_os": empty_list_i64,
                "qty_dc": empty_list_f64,
                "qty_os": empty_list_f64,
                "dir_dc": empty_list_i8,
                "dir_os": empty_list_i8,
            }
        
        def make_list_array(values: NDArray, offsets: NDArray, arrow_type: pa.DataType) -> pa.Array:
            """Helper para crear ListArray desde valores + offsets."""
            values_array = pa.array(values, type=arrow_type)
            offsets_array = pa.array(offsets, type=pa.int64())
            return pa.ListArray.from_arrays(offsets_array, values_array)
        
        return {
            "price_dc": make_list_array(dc_prices, dc_offsets, pa.float64()),
            "price_os": make_list_array(os_prices, os_offsets, pa.float64()),
            "time_dc": make_list_array(dc_times, dc_offsets, pa.int64()),
            "time_os": make_list_array(os_times, os_offsets, pa.int64()),
            "qty_dc": make_list_array(dc_quantities, dc_offsets, pa.float64()),
            "qty_os": make_list_array(os_quantities, os_offsets, pa.float64()),
            "dir_dc": make_list_array(dc_directions, dc_offsets, pa.int8()),
            "dir_os": make_list_array(os_directions, os_offsets, pa.int8()),
        }
    
    def _write_parquet(
        self,
        table: pa.Table,
        path: Path,
    ) -> None:
        """
        Escribe tabla Arrow a Parquet con codificaciones optimizadas.
        
        Configuración:
        - Compresión: ZSTD nivel 3
        - BYTE_STREAM_SPLIT para price/time (alta entropía)
        - RLE_DICTIONARY para qty/dir (baja cardinalidad)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configurar escritor con opciones avanzadas
        pq.write_table(
            table,
            path,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
    
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
    ) -> tuple[Optional[pl.DataFrame], int, int, Optional[ConvergenceResult]]:
        """
        Procesa un día de datos Bronze y genera salida Silver.
        
        Pipeline completo:
        1. (Opcional) Cargar datos previos para análisis de convergencia
        2. Cargar estado del día anterior (si existe)
        3. Stitch huérfanos + ticks nuevos
        4. Ejecutar kernel Numba
        5. Construir DataFrame Arrow nested
        6. Persistir data.parquet + state.arrow
        7. (Opcional) Comparar con datos previos y reportar convergencia
        
        Args:
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
            Tuple (df_silver, n_events, n_ticks, convergence_result)
        """
        self._warmup()
        
        # 0. (Convergencia) Calcular ruta y cargar datos previos si existen
        data_path = self._build_data_path(
            ticker, process_date.year, process_date.month, process_date.day
        )
        df_prev: Optional[pl.DataFrame] = None
        
        if analyze_convergence and data_path.exists():
            try:
                df_prev = pl.read_parquet(data_path, memory_map=True)
                if self.verbose:
                    print(f"  📊 Datos previos cargados: {len(df_prev)} eventos para análisis de convergencia")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ Error cargando datos previos: {e}")
                df_prev = None
        
        # 1. Extraer arrays de Bronze
        prices = df_bronze.get_column(price_col).cast(pl.Float64).to_numpy()
        
        # Manejar timestamps (pueden ser Datetime o Int64)
        time_dtype = df_bronze.schema[time_col]
        if isinstance(time_dtype, (pl.Datetime,)):
            times = df_bronze.get_column(time_col).cast(pl.Int64).to_numpy()
        else:
            times = df_bronze.get_column(time_col).to_numpy()
        
        quantities = df_bronze.get_column(quantity_col).cast(pl.Float64).to_numpy()
        directions = df_bronze.get_column(direction_col).cast(pl.Int8).to_numpy()
        
        # 2. Buscar y cargar estado anterior
        prev_state_result = find_previous_state(
            self.silver_base_path, ticker, self.theta, process_date
        )
        
        prev_state_path: Optional[Path] = None  # Para limpieza posterior
        
        if prev_state_result is not None:
            prev_state_path, prev_state = prev_state_result
            if self.verbose:
                print(f"  📂 Estado anterior encontrado: {prev_state.n_orphans} huérfanos")
        else:
            prev_state = create_empty_state(process_date)
            if self.verbose:
                print("  🆕 Sin estado anterior, iniciando desde cero")
        
        # 3. Stitch huérfanos con datos nuevos
        stitched_prices, stitched_times, stitched_quantities, stitched_directions = \
            self._stitch_data(
                prev_state,
                prices, times, quantities, directions
            )
        
        if self.verbose:
            print(f"  🔗 Datos combinados: {len(stitched_prices):,} ticks")
        
        # 4. Obtener estado inicial para el kernel
        ext_high, ext_low = prev_state.get_extreme_prices()
        
        # 5. Ejecutar kernel (nueva firma con búferes separados)
        (
            dc_prices, dc_times, dc_quantities, dc_directions,
            os_prices, os_times, os_quantities, os_directions,
            event_types,
            dc_offsets, os_offsets,
            n_events, final_trend, final_ext_high, final_ext_low, 
            final_last_os_ref, orphan_start_idx
        ) = segment_events_kernel(
            stitched_prices,
            stitched_times,
            stitched_quantities,
            stitched_directions,
            self.theta,
            prev_state.current_trend,
            np.float64(ext_high),
            np.float64(ext_low),
            prev_state.last_os_ref,
        )
        
        if self.verbose:
            print(f"  ✅ Eventos detectados: {n_events}")
        
        # 6. Identificar huérfanos del día actual
        orphan_prices = stitched_prices[orphan_start_idx:]
        orphan_times = stitched_times[orphan_start_idx:]
        orphan_quantities = stitched_quantities[orphan_start_idx:]
        orphan_directions = stitched_directions[orphan_start_idx:]
        
        if self.verbose:
            print(f"  🔚 Ticks huérfanos: {len(orphan_prices)}")
        
        # 7. Crear y guardar nuevo estado
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
            self.silver_base_path, ticker, self.theta,
            process_date.year, process_date.month, process_date.day
        )
        save_state(new_state, state_path)
        if self.verbose:
            print(f"  💾 Estado guardado: {state_path.name}")
        
        # 8. Construir y guardar datos Silver (si hay eventos)
        if n_events == 0:
            if self.verbose:
                print("  ⚠️ Sin eventos confirmados para este día")
            return None, 0, len(stitched_prices), None
        
        # Construir columnas Arrow anidadas (ahora con búferes separados)
        list_columns = self._build_arrow_lists(
            dc_prices, dc_times, dc_quantities, dc_directions,
            os_prices, os_times, os_quantities, os_directions,
            dc_offsets, os_offsets
        )
        
        # Crear tabla Arrow
        arrow_table = pa.table({
            "event_type": pa.array(event_types, type=pa.int8()),
            **list_columns,
        })
        
        # Guardar Parquet (data_path ya fue calculado al inicio)
        self._write_parquet(arrow_table, data_path)
        if self.verbose:
            print(f"  📁 Datos Silver: {data_path}")
        
        # 10. Limpiar archivo de estado anterior (ya es redundante)
        # Los ticks huérfanos del día anterior ahora están embebidos en el Parquet de hoy
        if not self.keep_state_files and prev_state_path is not None and prev_state_path.exists():
            prev_state_path.unlink()
            if self.verbose:
                print(f"  🧹 Estado anterior eliminado: {prev_state_path.name}")
        
        # 11. Convertir a Polars para retorno
        df_silver = pl.from_arrow(arrow_table)
        
        # 12. (Convergencia) Comparar con datos previos si se solicitó
        convergence_result: Optional[ConvergenceResult] = None
        
        if analyze_convergence and df_prev is not None:
            convergence_result = compare_dc_events(
                df_prev=df_prev,
                df_new=df_silver,
                ticker=ticker,
                theta=self.theta,
                day=process_date,
                strict_comparison=strict_comparison,
                tolerance_ns=tolerance_ns,
            )
            
            if self.verbose:
                status = "✅ Convergió" if convergence_result.converged else "⚠️ No convergió"
                print(f"  📊 Convergencia: {status} | Discrepantes: {convergence_result.n_discrepant_events}")
        
        return df_silver, n_events, len(stitched_prices), convergence_result
    
    def process_date_range(
        self,
        df_bronze: pl.DataFrame,
        ticker: str,
        time_col: str = "time",
        analyze_convergence: bool = False,
        strict_comparison: bool = True,
        tolerance_ns: int = 0,
    ) -> tuple[dict[date, Optional[pl.DataFrame]], Optional[ConvergenceReport]]:
        """
        Procesa un rango de datos Bronze particionándolos por día.
        
        Itera sobre cada día único en los datos y procesa secuencialmente,
        garantizando continuidad de estado entre días.
        
        Args:
            df_bronze: DataFrame con datos de múltiples días
            ticker: Símbolo del instrumento
            time_col: Columna de timestamps
            analyze_convergence: Si True, compara con datos Silver previos
            strict_comparison: Si True, comparación exacta (0 ns tolerancia)
            tolerance_ns: Tolerancia en nanosegundos si strict_comparison=False
            
        Returns:
            Tuple (results_dict, convergence_report)
            - results_dict: {date: DataFrame Silver} para cada día procesado
            - convergence_report: Reporte de convergencia si analyze_convergence=True
        """
        results = {}
        convergence_report: Optional[ConvergenceReport] = None
        
        if analyze_convergence:
            convergence_report = ConvergenceReport(ticker=ticker, theta=self.theta)
        
        # Asegurar que tenemos datetime
        if df_bronze.schema[time_col] == pl.Int64:
            df_bronze = df_bronze.with_columns(
                pl.col(time_col).cast(pl.Datetime("ns")).alias(time_col)
            )
        
        # Extraer fechas únicas
        df_bronze = df_bronze.with_columns(
            pl.col(time_col).dt.date().alias("_date")
        )
        
        unique_dates = df_bronze.get_column("_date").unique().sort().to_list()
        
        # Contadores para resumen
        total_events = 0
        total_ticks = 0
        early_exit = False
        
        if self.verbose:
            # Modo verbose: prints tradicionales
            print(f"📅 Procesando {len(unique_dates)} días...")
            if analyze_convergence:
                print("   📊 Análisis de convergencia activado")
            
            for d in unique_dates:
                print(f"\n🗓️ Día: {d}")
                df_day = df_bronze.filter(pl.col("_date") == d).drop("_date")
                
                result, n_events, n_ticks, conv_result = self.process_day(
                    df_day, ticker, d, time_col=time_col,
                    analyze_convergence=analyze_convergence,
                    strict_comparison=strict_comparison,
                    tolerance_ns=tolerance_ns,
                )
                
                results[d] = result
                total_events += n_events
                total_ticks += n_ticks
                
                # Agregar resultado de convergencia si existe
                if conv_result is not None and convergence_report is not None:
                    convergence_report.add_result(conv_result)
                    
                    # Early exit: si convergió, detener después de completar esta partición
                    if conv_result.converged:
                        print(f"  🛑 Convergencia alcanzada en {d}, deteniendo procesamiento")
                        early_exit = True
                        break
        else:
            # Modo silencioso: barra de progreso compacta
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Bronze → Silver"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total} días"),
                TimeElapsedColumn(),
                console=self._console,
                transient=True,
            ) as progress:
                task = progress.add_task("Procesando...", total=len(unique_dates))
                
                for d in unique_dates:
                    df_day = df_bronze.filter(pl.col("_date") == d).drop("_date")
                    
                    result, n_events, n_ticks, conv_result = self.process_day(
                        df_day, ticker, d, time_col=time_col,
                        analyze_convergence=analyze_convergence,
                        strict_comparison=strict_comparison,
                        tolerance_ns=tolerance_ns,
                    )
                    
                    results[d] = result
                    total_events += n_events
                    total_ticks += n_ticks
                    progress.advance(task)
                    
                    # Agregar resultado de convergencia si existe
                    if conv_result is not None and convergence_report is not None:
                        convergence_report.add_result(conv_result)
                        
                        # Early exit
                        if conv_result.converged:
                            early_exit = True
                            break
            
            # Mostrar resumen compacto
            days_with_events = sum(1 for r in results.values() if r is not None)
            
            if analyze_convergence and convergence_report is not None:
                conv_status = "✅ Convergió" if convergence_report.converged else "⚠️ No convergió"
                summary = (
                    f"[green]✓ {len(results)}/{len(unique_dates)} días procesados[/green] • "
                    f"[cyan]{days_with_events} con eventos[/cyan] • "
                    f"[dim]{total_events:,} eventos | {total_ticks:,} ticks[/dim]\n"
                    f"[yellow]📊 Convergencia: {conv_status} | "
                    f"Discrepantes: {convergence_report.total_discrepant_events}[/yellow]"
                )
            else:
                summary = (
                    f"[green]✓ {len(unique_dates)} días procesados[/green] • "
                    f"[cyan]{days_with_events} con eventos[/cyan] • "
                    f"[dim]{total_events:,} eventos | {total_ticks:,} ticks[/dim]"
                )
            
            self._console.print(Panel(summary, title=f"Engine θ={self.theta}", border_style="blue", box=box.ROUNDED))
        
        # Guardar reporte de convergencia si se solicitó
        if convergence_report is not None:
            theta_str = f"{self.theta:.6f}".rstrip('0').rstrip('.')
            report_path = (
                self.silver_base_path / ticker / f"theta={theta_str}" / "convergence_report.json"
            )
            convergence_report.save(report_path)
            
            if self.verbose:
                print(f"\n💾 Reporte de convergencia guardado: {report_path}")
                print(convergence_report.generate_summary())
        
        return results, convergence_report

