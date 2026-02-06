"""Sistema de reconciliación retroactiva para huérfanos entre días.

Maneja el escenario donde:
1. Un evento se confirma "virtualmente" al final del día
2. Ticks posteriores sugieren reversión pero no cruzan θ
3. Al día siguiente, se determina si fue reversión real o falsa alarma

Mitigaciones implementadas:
- Escritura atómica: archivo temporal + rename
- Backup automático antes de modificar
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import polars as pl

from .state import DCState


class ReconciliationType(Enum):
    """Tipo de reconciliación requerida."""

    NONE = "none"
    CONFIRM_REVERSAL = "confirm"  # Reversión confirmó, actualizar extreme_price
    EXTEND_OS = "extend_os"  # Falsa alarma, extender OS del evento anterior


@dataclass
class ReconciliationResult:
    """Resultado de una operación de reconciliación."""

    reconciliation_type: ReconciliationType
    previous_parquet_path: Path | None
    backup_path: Path | None
    success: bool
    error: str | None = None


def check_reconciliation_needed(
    prev_state: DCState,
    first_tick_price: float,
    theta: float,
) -> tuple[ReconciliationType, dict]:
    """Evalúa si se necesita reconciliación y de qué tipo.

    Lógica:
    1. Si el trend anterior era alcista (1):
       - Si el primer precio del día nuevo cruza θ hacia abajo → confirmar reversión
       - Si el primer precio supera el extremo alto → falsa alarma (extender OS)
    2. Si el trend anterior era bajista (-1):
       - Si el primer precio del día nuevo cruza θ hacia arriba → confirmar reversión
       - Si el primer precio está por debajo del extremo bajo → falsa alarma (extender OS)

    Args:
    ----
        prev_state: Estado del día anterior
        first_tick_price: Precio del primer tick del día actual
        theta: Umbral DC

    Returns:
    -------
        Tuple con el tipo de reconciliación y datos adicionales

    """
    if prev_state.n_orphans == 0:
        return ReconciliationType.NONE, {}

    ext_high, ext_low = prev_state.get_extreme_prices()
    trend = int(prev_state.current_trend)

    context = {
        "prev_trend": trend,
        "ext_high": ext_high,
        "ext_low": ext_low,
        "first_price": first_tick_price,
        "theta": theta,
    }

    if trend == 1:  # Trend alcista
        threshold_down = ext_high * (1.0 - theta)
        if first_tick_price <= threshold_down:
            # Confirmación de reversión bajista
            return ReconciliationType.CONFIRM_REVERSAL, context
        elif first_tick_price > ext_high:
            # Falsa alarma: el precio siguió subiendo
            return ReconciliationType.EXTEND_OS, context
    elif trend == -1:  # Trend bajista
        threshold_up = ext_low * (1.0 + theta)
        if first_tick_price >= threshold_up:
            # Confirmación de reversión alcista
            return ReconciliationType.CONFIRM_REVERSAL, context
        elif first_tick_price < ext_low:
            # Falsa alarma: el precio siguió bajando
            return ReconciliationType.EXTEND_OS, context

    return ReconciliationType.NONE, context


def reconcile_previous_day(
    silver_path: Path,
    reconciliation_type: ReconciliationType,
    new_extreme_price: float,
    new_extreme_time: int,
) -> ReconciliationResult:
    """Actualiza el Parquet del día anterior según el tipo de reconciliación.

    Mitigaciones:
    - Backup del archivo original antes de modificar
    - Escritura a archivo temporal, luego rename atómico

    Args:
    ----
        silver_path: Ruta al archivo Parquet del día anterior
        reconciliation_type: Tipo de reconciliación
        new_extreme_price: Nuevo precio extremo (si aplica)
        new_extreme_time: Nuevo timestamp del extremo (si aplica)

    Returns:
    -------
        ReconciliationResult con el resultado de la operación

    """
    if reconciliation_type == ReconciliationType.NONE:
        return ReconciliationResult(
            reconciliation_type=reconciliation_type,
            previous_parquet_path=None,
            backup_path=None,
            success=True,
        )

    if not silver_path.exists():
        return ReconciliationResult(
            reconciliation_type=reconciliation_type,
            previous_parquet_path=silver_path,
            backup_path=None,
            success=False,
            error=f"Archivo no encontrado: {silver_path}",
        )

    # 1. Crear backup
    backup_path = silver_path.with_suffix(".parquet.bak")
    try:
        shutil.copy2(silver_path, backup_path)
    except Exception as e:
        return ReconciliationResult(
            reconciliation_type=reconciliation_type,
            previous_parquet_path=silver_path,
            backup_path=None,
            success=False,
            error=f"Error creando backup: {e}",
        )

    try:
        # 2. Cargar DataFrame
        df = pl.read_parquet(silver_path)

        if df.is_empty():
            return ReconciliationResult(
                reconciliation_type=reconciliation_type,
                previous_parquet_path=silver_path,
                backup_path=backup_path,
                success=True,
            )

        # 3. Aplicar reconciliación
        if reconciliation_type == ReconciliationType.CONFIRM_REVERSAL:
            # Actualizar extreme_price del último evento
            df = df.with_columns(
                [
                    pl.when(pl.arange(0, df.height) == df.height - 1)
                    .then(pl.lit(new_extreme_price))
                    .otherwise(pl.col("extreme_price"))
                    .alias("extreme_price"),
                    pl.when(pl.arange(0, df.height) == df.height - 1)
                    .then(pl.lit(new_extreme_time))
                    .otherwise(pl.col("extreme_time"))
                    .alias("extreme_time"),
                ]
            )
        elif reconciliation_type == ReconciliationType.EXTEND_OS:
            # Falsa alarma: el precio siguió en la dirección del trend anterior
            # Actualizamos el extreme_price del último evento con el nuevo extremo
            # Nota: Los ticks huérfanos no se agregan a price_os aquí porque
            # serán procesados como parte del día actual por el kernel
            df = df.with_columns(
                [
                    pl.when(pl.arange(0, df.height) == df.height - 1)
                    .then(pl.lit(new_extreme_price))
                    .otherwise(pl.col("extreme_price"))
                    .alias("extreme_price"),
                    pl.when(pl.arange(0, df.height) == df.height - 1)
                    .then(pl.lit(new_extreme_time))
                    .otherwise(pl.col("extreme_time"))
                    .alias("extreme_time"),
                ]
            )

        # 4. Escribir a archivo temporal
        temp_path = silver_path.with_suffix(".parquet.tmp")
        df.write_parquet(temp_path, compression="zstd", compression_level=3)

        # 5. Rename atómico
        temp_path.rename(silver_path)

        return ReconciliationResult(
            reconciliation_type=reconciliation_type,
            previous_parquet_path=silver_path,
            backup_path=backup_path,
            success=True,
        )

    except Exception as e:
        # Restaurar backup si algo falla
        if backup_path.exists():
            shutil.copy2(backup_path, silver_path)

        return ReconciliationResult(
            reconciliation_type=reconciliation_type,
            previous_parquet_path=silver_path,
            backup_path=backup_path,
            success=False,
            error=f"Error durante reconciliación: {e}",
        )


def cleanup_backup(backup_path: Path) -> None:
    """Elimina el archivo de backup después de una reconciliación exitosa."""
    if backup_path and backup_path.exists():
        backup_path.unlink()


def reconcile_previous_month(
    silver_base_path: Path,
    ticker: str,
    theta: float,
    prev_year: int,
    prev_month: int,
    new_extreme_price: float,
    new_extreme_time: int,
) -> ReconciliationResult:
    """Actualiza el Parquet del mes anterior con el extremo definitivo.

    Similar a reconcile_previous_day pero opera sobre archivos mensuales.
    Busca el archivo mensual: .../month={MM}/data.parquet (sin day=)

    Mitigaciones:
    - Backup del archivo original antes de modificar
    - Escritura a archivo temporal, luego rename atómico

    Args:
    ----
        silver_base_path: Ruta base del Silver layer
        ticker: Ticker del activo
        theta: Umbral DC
        prev_year: Año del mes anterior
        prev_month: Mes anterior (1-12)
        new_extreme_price: Nuevo precio extremo definitivo
        new_extreme_time: Nuevo timestamp del extremo (nanosegundos)

    Returns:
    -------
        ReconciliationResult con el resultado de la operación

    """
    # Construir ruta al archivo mensual
    theta_str = f"{theta:.6f}".rstrip("0").rstrip(".")
    silver_path = (
        silver_base_path
        / ticker
        / f"theta={theta_str}"
        / f"year={prev_year}"
        / f"month={prev_month:02d}"
        / "data.parquet"
    )

    if not silver_path.exists():
        return ReconciliationResult(
            reconciliation_type=ReconciliationType.CONFIRM_REVERSAL,
            previous_parquet_path=silver_path,
            backup_path=None,
            success=False,
            error=f"Archivo mensual no encontrado: {silver_path}",
        )

    # 1. Crear backup
    backup_path = silver_path.with_suffix(".parquet.bak")
    try:
        shutil.copy2(silver_path, backup_path)
    except Exception as e:
        return ReconciliationResult(
            reconciliation_type=ReconciliationType.CONFIRM_REVERSAL,
            previous_parquet_path=silver_path,
            backup_path=None,
            success=False,
            error=f"Error creando backup: {e}",
        )

    try:
        # 2. Cargar DataFrame
        df = pl.read_parquet(silver_path)

        if df.is_empty():
            # Archivo vacío, nada que reconciliar
            cleanup_backup(backup_path)
            return ReconciliationResult(
                reconciliation_type=ReconciliationType.CONFIRM_REVERSAL,
                previous_parquet_path=silver_path,
                backup_path=None,
                success=True,
            )

        # 3. Verificar si el último evento tiene extremo provisional (-1.0)
        last_extreme = df["extreme_price"][-1]
        if last_extreme > 0:
            # Ya tiene extremo válido, no necesita reconciliación
            cleanup_backup(backup_path)
            return ReconciliationResult(
                reconciliation_type=ReconciliationType.NONE,
                previous_parquet_path=silver_path,
                backup_path=None,
                success=True,
            )

        # 4. Actualizar extreme_price del último evento
        df = df.with_columns(
            [
                pl.when(pl.arange(0, df.height) == df.height - 1)
                .then(pl.lit(new_extreme_price))
                .otherwise(pl.col("extreme_price"))
                .alias("extreme_price"),
                pl.when(pl.arange(0, df.height) == df.height - 1)
                .then(pl.lit(new_extreme_time))
                .otherwise(pl.col("extreme_time"))
                .alias("extreme_time"),
            ]
        )

        # 5. Escribir a archivo temporal
        temp_path = silver_path.with_suffix(".parquet.tmp")
        df.write_parquet(temp_path, compression="zstd", compression_level=3)

        # 6. Rename atómico
        temp_path.rename(silver_path)

        # 7. Limpiar backup
        cleanup_backup(backup_path)

        return ReconciliationResult(
            reconciliation_type=ReconciliationType.CONFIRM_REVERSAL,
            previous_parquet_path=silver_path,
            backup_path=None,
            success=True,
        )

    except Exception as e:
        # Restaurar backup si algo falla
        if backup_path.exists():
            shutil.copy2(backup_path, silver_path)

        return ReconciliationResult(
            reconciliation_type=ReconciliationType.CONFIRM_REVERSAL,
            previous_parquet_path=silver_path,
            backup_path=backup_path,
            success=False,
            error=f"Error durante reconciliación mensual: {e}",
        )
