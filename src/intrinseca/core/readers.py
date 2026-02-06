"""DC Event Readers.

Functions for reading DC event Parquet files with metadata extraction.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


@dataclass
class DCDataset:
    """DC event dataset with validated theta.

    Attributes:
        df: Concatenated DataFrame from all Parquet files in the range.
        theta: DC threshold used to generate this data, extracted from metadata.
    """

    df: pl.DataFrame
    theta: float


def read_dc_events(
    base_path: Path,
    ticker: str,
    year: int,
    month: int,
    day_start: int | None = None,
    day_end: int | None = None,
) -> DCDataset:
    """Read DC event Parquet files and extract theta from metadata.

    Supports both partitioning schemes:
    - Monthly: {base}/{ticker}/theta=X/year=Y/month=M/data.parquet
    - Daily: {base}/{ticker}/theta=X/year=Y/month=M/day=D/data.parquet

    Args:
        base_path: Base path where DC event data is stored.
        ticker: Instrument ticker (e.g., "BTCUSDT").
        year: Year to read.
        month: Month to read.
        day_start: Optional start day (inclusive). If None, reads all days.
        day_end: Optional end day (inclusive). If None, reads all days.

    Returns:
        DCDataset with concatenated DataFrame and validated theta.

    Raises:
        FileNotFoundError: If no Parquet files found for the range.
        ValueError: If any file lacks theta metadata or thetas are inconsistent.

    Example:
        >>> dataset = read_dc_events(Path("./data"), "BTCUSDT", 2025, 11)
        >>> print(f"Theta: {dataset.theta}, Events: {len(dataset.df)}")
    """
    base_path = Path(base_path)

    # Find all theta directories for this ticker
    ticker_path = base_path / ticker
    if not ticker_path.exists():
        raise FileNotFoundError(f"Ticker directory not found: {ticker_path}")

    files: list[Path] = []

    for theta_dir in ticker_path.glob("theta=*"):
        year_dir = theta_dir / f"year={year}"
        if not year_dir.exists():
            continue

        month_dir = year_dir / f"month={month:02d}"
        if not month_dir.exists():
            continue

        # First check for monthly partitioned file (no day subdirectory)
        month_parquet = month_dir / "data.parquet"
        if month_parquet.exists() and day_start is None and day_end is None:
            # Monthly partitioned data found
            files.append(month_parquet)
        else:
            # Fall back to daily partitioned files
            for day_dir in month_dir.glob("day=*"):
                day = int(day_dir.name.split("=")[1])
                if day_start is not None and day < day_start:
                    continue
                if day_end is not None and day > day_end:
                    continue

                parquet_file = day_dir / "data.parquet"
                if parquet_file.exists():
                    files.append(parquet_file)

    if not files:
        raise FileNotFoundError(f"No Parquet files found for {ticker} {year}-{month:02d}")

    # Sort files for deterministic order
    files.sort()

    # Extract theta from each file's metadata and validate consistency
    thetas: set[float] = set()
    dfs: list[pl.DataFrame] = []

    for f in files:
        # Read metadata without loading data
        parquet_meta = pq.read_metadata(f)
        schema_meta = parquet_meta.schema.to_arrow_schema().metadata or {}

        raw_theta = schema_meta.get(b"theta")
        if raw_theta is None:
            raise ValueError(
                f"Archivo sin metadata 'theta': {f}\n"
                "Este archivo fue creado antes de que theta se guardara en metadata. "
                "Debe reprocesarse con el Engine actualizado."
            )

        thetas.add(float(raw_theta.decode()))

        # Now read the actual data
        dfs.append(pl.read_parquet(f))

    if len(thetas) > 1:
        raise ValueError(
            f"Thetas inconsistentes en los archivos: {thetas}\n"
            "Todos los archivos del rango deben tener el mismo theta."
        )

    return DCDataset(
        df=pl.concat(dfs) if len(dfs) > 1 else dfs[0],
        theta=thetas.pop(),
    )


def read_dc_month(
    base_path: Path,
    ticker: str,
    year: int,
    month: int,
) -> DCDataset:
    """Read all DC event data for a month.

    Convenience wrapper around read_dc_events for full month reads.

    Args:
        base_path: Base path where DC event data is stored.
        ticker: Instrument ticker.
        year: Year to read.
        month: Month to read.

    Returns:
        DCDataset with concatenated DataFrame and validated theta.
    """
    return read_dc_events(base_path, ticker, year, month)
