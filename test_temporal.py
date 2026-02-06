"""Test simple del validador temporal corregido."""

import sys
from pathlib import Path

# Output a archivo
output_file = Path("/Users/byroncampo/My Drive/repositorio/eafit/intrinsica/test_output.txt")

with open(output_file, "w") as f:
    try:
        f.write("=== TEST DE VALIDADOR TEMPORAL ===\n")

        # Import
        from tests.quality.validators.temporal import TemporalValidator

        f.write("Import OK\n")

        # Crear instancia
        tv = TemporalValidator()
        f.write(f"Validator name: {tv.name}\n")

        # Verificar que tiene los métodos
        f.write(f"Has _test_zero_os_rate: {hasattr(tv, '_test_zero_os_rate')}\n")
        f.write(
            f"Has _test_confirm_before_extreme: {hasattr(tv, '_test_confirm_before_extreme')}\n"
        )

        # Cargar datos reales
        from datetime import date

        import polars as pl

        silver_path = Path(
            "/Users/byroncampo/My Drive/datalake/financial/02_silver/BTCUSDT/theta=0.005/year=2025/month=11/day=05/data.parquet"
        )
        if silver_path.exists():
            df = pl.read_parquet(silver_path)
            f.write(f"\nCargados {len(df)} eventos del 2025-11-05\n")

            # Ejecutar tests
            day = date(2025, 11, 5)
            result1 = tv._test_confirm_before_extreme(df, day)
            f.write(f"\nTest confirm_before_extreme:\n")
            f.write(f"  passed: {result1.passed}\n")
            f.write(f"  message: {result1.message}\n")
            f.write(f"  affected: {result1.affected_events}\n")

            result2 = tv._test_zero_os_rate(df, day)
            f.write(f"\nTest zero_os_rate:\n")
            f.write(f"  passed: {result2.passed}\n")
            f.write(f"  message: {result2.message}\n")
            f.write(f"  details: {result2.details}\n")
        else:
            f.write(f"\nArchivo no existe: {silver_path}\n")

        f.write("\n=== FIN ===\n")

    except Exception as e:
        f.write(f"\nERROR: {type(e).__name__}: {e}\n")
        import traceback

        f.write(traceback.format_exc())

print(f"Output written to {output_file}")
