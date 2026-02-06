"""Script para analizar eventos donde confirm_time >= extreme_time."""

from pathlib import Path

import polars as pl

silver_base = Path("/Users/byroncampo/My Drive/datalake/financial/02_silver/BTCUSDT/theta=0.005")

# Días con fallos reportados
problem_days = [
    (2025, 11, 5),
    (2025, 11, 13),
    (2025, 11, 14),
]

print("=" * 80)
print("ANÁLISIS DE EVENTOS DONDE confirm_time >= extreme_time")
print("=" * 80)
print()

for year, month, day in problem_days:
    path = silver_base / f"year={year}/month={month:02d}/day={day:02d}/data.parquet"

    if not path.exists():
        print(f"No existe: {path}")
        continue

    df = pl.read_parquet(path)

    # Filtrar eventos problemáticos (excluyendo provisionales)
    invalid = df.filter(
        (pl.col("extreme_time") != -1) & (pl.col("confirm_time") >= pl.col("extreme_time"))
    )

    print(f"--- {year}-{month:02d}-{day:02d} ---")
    print(f"Total eventos: {len(df)}")
    print(f"Eventos con confirm_time >= extreme_time: {len(invalid)}")

    if len(invalid) > 0:
        for i, row in enumerate(invalid.iter_rows(named=True)):
            print(f"\n  Evento problemático #{i + 1}:")
            print(f"    confirm_price  = {row['confirm_price']:.4f}")
            print(f"    extreme_price  = {row['extreme_price']:.4f}")
            print(f"    diff_price     = {row['extreme_price'] - row['confirm_price']:.8f}")
            print(f"    confirm_time   = {row['confirm_time']}")
            print(f"    extreme_time   = {row['extreme_time']}")
            print(f"    diff_time (ns) = {row['extreme_time'] - row['confirm_time']}")
            print(f"    event_type     = {row['event_type']}")
            price_os = row.get("price_os", []) or []
            print(f"    len(price_os)  = {len(price_os)}")

            # ¿Es un overshoot cero?
            if abs(row["extreme_price"] - row["confirm_price"]) < 1e-6:
                print("    >> POSIBLE OVERSHOOT CERO (extreme == confirm)")
            elif row["extreme_time"] == row["confirm_time"]:
                print("    >> TIEMPOS IGUALES (extreme_time == confirm_time)")

    print()

print("=" * 80)
print("FIN DEL ANÁLISIS")
print("=" * 80)
