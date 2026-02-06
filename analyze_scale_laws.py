"""Análisis profundo de las leyes de escala DC.

Glattfelder et al. (2011) establecieron que:
- avg(|OS|) ≈ θ  (Ley del Factor 2)
- avg(TMV) ≈ 2θ

Los warnings reportan:
- 2025-11-01: avg(|OS|) = 0.002276, avg(TMV) = 0.007276
- 2025-11-08: avg(|OS|) = 0.002304, avg(TMV) = 0.007306
- 2025-11-30: avg(|OS|) = 0.002372, avg(TMV) = 0.007374

Con θ = 0.005:
- Rango esperado OS: [0.0025, 0.01]  (0.5θ a 2θ)
- Rango esperado TMV: [0.0075, 0.015] (1.5θ a 3θ)

Valores están LIGERAMENTE por debajo del límite inferior.
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

silver_base = Path("/Users/byroncampo/My Drive/datalake/financial/02_silver/BTCUSDT/theta=0.005")
THETA = 0.005

# Días con warnings
warning_days = [(2025, 11, 1), (2025, 11, 8), (2025, 11, 30)]

# También analizar días que pasaron para comparación
passing_days = [(2025, 11, 15), (2025, 11, 20), (2025, 11, 25)]

print("=" * 90)
print("ANÁLISIS PROFUNDO DE LEYES DE ESCALA DC")
print("=" * 90)
print(f"\nθ = {THETA}")
print(f"Rango esperado avg(|OS|): [{0.5 * THETA:.6f}, {2 * THETA:.6f}]")
print(f"Rango esperado avg(TMV):  [{1.5 * THETA:.6f}, {3 * THETA:.6f}]")
print()


def analyze_day(year: int, month: int, day: int) -> dict:
    """Analiza un día en detalle."""
    path = silver_base / f"year={year}/month={month:02d}/day={day:02d}/data.parquet"
    if not path.exists():
        return None

    df = pl.read_parquet(path)

    os_magnitudes = []
    dc_magnitudes = []
    tmv_values = []
    zero_os_count = 0

    for row in df.iter_rows(named=True):
        extreme_p = row["extreme_price"]
        confirm_p = row["confirm_price"]
        reference_p = row["reference_price"]
        price_os = row.get("price_os") or []

        # DC magnitude (siempre calculable)
        dc_mag = abs(confirm_p - reference_p) / reference_p
        dc_magnitudes.append(dc_mag)

        # OS magnitude (solo si extreme válido)
        if extreme_p != -1.0:
            if len(price_os) > 0:
                os_mag = abs(extreme_p - confirm_p) / confirm_p
                os_magnitudes.append(os_mag)
                tmv_values.append(dc_mag + os_mag)
            else:
                # Zero OS
                zero_os_count += 1
                # TMV para Zero OS = DC magnitude solamente
                tmv_values.append(dc_mag)

    return {
        "date": f"{year}-{month:02d}-{day:02d}",
        "total_events": len(df),
        "events_with_os": len(os_magnitudes),
        "zero_os_count": zero_os_count,
        "zero_os_rate": zero_os_count / len(df) if len(df) > 0 else 0,
        "dc_magnitudes": dc_magnitudes,
        "os_magnitudes": os_magnitudes,
        "tmv_values": tmv_values,
        "avg_dc": np.mean(dc_magnitudes) if dc_magnitudes else 0,
        "avg_os": np.mean(os_magnitudes) if os_magnitudes else 0,
        "avg_tmv": np.mean(tmv_values) if tmv_values else 0,
        "std_os": np.std(os_magnitudes) if os_magnitudes else 0,
        "median_os": np.median(os_magnitudes) if os_magnitudes else 0,
        "min_os": np.min(os_magnitudes) if os_magnitudes else 0,
        "max_os": np.max(os_magnitudes) if os_magnitudes else 0,
    }


print("=" * 90)
print("DÍAS CON WARNINGS (avg(|OS|) < 0.5θ)")
print("=" * 90)

warning_results = []
for y, m, d in warning_days:
    result = analyze_day(y, m, d)
    if result:
        warning_results.append(result)
        print(f"\n--- {result['date']} ---")
        print(f"  Total eventos:     {result['total_events']}")
        print(f"  Eventos con OS:    {result['events_with_os']}")
        print(f"  Zero OS:           {result['zero_os_count']} ({result['zero_os_rate']:.1%})")
        print(f"  avg(|DC|):         {result['avg_dc']:.6f} (esperado ≥ θ = {THETA})")
        print(f"  avg(|OS|):         {result['avg_os']:.6f} (esperado ≈ θ = {THETA})")
        print(f"  avg(TMV):          {result['avg_tmv']:.6f} (esperado ≈ 2θ = {2 * THETA})")
        print(f"  std(|OS|):         {result['std_os']:.6f}")
        print(f"  median(|OS|):      {result['median_os']:.6f}")
        print(f"  min/max(|OS|):     [{result['min_os']:.6f}, {result['max_os']:.6f}]")

print("\n" + "=" * 90)
print("DÍAS QUE PASARON (para comparación)")
print("=" * 90)

passing_results = []
for y, m, d in passing_days:
    result = analyze_day(y, m, d)
    if result:
        passing_results.append(result)
        print(f"\n--- {result['date']} ---")
        print(f"  Total eventos:     {result['total_events']}")
        print(f"  Zero OS:           {result['zero_os_count']} ({result['zero_os_rate']:.1%})")
        print(f"  avg(|OS|):         {result['avg_os']:.6f}")
        print(f"  avg(TMV):          {result['avg_tmv']:.6f}")

print("\n" + "=" * 90)
print("ANÁLISIS DE DISTRIBUCIÓN - DÍAS CON WARNING")
print("=" * 90)

for result in warning_results:
    os_mags = result["os_magnitudes"]
    if os_mags:
        percentiles = np.percentile(os_mags, [5, 10, 25, 50, 75, 90, 95])
        print(f"\n{result['date']} - Distribución de |OS|:")
        print(f"  P5:  {percentiles[0]:.6f}")
        print(f"  P10: {percentiles[1]:.6f}")
        print(f"  P25: {percentiles[2]:.6f}")
        print(f"  P50: {percentiles[3]:.6f} (mediana)")
        print(f"  P75: {percentiles[4]:.6f}")
        print(f"  P90: {percentiles[5]:.6f}")
        print(f"  P95: {percentiles[6]:.6f}")

        # Cuántos eventos tienen OS < theta/2?
        below_half_theta = sum(1 for x in os_mags if x < THETA / 2)
        below_theta = sum(1 for x in os_mags if x < THETA)
        print(
            f"\n  Eventos con |OS| < θ/2:  {below_half_theta} ({below_half_theta / len(os_mags):.1%})"
        )
        print(f"  Eventos con |OS| < θ:    {below_theta} ({below_theta / len(os_mags):.1%})")

# Análisis de todos los días del mes
print("\n" + "=" * 90)
print("RESUMEN MENSUAL - TODOS LOS DÍAS")
print("=" * 90)

all_results = []
for day in range(1, 31):
    result = analyze_day(2025, 11, day)
    if result:
        all_results.append(result)

# Calcular estadísticas globales
all_os = [r["avg_os"] for r in all_results]
all_tmv = [r["avg_tmv"] for r in all_results]
all_zero_rate = [r["zero_os_rate"] for r in all_results]

print(f"\nDías analizados: {len(all_results)}")
print(f"\navg(|OS|) por día:")
print(f"  Media:   {np.mean(all_os):.6f}")
print(f"  Mediana: {np.median(all_os):.6f}")
print(f"  Min:     {np.min(all_os):.6f}")
print(f"  Max:     {np.max(all_os):.6f}")
print(f"  Std:     {np.std(all_os):.6f}")

print(f"\navg(TMV) por día:")
print(f"  Media:   {np.mean(all_tmv):.6f}")
print(f"  Mediana: {np.median(all_tmv):.6f}")
print(f"  Min:     {np.min(all_tmv):.6f}")
print(f"  Max:     {np.max(all_tmv):.6f}")

print(f"\nTasa Zero OS por día:")
print(f"  Media:   {np.mean(all_zero_rate):.1%}")
print(f"  Max:     {np.max(all_zero_rate):.1%}")

# Días bajo el límite
days_below_limit = [r["date"] for r in all_results if r["avg_os"] < 0.5 * THETA]
print(f"\nDías con avg(|OS|) < 0.5θ: {len(days_below_limit)}")
for d in days_below_limit:
    r = next(x for x in all_results if x["date"] == d)
    print(f"  {d}: {r['avg_os']:.6f} (deficit: {0.5 * THETA - r['avg_os']:.6f})")

print("\n" + "=" * 90)
print("DIAGNÓSTICO")
print("=" * 90)

# Calcular desviación del valor teórico
avg_all_os = np.mean(all_os)
theoretical_os = THETA
deviation_os = (avg_all_os - theoretical_os) / theoretical_os * 100

print(f"""
HALLAZGO PRINCIPAL:
- avg(|OS|) mensual = {avg_all_os:.6f}
- Valor teórico (Glattfelder 2011) = {THETA:.6f}
- Desviación = {deviation_os:.1f}%

POSIBLES CAUSAS:
1. Alta tasa de Zero OS: {np.mean(all_zero_rate):.1%} promedio
   - Zero OS reduce el promedio de |OS| porque no contribuyen a os_magnitudes
   - Pero sí afectan la distribución general de eventos

2. Eventos con OS muy pequeños
   - Eventos donde la reversión ocurre casi inmediatamente después de confirmación
   - Pueden indicar alta volatilidad microestructural

3. Límite del test posiblemente muy estrecho
   - Límite actual: [0.5θ, 2θ]
   - Glattfelder (2011) observó variabilidad considerable entre activos
""")

print("=" * 90)
print("FIN DEL ANÁLISIS")
print("=" * 90)
