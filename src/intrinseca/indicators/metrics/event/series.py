"""
Series Metrics.

Indicators based on signal processing of intra-event price/time series.

These indicators operate on the nested list columns:
- price_dc: List of prices during DC phase
- time_dc: List of timestamps during DC phase
- price_os: List of prices during OS phase
- time_os: List of timestamps during OS phase

Future indicators may include:
- FourierDominantFreq: Dominant frequency via FFT
- WaveletEnergy: Energy per wavelet scale
- AutoCorrelation: Lag-1 autocorrelation
- SeriesEntropy: Entropy of intra-event returns
"""

from intrinseca.indicators.base import BaseIndicator, IndicatorMetadata
import polars as pl

# Placeholder module - add signal processing indicators here
