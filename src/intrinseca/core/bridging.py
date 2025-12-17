"""
Puente entre estructuras de datos: Polars <-> NumPy.

Este módulo gestiona la conversión eficiente entre DataFrames de Polars
y arrays de NumPy, minimizando copias de memoria cuando es posible.

La filosofía es: recibir cualquier formato razonable de entrada,
convertir internamente a NumPy para cálculos Numba, y retornar
en el formato solicitado.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import polars as pl


def to_numpy(
    data: Union[NDArray, Sequence[float], "pl.Series", "pl.DataFrame"],
    column: str = "close",
    dtype: np.dtype = np.float64
) -> NDArray:
    """
    Convierte datos de entrada a array NumPy.
    
    Esta función acepta múltiples formatos y los normaliza a un array
    NumPy del tipo especificado, optimizando para evitar copias
    innecesarias cuando es posible.
    
    Args:
        data: Datos de entrada. Soporta:
            - numpy.ndarray
            - Lista o secuencia de Python
            - polars.Series
            - polars.DataFrame (extrae columna especificada)
        column: Nombre de columna si data es DataFrame.
        dtype: Tipo de dato del array resultante.
    
    Returns:
        Array NumPy con los datos.
    
    Raises:
        TypeError: Si el tipo de data no es soportado.
        KeyError: Si la columna no existe en el DataFrame.
    
    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
        >>> arr = to_numpy(df, column="close")
        >>> arr
        array([1., 2., 3.])
    """
    # NumPy array
    if isinstance(data, np.ndarray):
        if data.dtype == dtype:
            return data
        return data.astype(dtype)
    
    # Lista o secuencia
    if isinstance(data, (list, tuple)):
        return np.array(data, dtype=dtype)
    
    # Polars
    try:
        import polars as pl
        
        if isinstance(data, pl.Series):
            # to_numpy() en Polars puede ser zero-copy para algunos tipos
            arr = data.to_numpy()
            if arr.dtype != dtype:
                arr = arr.astype(dtype)
            return arr
        
        if isinstance(data, pl.DataFrame):
            if column not in data.columns:
                raise KeyError(f"Columna '{column}' no encontrada. Disponibles: {data.columns}")
            return to_numpy(data[column], dtype=dtype)
    
    except ImportError:
        pass
    
    # Intentar conversión genérica
    try:
        return np.asarray(data, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"No se puede convertir tipo {type(data).__name__} a NumPy array: {e}"
        ) from e


def to_polars(
    data: Union[NDArray, dict[str, NDArray], Sequence],
    schema: dict[str, type] | None = None
) -> "pl.DataFrame":
    """
    Convierte datos a DataFrame de Polars.
    
    Args:
        data: Datos a convertir:
            - dict de arrays -> DataFrame con esas columnas
            - array 2D -> DataFrame con columnas col_0, col_1, ...
            - array 1D -> DataFrame con una columna "value"
        schema: Esquema opcional de tipos {columna: tipo}.
    
    Returns:
        DataFrame de Polars.
    
    Example:
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> df = to_polars(arr)
        >>> df
        shape: (3, 1)
        ┌───────┐
        │ value │
        │ ---   │
        │ f64   │
        ╞═══════╡
        │ 1.0   │
        │ 2.0   │
        │ 3.0   │
        └───────┘
    """
    import polars as pl
    
    if isinstance(data, dict):
        return pl.DataFrame(data, schema=schema)
    
    arr = np.asarray(data)
    
    if arr.ndim == 1:
        return pl.DataFrame({"value": arr}, schema=schema)
    
    if arr.ndim == 2:
        columns = {f"col_{i}": arr[:, i] for i in range(arr.shape[1])}
        return pl.DataFrame(columns, schema=schema)
    
    raise ValueError(f"Array de {arr.ndim} dimensiones no soportado")


def extract_price_column(
    data: Union[NDArray, Sequence[float], "pl.Series", "pl.DataFrame"],
    column: str = "close"
) -> NDArray[np.float64]:
    """
    Extrae columna de precios de cualquier estructura de datos.
    
    Función de conveniencia que encapsula la lógica común de extraer
    una serie de precios para análisis DC.
    
    Args:
        data: Fuente de datos (array, Series, DataFrame).
        column: Nombre de columna si es DataFrame.
    
    Returns:
        Array NumPy float64 con precios.
    """
    return to_numpy(data, column=column, dtype=np.float64)


def events_to_polars(events: list) -> "pl.DataFrame":
    """
    Convierte lista de DCEvent a DataFrame de Polars.
    
    Args:
        events: Lista de objetos DCEvent.
    
    Returns:
        DataFrame con una fila por evento.
    """
    import polars as pl
    
    if not events:
        return pl.DataFrame(schema={
            "index": pl.Int64,
            "price": pl.Float64,
            "event_type": pl.Utf8,
            "extreme_index": pl.Int64,
            "extreme_price": pl.Float64,
            "dc_return": pl.Float64,
            "dc_duration": pl.Int64
        })
    
    return pl.DataFrame({
        "index": [e.index for e in events],
        "price": [e.price for e in events],
        "event_type": [e.event_type for e in events],
        "extreme_index": [e.extreme_index for e in events],
        "extreme_price": [e.extreme_price for e in events],
        "dc_return": [e.dc_return for e in events],
        "dc_duration": [e.dc_duration for e in events]
    })


def result_to_polars(result, prices: NDArray) -> "pl.DataFrame":
    """
    Convierte DCResult completo a DataFrame enriquecido.
    
    Genera un DataFrame con la serie de precios original más
    las columnas de tendencia y señales de eventos.
    
    Args:
        result: Objeto DCResult.
        prices: Array de precios original.
    
    Returns:
        DataFrame con columnas: index, price, trend, is_event, event_type.
    """
    import polars as pl
    
    n = len(prices)
    
    # Crear arrays de señales
    is_event = np.zeros(n, dtype=np.int8)
    event_type = np.full(n, "", dtype=object)
    
    for event in result.events:
        is_event[event.index] = 1
        event_type[event.index] = event.event_type
    
    return pl.DataFrame({
        "index": np.arange(n),
        "price": prices,
        "trend": result.trends,
        "is_event": is_event,
        "event_type": event_type
    })