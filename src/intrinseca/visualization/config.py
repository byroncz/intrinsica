"""
Configuración centralizada del módulo de visualización.
Constantes y parámetros por defecto para dashboards.
"""

# --- Límites de Ventana Temporal ---
MAX_WINDOW_HOURS: int = 168  # 1 semana - límite máximo de datos procesables
INITIAL_WINDOW_HOURS: int = 24  # Ventana por defecto al cargar

# --- Dimensiones de Paneles ---
MAIN_PANEL_HEIGHT: int = 350  # Altura Panel principal (intrínseco)
SECONDARY_PANEL_HEIGHT: int = 350  # Altura Panel secundario (físico)

# --- Control de Streams ---
DEFAULT_THROTTLE_MS: int = 200  # Debounce para streams (fallback)

# --- Paleta de Colores DC ---
COLOR_UPTURN: str = "#006400"  # Verde oscuro
COLOR_DOWNTURN: str = "#8b0000"  # Rojo oscuro
COLOR_UPWARD_OS: str = "#90ee90"  # Verde claro (overshoot)
COLOR_DOWNWARD_OS: str = "#ffb6c1"  # Rosa claro (overshoot)
COLOR_NEUTRAL: str = "#bdc3c7"  # Gris
