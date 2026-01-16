"""
Configuración centralizada del módulo de visualización.
Constantes y parámetros por defecto para dashboards.
"""

# =============================================================================
# LÍMITES DE VENTANA TEMPORAL
# =============================================================================
MAX_WINDOW_HOURS: int = 168  # 1 semana - límite máximo de datos procesables
INITIAL_WINDOW_HOURS: int = 24  # Ventana por defecto al cargar

# =============================================================================
# DIMENSIONES DE PANELES
# =============================================================================
MAIN_PANEL_HEIGHT: int = 350  # Altura Panel principal (intrínseco)
SECONDARY_PANEL_HEIGHT: int = 350  # Altura Panel secundario (físico)

# =============================================================================
# CONTROL DE STREAMS
# =============================================================================
DEFAULT_THROTTLE_MS: int = 200  # Debounce para streams (fallback)

# =============================================================================
# PALETA DE COLORES DC (Eventos Directional Change)
# =============================================================================
# Colores base para eventos DC
COLOR_UPTURN: str = "#006400"       # Verde oscuro - confirmación upturn
COLOR_DOWNTURN: str = "#8b0000"     # Rojo oscuro - confirmación downturn
COLOR_UPWARD_OS: str = "#90ee90"    # Verde claro - overshoot alcista
COLOR_DOWNWARD_OS: str = "#ffb6c1"  # Rosa claro - overshoot bajista
COLOR_NEUTRAL: str = "#bdc3c7"      # Gris - sin tendencia definida

# Diccionario de color_key para Datashader (mapeo status_cat -> color)
DATASHADER_COLOR_KEY: dict[str, str] = {
    'Upward': COLOR_UPTURN,      # Overshoot alcista usa verde oscuro
    'Upturn': COLOR_UPWARD_OS,   # Evento upturn usa verde claro
    'Downward': COLOR_DOWNTURN,  # Overshoot bajista usa rojo oscuro
    'Downturn': COLOR_DOWNWARD_OS,  # Evento downturn usa rosa
    'Neutral': COLOR_NEUTRAL
}

# Diccionario de colores para cajas del panel intrínseco (por tipo de evento)
INTRINSIC_BOX_COLORS: dict[str, dict[str, str]] = {
    'upturn': {'change': COLOR_UPWARD_OS, 'overshoot': COLOR_UPTURN},   # Verdes
    'downturn': {'change': COLOR_DOWNWARD_OS, 'overshoot': COLOR_DOWNTURN}  # Rosas/Rojos
}

# Diccionario de colores para VSpan bands del panel físico
VSPAN_COLORS: dict[str, dict[str, str]] = {
    'upturn': {'dc_event': COLOR_UPWARD_OS, 'overshoot': COLOR_UPTURN},
    'downturn': {'dc_event': COLOR_DOWNWARD_OS, 'overshoot': COLOR_DOWNTURN}
}

# Colores para marcadores de eventos en Panel B
EVENT_MARKER_COLORS: dict[str, str] = {
    'upturn': COLOR_UPTURN,
    'downturn': COLOR_DOWNTURN
}

# =============================================================================
# COLORES DE TEXTO Y ETIQUETAS
# =============================================================================
TEXT_COLOR_DARK: str = "#333333"     # Texto oscuro para fondos claros
TEXT_COLOR_LIGHT: str = "#FFFFFF"    # Texto blanco para fondos oscuros
TEXT_COLOR_DEFAULT: str = "#555555"  # Texto gris por defecto

# =============================================================================
# COLORES DE LÍNEAS Y BORDES
# =============================================================================
LINE_COLOR_CHANGE: str = "#666666"    # Borde de rectángulos DC Event
LINE_COLOR_OVERSHOOT: str = "#444444" # Borde de rectángulos Overshoot
LINE_COLOR_VLINE: str = "gray"        # Líneas verticales de tiempo
TICK_MINOR_COLOR: str = "gray"        # Marcas menores de ejes

# =============================================================================
# COLORES PARA GRÁFICOS ESTÁTICOS (Matplotlib)
# =============================================================================
STATIC_COLOR_UPTURN: str = "#2ecc71"   # Verde brillante para plots estáticos
STATIC_COLOR_DOWNTURN: str = "#e74c3c" # Rojo brillante para plots estáticos
STATIC_COLOR_PRICE: str = "#3498db"    # Azul para línea de precios
STATIC_COLOR_DURATION: str = "#9b59b6" # Púrpura para histograma de duraciones
STATIC_COLOR_OVERSHOOT: str = "#f39c12" # Naranja para barras de overshoot

# =============================================================================
# COLORES DE FONDO Y UI
# =============================================================================
BACKGROUND_LABEL: str = "white"        # Fondo de etiquetas
BACKGROUND_LABEL_ALPHA: float = 0.9    # Opacidad de fondo de etiquetas
BORDER_LABEL_ALPHA: float = 0.8        # Opacidad de borde de etiquetas
