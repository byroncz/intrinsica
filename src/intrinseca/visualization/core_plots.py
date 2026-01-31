import datashader as ds
import holoviews as hv
from bokeh.models import HoverTool
from holoviews import opts
from holoviews.operation.datashader import datashade, spread

from .config import (
    COLOR_DOWNTURN,
    COLOR_NEUTRAL,
    COLOR_UPTURN,
    DATASHADER_COLOR_KEY,
    EVENT_MARKER_COLORS,
    INTRINSIC_BOX_COLORS,
    LINE_COLOR_VLINE,
    TEXT_COLOR_DARK,
    TEXT_COLOR_DEFAULT,
    VSPAN_COLORS,
)
from .hooks import _apply_30min_xticks_hook, _apply_integer_xticks_hook, _apply_x_zoom_hook
from .utils import _get_vlines, _prepare_price_data

hv.extension("bokeh")


def _build_hv_plot(df_ticks, df_events, title, theta, height):
    custom_hover = HoverTool(
        tooltips=[("Fecha", "$x{%F %T}"), ("Precio", "$y{0,0.00}")],
        formatters={"$x": "datetime", "$y": "numeral"},
    )

    pdf = _prepare_price_data(df_ticks)

    # Capas de Datashader
    price_dots = hv.Points(pdf, ["time", "price"], vdims=["status_cat"])
    price_shaded = spread(
        datashade(
            price_dots, aggregator=ds.count_cat("status_cat"), color_key=DATASHADER_COLOR_KEY
        ),
        px=1,
    ).opts(xlabel="Time (UTC)", ylabel="Price", show_legend=False)

    # Capas de Eventos
    up_ticks = hv.Scatter(pdf[pdf["status_cat"] == "Upward"], ["time"], ["price"]).opts(
        color=COLOR_UPTURN, size=2
    )
    down_ticks = hv.Scatter(pdf[pdf["status_cat"] == "Downward"], ["time"], ["price"]).opts(
        color=COLOR_DOWNTURN, size=2
    )

    # Líneas de tiempo
    mid, noon = _get_vlines(df_ticks)
    v_lines = hv.Overlay(
        [hv.VLine(t).opts(color=LINE_COLOR_VLINE, line_width=0.5) for t in mid]
        + [
            hv.VLine(t).opts(color=LINE_COLOR_VLINE, line_dash="dotted", line_width=0.5)
            for t in noon
        ]
    )

    layout = price_shaded * up_ticks * down_ticks * v_lines

    return layout.opts(
        opts.RGB(
            height=height,
            responsive=True,
            bgcolor="white",
            tools=[custom_hover, "crosshair", "xbox_zoom", "xwheel_zoom", "reset"],
            hooks=[_apply_x_zoom_hook],
        )
    )


def _build_intrinsic_panel(pdf_segments, height=400, ylim=None):
    """Panel Superior: Cajas bi-particionadas por evento DC (definiciones Tsang).

    Cada evento N se visualiza como dos rectángulos apilados verticalmente:
    - DC Event N (change): ext_price(N) → price(N) [extremo → confirmación]
    - Overshoot N: price(N) → next_ext_price [confirmación → extremo siguiente]

    Colores:
    - Upturn: verde claro (change) / verde oscuro (overshoot)
    - Downturn: rosa (change) / rojo oscuro (overshoot)

    Args:
    ----
        pdf_segments: DataFrame con columnas seq_idx, ext_price, price, next_ext_price,
                      next_ext_time, type_desc, time, ext_time
        height: Altura en píxeles, o None para modo responsivo completo
        ylim: Tupla (y_min, y_max) opcional.

    """
    import numpy as np

    # Parámetros de visualización
    box_half_width = 0.35  # Mitad del ancho de la caja

    # Colores por tipo de evento (desde config)
    colors = INTRINSIC_BOX_COLORS

    # =========================================================================
    # DC Event N (change): ext_price(N) → price(N)
    # Representa el movimiento desde el extremo hasta la confirmación
    # =========================================================================
    change_data = pdf_segments.copy()
    change_data["x0"] = change_data["seq_idx"] - box_half_width
    change_data["x1"] = change_data["seq_idx"] + box_half_width
    # y0/y1 son el rango de precios (min/max de ext_price y price)
    change_data["y0"] = np.minimum(change_data["ext_price"], change_data["price"])
    change_data["y1"] = np.maximum(change_data["ext_price"], change_data["price"])
    change_data["color"] = change_data["type_desc"].map(
        lambda t: colors.get(t, {}).get("change", COLOR_NEUTRAL)
    )
    change_data["part"] = "DC Event"

    # Tiempo: el DC Event va desde ext_time hasta time (confirmación)
    if "ext_time" in change_data.columns:
        change_data["start_time"] = change_data["ext_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        change_data["start_time"] = "N/A"
    if "time" in change_data.columns:
        change_data["end_time"] = change_data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        change_data["end_time"] = "N/A"

    # Delta P y Delta T para DC Event
    change_data["delta_p"] = np.abs(change_data["price"] - change_data["ext_price"])
    if "time" in change_data.columns and "ext_time" in change_data.columns:
        change_data["delta_t"] = (
            change_data["time"] - change_data["ext_time"]
        ).dt.total_seconds() / 3600  # en horas
        change_data["delta_t_str"] = change_data["delta_t"].apply(lambda h: f"{h:.2f} h")
    else:
        change_data["delta_t_str"] = "N/A"

    # =========================================================================
    # Overshoot N: price(N) → next_ext_price
    # Representa el movimiento desde confirmación hasta el siguiente extremo
    # =========================================================================
    os_data = pdf_segments.copy()
    os_data["x0"] = os_data["seq_idx"] - box_half_width
    os_data["x1"] = os_data["seq_idx"] + box_half_width
    # y0/y1 son el rango de precios (min/max de price y next_ext_price)
    os_data["y0"] = np.minimum(os_data["price"], os_data["next_ext_price"])
    os_data["y1"] = np.maximum(os_data["price"], os_data["next_ext_price"])
    os_data["color"] = os_data["type_desc"].map(
        lambda t: colors.get(t, {}).get("overshoot", COLOR_NEUTRAL)
    )
    os_data["part"] = "Overshoot"

    # Tiempo: el Overshoot va desde time (confirmación) hasta next_ext_time (extremo del evento N+1)
    if "time" in os_data.columns:
        os_data["start_time"] = os_data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        os_data["start_time"] = "N/A"
    if "next_ext_time" in os_data.columns:
        os_data["end_time"] = os_data["next_ext_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        os_data["end_time"] = "N/A"

    # Delta P y Delta T para Overshoot
    os_data["delta_p"] = np.abs(os_data["next_ext_price"] - os_data["price"])
    if "next_ext_time" in os_data.columns and "time" in os_data.columns:
        os_data["delta_t"] = (
            os_data["next_ext_time"] - os_data["time"]
        ).dt.total_seconds() / 3600  # en horas
        os_data["delta_t_str"] = os_data["delta_t"].apply(lambda h: f"{h:.2f} h")
    else:
        os_data["delta_t_str"] = "N/A"

    # Custom HoverTool con formato legible
    custom_hover = HoverTool(
        tooltips=[
            # ('Tipo', '@type_desc'),
            # ('Parte', '@part'),
            # ('Inicio', '@start_time'),
            # ('Fin', '@end_time'),
            # ('Precio Alto', '@y1{0.0,00}'),
            # ('Precio Bajo', '@y0{0.0,00}'),
            ("ΔP", "@delta_p{0.0,00}"),
            ("ΔT", "@delta_t_str"),
        ]
    )

    # Crear rectángulos HoloViews
    change_rects = hv.Rectangles(
        change_data,
        ["x0", "y0", "x1", "y1"],
        ["type_desc", "part", "start_time", "end_time", "delta_p", "delta_t_str", "color"],
    ).opts(
        color="color",
        line_width=0,  # Sin borde
        alpha=0.85,
    )

    os_rects = hv.Rectangles(
        os_data,
        ["x0", "y0", "x1", "y1"],
        ["type_desc", "part", "start_time", "end_time", "delta_p", "delta_t_str", "color"],
    ).opts(
        color="color",
        line_width=0,  # Sin borde
        alpha=0.95,
    )

    # =========================================================================
    # Etiquetas permanentes de ΔP y ΔT dentro de cada rectángulo
    # =========================================================================

    # Preparar datos para etiquetas de DC Event (centradas en el rectángulo)
    change_labels_data = change_data.copy()
    change_labels_data["x_center"] = change_labels_data["seq_idx"]
    change_labels_data["y_center"] = (change_labels_data["y0"] + change_labels_data["y1"]) / 2
    change_labels_data["label_text"] = change_labels_data.apply(
        lambda r: f"ΔP: {r['delta_p']:.2f}\nΔT: {r['delta_t_str']}", axis=1
    )
    # Color de texto según tipo (blanco para fondos oscuros, negro para claros)
    change_labels_data["text_color"] = TEXT_COLOR_DARK  # Texto oscuro para DC Event (fondo claro)

    # Preparar datos para etiquetas de Overshoot (posicionadas fuera del rectángulo)
    os_labels_data = os_data.copy()
    os_labels_data["x_center"] = os_labels_data["seq_idx"]
    # Posición Y: arriba del box para upturn, abajo del box para downturn
    # Para upturn (overshoot hacia arriba): la etiqueta va ARRIBA del box (y1 + padding)
    # Para downturn (overshoot hacia abajo): la etiqueta va ABAJO del box (y0 - padding)
    y_padding_fixed = 15  # Distancia fija en unidades de precio
    os_labels_data["y_label"] = os_labels_data.apply(
        lambda r: r["y1"] + y_padding_fixed
        if r["type_desc"] == "upturn"
        else r["y0"] - y_padding_fixed,
        axis=1,
    )
    os_labels_data["label_text"] = os_labels_data.apply(
        lambda r: f"ΔP: {r['delta_p']:.2f}\nΔT: {r['delta_t_str']}", axis=1
    )
    # Color de texto: verde oscuro para upturn, rojo oscuro para downturn
    os_labels_data["text_color"] = os_labels_data["type_desc"].map(
        lambda t: COLOR_UPTURN if t == "upturn" else COLOR_DOWNTURN
    )

    # Crear Labels HoloViews para DC Events
    change_labels = hv.Labels(change_labels_data, ["x_center", "y_center"], "label_text").opts(
        text_font_size="7pt",
        text_color=TEXT_COLOR_DARK,
        text_align="center",
        text_baseline="middle",
    )

    # Crear Labels HoloViews para Overshoots (separados por tipo para diferente baseline)
    # Upturn: etiqueta arriba del box (baseline='bottom' para que el texto quede sobre la línea)
    os_upturn = os_labels_data[os_labels_data["type_desc"] == "upturn"]
    os_downturn = os_labels_data[os_labels_data["type_desc"] == "downturn"]

    os_labels_upturn = hv.Labels(os_upturn, ["x_center", "y_label"], "label_text").opts(
        text_font_size="7pt",
        text_color=COLOR_UPTURN,
        text_align="center",
        text_baseline="bottom",  # Texto arriba de la posición
    )

    # Downturn: etiqueta abajo del box (baseline='top' para que el texto quede bajo la línea)
    os_labels_downturn = hv.Labels(os_downturn, ["x_center", "y_label"], "label_text").opts(
        text_font_size="7pt",
        text_color=COLOR_DOWNTURN,
        text_align="center",
        text_baseline="top",  # Texto abajo de la posición
    )

    # Combinar overlays: rectángulos primero, luego etiquetas encima
    overlay = change_rects * os_rects * change_labels * os_labels_upturn * os_labels_downturn

    # Si no se provee ylim, calcularlo automáticamente con margen para etiquetas
    if ylim is None:
        # Calcular límites Y explícitos incluyendo espacio para etiquetas externas
        # Las etiquetas de OS están a y_padding_fixed (15) fuera del box, más espacio para el texto
        label_space = y_padding_fixed + 35  # 15 de padding + 35 para el texto (2 líneas)
        y_min_data = min(change_data["y0"].min(), os_data["y0"].min())
        y_max_data = max(change_data["y1"].max(), os_data["y1"].max())
        # Agregar espacio para etiquetas
        ylim = (y_min_data - label_space, y_max_data + label_space)

    # Opciones base (usar custom_hover en lugar de 'hover' genérico)
    base_opts = {
        "responsive": True,
        "padding": 0,  # Sin padding adicional, usamos ylim explícito
        "ylim": ylim,
        "tools": ["xbox_select", "xwheel_zoom", "reset", custom_hover],
        "hooks": [_apply_integer_xticks_hook],
        "xlabel": "Event Index (n)",
        "ylabel": "Price",
        "xformatter": "%d",
        "yformatter": "%.0f",
    }

    # Altura explícita solo si se especifica
    if height is not None:
        base_opts["height"] = height

    return overlay.opts(opts.Rectangles(**base_opts))


def _build_physical_panel(
    pdf_ticks_win, event_markers=None, event_segments=None, xlim=None, ylim=None, height=250
):
    """Panel Inferior: Recibe ticks con columna status_cat para colorización.

    Args:
    ----
        pdf_ticks_win: DataFrame Pandas con time, price, status_cat
        event_markers: Lista de tuplas (timestamp, event_idx) para VLines
        event_segments: Lista de dicts con datos de eventos para VSpan bands:
            - ext_time: inicio del DC Event
            - time: fin del DC Event / inicio del Overshoot
            - next_ext_time: fin del Overshoot
            - type_desc: 'upturn' o 'downturn'
        xlim: Tupla (t_start, t_end) para limitar el eje X
        ylim: Tupla (y_min, y_max) para limitar el eje Y (sincronizado con Panel A)
        height: Altura en píxeles, o None para modo responsivo completo

    """
    # Colores para VSpan bands (desde config, consistentes con Panel A)
    vspan_colors = VSPAN_COLORS

    # =========================================================================
    # 1. Crear VSpan bands para DC Events y Overshoots (capa inferior)
    # =========================================================================
    vspan_elements = []
    if event_segments:
        for seg in event_segments:
            type_desc = seg.get("type_desc", "").lower()
            colors = vspan_colors.get(
                type_desc, {"dc_event": COLOR_NEUTRAL, "overshoot": COLOR_NEUTRAL}
            )

            # Remover timezone de los timestamps
            def naive(ts):
                return ts.replace(tzinfo=None) if hasattr(ts, "tzinfo") and ts.tzinfo else ts

            ext_time = naive(seg["ext_time"])
            time = naive(seg["time"])
            next_ext_time = seg.get("next_ext_time")

            # DC Event band: ext_time → time
            if ext_time and time and ext_time < time:
                dc_vspan = hv.VSpan(ext_time, time).opts(color=colors["dc_event"], alpha=0.3)
                vspan_elements.append(dc_vspan)

            # Overshoot band: time → next_ext_time
            if next_ext_time:
                next_ext_time = naive(next_ext_time)
                if time and next_ext_time and time < next_ext_time:
                    os_vspan = hv.VSpan(time, next_ext_time).opts(
                        color=colors["overshoot"], alpha=0.3
                    )
                    vspan_elements.append(os_vspan)

    # =========================================================================
    # 2. Datashader layer (ticks colorizados)
    # =========================================================================
    points = hv.Points(pdf_ticks_win, ["time", "price"], vdims=["status_cat"])

    shaded = spread(
        datashade(points, aggregator=ds.count_cat("status_cat"), color_key=DATASHADER_COLOR_KEY),
        px=1,
    )

    # =========================================================================
    # 3. Preparar datos para etiquetas de eventos (fondo blanco via Bokeh hook)
    # =========================================================================
    label_data = []
    if event_markers:
        max_markers = 30
        num_events = len(event_markers)

        # Densidad adaptativa: mostrar cada N eventos si hay muchos
        step = max(1, (num_events + max_markers - 1) // max_markers)
        sampled_markers = event_markers[::step]

        # Obtener rango de precios para posicionar etiquetas DENTRO del área visible
        if not pdf_ticks_win.empty:
            y_min = pdf_ticks_win["price"].min()
            y_max = pdf_ticks_win["price"].max()
            y_range = y_max - y_min
            y_pos = y_max - 0.05 * y_range  # 5% debajo del máximo (dentro del rango)
        else:
            y_pos = 0

        for ts, idx, type_desc in sampled_markers:
            # Remover timezone si existe
            ts_naive = ts.replace(tzinfo=None) if hasattr(ts, "tzinfo") and ts.tzinfo else ts
            color = EVENT_MARKER_COLORS.get(
                type_desc.lower() if type_desc else "", TEXT_COLOR_DEFAULT
            )
            label_data.append({"x": ts_naive, "y": y_pos, "text": f" {idx} ", "color": color})

    # =========================================================================
    # 4. Combinar layers: VSpans (fondo) → Datashader (medio)
    # =========================================================================
    if vspan_elements:
        vspan_overlay = hv.Overlay(vspan_elements)
        layout = vspan_overlay * shaded
    else:
        layout = shaded

    # Custom HoverTool para Panel B con formato de tiempo y precio
    panel_b_hover = HoverTool(
        tooltips=[
            ("Tiempo", "$x{%F %H:%M:%S}"),
            ("Precio", "$y{0.0,00}"),
        ],
        formatters={"$x": "datetime"},
    )

    # Crear lista de hooks: siempre incluir 30min ticks, opcionalmente labels
    from .hooks import create_event_labels_hook

    hooks_list = [_apply_30min_xticks_hook]
    if label_data:
        hooks_list.append(create_event_labels_hook(label_data))

    # Opciones base para RGB
    rgb_opts = {
        "responsive": True,
        "xlabel": "Time (UTC)",
        "tools": ["xwheel_zoom", "reset", "crosshair", panel_b_hover],
        "hooks": hooks_list,
        "show_legend": False,  # Ocultar leyenda para evitar obstruir ticks
        "padding": 0,  # Sin padding para ajustar exactamente al rango de datos
    }

    # Altura explícita solo si se especifica
    if height is not None:
        rgb_opts["height"] = height

    # Limitar eje X si se proporciona xlim
    if xlim is not None:
        t_start, t_end = xlim
        # Remover timezone si existe
        if hasattr(t_start, "tzinfo") and t_start.tzinfo:
            t_start = t_start.replace(tzinfo=None)
        if hasattr(t_end, "tzinfo") and t_end.tzinfo:
            t_end = t_end.replace(tzinfo=None)
        rgb_opts["xlim"] = (t_start, t_end)

    # Limitar eje Y si se proporciona ylim (Sincronización Panel A <-> Panel B)
    if ylim is not None:
        rgb_opts["ylim"] = ylim

    return layout.opts(opts.RGB(**rgb_opts))
