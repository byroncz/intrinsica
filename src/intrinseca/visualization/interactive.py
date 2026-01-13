import panel as pn
import holoviews as hv
import polars as pl
from datetime import datetime, timezone, timedelta
from .core_plots import _build_hv_plot

# Inicialización de extensiones
pn.extension('mathjax') 
hv.extension('bokeh')

def create_dashboard_app(df_ticks, df_events, title="Análisis DC", theta=0.001, height=800):
    # Rangos para los widgets
    abs_min = df_ticks["time"].min().replace(tzinfo=None)
    abs_max = df_ticks["time"].max().replace(tzinfo=None)
    init_start = abs_max - timedelta(hours=24)

    # Widgets
    start_p = pn.widgets.DatetimePicker(name="Inicio", value=init_start, start=abs_min, end=abs_max)
    end_p = pn.widgets.DatetimePicker(name="Fin", value=abs_max, start=abs_min, end=abs_max)
    update_btn = pn.widgets.Button(name="🚀 Actualizar", button_type="primary", width=150)

    def _update_view(clicks):
        s_utc = start_p.value.replace(tzinfo=timezone.utc)
        e_utc = end_p.value.replace(tzinfo=timezone.utc)
        
        df_tk_win = df_ticks.filter(pl.col("time").is_between(s_utc, e_utc))
        df_ev_win = df_events.filter(pl.col("time").is_between(s_utc, e_utc))

        if df_tk_win.is_empty():
            return pn.pane.Markdown("### ⚠️ Sin datos en el rango.")

        return _build_hv_plot(df_tk_win, df_ev_win, title, theta, height)

    interactive_plot = pn.bind(_update_view, update_btn)

    return pn.Column(
        pn.pane.Markdown(f"### {title}"), 
        pn.Row(
            start_p, end_p, 
            pn.Column(pn.Spacer(height=15), update_btn),
            styles={"background": "#f8f9fa", "padding": "10px"}
        ),
        pn.panel(interactive_plot, loading_indicator=True),
        sizing_mode="stretch_both"
    )

def serve_dashboard(app, port=5006):
    pn.serve(app, port=port, threaded=True)