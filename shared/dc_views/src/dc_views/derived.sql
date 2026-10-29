-- derived_version: 1.0.0
-- Familia D: aritmética por fila sobre summaries unido a events. Precios y
-- cantidades se castean a DOUBLE dentro de cada razón; las duraciones están en
-- µs y se pasan a segundos. Una división por cero da nulo con NULLIF.
-- D-02 y D-08 no necesitan NULLIF (c_ticks >= 1 y las cantidades de la
-- confirmación son > 0). D-05, D-07, D-11, D-13 y D-18 salen nulos porque los
-- os_* de summaries son nulos con overshoot vacío.
CREATE OR REPLACE VIEW derived AS
SELECT
    theta,
    confirm_agg_trade_id,
    s.* EXCLUDE (
        theta, confirm_agg_trade_id,
        c_ticks, c_quantity_buy, c_quantity_sell, c_price_max, c_price_min,
        c_duration, c_vwap,
        os_ticks, os_quantity_buy, os_quantity_sell, os_price_max, os_price_min,
        os_duration, os_vwap,
        confirm_instant_ticks, confirm_instant_price_best,
        confirm_instant_price_worst
    ),
    -- D-01
    e.direction
        * (s.confirm_instant_price_best::DOUBLE - e.confirm_price::DOUBLE)
        / e.reference_price::DOUBLE AS c_instant_bias,
    -- D-02, D-03
    (s.c_quantity_buy::DOUBLE - s.c_quantity_sell::DOUBLE)
        / (s.c_quantity_buy::DOUBLE + s.c_quantity_sell::DOUBLE)
        AS c_flow_imbalance,
    (s.os_quantity_buy::DOUBLE - s.os_quantity_sell::DOUBLE)
        / NULLIF(s.os_quantity_buy::DOUBLE + s.os_quantity_sell::DOUBLE, 0)
        AS os_flow_imbalance,
    -- D-04, D-05
    s.c_ticks::DOUBLE / NULLIF(s.c_duration::DOUBLE / 1e6, 0) AS c_tick_rate,
    s.os_ticks::DOUBLE / (s.os_duration::DOUBLE / 1e6) AS os_tick_rate,
    -- D-06, D-07
    (s.c_quantity_buy::DOUBLE + s.c_quantity_sell::DOUBLE)
        / NULLIF(s.c_duration::DOUBLE / 1e6, 0) AS c_quantity_rate,
    (s.os_quantity_buy::DOUBLE + s.os_quantity_sell::DOUBLE)
        / (s.os_duration::DOUBLE / 1e6) AS os_quantity_rate,
    -- D-08, D-09
    (s.c_quantity_buy::DOUBLE + s.c_quantity_sell::DOUBLE)
        / s.c_ticks::DOUBLE AS c_mean_trade,
    (s.os_quantity_buy::DOUBLE + s.os_quantity_sell::DOUBLE)
        / NULLIF(s.os_ticks::DOUBLE, 0) AS os_mean_trade,
    -- D-10, D-11
    (s.c_price_max::DOUBLE - s.c_price_min::DOUBLE)
        / e.reference_price::DOUBLE AS c_range,
    (s.os_price_max::DOUBLE - s.os_price_min::DOUBLE)
        / e.reference_price::DOUBLE AS os_range,
    -- D-12, D-13
    (s.c_vwap::DOUBLE - e.confirm_price::DOUBLE)
        / e.reference_price::DOUBLE AS c_vwap_distance,
    (s.os_vwap::DOUBLE - e.confirm_price::DOUBLE)
        / e.reference_price::DOUBLE AS os_vwap_distance,
    -- D-17, D-18, D-19
    (e.confirm_price::DOUBLE - e.reference_price::DOUBLE)
        / (e.reference_price::DOUBLE * NULLIF(s.c_duration::DOUBLE / 1e6, 0))
        AS c_speed,
    (e.extreme_price::DOUBLE - e.confirm_price::DOUBLE)
        / (e.reference_price::DOUBLE * (s.os_duration::DOUBLE / 1e6)) AS os_speed,
    (e.extreme_price::DOUBLE - e.reference_price::DOUBLE)
        / (
            e.reference_price::DOUBLE
            * NULLIF((e.extreme_time - e.reference_time)::DOUBLE / 1e6, 0)
        ) AS ev_speed,
    -- D-14, D-15, D-16
    s.c_duration = 0 AS atomic_sweep,
    s.os_ticks = 0 AS empty_overshoot,
    s.c_ticks < e.confirm_agg_trade_id - e.reference_agg_trade_id
        OR s.os_ticks < e.extreme_agg_trade_id - e.confirm_agg_trade_id
        AS provider_gap
FROM summaries AS s
JOIN events AS e USING (theta, confirm_agg_trade_id);
