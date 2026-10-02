from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from itertools import pairwise

import pytest
from l2_dc_events.thetas import (
    SCALE,
    THETA_MAX,
    THETA_MIN,
    THETAS_CONFIG,
    ThetasError,
    load_thetas,
    parse_thetas,
)


def rule(n=50, theta_min="0.0001", theta_max="0.05"):
    """ADR-L2-10: θᵢ = θ_min · (θ_max / θ_min)^(i / (N-1)), en enteros de 10⁻⁸.

    Con 60 dígitos, para que ningún redondeo de float decida un empate.
    """
    with localcontext() as ctx:
        ctx.prec = 60
        low, high = Decimal(theta_min), Decimal(theta_max)
        return [
            int(
                (
                    low * (high / low) ** (Decimal(i) / (n - 1)) * SCALE
                ).to_integral_value(ROUND_HALF_EVEN)
            )
            for i in range(n)
        ]


def test_the_config_holds_the_50_thetas_of_the_rule():
    assert load_thetas() == rule()


def test_extremes_and_order():
    thetas = load_thetas()
    assert len(thetas) == 50
    assert thetas[0] == 10_000  # 0.0001
    assert thetas[-1] == 5_000_000  # 0.05
    assert all(a < b for a, b in pairwise(thetas))


def test_the_ratio_between_thetas_is_the_geometric_step():
    # r = (0.05 / 0.0001)^(1/49) ≈ 1,13522: cada θ es ~13,5 % mayor que el anterior.
    ratios = [b / a for a, b in pairwise(load_thetas())]
    assert all(1.13 < r < 1.14 for r in ratios)


def test_the_config_scale_is_the_price_scale():
    assert SCALE == 10**8
    assert f"scale: {SCALE}" in THETAS_CONFIG.read_text()


@pytest.mark.parametrize(
    "body",
    [
        "scale: 1000\nthetas: [1, 2]\n",
        "scale: 100000000\nthetas: []\n",
        "scale: 100000000\nthetas: [2, 1]\n",
        "scale: 100000000\nthetas: [1, 1]\n",
        "scale: 100000000\nthetas: [0, 1]\n",
        "scale: 100000000\nthetas: [1, 100000000]\n",
        "scale: 100000000\nthetas: [0.5, 1]\n",
        "scale: 100000000\nthetas: [true]\n",
        "thetas: [1]\n",
        "[]\n",
    ],
)
def test_load_rejects_a_malformed_file(tmp_path, body):
    path = tmp_path / "thetas.yaml"
    path.write_text(body)
    with pytest.raises(ThetasError):
        load_thetas(path)


def _catalog(*thetas, scale=SCALE):
    return f"scale: {scale}\nthetas: {list(thetas)}\n"


def test_the_catalog_order_is_free_and_the_result_is_sorted():
    assert parse_thetas(_catalog(500_000, 10_000, 31_313)) == [10_000, 31_313, 500_000]


def test_the_range_is_closed_at_both_ends():
    assert parse_thetas(_catalog(THETA_MIN, THETA_MAX)) == [THETA_MIN, THETA_MAX]


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        (_catalog(THETA_MIN - 1), "fuera de"),
        (_catalog(THETA_MAX + 1), "fuera de"),
        (_catalog(0, 20_000), "fuera de"),
        (_catalog(-5, 20_000), "fuera de"),
        (_catalog(20_000, 10_000, 20_000), "repetidos: [20000]"),
        (_catalog(20_000, 0.0003), "debe ser un entero"),
        (_catalog(20_000, scale=1000), "`scale` debe ser"),
        ("scale: 100000000\nthetas: [10000\n", "no es YAML válido"),
    ],
)
def test_the_catalog_names_what_is_wrong(body, problem):
    with pytest.raises(ThetasError) as error:
        parse_thetas(body, "gs://b/l2/thetas.yaml")
    assert problem in str(error.value)
    assert error.value.source == "gs://b/l2/thetas.yaml"
    assert any(problem in p for p in error.value.problems)


def test_a_missing_catalog_is_an_error_not_a_crash(tmp_path):
    with pytest.raises(ThetasError, match="no se pudo leer"):
        load_thetas(tmp_path / "no-existe.yaml")
