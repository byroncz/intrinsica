from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from itertools import pairwise

import pytest
from l2_dc_events.thetas import SCALE, THETAS_CONFIG, ThetasError, load_thetas


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
