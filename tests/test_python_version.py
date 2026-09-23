import sys


def test_python_cumple_requires_python():
    assert sys.version_info >= (3, 14)
