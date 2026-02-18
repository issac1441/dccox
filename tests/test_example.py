"""Tests for `dccox` package."""

from dccox import main


def test_hello_world() -> None:
    """Test hello_world."""
    assert main.hello_world() == "Hello from dccox!"
