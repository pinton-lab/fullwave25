import importlib
import logging
from unittest.mock import patch

import fullwave

"""Tests for fullwave.__init__ module."""


def test_imports():
    """Test that all expected modules and classes can be imported."""
    assert hasattr(fullwave, "Domain")
    assert hasattr(fullwave, "Grid")
    assert hasattr(fullwave, "Medium")
    assert hasattr(fullwave, "MediumBuilder")
    assert hasattr(fullwave, "MediumExponentialAttenuation")
    assert hasattr(fullwave, "MediumRelaxationMaps")
    assert hasattr(fullwave, "Sensor")
    assert hasattr(fullwave, "Solver")
    assert hasattr(fullwave, "Source")
    assert hasattr(fullwave, "Transducer")
    assert hasattr(fullwave, "TransducerGeometry")
    assert hasattr(fullwave, "presets")
    assert hasattr(fullwave, "utils")


def test_all_exports():
    """Test that __all__ contains expected exports."""
    expected_all = [
        "Domain",
        "Grid",
        "Medium",
        "MediumBuilder",
        "MediumExponentialAttenuation",
        "MediumRelaxationMaps",
        "Sensor",
        "Solver",
        "Source",
        "Transducer",
        "TransducerGeometry",
        "presets",
        "utils",
    ]

    assert fullwave.__all__ == expected_all


@patch("platform.system")
def test_linux_warning_on_non_linux(mock_system, caplog):
    """Test that warning is logged on non-Linux systems."""
    mock_system.return_value = "Windows"

    # Reload the module to trigger the platform check

    with caplog.at_level(logging.WARNING):
        importlib.reload(fullwave)

    # Check if warning was logged
    assert any(
        "fullwave is primarily developed for Linux" in record.message for record in caplog.records
    )


@patch("platform.system")
def test_no_warning_on_linux(mock_system, caplog):
    """Test that no warning is logged on Linux systems."""
    mock_system.return_value = "Linux"

    # Reload the module to trigger the platform check

    with caplog.at_level(logging.WARNING):
        importlib.reload(fullwave)

    # Check that the specific warning was not logged
    assert not any(
        "fullwave is primarily developed for Linux" in record.message for record in caplog.records
    )


@patch("platform.system")
def test_warning_on_mac(mock_system, caplog):
    """Test that warning is logged on Mac systems."""
    mock_system.return_value = "Darwin"

    # Reload the module to trigger the platform check

    with caplog.at_level(logging.WARNING):
        importlib.reload(fullwave)

    # Check if warning was logged
    assert any(
        "fullwave is primarily developed for Linux" in record.message for record in caplog.records
    )
