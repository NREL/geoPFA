"""Tests for the geoPFA package"""

import geopfa


def test_version_available():
    """Confirm that the version attribute is available"""
    assert hasattr(geopfa, "__version__")
