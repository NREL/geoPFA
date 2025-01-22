"""Tests for the geoPFA package"""

import geoPFA


def test_version_available():
    """Confirm that the version attribute is available"""
    assert hasattr(geoPFA, '__version__')
