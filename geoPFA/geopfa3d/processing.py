"""Transition module

All functionalities from this module were moved to
:module:`~geoPFA.processing`.
"""

import warnings

import geoPFA.processing


class Cleaners(geoPFA.processing.Cleaners):
    """Alias for geoPFA.processing.Cleaners

    .. deprecated:: 0.1.0
       :class:`~geoPFA.processing.Cleaners` instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Cleaners class"""
        warnings.warn(
            "The geopfa3d.processing.Cleaners class is deprecated"
            " and will be removed in a future version."
           " Please use the geoPFA.processing module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class Exclusions(geoPFA.processing.Exclusions):
    """Alias for geoPFA.processing.Exclusions

    .. deprecated:: 0.1.0
       :class:`~geoPFA.processing.Exclusions` instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Exclusions class"""
        warnings.warn(
            "The geopfa3d.processing.Exclusions class is deprecated"
            " and will be removed in a future version."
           " Please use the geoPFA.processing module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class Processing(geoPFA.processing.Processing):
    """Alias for geoPFA.processing.Processing

    .. deprecated:: 0.1.0
       :class:`~geoPFA.processing.Processing` instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Processing class"""
        warnings.warn(
            "The geopfa3d.processing.Processing class is deprecated"
            " and will be removed in a future version."
           " Please use the geoPFA.processing module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
