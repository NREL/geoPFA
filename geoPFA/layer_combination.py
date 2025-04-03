"""
Set of various methods to weight and combine data layers for use in PFA.
The methods included in this class are based on those outlined by the PFA
Best Practices Report (Pauling et al. 2023).
"""

import warnings

import numpy as np


"""Class of functions to weight and combine data layers using the voter-veto method.
    This method is based on a generalized linear model and is defined as a best practice
    in the PFA Best Practices Report (Pauling et al. 2023)."""


def get_w0(Pr0):
    """
    Derives w0 value from reference 'favorability', or prior 'favorability', using logit
    function.  Is specific to a required component of a resource.

    Parameters
    ----------
    Pr0 : float
        Reference 'favorability', or prior 'favorability'. Can be defined using expert
        opinion or other means. Is specific to a required component of a resource.

    Returns
    -------
    w0 : float
        Value used to incorporate a reference 'favorability' (prior 'favorability') into
        the voter equation (generalized linear model). Is specific to a required
        component of a resource.
    """

    w0 = np.log(Pr0 / (1 - Pr0))
    return w0


class WeightsOfEvidence:
    """Class of functions to weight and combine data layers using the weights of evidence
    method. This method examines multiple layers of evidence, calculates weights for each
    evidential layer based upon the spatial relationships of training points, which are
    located at known geothermal systems, and then produces a posterior 'favorability' raster
    surface and other related statistics. Weights of Evidence is defined as a best
    practice in the PFA Best Practices Report (Pauling et al. 2023)."""

    @classmethod
    def do_weights_of_evidence(cls):
        """
        Combine individual data layers into a resource 'favorability' map,
        using WoE.

        Parameters
        ----------
        pfa : dict

        Returns
        __________
        pfa : dict
        """
        print("NOT YET IMPLEMENTED")
        # # # Example below from: https://ishanjainoffical.medium.com/understanding-weight-of-evidence-woe-with-python-code-cd0df0e4001e
        # # # TODO: Enhance with this article: https://www.sciencedirect.com/science/article/pii/S0377027313002941?via%3Dihub
        # # # and Tularosa Basin reports/papers. Maybe Faulds work??
        # # Calculate WOE for Category 'A' and 'B'
        # category_counts = data['Category'].value_counts()
        # category_counts_pos = data[data['Target'] == 1]['Category'].value_counts()
        # category_counts_neg = data[data['Target'] == 0]['Category'].value_counts()

        # # Calculate WOE
        # woe_pos = np.log((category_counts_pos['A'] / category_counts['A']) / (category_counts_neg['A'] / category_counts['A']))
        # woe_neg = np.log((category_counts_pos['B'] / category_counts['B']) / (category_counts_neg['B'] / category_counts['B']))
        PrR = None
        return PrR


def voter(w, z, w0):
    """
    Combine processed, transformed, and scaled 2D data layers into a 'favorability'
    grid for a specific required resource component using a generalized linear model.

    Parameters
    ----------
    w : ndarray
        Array of weights of shape (n,1), where n is the number of input data layers,
        sorted in order of the input data layers.
    z : np.array
        Array containing processed, transformed, and scaled 2D data layers rasterized in
        np.arrays - all of which should be on the same grid. Shape (m,x,y), where m is
        the number of data layers
    w0 : float
        Value used to incorporate a reference 'favorability' (prior 'favorability') into
        the voter equation (generalized linear model). Is specific to a required
        component of a resource.
    Returns
    -------
    PrX : np.array
        Rasterized array of 'favorabilities' for an individual required resource component being
        present (i.e., heat, fluid, perm, etc.). Shape (x,y)
    """
    if w.ndim > 1:
        warnings.warn(
            "Weights array should be 1D, i.e. one value per data layer.",
            stacklevel=2,
        )
        w = np.atleast_1d(w.squeeze())

    assert w.shape[0] == z.shape[0], (
        "Number of weights must match number of data layers"
    )
    e = -w0 - np.nansum((w * z.T).T, axis=0)
    PrX = 1 / (1 + np.exp(e))
    return PrX
