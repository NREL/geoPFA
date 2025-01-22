# -*- coding: utf-8 -*-
"""
Set of various methods to weight and combine data layers for use in PFA.
The methods included in this class are based on those outlined by the PFA
Best Practices Report (Pauling et al. 2023).
"""

import numpy as np
from geoPFA3d.transformation import VoterVetoTransformation3D

class VoterVeto:
    """Class of functions to weight and combine data layers using the voter-veto method.
    This method is based on a generalized linear model and is defined as a best practice
    in the PFA Best Practices Report (Pauling et al. 2023)."""

    @staticmethod
    def get_w0(Pr0):
        """
        Derives w0 value from reference 'favorability', or prior 'favorability', using logit
        function. Is specific to a required component of a resource.
        
        Parameters
        ----------
        Pr0 : float
            Reference 'favorability', or prior 'favorability'.

        Returns
        -------
        w0 : float
        """
        w0 = np.log(Pr0 / (1 - Pr0))
        return w0

    @staticmethod
    def voter(w, z, w0):
        """
        Combine processed, transformed, and scaled 3D data layers into a 'favorability'
        grid for a specific required resource component using a generalized linear model.

        Parameters
        ----------
        w : ndarray
            Array of weights of shape (n, 1, 1, 1), where n is the number of input data layers.
        z : np.array
            Array containing processed, transformed, and scaled 3D data layers rasterized in
            np.arrays - all of which should be on the same grid. Shape (m, x, y, z).
        w0 : float
            Value used to incorporate a reference 'favorability'.

        Returns
        -------
        PrX : np.array
            3D rasterized array of 'favorabilities'. Shape (x, y, z).
        """
        w = w.reshape(-1, 1, 1, 1)  # Reshape weights for broadcasting
        e = -w0 - np.nansum(w * z, axis=0)
        PrX = 1 / (1 + np.exp(e))
        return PrX

    @staticmethod
    def veto(PrXs):
        """
        Combine component 'favorability' grids into a resource 'favorability' map,
        vetoing areas where any one component is not present.

        Parameters
        ----------
        PrXs : np.array
            Array of rasterized 3D 'favorability' arrays for each required component of a resource.

        Returns
        -------
        PrR : np.array
            3D array of 'favorabilities', taking into account all components.
        """
        PrR = np.ones(PrXs[0].shape)
        for PrX in PrXs:
            PrR *= PrX  # Element-wise multiplication for veto
        return PrR

    @staticmethod
    def modified_veto(PrXs, w, veto=True):
        """
        Combine component 'favorability' grids into a resource 'favorability' map,
        optionally vetoing areas where any one component is not present.

        Parameters
        ----------
        PrXs : np.array
            Array of rasterized 3D 'favorability' arrays for each required component or criteria.
        w : np.array
            Array of weights for each component or criteria.
        veto : bool
            Whether or not to veto indices where one component does not exist.

        Returns
        -------
        PrR : np.array
            3D array of 'favorabilities', considering all components.
        """
        PrR = np.zeros(PrXs[0].shape)
        max_PrR = np.ones_like(PrR)

        for i, PrX in enumerate(PrXs):
            max_PrR *= PrX
            PrR += w[i] * PrX
            if veto:
                PrR[PrX == 0] = 0

        # Normalize to maintain valid distribution
        PrR = PrR / np.max(PrR) * np.max(max_PrR)
        return PrR

    @classmethod
    def do_voter_veto(cls, pfa, normalize_method, component_veto=False, criteria_veto=True, normalize=True, norm_to=5):
        """
        Combine individual data layers into a resource 'favorability' map, 
        vetoing areas where any one component is not present (0% 'favorability').

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
        normalize_method : str
            Method to use to normalize data layers.

        Returns
        -------
        pfa : dict
            Config updated with new 'favorability' maps.
        """
        PrRs = []
        w_criteria = []
        for criteria in pfa['criteria']:
            PrXs = []
            w_components = []
            for component in pfa['criteria'][criteria]['components']:
                z = []
                w_layers = []
                Pr0 = pfa['criteria'][criteria]['components'][component]['pr0']
                w0 = cls.get_w0(Pr0)
                for layer in pfa['criteria'][criteria]['components'][component]['layers']:
                    print(layer)
                    map = pfa['criteria'][criteria]['components'][component]['layers'][layer]['model']
                    col = pfa['criteria'][criteria]['components'][component]['layers'][layer]['map_data_col']
                    transformation_method = pfa['criteria'][criteria]['components'][component]['layers'][layer]['transformation_method']

                    map_array = VoterVetoTransformation3D.rasterize_map_3d(gdf=map, col=col)
                    if transformation_method != "none":
                        map_array = VoterVetoTransformation3D.transform(map_array, transformation_method)
                    map_array = VoterVetoTransformation3D.normalize_array(map_array, method=normalize_method)
                    z.append(map_array)
                    w_layers.append(pfa['criteria'][criteria]['components'][component]['layers'][layer]['weight'])

                PrX = cls.voter(np.array(w_layers), np.array(z), w0)
                pfa['criteria'][criteria]['components'][component]['pr'] = VoterVetoTransformation3D.derasterize_map_3d(PrX, gdf_geom=map)
                if normalize:
                    pfa['criteria'][criteria]['components'][component]['pr_norm'] = VoterVetoTransformation3D.normalize_gdf(
                        pfa['criteria'][criteria]['components'][component]['pr'], col='favorability', norm_to=norm_to)
                PrXs.append(PrX)
                w_components.append(pfa['criteria'][criteria]['components'][component]['weight'])

            PrR = cls.modified_veto(PrXs, np.array(w_components), veto=component_veto)
            pfa['criteria'][criteria]['pr'] = VoterVetoTransformation3D.derasterize_map_3d(PrR, gdf_geom=map)
            if normalize:
                pfa['criteria'][criteria]['pr_norm'] = VoterVetoTransformation3D.normalize_gdf(
                    pfa['criteria'][criteria]['pr'], col='favorability', norm_to=norm_to)
            PrRs.append(PrR)
            w_criteria.append(pfa['criteria'][criteria]['weight'])

        PrR = cls.modified_veto(PrRs, np.array(w_criteria), veto=criteria_veto)
        pfa['pr'] = VoterVetoTransformation3D.derasterize_map_3d(PrR, gdf_geom=map)
        if normalize:
            pfa['pr_norm'] = VoterVetoTransformation3D.normalize_gdf(pfa['pr'], col='favorability', norm_to=norm_to)
        return pfa


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
        print('NOT YET IMPLEMENTED')
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
        