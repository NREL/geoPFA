
[![PyPi](https://badge.fury.io/py/geoPFA.svg)](https://pypi.org/project/geoPFA/)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17316283.svg)](https://doi.org/10.5281/zenodo.17316283)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
![SWR](https://img.shields.io/badge/SWR--25--73_-blue?label=NLR)

[![PythonV](https://img.shields.io/pypi/pyversions/geoPFA.svg)](https://pypi.org/project/geoPFA/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![codecov](https://codecov.io/gh/NatLabRockies/geoPFA/graph/badge.svg?token=W2COSPBX4Z)](https://codecov.io/gh/NatLabRockies/geoPFA)

# Geothermal PFA

geoPFA is an open-source Python library for conducting Play Fairway Analysis
(PFA) in 2D and 3D, designed to reduce exploration risk by integrating surface
and subsurface considerations into a single, transparent workflow. Built around
NLR’s Geothermal PFA Best Practices and aligned with FAIR software principles,
geoPFA provides modular, extensible tools for cleaning, processing, weighting,
and combining diverse datasets into quantitative favorability maps. These
datasets can include geological, geophysical, geochemical, and
thermo-hydro-mechanical-chemical simulation results, as well as surface-level
factors such as energy demand, transmission access, and natural hazard
exposure.

The framework is fully customizable, enabling users to define criteria,
components, and indicators for any geothermal resource type—from
low-temperature and conventional hydrothermal to superhot systems—and to extend
the methodology to other subsurface applications if desired. geoPFA supports multiple data
processing approaches, including interpolation, density mapping, distance-based
scoring, extrapolation, and thermal modeling, while allowing integration of
expert-derived weightings or analytical hierarchy methods.

geoPFA has been successfully demonstrated in diverse contexts: a 3D PFA for
the Nesjavellir field in Iceland, where results aligned with known subsurface
conditions and guided scenario-based development strategies (Taverna et al.,
2025); and 2D PFAs of the Denver Basin and Alaska for lower-enthalpy geothermal
with greater emphasis on surface constraints (Davalos-Elizondo et al., 2024;
in work). By making advanced exploration workflows reproducible, transparent,
and openly accessible, geoPFA enables research teams, developers, and agencies
to make better-informed decisions through reducing time required for developing
workflows, allowing more time to be spent on feature engineering and interpretation
of results.

<!-- start-license -->
# NOTICE

Copyright © 2025 Alliance for Energy Innovation, LLC

This work was authored by the National Laboratory of the Rockies for the 
U.S. Department of Energy (DOE), operated under Contract No. DE-AC36-08GO28308. 
Funding provided by Department of Energy Hydrocarbons and Geothermal Energy Office, 
Office of Geothermal. The views expressed in the article do not necessarily represent 
the views of the DOE or the U.S. Government. The U.S. Government retains and the 
publisher, by accepting the article for publication, acknowledges that the U.S. 
Government retains a nonexclusive, paid-up, irrevocable, worldwide license to 
publish or reproduce the published form of this work, or allow others to do so, 
for U.S. Government purposes. 
