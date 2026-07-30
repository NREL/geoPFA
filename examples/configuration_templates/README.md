# PFA configuration templates

This directory demonstrates the flat Excel format accepted by
`GeospatialDataWriters.excel_to_pfa_json()` and the nested JSON configuration
consumed by geoPFA.

- `pfa_config_example.xlsx` and `pfa_config_example.json` are a matching,
  populated example.
- `pfa_config_template.xlsx` is a reusable workbook with `Configuration` and
  `Field Guide` sheets.
- `pfa_config_template.json` is the equivalent minimal JSON skeleton.

```python
from geopfa.io import GeospatialDataWriters

pfa = GeospatialDataWriters.excel_to_pfa_json(
    "pfa_config_example.xlsx",
    "pfa_config_example.json",
    sheet_name="Configuration",
)
```

## Field requirements

| Field | Requirement |
| --- | --- |
| `criteria`, `component`, `layer` | Required hierarchy names. Criteria and component cells may be blank after their first row because the converter forward-fills them. These must match the folder and file names in the structure data/ folder  |
| `criteria_weight`, `component_weight`, `layer_weight` | Required numeric voter-veto weights. |
| `pr0` | Required numeric prior probability for each component. This is the estimated propobability that the corresponding component exists anywhere in the study area before any further knowledge (data/evidence layers) is added. |
| `transformation_method` | Required before layer combination. Use `none` for no transformation. Supported methods are `none`, `inverse`, `negate`, `ln`, `hill`, and `valley`. |
| `processing_method` | Recommended for workflow use and documentation |
| `units` | Recommended and used by several processing functions to label output. |
| `data_col` | Required for value-based interpolation, kriging, polygon aggregation, and 3D-to-2D aggregation. Leave blank or use `none` for geometry-only distance/density layers where data values are unnecessary. |
| `crs` | Required for CSV/TEC loading. Shapefiles carry their CRS, though recording it remains useful. |
| `x_col`, `y_col` | Required together for CSV/TEC coordinate columns. Omit both for shapefiles or CSV/TEC files with WKT in a `geometry` column. |
| `z_col` | Required only when 3D coordinates are stored in a separate CSV/TEC column. |
| `z_meas` | Required only when vertical-reference conversion is needed. |
| `notes` | Optional free-text documentation. |

Blank optional Excel cells are omitted from JSON. A case-insensitive `none`
cell is written as JSON `null`. For coordinate fields that do not apply, blank
cells are preferred so those keys are omitted entirely.
