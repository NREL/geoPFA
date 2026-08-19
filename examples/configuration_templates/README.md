# PFA configuration templates

This directory demonstrates the flat Excel format accepted by
`GeospatialDataWriters.excel_to_pfa_json()` and the nested JSON configuration
consumed by geoPFA.

- `pfa_config_example.xlsx` and `pfa_config_example.json` are a matching,
  populated example.
- `pfa_config_template.xlsx` is a reusable workbook with `Configuration` and
  `Field Guide` sheets.
- `pfa_config_template.json` is the equivalent minimal JSON skeleton.

This is how to use the Excel format: fill in the workbook's `Configuration`
sheet, then convert it to geoPFA's nested JSON format with the function below.

```python
from geopfa.io import GeospatialDataWriters

pfa = GeospatialDataWriters.excel_to_pfa_json(
    "pfa_config_example.xlsx",
    "pfa_config_example.json",
    sheet_name="Configuration",
)
```

## Field requirements

| Field | Definition | Requirement | When used |
| --- | --- | --- | --- |
| `criteria` | Name of top-level PFA criterion | Required | Organizes related components and corresponds to the criterion directory in structured input data folder. Blank continuation cells are forward-filled. |
| `criteria_weight` | Numeric weight assigned to the criterion. | Required numeric | Used when combining multiple criteria to determine their relative contributions. |
| `component` | The name of a resource component within a criterion, such as heat, permeability, or infrastructure. | Required | Every layer belongs to a component. In structured input data, this value should match the component directory name. In Excel, blank continuation cells are forward-filled. |
| `component_weight` | The numeric weight assigned to a component. | Required numeric | Used when combining multiple components in a criterion to determine their relative contributions. |
| `pr0` | The estimated prior probability that a component exists anywhere in the study area before data-layer evidence is applied. | Required numeric | Used by layer combination to establish the component’s prior favorability. |
| `layer` | The name of an individual data or evidence layer. | Required | Identifies the layer in the PFA dictionary. For structured input data, it should match the input file’s name without its extension. |
| `layer_weight` | The numeric weight assigned to an individual layer. | Required numeric | Determines the layer’s relative contribution when layers are combined within a component. |
| `transformation_method` | The mathematical transformation applied to processed layer values before layer combination. | Required for combination | Use none when values should remain unchanged. Supported methods are none, inverse, negate, ln, hill, and valley. Case-insensitive forms of none and JSON null represent no transformation. See geopfa.transformation.transform for info. |
| `processing_method` | A label describing the processing operation intended for the layer. | Recommended | Optionally used by workflow code for processing dispatch or documentation. The Excel-to-JSON converter stores this value but does not execute the processing operation. |
| `units` | The measurement units of the configured data values. | Recommended | For documenting the input layer and is used by several processing functions when labeling output. |
| `data_col` | The name of the input attribute column containing values to process. | Conditional | Required for value-based interpolation, kriging, polygon aggregation, and 3D-to-2D aggregation. It may be blank, omitted, or none for geometry-only operations such as unweighted distance or density calculations. |
| `crs` | The coordinate reference system of the input coordinates, generally expressed as an EPSG code. | Conditional | Required for CSV and TEC inputs that must be converted into spatial geometry. Shapefiles normally include CRS information internally, although documenting it may still be useful. |
| `x_col and y_col` | Name of the input columns containing X and Y or longitude and latitude coordinates. | Conditional pair | Used for coordinate-based CSV or TEC input. Omit for shapefiles or WKT geometry input. |
| `z_col` | Name of the input column containing Z, elevation, or depth coordinates. | Conditional | Required when 3D coordinates are stored in a separate column. |
| `z_meas` | Vertical measurement unit or reference system associated with Z values. | Conditional | Used when vertical-reference conversion is needed. Supported conversion references are m-msl, ft-msl, and epsg:&lt;integer&gt;. |
| `notes` | Free-text information about the layer or its configuration. | Optional | Used for human-readable documentation; not interpreted by the converter. |

Blank optional Excel cells are omitted from JSON when automatically converted. A case-insensitive `none`
cell is written as JSON `null`. For coordinate fields that do not apply, blank
cells are preferred so those keys are omitted entirely.
