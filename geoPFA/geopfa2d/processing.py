# -*- coding: utf-8 -*-
"""
Set of interp_methods to process data from various formats into 2d images.
"""

import warnings

import geopandas as gpd
import pandas as pd
import scipy
import numpy as np
import shapely
from geoPFA.transformation import transform
# from pygem import IDW

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

    @staticmethod
    def get_extent(gdf: gpd.GeoDataFrame):
        """
        Get extent (i.e., bounding box) of a set of points or polygons.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame of Point or Polygon geometry type to get extent from.

        Returns
        -------
        extent : list
            List of length 4 containing the extent (i.e., bounding box) of the gdf,
            in this order: [x_min, y_min, x_max, y_max]

        Notes
        -----
        - If the geometry type is Polygon, the function uses the total bounds of the polygons.
        - If the geometry type is Point, the function uses the coordinates of the points.
        """

        if gdf.geometry.iloc[0].geom_type == "Point":
            xmin = gdf.geometry.x.min()
            xmax = gdf.geometry.x.max()
            ymin = gdf.geometry.y.min()
            ymax = gdf.geometry.y.max()
        elif gdf.geometry.iloc[0].geom_type == "Polygon":
            xmin, ymin, xmax, ymax = gdf.total_bounds
        else:
            raise TypeError(
                "Unsupported geometry type. The GeoDataFrame should contain Points or Polygons."
            )

        extent = [xmin, ymin, xmax, ymax]
        return extent

    @staticmethod
    def set_extent(gdf, extent):
        """Clip a GeoPandas DataFrame to a specified extent.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The GeoDataFrame to be clipped. It can contain any geometry type (e.g., Point, LineString, Polygon).
        extent : list or tuple
            The extent to clip to, specified as [xmin, ymin, xmax, ymax].

        Returns
        -------
        geopandas.GeoDataFrame
            A new GeoDataFrame clipped to the specified extent.
        """
        # Create a bounding box from the extent
        bbox = shapely.geometry.box(extent[0], extent[1], extent[2], extent[3])
        # Clip the GeoDataFrame using the bounding box
        gdf_clipped = gdf.clip(bbox)
        return gdf_clipped


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


class Processing:
    """Class of functions for use in processing data into 2D models"""

    @staticmethod
    def interpolate_points(
        pfa, criteria, component, layer, interp_method, nx, ny, extent=None
    ):
        """Funtion to interpolate, or go from points to a 2D image.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames.
        criteria : str
            Criteria associated with data to interpolate.
        component : str
            Component associated with data to interpolate.
        layer : str
            Layer associated with data to interpolate.
        interp_method : str
            Method to use for interpolation. Can be one of: 'nearest' (KNN), 'linear',
            or 'cubic'.
        nx : int
            Number of points in the x-direction
        ny : int
            Number of points in the y-direction
        extent : list
            List of length 4 containing the extent (i.e., bounding box) of the gdf,
            in this order: [x_min, y_min, x_max, y_max]
        power: int
            Determines how the distance between data points affects the interpolation result.
            *Specifically, it defines the rate at which the influence of a data point decreases
            as the distance from the interpolation location increases.
            *Adjust power parameter accordingly to change how strongly the distance influences the
            interpolation (default is 2).


        Returns
        -------
        pfa : dict
            Updated pfa config which includes interpolation
        """
        gdf = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        data_col = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data_col"]

        if gdf.geometry.type.iloc[0] == "Polygon":
            print(
                "Notice: interpolate_points() function recieved GeoDataFrame with geometry type 'Polygon.' Converting geometry to 'Point' geometry using centroids."
            )
            gdf.geometry = gdf.geometry.centroid

        # Extract coordinates and values from the GeoDataFrame
        x = gdf.geometry.x
        y = gdf.geometry.y
        values = gdf[data_col]

        # Define the grid for interpolation
        if extent is None:
            x_min = min(x)
            x_max = max(x)
            y_min = min(y)
            y_max = max(y)
        else:
            x_min, y_min, x_max, y_max = extent
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        xv, yv = np.meshgrid(x_grid, y_grid)

        ## TODO: Properly implement IDW. The commented out code below does not work
        # Choose interpolation method
        # if interp_method == 'idw':
        #     # IDW interpolation inline
        #     grid = np.zeros_like(xv)
        #     distances = np.sqrt((x[:, None, None] - xv[None, :, :])**2 + (y[:, None, None] - yv[None, :, :])**2)
        #     weights = 1 / np.power(distances, power)
        #     weighted_values = weights * values[:, None, None]
        #     grid = np.sum(weighted_values, axis=0) / np.sum(weights, axis=0)
        # pygem IDW option
        # if interp_method == 'idw':
        # mesh_points = np.array([x.values, y.values, values.values])
        # idw = IDW(power)
        # idw.read_parameters('tests/test_datasets/parameters_idw_cube.prm')
        # new_mesh_points = idw(mesh_points.T)
        # grid = idw(xv.flatten(), yv.flatten()).reshape(nx, ny)
        # else:
        # Otherwise, default to scipy interpolation

        # Clean out any rows with NaNs before interpolation
        valid = ~(x.isna() | y.isna() | values.isna())
        x, y, values = x[valid], y[valid], values[valid]

        grid = scipy.interpolate.griddata(
            (x, y), values, (xv, yv), method=interp_method
        )

        # Create a new GeoDataFrame with the interpolated values
        interpolated_gdf = gpd.GeoDataFrame(
            {"value_interpolated": grid.flatten()},
            geometry=gpd.points_from_xy(xv.flatten(), yv.flatten()),
            crs=gdf.crs,
        )
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = interpolated_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "value_interpolated"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["units"]
        return pfa

    @staticmethod
    def mark_buffer_areas(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nx,
        ny,
        buffer_distance,
        polygon_value,
        buffer_value,
        background_value,
    ):
        """
        Marks grid points within a spatial extent based on their location relative to polygons and buffer areas,
        using vectorized operations with GeoPandas for performance optimization.

        This function creates a grid of points within a given spatial extent and classifies each point as either:
        - Inside a polygon (assigned `polygon_value`)
        - Inside a buffer around a polygon but outside the polygon itself (assigned `buffer_value`)
        - Outside both the polygon and its buffer (assigned `background_value`)

        The classifications are performed using vectorized operations for efficiency, and the results are stored
        in the `pfa` dictionary under the specified `criteria`, `component`, and `layer`.

        Parameters:
        ----------
        pfa : dict
            A dictionary containing spatial data, including:
            - 'criteria': Holds various components and layers, where polygon and buffer data are stored.
        criteria : str
            The key for the specific criterion in the `pfa` dictionary under which the polygon data is stored.
        component : str
            The key for the specific component in the `pfa['criteria']` dictionary where the data layer is located.
        layer : str
            The key for the specific layer within the component where the polygon geometries are stored.
        extent : tuple of float
            The spatial extent in which the grid of points will be created, defined as (x_min, y_min, x_max, y_max).
        nx : int
            The number of points to generate along the x-axis within the extent.
        ny : int
            The number of points to generate along the y-axis within the extent.
        buffer_distance : float
            The distance to create buffer zones around the polygons.
        polygon_value : float
            The classification value to assign to points inside a polygon.
        buffer_value : float
            The classification value to assign to points inside a buffer but outside the polygon.
        background_value : float
            The classification value to assign to points outside both the polygon and buffer areas.

        Returns:
        -------
        dict
            The updated `pfa` dictionary, where the specified layer's grid points are classified based on their
            spatial relationship to the polygons and buffers. The classification is stored in the `model` attribute
            of the layer, with the 'classification' column representing the assigned values.

        Notes:
        ------
        - The function generates a grid of points within the provided extent using `numpy` and classifies the
        points based on spatial relationships to the polygons and buffers.
        - The polygon geometries and their buffers are extracted from the `pfa` dictionary and processed with
        vectorized GeoPandas operations for performance optimization.
        - The results are stored back in the `pfa` dictionary, with the classifications as part of the layer's model data.
        """

        gdf_polygons = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Create a grid of points within the spatial extent
        x_min, y_min, x_max, y_max = extent
        x_points = np.linspace(x_min, x_max, nx)
        y_points = np.linspace(y_min, y_max, ny)
        xv, yv = np.meshgrid(x_points, y_points)
        points = np.c_[xv.ravel(), yv.ravel()]

        # Create a GeoDataFrame from the generated points
        gdf_points = gpd.GeoDataFrame(
            geometry=[shapely.geometry.Point(p) for p in points],
            crs=gdf_polygons.crs,
        )

        # Create buffer areas around polygons
        buffers = gdf_polygons.buffer(buffer_distance)

        # Vectorized operations to classify points
        gdf_points["inside_polygon"] = gdf_points.geometry.apply(
            lambda point: gdf_polygons.contains(point).any()
        )
        gdf_points["inside_buffer"] = gdf_points.geometry.apply(
            lambda point: buffers.contains(point).any()
        )

        # Assign classifications based on the spatial relationship
        gdf_points["classification"] = np.where(
            gdf_points["inside_polygon"],
            polygon_value,
            np.where(
                gdf_points["inside_buffer"], buffer_value, background_value
            ),
        )

        # Drop the temporary columns
        gdf_points = gdf_points.drop(
            columns=["inside_polygon", "inside_buffer"]
        )

        # Update the pfa dictionary with the new model
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = gdf_points
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "classification"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = f"binary ({polygon_value}=polygon, {buffer_value}=buffer, 1=outside)"

        return pfa

    @staticmethod
    def distance_from_polygons(
        pfa, criteria, component, layer, extent, nx, ny
    ):
        """
        Calculate the true distance from a grid of points to a set of polygons,
        using Shapely’s .distance() method on the union of all polygons.
        A point inside any polygon will get a distance of 0.

        Parameters
        ----------
        pfa : dict
            Configuration dictionary specifying relationships between criteria, components,
            and data layers.
        criteria : str
            Criteria associated with Polygon data to calculate distances from.
        component : str
            Component associated with Polygon data to calculate distances from.
        layer : str
            Layer associated with Polygon data to calculate distances from.
        extent : list
            [x_min, y_min, x_max, y_max] bounding box for the grid.
        nx : int
            Number of points in the x-direction.
        ny : int
            Number of points in the y-direction.

        Returns
        -------
        pfa : dict
            Updated pfa config, with a new GeoDataFrame in
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['distance_model']
            that contains the grid points plus a 'distance' column.
        """
        gdf_polygons = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Create a single geometry (union) from all polygons
        polygons_union = gdf_polygons.geometry.unary_union

        # Create a grid of points within the specified extent
        x_min, y_min, x_max, y_max = extent
        x_points = np.linspace(x_min, x_max, nx)
        y_points = np.linspace(y_min, y_max, ny)
        points = [
            shapely.geometry.Point(x, y) for x in x_points for y in y_points
        ]

        # Convert that list of points to a GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(geometry=points, crs=gdf_polygons.crs)

        # Compute distances and store in a new GeoDataFrame
        distances = [pt.distance(polygons_union) for pt in grid_gdf.geometry]
        distance_gdf = grid_gdf.copy()
        distance_gdf["distance"] = distances

        # Save the result in the pfa dictionary (under 'distance_model')
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "distance_model"
        ] = distance_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "distance"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "distance (m)"

        return pfa

    @staticmethod
    def distance_from_lines(pfa, criteria, component, layer, extent, nx, ny):
        """
        Function to calculate the minimum distance from each point in a grid
        to any LineString in the specified GeoDataFrame. The result is stored
        in `pfa` as a new GeoDataFrame with a 'distance' column.

        Parameters
        ----------
        pfa : dict
            Data structure specifying criteria, components, and layers (with
            associated GeoDataFrames). Must include a 'data' GeoDataFrame
            containing LineString geometries.
        criteria : str
            Criteria name.
        component : str
            Component name.
        layer : str
            Layer name (which holds the GeoDataFrame to measure distance from).
        extent : list
            [x_min, y_min, x_max, y_max] bounding box for the grid points.
        nx : int
            Number of points along the x-direction.
        ny : int
            Number of points along the y-direction.

        Returns
        -------
        pfa : dict
            Updated dictionary that now includes the distance model under:
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['model']
        """
        gdf_linestrings = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Create a grid of points within the spatial extent
        x_min, y_min, x_max, y_max = extent
        x_points = np.linspace(x_min, x_max, nx)
        y_points = np.linspace(y_min, y_max, ny)
        points = [
            shapely.geometry.Point(x, y) for x in x_points for y in y_points
        ]

        # Create GeoDataFrame of the grid points
        gdf_points = gpd.GeoDataFrame(geometry=points, crs=gdf_linestrings.crs)

        # Combine all LineString geometries into a single (multi-)geometry
        lines_union = gdf_linestrings.geometry.unary_union

        # Calculate distance from each point to the union of lines
        distances = [pt.distance(lines_union) for pt in gdf_points.geometry]

        # Create a new GeoDataFrame with distances
        gdf_distances = gdf_points.copy()
        gdf_distances["distance"] = distances

        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = gdf_distances
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "distance"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "distance (m)"

        return pfa

    @staticmethod
    def generate_grid_points(extent, nx, ny, crs):
        """
        Generate grid points (centroids) for a regular grid within a given extent.

        Args:
            extent (tuple): A tuple of (xmin, ymin, xmax, ymax) defining the bounding box.
            nx (int): Number of grid cells in the x direction.
            ny (int): Number of grid cells in the y direction.
            crs: Coordinate reference system for the GeoDataFrame.

        Returns:
            GeoDataFrame containing centroids of the grid cells.
        """
        xmin, ymin, xmax, ymax = extent

        # Generate the x and y coordinates for the grid
        x_coords = np.linspace(xmin, xmax, nx)
        y_coords = np.linspace(ymin, ymax, ny)

        # Create centroids by calculating the middle of each grid cell
        centroids = []
        for x in x_coords:
            for y in y_coords:
                centroids.append(shapely.geometry.Point(x, y))

        # Create a GeoDataFrame with the centroids
        gdf_points = gpd.GeoDataFrame(geometry=centroids, crs=crs)

        return gdf_points

    @staticmethod
    def calculate_intersections(gdf_lines):
        """
        Calculate intersection points between line geometries in a GeoDataFrame.

        This function takes a GeoDataFrame of lines (such as faults or other linear features)
        and calculates the intersection points between them. If two lines overlap (instead
        of intersecting at a single point), the midpoint of the overlapping segment is taken
        as the intersection point.

        Parameters:
        -----------
        gdf_lines : geopandas.GeoDataFrame
            A GeoDataFrame containing line geometries (e.g., faults) stored in the 'geometry' column.
            It is assumed that these are LineString geometries.

        Returns:
        --------
        gdf_intersections : geopandas.GeoDataFrame
            A GeoDataFrame containing the points where the input lines intersect.
            If lines overlap, the midpoint of the overlapping segment is included.
            The returned GeoDataFrame will use the same CRS (coordinate reference system)
            as the input GeoDataFrame.
        """
        intersections = []
        # Extract the geometries from the GeoDataFrame
        lines = gdf_lines["geometry"].tolist()

        # Get all unique pairs of lines to check for intersections
        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1 :]:
                if line1.intersects(line2):
                    # Find the intersection point (could be a Point or a LineString if overlapping)
                    intersection = line1.intersection(line2)
                    if isinstance(intersection, shapely.geometry.Point):
                        intersections.append(intersection)
                    elif isinstance(intersection, shapely.geometry.LineString):
                        # If lines overlap, get the mid-point of the overlapping segment
                        midpoint = shapely.geometry.Point(
                            intersection.centroid
                        )
                        intersections.append(midpoint)

        # Return a GeoDataFrame of intersection points
        return gpd.GeoDataFrame(geometry=intersections, crs=gdf_lines.crs)

    @staticmethod
    def vectorized_distance_calculation(
        gdf_points, tree, intersection_tree=None
    ):
        """Calculates the nearest line and intersection distances for each point in a GeoDataFrame using vectorized operations
        and spatial indexing with STRtree.

        This function computes the shortest distance from each point in the `gdf_points` GeoDataFrame to the nearest line
        in the `tree` (a spatial index of line geometries). Optionally, it can also compute the nearest distance to intersections
        (if an `intersection_tree` is provided).

        Parameters:
        ----------
        gdf_points : GeoDataFrame
            A GeoDataFrame containing point geometries for which distances will be calculated.
        tree : STRtree
            A spatial index (STRtree) containing line geometries. This allows for efficient querying of the nearest line
            for each point.
        intersection_tree : STRtree, optional
            An optional spatial index (STRtree) containing intersection geometries. If provided, the function calculates
            the nearest distance to intersections as well. If not provided, the intersection distances are set to infinity.

        Returns:
        -------
        tuple
            A tuple containing two pandas Series:
            - nearest_line_distances: The nearest distance from each point to the nearest line.
            - nearest_intersection_distances: The nearest distance from each point to the nearest intersection (or infinity
            if no intersection tree is provided).

        Notes:
        ------
        - The function uses an inner helper `get_nearest_line_distance` to query the spatial index (`tree`) and calculate the
        distance between a point and its nearest line.
        - If no line is found for a point, the distance is set to infinity (`float('inf')`).
        - When an `intersection_tree` is provided, it computes the minimum distance between a point and the intersection geometries.
        - The function returns `float('inf')` for intersection distances if no intersections are found or if the `intersection_tree`
        is not provided.

        """

        # Function to get the nearest line distance for each point
        def get_nearest_line_distance(point):
            # Query the STRtree for the nearest lines
            nearest_lines = tree.query(point)
            if nearest_lines:  # If lines are found
                # Ensure nearest_lines are geometries, unpacking from STRtree structure if needed
                nearest_line_geom = (
                    nearest_lines[0].geometry
                    if hasattr(nearest_lines[0], "geometry")
                    else nearest_lines[0]
                )
                return point.distance(
                    nearest_line_geom
                )  # Calculate distance between point and line
            else:
                return float("inf")  # If no lines are found, return infinity

        # Calculate the nearest line distance for each point in the GeoDataFrame
        nearest_line_distances = gdf_points.geometry.apply(
            get_nearest_line_distance
        )

        # Handle intersections similarly (if available)
        if intersection_tree:
            nearest_intersection_distances = gdf_points.geometry.apply(
                lambda point: min(
                    [
                        point.distance(geom)
                        for geom in intersection_tree.query(point)
                    ],
                    default=float("inf"),
                )
            )
        else:
            nearest_intersection_distances = pd.Series(
                [float("inf")] * len(gdf_points)
            )

        return nearest_line_distances, nearest_intersection_distances

    @staticmethod
    def distance_from_lines_with_intersections(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nx,
        ny,
        weight_line=1.0,
        weight_intersection=0.5,
    ):
        """
        TODO: Get this to work!!!
        Calculate the weighted distances from points to the nearest lines and intersections,
        and update the provided PFA (Potential Field Analysis) dictionary with the computed distance model.

        This function creates a grid of points over a specified spatial extent, computes the distance
        from each point to the nearest line and nearest line intersection, and then combines the distances
        using specified weights. The result is stored in the 'pfa' dictionary under the specified criteria,
        component, and layer.

        Parameters:
        ----------
        pfa : dict
            The PFA (Potential Field Analysis) dictionary that contains geospatial data for various criteria,
            components, and layers.
        criteria : str
            The criteria key within the PFA dictionary to access the desired data.
        component : str
            The component key within the criteria to access the desired layer data.
        layer : str
            The layer key within the component to access the lines for distance calculation.
        extent : tuple
            A tuple (x_min, y_min, x_max, y_max) specifying the spatial extent for the grid of points.
        nx : int
            Number of points along the x-axis for the grid.
        ny : int
            Number of points along the y-axis for the grid.
        weight_line : float, optional (default=0.7)
            The weight assigned to the distance from the nearest line.
        weight_intersection : float, optional (default=0.3)
            The weight assigned to the distance from the nearest line intersection.

        Returns:
        -------
        pfa : dict
            The updated PFA dictionary with a new distance model stored in the specified layer.
        """
        # Extract lines and intersections from the PFA
        gdf_lines = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]  # Lines data
        gdf_intersections = Processing.calculate_intersections(gdf_lines)

        # Define the CRS (Coordinate Reference System) to match the lines
        crs = gdf_lines.crs
        # Generate the grid of centroids within the given extent
        gdf_points = Processing.generate_grid_points(extent, nx, ny, crs)

        # Build spatial index using STRtree for lines and intersections
        tree = shapely.strtree.STRtree(gdf_lines.geometry)  # STRtree for lines
        intersection_tree = (
            shapely.strtree.STRtree(gdf_intersections.geometry)
            if gdf_intersections is not None
            else None
        )  # STRtree for intersections

        # Perform vectorized distance calculations for both lines and intersections
        line_distances, intersection_distances = (
            Processing.vectorized_distance_calculation(
                gdf_points, tree, intersection_tree
            )
        )

        # Combine the distances with weights (line and intersection distances)
        combined_distances = (
            weight_line * line_distances
            + weight_intersection * intersection_distances
        )

        # Assign combined distances to the points
        gdf_points["distance"] = combined_distances

        # Update the PFA dictionary with the calculated distances
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = gdf_points
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "distance"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "distance (weighted by line and intersection proximity)"

        return pfa

    @staticmethod
    def distance_from_points(pfa, criteria, component, layer, extent, nx, ny):
        """Function to calculate distance from Point objects.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames, particularly the gdf with Point
            geometry to calculate distances from.
        criteria : str
            Criteria associated with Point data to calculate distances from.
        component : str
            Component associated with Point data to calculate distances from.
        layer : str
            Layer associated with Point data to calculate distances from.
        extent : list
            List of length 4 containing the extent (i.e., bounding box) to use to produce the
            distance model. Can be produced using get_extent() function below. Should be in
            this order: [x_min, y_min, x_max, y_max]
        nx : int
            Number of points in the x-direction
        ny : int
            Number of points in the y-direction

        Returns
        -------
        pfa : dict
            Updated pfa config which includes interpolation
        """
        gdf_points = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Ensure we have only simple Point geometries
        points = []
        for geom in gdf_points.geometry:
            if geom.geom_type == "Point":
                points.append(geom)
            elif geom.geom_type == "MultiPoint":
                points.extend(geom.geoms)
            else:
                raise ValueError(
                    f"Unsupported geometry type: {geom.geom_type}"
                )

        if not points:
            raise ValueError(
                "No valid Point geometries found in the GeoDataFrame"
            )

        # Convert points to a numpy array of coordinates, ignoring the z-dimension if it exists
        points_coords = np.array([(point.x, point.y) for point in points])

        # Create a grid of points within the spatial extent
        x_min, y_min, x_max, y_max = extent
        x_points = np.linspace(x_min, x_max, nx)
        y_points = np.linspace(y_min, y_max, ny)
        grid_points = [
            shapely.geometry.Point(x, y) for x in x_points for y in y_points
        ]

        # Create a GeoDataFrame from the generated points
        gdf_grid_points = gpd.GeoDataFrame(
            geometry=grid_points, crs=gdf_points.crs
        )

        # Convert grid points to a numpy array of coordinates
        grid_coords = np.array(
            [(point.x, point.y) for point in gdf_grid_points.geometry]
        )

        # Ensure both coordinate arrays have the same number of dimensions
        assert points_coords.shape[1] == 2, (
            "Points coordinates should have two columns (x, y)"
        )
        assert grid_coords.shape[1] == 2, (
            "Grid coordinates should have two columns (x, y)"
        )

        # Calculate distances between the grid points and the specified points
        distances = scipy.spatial.distance.cdist(grid_coords, points_coords)

        # Store minimum distances in a GeoDataFrame
        gdf_distances = gdf_grid_points.copy()
        gdf_distances["distance"] = distances.min(axis=1)

        # Update pfa dictionary with the distance model
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = gdf_distances
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "distance"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "distance (m)"

        return pfa

    @staticmethod
    def weighted_distance_from_points(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nx,
        ny,
        alpha=1000.0,
        transform_method="none",
        weight_points=True,
        weight_min=1.0,
        weight_max=2.0,
    ):
        """
        Compute a "weighted distance score" from a set of points,
        where each point has a data value in `data_col`. That data
        value is normalized/transformed to produce a point weight (0-1),
        then each cell's score is sum_i [ w_i * exp(-dist_i / alpha ) ].

        Parameters
        ----------
        pfa : dict
            The PFA dictionary, which must contain:
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['data']
            ...a GeoDataFrame of points, each with a relevant 'data_col'.
        criteria : str
            E.g. 'geologic'
        component : str
            E.g. 'fluid'
        layer : str
            E.g. 'hot_springs'
        extent : list of float
            [x_min, y_min, x_max, y_max] bounding box for the grid
        nx, ny : int
            Number of points along x and y in the final grid
        alpha : float, default=1000
            Decay constant for distance (in meters).
        transform_method: str, default="none"
            How to transform point data into proximity favorability via transformation
            function (e.g."none", "negate", "inverse", etc.).
        weight_points : bool, default=True
            Whether to apply data-based point weighting or not.
        weight_min : float, default=1.0
            Minimum weight for a transformed point.
        weight_max : float, default=2.0
            Maximum weight for a transformed point.

        Returns
        -------
        pfa : dict
            The updated dictionary, storing a new GeoDataFrame with column
            "weighted_point_score" in:
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['model'].
        """

        def normalize_point_weights(values, out_min=1.0, out_max=5.0):
            # Scale a 1D array of values to a user-defined positive range [out_min..out_max].
            vals = values.copy()
            valid_mask = ~np.isnan(vals)
            valid_data = vals[valid_mask]
            if valid_data.size == 0:
                scaled = np.full_like(vals, out_min)
                return scaled
            vmin = valid_data.min()
            vmax = valid_data.max()
            rng = vmax - vmin
            if rng == 0:
                midpoint = (out_min + out_max) / 2.0
                scaled = np.full_like(vals, midpoint)
                return scaled

            # Normal minmax to [0..1] then scale to [out_min..out_max]
            mm = (valid_data - vmin) / rng
            scaled_valid = mm * (out_max - out_min) + out_min

            scaled = np.full_like(vals, np.nan)
            scaled[valid_mask] = scaled_valid

            return scaled

        layer_dict = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]
        data_col = layer_dict.get("data_col", None)
        gdf_points = layer_dict["data"].copy()

        if data_col is None:
            print("No data_col => defaulting all weights=1.0")
            gdf_points["temp_val"] = 1.0
            data_col = "temp_val"

        # Collect geometry + data
        point_list, data_list = [], []
        for geom, val in zip(gdf_points.geometry, gdf_points[data_col]):
            if geom.geom_type == "Point":
                point_list.append(geom)
                data_list.append(val)
            elif geom.geom_type == "MultiPoint":
                for subg in geom.geoms:
                    point_list.append(subg)
                    data_list.append(val)
            else:
                raise ValueError(f"Unsupported geometry: {geom.geom_type}")

        if not point_list:
            print("No valid points found.")
            return pfa

        # Convert data_list to 1D NumPy
        data_array_1d = np.array(data_list, dtype=float)

        # Transform data values into weight
        arr_2d = data_array_1d.reshape(-1, 1)
        arr_2d_trans = transform(arr_2d, transform_method)  # e.g. "negate"
        arr_2d_trans = np.nan_to_num(arr_2d_trans, nan=0)  # handle NaNs
        trans_1d = arr_2d_trans.ravel()

        # scale them to [out_min..out_max]
        if not weight_points:
            print("User not weighting points. Defaulting all weights=1.0")
            out_min = out_max = 1.0  # all points get equal weight
        else:
            out_min = weight_min
            out_max = weight_max

        # Scale once with the chosen range
        weights_1d = normalize_point_weights(
            trans_1d,
            out_min=out_min,
            out_max=out_max,
        )

        # Create grid, compute distances
        x_min, y_min, x_max, y_max = extent
        x_coords = np.linspace(x_min, x_max, nx)
        y_coords = np.linspace(y_min, y_max, ny)
        grid_points = [
            shapely.geometry.Point(x, y) for x in x_coords for y in y_coords
        ]

        gdf_grid = gpd.GeoDataFrame(geometry=grid_points, crs=gdf_points.crs)
        point_coords = np.array([(p.x, p.y) for p in point_list])
        grid_coords = np.array([(pt.x, pt.y) for pt in gdf_grid.geometry])

        dist_matrix = scipy.spatial.distance.cdist(
            grid_coords, point_coords
        )  # shape=(nx*ny, n_points)

        # Weighted sum => sum_i [ w_i * exp(-dist_ij / alpha) ]
        decays = np.exp(-dist_matrix / alpha)
        weighted = (
            decays * weights_1d
        )  # broadcasting => shape=(n_grid, n_points)
        score_1d = np.sum(weighted, axis=1)

        gdf_grid["weighted_point_score"] = score_1d
        layer_dict["model"] = gdf_grid
        layer_dict["model_data_col"] = "weighted_point_score"
        layer_dict["model_units"] = "score (0-∞)"

        return pfa

    @staticmethod
    def point_density(
        pfa, criteria, component, layer, extent, cell_size, nx, ny
    ):
        """Calculate point density within a specified extent and return as a GeoDataFrame.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames, particularly the gdf with Point
            geometry to calculate distances from.
        criteria : str
            Criteria associated with Point data to calculate distances from.
        component : str
            Component associated with Point data to calculate distances from.
        layer : str
            Layer associated with Point data to calculate distances from.
        extent : list
            List of length 4 containing the extent (i.e., bounding box) to use to produce the
            distance model. Can be produced using get_extent() function below. Should be in
            this order: [x_min, y_min, x_max, y_max]
        - cell_size : float
            Size of each cell in the grid used for density calculation.
            Example Cell Sizes for EPSG:3857
                High Detail: 50 meters
                Moderate Detail: 100 meters
                Lower Detail: 500 meters - 1 kilometer
        nx : int
            Number of cells in the x direction for the output grid.
        ny : int
            Number of cells in the y direction for the output grid.

        Returns
        -------
        density_gdf : GeoDataFrame
            GeoDataFrame populated with point density within the specified extent, with
            point geometry.
        """

        # Extract GeoDataFrame from pfa dictionary
        gdf = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]

        # Define the extent
        xmin, ymin, xmax, ymax = extent

        # Create grid for density calculation using cell_size
        x_cells = int((xmax - xmin) / cell_size)
        y_cells = int((ymax - ymin) / cell_size)

        density_grid = np.zeros((y_cells, x_cells))

        for geom in gdf.geometry:
            # Check if geometry is Point or MultiPoint
            if isinstance(geom, shapely.geometry.Point):
                points = [geom]
            elif isinstance(geom, shapely.geometry.MultiPoint):
                points = geom.geoms
            else:
                points = []

            # Iterate through each point in the geometry
            for point in points:
                # Ignore the z component by only considering the x and y coordinates
                if (xmin <= point.x <= xmax) and (ymin <= point.y <= ymax):
                    x_index = int((point.x - xmin) / cell_size)
                    y_index = int((point.y - ymin) / cell_size)
                    if (x_index > x_cells - 1) | (y_index > y_cells - 1):
                        continue
                    density_grid[y_index, x_index] += 1

        # Calculate the size of each cell in the high-resolution grid
        x_step = (xmax - xmin) / nx
        y_step = (ymax - ymin) / ny

        # Initialize high-resolution density grid
        high_res_density = np.zeros((ny, nx))

        # Fill high-resolution density grid with aggregated values from density_grid
        for i in range(x_cells):
            for j in range(y_cells):
                value = density_grid[j, i]
                x_start = int(i * cell_size / x_step)
                y_start = int(j * cell_size / y_step)
                x_end = int((i + 1) * cell_size / x_step)
                y_end = int((j + 1) * cell_size / y_step)
                high_res_density[y_start:y_end, x_start:x_end] = value

        # Create points for the high-resolution grid
        points = [
            shapely.geometry.Point(
                xmin + (i + 0.5) * x_step, ymin + (j + 0.5) * y_step
            )
            for j in range(ny)
            for i in range(nx)
        ]

        # Flatten the high-resolution density grid to match the points
        densities = high_res_density.flatten()

        # Ensure the lengths match
        assert len(points) == len(densities), (
            f"Points length {len(points)} and densities length {len(densities)} do not match."
        )

        # Create the GeoDataFrame
        density_gdf = gpd.GeoDataFrame(
            {"geometry": points, "density": densities}
        )
        density_gdf = density_gdf.set_crs(gdf.crs)

        # Update the pfa dictionary with the new layer
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = density_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "density"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "density per" + str(cell_size) + " m^2"

        return pfa

    @staticmethod
    def polygon_density(
        pfa, criteria, component, layer, extent, cell_size, nx, ny
    ):
        """Calculate polygon density within a specified extent and return as a GeoDataFrame.

        Parameters
        ----------
        pfa : dict
            Configuration specifying criteria, components, and data layers' relationship.
        criteria : str
            Criteria associated with Polygon data to calculate density from.
        component : str
            Component associated with Polygon data to calculate density from.
        layer : str
            Layer associated with Polygon data to calculate density from.
        extent : list
            List of length 4 containing the extent [x_min, y_min, x_max, y_max].
        cell_size : float
            Size of each cell in the grid used for density calculation.
        nx : int
            Number of cells in the x direction for the output grid.
        ny : int
            Number of cells in the y direction for the output grid.

        Returns
        -------
        pfa : dict
            Updated pfa config which includes polygon density as a GeoDataFrame.
        """
        # Extract GeoDataFrame containing polygons
        gdf_polygons = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Define the extent
        xmin, ymin, xmax, ymax = extent

        # Create grid for density calculation using cell_size
        x_cells = int((xmax - xmin) / cell_size)
        y_cells = int((ymax - ymin) / cell_size)

        density_grid = np.zeros((y_cells, x_cells))

        for geom in gdf_polygons.geometry:
            # Check if geometry is Polygon or MultiPolygon
            if isinstance(geom, shapely.geometry.Polygon):
                polygons = [geom]
            elif isinstance(geom, shapely.geometry.MultiPolygon):
                polygons = geom.geoms
            else:
                polygons = []

            # Iterate through each polygon in the geometry
            for polygon in polygons:
                # Get bounding box of polygon
                bbox = polygon.bounds
                xmin_poly, ymin_poly, xmax_poly, ymax_poly = bbox

                # Calculate intersecting grid cells
                x_start = max(0, int((xmin_poly - xmin) / cell_size))
                x_end = min(x_cells, int((xmax_poly - xmin) / cell_size)) + 1
                y_start = max(0, int((ymin_poly - ymin) / cell_size))
                y_end = min(y_cells, int((ymax_poly - ymin) / cell_size)) + 1

                # Increment density count for intersecting grid cells
                density_grid[y_start:y_end, x_start:x_end] += 1

        # Calculate the size of each cell in the high-resolution grid
        x_step = (xmax - xmin) / nx
        y_step = (ymax - ymin) / ny

        # Initialize high-resolution density grid
        high_res_density = np.zeros((ny, nx))

        # Fill high-resolution density grid with aggregated values from density_grid
        for i in range(x_cells):
            for j in range(y_cells):
                value = density_grid[j, i]
                x_start = int(i * cell_size / x_step)
                y_start = int(j * cell_size / y_step)
                x_end = int((i + 1) * cell_size / x_step)
                y_end = int((j + 1) * cell_size / y_step)
                high_res_density[y_start:y_end, x_start:x_end] = value

        # Create points for the high-resolution grid
        points = [
            shapely.geometry.Point(
                xmin + (i + 0.5) * x_step, ymin + (j + 0.5) * y_step
            )
            for j in range(ny)
            for i in range(nx)
        ]

        # Flatten the high-resolution density grid to match the points
        densities = high_res_density.flatten()

        # Ensure the lengths match
        assert len(points) == len(densities), (
            f"Points length {len(points)} and densities length {len(densities)} do not match."
        )

        # Create the GeoDataFrame
        density_gdf = gpd.GeoDataFrame(
            {"geometry": points, "density": densities}
        )
        density_gdf = density_gdf.set_crs(gdf_polygons.crs)

        # Update the pfa dictionary with the new layer
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = density_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "density"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "density per " + str(cell_size) + " m^2"

        return pfa

    def convert_3d_to_2d(pfa, criteria, component, layer):
        """
        Convert a 3D GeoDataFrame layer to a 2D representation by collapsing along the Z-dimension.

        This function processes a 3D layer stored in the `pfa` dictionary structure, filtering out empty geometries,
        sorting the data by X, Y, and Z coordinates, and aggregating data by grouping on X and Y. The resulting
        2D GeoDataFrame replaces the original 3D data in the `pfa` dictionary.

        Parameters:
        ----------
        pfa : dict
            A nested dictionary containing geospatial data organized by criteria, components, and layers.
            The specific layer to be processed is accessed using the given `criteria`, `component`, and `layer` keys.
        criteria : str
            The key in `pfa['criteria']` identifying the specific criteria containing the 3D layer.
        component : str
            The key in `pfa['criteria'][criteria]['components']` identifying the specific component containing the 3D layer.
        layer : str
            The key in `pfa['criteria'][criteria]['components'][component]['layers']` identifying the specific 3D layer.

        Returns:
        -------
        dict
            The updated `pfa` dictionary with the 3D layer converted to a 2D representation. The resulting 2D GeoDataFrame
            is stored in `pfa['criteria'][criteria]['components'][component]['layers'][layer]['data']`.

        Raises:
        ------
        ValueError
            If the `geometry` column in the GeoDataFrame is not of type `Point` or contains invalid geometries.

        Notes:
        ------
        - Empty geometries (e.g., `POINT EMPTY`) are filtered out before processing.
        - Sorting is performed by X, Y, and Z coordinates. Empty points are handled gracefully.
        - Aggregation sums the values in the specified data column while keeping a representative geometry for each (X, Y) pair.
        - The units of the data are updated to reflect the aggregation performed.
        """
        # Extract GeoDataFrame containing 3D data
        gdf_3d = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        col = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data_col"]
        units = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["units"]
        crs = gdf_3d.crs
        # Ensure the geometry column is of Point type
        if not gdf_3d.geometry.geom_type.isin(["Point"]).all():
            raise ValueError("Geometry column must contain Point geometries.")

        # Filter out empty geometries
        gdf_3d = gdf_3d[~gdf_3d.geometry.is_empty].reset_index(drop=True)

        # Sort geometries by X, Y, Z, handling empty points gracefully
        gdf_3d = gdf_3d.sort_values(
            by=["geometry"],
            key=lambda col: col.apply(
                lambda geom: (geom.x, geom.y, geom.z)
                if not geom.is_empty
                else (float("nan"), float("nan"), float("nan"))
            ),
        ).reset_index(drop=True)

        # Step 1: Aggregate by unique (X, Y, Z) points
        gdf_3d = gdf_3d.groupby(
            gdf_3d.geometry.apply(lambda geom: (geom.x, geom.y, geom.z)),
            as_index=False,
        ).agg(
            {
                "geometry": "first",  # Keep the representative geometry
                col: "sum",  # Sum the data column for unique (X, Y, Z)
            }
        )
        gdf_3d = gdf_3d.set_geometry("geometry")
        gdf_3d.set_crs(crs, inplace=True)
        # Step 2: Collapse Z dimension to (X, Y)
        gdf_2d = gdf_3d.groupby(
            gdf_3d.geometry.apply(lambda geom: (geom.x, geom.y)),
            as_index=False,
        ).agg(
            {
                "geometry": "first",  # Keep the representative geometry
                col: "sum",  # Sum across Z for unique (X, Y)
            }
        )
        gdf_2d = gdf_2d.set_geometry("geometry")
        gdf_2d.set_crs(crs, inplace=True)

        # Update the pfa dictionary with the new layer
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "data"
        ] = gdf_2d
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "units"
        ] = "summed " + units
        return pfa

    def extract_fault_traces_slice(
        pfa, criteria, component, layer, min_z=None, tolerance=1e6
    ):
        """
        Extracts a 2D representation of faults by taking a slice at the bottom of the model.

        Parameters:
        ----------
        gdf : GeoDataFrame
            A GeoDataFrame containing 3D fault geometries (Point geometries).
        fault_id_col : str
            The column name representing fault IDs to group points into separate faults.
        bottom_z_threshold : float
            A Z-value threshold used to select the bottom slice of the model. Points with
            Z-values within this threshold from the minimum Z will be included.

        Returns:
        -------
        GeoDataFrame
            A GeoDataFrame with LineString geometries representing the bottom traces of each fault.
        """
        # Extract GeoDataFrame containing 3D data
        gdf_3d = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        fault_id_col = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["id_col"]

        # Ensure the geometry column contains 3D Points
        if not all(gdf_3d.geometry.geom_type == "Point"):
            raise ValueError("All geometries must be Point geometries.")
        if not all(gdf_3d.geometry.apply(lambda geom: geom.has_z)):
            raise ValueError("All geometries must be 3D (with Z-coordinates).")

        if min_z is None:
            # Find the minimum Z value in the dataset
            min_z = gdf_3d.geometry.apply(lambda geom: geom.z).min()
            print(min_z)

        # Filter points at exactly min_z (or within a small tolerance)
        bottom_points = gdf_3d[
            gdf_3d.geometry.apply(
                lambda geom: abs(geom.z - min_z) <= tolerance
            )
        ]

        # Extract fault traces (LineStrings) for each unique fault ID
        fault_traces = []
        for fault_id, group in bottom_points.groupby(fault_id_col):
            # Ensure there are enough points to create a LineString
            if len(group) > 1:
                # Sort points by X and Y for proper LineString creation
                group = group.sort_values(
                    by=["geometry"],
                    key=lambda col: col.apply(lambda geom: (geom.x, geom.y)),
                )
                trace = shapely.geometry.LineString(
                    group.geometry.apply(
                        lambda geom: (geom.x, geom.y, geom.z)
                    ).tolist()
                )
                fault_traces.append({"fault_id": fault_id, "geometry": trace})

        if not fault_traces:
            raise ValueError(
                "No valid fault traces found at the minimum Z value."
            )

        # Create a GeoDataFrame from the fault traces
        fault_traces_gdf = gpd.GeoDataFrame(
            fault_traces, geometry="geometry", crs=gdf_3d.crs
        )

        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "data"
        ] = fault_traces_gdf
        return pfa

    def process_faults(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nx,
        ny,
        alpha_fault=1000.0,
        alpha_intersection=500.0,
        weight_fault=0.7,
        weight_intersection=0.3,
        use_intersections=True,
    ):
        """
        Process faults by:
        1) Computing distance from faults (via distance_from_lines).
        2) (Optional) Finding fault intersections, computing distance to those intersections.
        3) Applying exponential decays to fault distance and (optionally) intersection distance.
        4) Combining via a weighted sum to produce a final favorability.

        Parameters
        ----------
        pfa : dict
            The PFA dictionary containing fault data in:
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['data']
            Must have valid line geometry for distance_from_lines to work.
        criteria : str
            E.g. 'geologic'
        component : str
            E.g. 'fluid'
        layer : str
            E.g. 'intersecting_faults'
        extent : list
            [x_min, y_min, x_max, y_max] bounding box
        nx, ny : int
            Grid dimensions in x and y directions
        alpha_fault : float
            Decay constant for fault distance
        alpha_intersection : float
            Decay constant for intersection distance
        weight_fault : float
            Weight for the fault distance favorability
        weight_intersection : float
            Weight for the intersection distance favorability
        use_intersections : bool
            If False, ignore intersection distances (treat them as zero).
            If True, do the intersection calculation + decay.

        Returns
        -------
        pfa : dict
            Updated dictionary with a new 'model' containing 'favorability'
            under pfa['criteria'][criteria]['components'][component]['layers'][layer].
        """
        # Compute distance from faults
        pfa = Processing.distance_from_lines(
            pfa, criteria, component, layer, extent, nx, ny
        )
        fault_layer_dict = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]
        fault_model_gdf = fault_layer_dict.get("model", None)
        dist_col = fault_layer_dict.get("model_data_col", None)

        if fault_model_gdf is None or dist_col != "distance":
            print(
                "ERROR: distance_from_lines did not produce a valid 'model' with 'distance' column."
            )
            return pfa

        fault_dist_1d = fault_model_gdf["distance"].to_numpy()
        try:
            fault_dist_2d = fault_dist_1d.reshape((ny, nx))
        except ValueError as e:
            print("ERROR reshaping fault distance array:", e)
            return pfa

        # find fault intersections
        gdf_lines = fault_layer_dict["data"]
        if not use_intersections or gdf_lines.empty:
            # skip intersection logic
            intersect_score_2d = np.zeros((ny, nx))
            if not use_intersections:
                print(
                    "use_intersections=False => intersection favorability is zero."
                )
            else:
                print(
                    "WARNING: No line geometry found, skipping intersections."
                )
        else:
            gdf_intersections = Processing.calculate_intersections(gdf_lines)
            if gdf_intersections.empty:
                print(
                    "No fault intersections found => intersection favorability is zero."
                )
                intersect_favor_2d = np.zeros((ny, nx))
            else:
                intersections_union = gdf_intersections.geometry.union_all()
                grid_points = Processing.generate_grid_points(
                    extent, nx, ny, crs=gdf_lines.crs
                )
                dist_intersect_1d = np.array(
                    [
                        pt.distance(intersections_union)
                        for pt in grid_points.geometry
                    ]
                )
                try:
                    dist_intersect_2d = dist_intersect_1d.reshape((ny, nx))
                except ValueError as e:
                    print("ERROR reshaping intersection distances:", e)
                    dist_intersect_2d = np.zeros((ny, nx))
                # Exponential decay for intersection distance
                intersect_score_2d = np.exp(
                    -dist_intersect_2d / alpha_intersection
                )

        # Exponential decay for fault distances
        fault_score_2d = np.exp(-fault_dist_2d / alpha_fault)

        # Combine with weights in a weighted sum
        composite_2d = (
            weight_fault * fault_score_2d
            + weight_intersection * intersect_score_2d
        )

        composite_1d = composite_2d.flatten()
        fault_model_gdf = fault_model_gdf.copy()
        fault_model_gdf["fault score"] = composite_1d

        fault_layer_dict["model"] = fault_model_gdf
        fault_layer_dict["model_data_col"] = "fault score"
        fault_layer_dict["model_units"] = "fault score (0-1)"

        return pfa
