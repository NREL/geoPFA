# -*- coding: utf-8 -*-
"""
Set of methods to read in data in various formats.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
import contextily as ctx
import shapely

class GeospatialDataPlotters:
    """Class of functions to plot geospatial data in various formats"""
    @staticmethod
    def geo_plot(gdf,col,units,title,area_outline=None,overlay=None,xlabel='default',ylabel='default',\
        cmap='jet',xlim=None,ylim=None,extent=None,basemap=False,markersize=15,figsize=(10, 10),vmin=None,vmax=None):
        """Plots data using gdf.plot(). Preserves geometry, but does not look 
        smoothe.
        
        Parameters
        ----------
        gdf : pandas geodataframe
            Geodataframe containing data to plot, including a geometry column and crs.
        col : str
            Name of column containing data value to plot, if applicable.
        units : str
            Units of data to plot.
        title : str
            Title to add to plot.
        area_outline : geodataframe
            Optional, Geodataframe contatining outline of area to overlay on plot.
        overlay : geodataframe
            Optional, Geodataframe containing data locations to plot over map data.
        xlabel, ylabel : str
            Optional, label for x-axis and y-axis.
        cmap : str
            Optional, colormap to use instead of the default 'jet'.
        xlim, ylim : tuple
            Optional, limits to use for x and y axes.
        extent : list
            List of length 4 containing the extent (i.e., bounding box) to use in 
            lieau of xlim and ylim, in this order: [x_min, y_min, x_max, y_max].
        basemap : bool
            Option to add a basemap, defaults to False.
        markersize : int
            Option to specify marker size to use in plot. Defaults to 15.
        figsize : tuple
            Option to specify figure size. Defaults to (10,10).
        vmin, vmax : float
            Optional minimum and maximum values to include in colorbar. If not provided,
            will use min and max value of data in the column to plot.

        """
        fig, ax = plt.subplots(figsize=figsize)
        if col is None or str(col).lower() == "none":
            gdf.plot(ax=ax)
        else:
            if vmin is None:
                norm=plt.Normalize(vmin=gdf[col].min(), vmax=gdf[col].max())
            else:
                norm=plt.Normalize(vmin=vmin, vmax=vmax)
            gdf.plot(ax=ax, marker='s', markersize=markersize,
                    column=col,cmap=cmap,norm=norm,legend=False)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(units)
        if area_outline is not None:
            area_outline.boundary.plot(ax=ax,color='black')
        if overlay is not None:
            overlay.plot(ax=ax,color='gray',markersize=3,alpha=0.5)
        if xlabel == 'default':
            xlabel = gdf.crs.axis_info[1].name
        if ylabel == 'default':
            ylabel = gdf.crs.axis_info[0].name
        if basemap is True:
            ctx.add_basemap(ax)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        elif extent is not None:
            plt.xlim(extent[0],extent[2])
            plt.ylim(extent[1],extent[3])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def geo_plot_3d(
        gdf, col, units, title, area_outline=None, overlay=None, xlabel='default', ylabel='default', zlabel='Z-axis',
        cmap='jet', xlim=None, ylim=None, zlim=None, extent=None, markersize=15, figsize=(10, 10),
        vmin=None, vmax=None, filter_threshold=None, x_slice=None, y_slice=None, z_slice=None
    ):
        """
        Plots 3D geospatial data using matplotlib's 3D plotting capabilities with optional filtering.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        gdf_copy = gdf.copy()

        # Normalize color data
        if col is not None and str(col).lower() != "none":
            if vmin is None:
                vmin = gdf_copy[col].min()
            if vmax is None:
                vmax = gdf_copy[col].max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(cmap)
            colors = cmap(norm(gdf_copy[col]))
        else:
            colors = 'blue'

        # Apply slicing for x, y, and z axes
        if x_slice is not None:
            gdf_copy = gdf_copy[gdf_copy.geometry.apply(lambda geom: geom.coords[0][0] <= x_slice)]
        if y_slice is not None:
            gdf_copy = gdf_copy[gdf_copy.geometry.apply(lambda geom: geom.coords[0][1] <= y_slice)]
        if z_slice is not None:
            gdf_copy = gdf_copy[gdf_copy.geometry.apply(lambda geom: geom.coords[0][2] <= z_slice)]

        # Apply filtering if filter_threshold is set
        if filter_threshold is not None and col != "None":
            mask = gdf_copy[col] >= filter_threshold
            gdf_filtered = gdf_copy[mask]
        else:
            gdf_filtered = gdf_copy

        # Check if gdf_filtered is empty
        if gdf_filtered.empty:
            print("No data to plot after filtering and slicing.")
            return

        # Update colors for filtered data
        if col is not None and str(col).lower() != "none":
            filtered_colors = cmap(norm(gdf_filtered[col]))
        else:
            filtered_colors = colors

        # Extract 3D coordinates from filtered geometry
        if gdf_filtered.geometry.iloc[0].geom_type == 'Point':
            xs, ys, zs = zip(*[geom.coords[0] for geom in gdf_filtered.geometry])
            ax.scatter(
                xs,
                ys,
                zs,
                c=filtered_colors if (col is not None and str(col).lower() != "none") else None,
                s=markersize,
            )

        # Plot polygons as filled surfaces
        elif gdf_filtered.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
            for geom in gdf_filtered.geometry:
                if geom.geom_type == 'Polygon':
                    rings = [geom.exterior] + list(geom.interiors)
                elif geom.geom_type == 'MultiPolygon':
                    rings = [ring for polygon in geom.geoms for ring in [polygon.exterior] + list(polygon.interiors)]

                for ring in rings:
                    vertices = [(coord[0], coord[1], coord[2] if len(coord) == 3 else 0) for coord in ring.coords]
                    poly = Poly3DCollection([vertices], alpha=0.5, edgecolor='grey', facecolor='lightblue')
                    ax.add_collection3d(poly)

        # Add colorbar if a column is specified
        if col is not None and str(col).lower() != "none":
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = fig.colorbar(sm, ax=ax, pad=0.1)
            cbar.set_label(units)

        # Plot overlay if provided
        if overlay is not None:
            overlay_xs, overlay_ys, overlay_zs = zip(*[geom.coords[0] for geom in overlay.geometry])
            ax.scatter(overlay_xs, overlay_ys, overlay_zs, color='gray', s=5, alpha=0.5)

        # Plot area outline if provided
        if area_outline is not None:
            # Compute zmax across all geometries in the gdf_copy
            if gdf_copy.geometry.iloc[0].geom_type == 'Point':
                zmax = max(geom.z for geom in gdf_copy.geometry)
            elif gdf_copy.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
                zmax = max(
                    max(coord[2] for coord in ring.coords if len(coord) == 3)
                    for geom in gdf_copy.geometry
                    for ring in ([geom.exterior] + list(geom.interiors))
                )
            else:
                zmax = 0  # Default if geometry type is not supported

            # Add the outline at zmax + 1
            for geom in area_outline.geometry:
                outline_xs, outline_ys = zip(*[(coord[0], coord[1]) for coord in geom.exterior.coords])
                outline_zs = [zmax + 1] * len(outline_xs)
                ax.plot(outline_xs, outline_ys, outline_zs, color='black')

        # Set axis labels
        if xlabel == 'default':
            xlabel = gdf_copy.crs.axis_info[1].name if gdf_copy.crs else 'X-axis'
        if ylabel == 'default':
            ylabel = gdf_copy.crs.axis_info[0].name if gdf_copy.crs else 'Y-axis'
        if zlabel == 'default':
            zlabel = 'Z-axis'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        # Set axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
        elif extent is not None:
            ax.set_xlim(extent[0], extent[3])
            ax.set_ylim(extent[1], extent[4])
            ax.set_zlim(extent[2], extent[5])

        # Add title and grid
        ax.set_title(title)
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_zoom_in(gdf, col, units, title, xlim, ylim, figsize, markersize, xlabel, ylabel, cmap):
        """Method to plot zoomed in version of geopfa maps, using xlim and ylim to determine the extent. 
        Also adds a basemap."""
        fig, ax = plt.subplots(figsize=figsize)
        if col is None or str(col).lower() == "none":
            gdf.plot(ax=ax)
        else:
            gdf.plot(ax=ax, marker='s', markersize=markersize,
                    column=col,cmap=cmap,legend=False, alpha=0.25)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gdf[col].min(), vmax=gdf[col].max()))
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(units)
        if xlabel == 'default':
            xlabel = gdf.crs.axis_info[1].name
        if ylabel == 'default':
            ylabel = gdf.crs.axis_info[0].name
        ## TODO: Basemap is causing problems. Fix at a later date.
        # Add the basemap
        # ctx.add_basemap(ax=ax)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def raster_plot(gdf, col, units, layer):
        """Plots data using pcolormesh. Creates a smoother plot, but does not 
        preserve geometry in plot"""
        x = gdf.geometry.x
        y = gdf.geometry.y
        z = gdf[col]

        # grid coordinates
        xi = np.linspace(x.min(), x.max(), 500)
        yi = np.linspace(y.min(), y.max(), 500)
        xi, yi = np.meshgrid(xi, yi)

        # interpolate 
        zi = griddata((x, y), z, (xi, yi), method='linear')

        fig, ax = plt.subplots(figsize=(10, 10))
        c = ax.pcolormesh(xi, yi, zi, shading='auto', cmap='jet') 
        fig.colorbar(c, ax=ax, label=units)

        plt.title(f'{layer}: heatmap')
        plt.xlabel('easting (m)')
        plt.ylabel('northing (m)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


