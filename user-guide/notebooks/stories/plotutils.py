import folium
import folium.plugins
from folium import Map, TileLayer, Element
from folium.raster_layers import ImageOverlay
from folium.plugins import FloatImage
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # ensure proper norm
import matplotlib.ticker as mticker
import io
import os
import base64
import pandas as pd  # needed for date formatting
import rioxarray as rxr
from typing import Tuple
from branca.colormap import LinearColormap
import matplotlib.cm as cm
import imageio.v2 as imageio
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from IPython.display import Image
from PIL import Image as pilImage
import plotly.graph_objects as go


def plot_folium_from_xarray(dataset, day_select, bbox, var_name_for_title, flipud, matplot_ramp, zoom_level, save_tif=False, tif_filename=None, crs=None, opacity=0.8):
    """
    Plot a selected day's xarray data on an interactive Folium map with a colorbar.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        The dataset containing time, lat, lon dimensions.
    day_select : str
        Date string (e.g., '2022-05-11') to select.
    bbox : list
        Bounding box [lon_min, lat_min, lon_max, lat_max].
    var_name_for_title : str
        Variable name to show in the title.
    flipud : bool, optional
        If True, flips the latitude axis upside down.
    save_tif : bool, optional
        If True, saves the selected slice to GeoTIFF file.
    tif_filename : str, optional
        Full output filename for the GeoTIFF (must end in .tif).
        
    Returns
    -------
    folium.Map
        Interactive Folium map object.
    """

    lon_min, lat_min, lon_max, lat_max = bbox

    # Select slice
    da = dataset.sel(time=day_select).sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max)
    )

    lons = da.lon.values
    lats = da.lat.values
    data = da.values

    # after you select da and extract lons/lats …
    dx = float(lons[1] - lons[0])
    dy = float(lats[1] - lats[0])

    #build edge bounds for plotting
    lon_left   = lon_min - dx/2
    lon_right  = lon_max + dx/2
    lat_bottom = lat_min - dy/2
    lat_top    = lat_max + dy/2

    #center on the true middle
    center_lat = (lat_bottom + lat_top) / 2
    center_lon = (lon_left   + lon_right) / 2


    # Flip latitudes if needed
    if flipud and lats[0] < lats[-1]:
        data = np.flipud(data)

    # Normalize 0-1
    normed = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    # ======================
    # SAVE TIF IF REQUESTED
    # ======================
    if save_tif:
        if tif_filename is None:
            tif_filename = f"{var_name_for_title.replace(' ', '_').lower()}_{day_select}.tif"

        # Create new xarray.DataArray with coordinates
        da_out = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            name=var_name_for_title
        )

        # Set CRS and spatial dimensions properly
        da_out.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        da_out.rio.write_crs(crs, inplace=False)  
        
        # Save as GeoTIFF
        da_out.rio.to_raster(tif_filename)
        print(f"Saved GeoTIFF to {tif_filename}")
    
    # ========== Plot and Save Main Image ==========
    fig, ax = plt.subplots(figsize=(8, 6))

    # — in your plotting section —
    extent_edges = [lon_left, lon_right, lat_bottom, lat_top]
    ax.imshow(
        normed,
        cmap=matplot_ramp,
        extent=extent_edges,
        origin='lower'
    )

    ax.imshow(
        normed,
        cmap=matplot_ramp,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin='lower'
    )
    ax.axis('off')
    ax.set_title(f"{var_name_for_title} on {pd.to_datetime(day_select).strftime('%B %d, %Y')}", fontsize=16, pad=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    # ========== Plot and Save Colorbar ==========
    fig_cbar, ax_cbar = plt.subplots(figsize=(0.8, 4))  # narrower colorbar
    norm = mcolors.Normalize(vmin=float(da.min()), vmax=float(da.max()))
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=matplot_ramp, norm=norm),
        cax=ax_cbar,
        orientation='vertical'
    )
    ax_cbar.set_ylabel(var_name_for_title, rotation=270, labelpad=15)
    ax_cbar.yaxis.set_label_position('left')

    buf_cbar = io.BytesIO()
    plt.savefig(buf_cbar, format='png', bbox_inches='tight', pad_inches=0.05, transparent=True)
    buf_cbar.seek(0)
    encoded_cbar = base64.b64encode(buf_cbar.read()).decode('utf-8')
    buf_cbar.close()
    plt.close(fig_cbar)

    # ========== Create Folium Map ==========
    # — after you've computed `normed` (shape [ny, nx]) and the half-cell edges:
    rgba = cm.get_cmap(matplot_ramp)(normed)         # shape [ny, nx, 4], floats 0–1
    
    m = Map(
        location=[ (lat_bottom+lat_top)/2, (lon_left+lon_right)/2 ],
        zoom_start=zoom_level
    )
    TileLayer("CartoDB positron").add_to(m)
    
    ImageOverlay(
        image=rgba,
        bounds=[[lat_bottom, lon_left], [lat_top, lon_right]],
        opacity=opacity,
        origin="lower",                # ensures array[0] is the bottom row
        mercator_project=True          # ← this warps your Plate Carrée array into 3857
    ).add_to(m)
    
    m

    # 1) build a little HTML <div> with your title
    title_html = f"""
         <div style="
             position: fixed;
             top: 10px;
             left: 50%;
             transform: translateX(-50%);
             z-index: 1000;
             font-size: 20px;
             font-weight: bold;
             background-color: rgba(255,255,255,0.7);
             padding: 5px 10px;
             border-radius: 5px;
         ">
             {var_name_for_title} on {pd.to_datetime(day_select).strftime('%B %d, %Y')}
         </div>
    """
    
    # 2) inject it into the map’s HTML
    m.get_root().html.add_child(Element(title_html))

    # Add Colorbar FloatImage
    colorbar = FloatImage(f"data:image/png;base64,{encoded_cbar}", bottom=30, left=85)
    colorbar.add_to(m)

    #adds a little widgetthat lists all of the map’s named layers—both base‐layers (TileLayer) and overlays (ImageOverlay, WMS layers, etc.)
    folium.LayerControl().add_to(m)

    return m



def plot_folium_from_STAC(
    tile_url: str,
    center: Tuple[float, float],
    zoom_start: int = 6,
    width: str = "100%",
    height: str = "600px",
    attribution: str = "",
    layer_name: str = "Base Layer",
    show_control: bool = True
) -> folium.Map:
    """
    Create and return a Folium map with a single TileLayer.

    Args:
        tile_url: URL template for the tile layer (e.g. from TileJSON 'tiles'[0]).
        center: Tuple of (latitude, longitude) to center the map on.
        zoom_start: Initial zoom level (default 6).
        width: Width of the map container (e.g. '100%' or '800px').
        height: Height of the map container (e.g. '600px' or '400px').
        attribution: Text to show in the lower-right attribution control.
        layer_name: Name for the TileLayer in the layer control.
        show_control: Whether to add a LayerControl (default True).

    Returns:
        A folium.Map object ready for display in Jupyter (or to save to HTML).
    """
    # Initialize map
    m = Map(
        location=center,
        zoom_start=zoom_start,
        width=width,
        height=height,
        control_scale=True
    )

    # Add the tile layer
    TileLayer(
        tiles=tile_url,
        attr=attribution,
        name=layer_name,
        overlay=False,
        control=show_control
    ).add_to(m)

    # Optionally add the layer control
    if show_control:
        folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m


def plot_folium_from_STAC_with_legend(
    tiles_url: dict,
    center: Tuple[float,float],
    vmin: float,
    vmax: float,
    day_select,
    colormap_name: str = "viridis",
    steps: int = 10,
    zoom_start: int = 1,
    width: str = "100%",
    height: str = "500px",
    layer_name: str= "",
    attribution: str = "",
    show_control: bool = True,
    crs: str = "EPSG4326",
    opacity: float = 0.8
    
) -> folium.Map:

    # Initialize map
    m = Map(
        location=center,
        zoom_start=zoom_start,
        width=width,
        height=height,
        control_scale=True,
        crs = crs
    )
    # Add the tile layer
    TileLayer(
        tiles=tiles_url,
        attr=attribution,
        name=layer_name,
        overlay=False,
        control=show_control,
        tms=False,
        opacity = opacity
    ).add_to(m)

    # sample the Matplotlib colormap
    import matplotlib.pyplot as plt
    mpl_cmap = plt.get_cmap(colormap_name)
    colors = [mpl_cmap(i/(steps-1)) for i in range(steps)]

    # build a stepped legend
    legend = LinearColormap(
        colors=colors,
        vmin=vmin,
        vmax=vmax,
        caption="Value"
    ).to_step(steps)

    # ——— NEW TITLE INJECTION ———
    title_html = f"""
      <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        font-size: 18px;
        font-weight: bold;
        background: rgba(255,255,255,0.8);
        padding: 4px 8px;
        border-radius: 4px;
      ">
        {layer_name} — {pd.to_datetime(day_select).strftime('%B %d, %Y')}
      </div>
    """
    m.get_root().html.add_child(Element(title_html))
    
    legend.add_to(m)
    folium.LayerControl().add_to(m)
    return m



def plot_hdf4_as_png(directory, extension, variable_name, colorbar_label, plot_title):

    # List of HDF files
    group_dict = dateutils.group_files_by_year_and_day_EARTHDATA(directory, extension)

    for (year, doy), files in group_dict.items():
        plt.figure(figsize=(12,6))
        for f in files:
            # print("Reading", f)
            try:
                hdf = SD(f, SDC.READ)
                data = hdf.select(variable_name)[:]
                lat = hdf.select('Latitude')[:]
                lon = hdf.select('Longitude')[:]
                data = np.ma.masked_where(data < -9000, data)
                plt.scatter(lon, lat, c=data, s=1, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            except Exception as e:
                print(f"Failed on {f}: {e}")

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f"{plot_title} {year} Day {doy}")
        plt.colorbar(label=colorbar_label)
        plt.tight_layout()
        png_name = f"{directory}_{year}_DOY{doy:03d}_{variable_name}.png"
        plt.savefig(png_name, dpi=150)
        plt.close()
        print(f"Saved {png_name}")


def matplotlib_gif(
    data: xr.DataArray,
    bbox: list[float],
    gif_savename="testing.gif",
    duration=2,
    cmap="viridis",
):
    lon_min, lat_min, lon_max, lat_max = bbox
    vmin, vmax = float(np.nanmin(data.values)), float(np.nanmax(data.values))
    frames = []

    for t in data.time.values:
        da = data.sel(time=t).sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max)
        )

        fig = plt.figure(figsize=(6,5))
        ax = plt.axes(projection=ccrs.PlateCarree())

        da.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            add_colorbar=True
        )
        ax.coastlines()

        # add gridlines with labels
        gl = ax.gridlines(
            draw_labels=True,
            x_inline=False, y_inline=False,
            linewidth=0.5, color='gray', alpha=0.7, linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False

        # set tick locations every 5° (or whatever interval you prefer)
        gl.xlocator = mticker.FixedLocator(np.arange(lon_min, lon_max+1, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(lat_min, lat_max+1, 5))

        # hook up nice formatters
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.set_title(f"{pd.to_datetime(t).strftime('%Y-%m-%d')}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    imageio.mimsave(gif_savename, frames, duration=duration, loop=0)
    display(Image(filename=gif_savename))
    print("✅ Saved GIF →", gif_savename)


def load_preview(path: str, target_width: int = 800) -> np.ndarray:
    """
    Load an image from disk, resize it to a given width while preserving aspect ratio,
    and return its pixel data as a NumPy array.

    Parameters
    ----------
    path : str
        Filesystem path to the input image.
    target_width : int, optional
        Desired width of the output image in pixels. The height will be scaled
        to preserve the original aspect ratio. Default is 800.

    Returns
    -------
    np.ndarray
        A 3-dimensional NumPy array representing the resized image (height, width, channels).
    """
    # Open the image and compute new height to preserve aspect ratio
    img = pilImage.open(path)
    original_width, original_height = img.size
    new_height = int(target_width * original_height / original_width)

    # Resize with high-quality resampling and convert to array
    resized = img.resize((target_width, new_height), pilImage.LANCZOS)
    return np.array(resized)


def plotly_dual_slider_window(
    image1_path: str,
    image2_path: str,
    target_width: int,
    fig_width: int,
    fig_height: int
) -> None:
    """
    Display two images in a Plotly figure with a blend slider that transitions
    the opacity from the first image to the second.

    Parameters
    ----------
    image1_path : str
        Filesystem path to the first image (shown at 100% opacity initially).
    image2_path : str
        Filesystem path to the second image (hidden initially).
    target_width : int
        Width to which both images will be resized (in pixels).
    fig_width : int
        Width of the resulting Plotly figure (in pixels).
    fig_height : int
        Height of the resulting Plotly figure (in pixels).

    Returns
    -------
    None
        This function renders the figure in a Jupyter notebook or interactive session.
    """
    # Load and resize both images
    img1 = load_preview(image1_path, target_width)
    img2 = load_preview(image2_path, target_width)

    # Create the Plotly figure with two image layers
    fig = go.Figure([
        go.Image(z=img1, opacity=1.0),
        go.Image(z=img2, opacity=0.0),
    ])

    # Build slider steps (0%→100%)
    steps = []
    for pct in range(101):
        steps.append({
            "method": "restyle",
            "args": ["opacity", [1.0, pct / 100]],
            "label": f"{pct}%"
        })

    slider = {
        "active": 50,  # start halfway blended
        "pad": {"t": 30},
        "currentvalue": {"prefix": "Blend: "},
        "steps": steps
    }

    # Update layout: add slider, remove axes, set figure size and margins
    fig.update_layout(
        sliders=[slider],
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        width=fig_width,
        height=fig_height
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()