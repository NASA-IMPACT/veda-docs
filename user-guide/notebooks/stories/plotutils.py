import folium
from folium import Map, TileLayer, Element
from folium.raster_layers import ImageOverlay
from folium.plugins import FloatImage
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # ensure proper norm
import matplotlib.ticker as mticker
import io
import base64
import pandas as pd  # needed for date formatting
import rioxarray as rxr
from typing import Tuple
from branca.colormap import LinearColormap
import matplotlib.cm as cm
import imageio.v2 as imageio
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from IPython.display import Image, display
from PIL import Image as pilImage
import ipywidgets as widgets
import io


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

    # 1) true grid‐cell edges, as Python floats
    dx = float(lons[1] - lons[0])
    dy = float(lats[1] - lats[0])

    lon_left   = float(lons.min() - dx/2)
    lon_right  = float(lons.max() + dx/2)
    lat_bottom = float(lats.min() - dy/2)
    lat_top    = float(lats.max() + dy/2)



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
    
    # convert RGBA [0–1] → uint8 [0–255] → pure‐Python nested lists
    rgba_uint8 = (rgba * 255).astype("uint8")
    image_list = rgba_uint8.tolist()
    
    m = Map(
        location=[ (lat_bottom+lat_top)/2, (lon_left+lon_right)/2 ],
        zoom_start=zoom_level
    )
    TileLayer("CartoDB positron").add_to(m)
    
    ImageOverlay(
        image=image_list,
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


def plot_hdf4_as_png(directory, extension, variable_name, colorbar_label, plot_title):
    """
    NOTE: This function requires the following imports to work:
    - dateutils (custom module)
    - from pyhdf.SD import SD, SDC
    """
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

        ax.set_title(f"{pd.to_datetime(t).strftime('%Y-%m-%d %H:%M')}")

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


def create_pausable_blend_slider(img1_path, img2_path, width=800):
    """
    Creates an interactive slider to blend between two images,
    allowing the user to "pause" on any blend percentage.

    Parameters:
        img1_path (str): Filesystem path to the first input image (bottom layer).
        img2_path (str): Filesystem path to the second input image (top layer).
        width (int): Desired width of the output image in pixels. Height scaled to preserve aspect ratio.

    Returns:
        ipywidgets.VBox: An interactive widget containing the slider and the blended image.
    """
    # Load images using the helper function
    try:
        img1 = load_preview(img1_path, width)
        img2 = load_preview(img2_path, width)
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        # Return an empty widget or raise the error further
        return widgets.Label("Error: Image files not found. Cannot create blend slider.")
    
    # Ensure same dimensions for blending
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])
    img1 = img1[:min_height, :min_width]
    img2 = img2[:min_height, :min_width]

    # Create an Output widget to display the dynamically updated image
    output_image_widget = widgets.Output()

    # Create a FloatSlider widget for blending
    blend_slider = widgets.FloatSlider(
        value=0.50,  # Start with img1 fully visible (0% blend towards img2)
        min=0.0,
        max=1.0,
        step=0.01,
        description='Blend:',
        continuous_update=True, # Update image as slider is dragged
        orientation='horizontal',
        readout=True,
        readout_format='.0%', # Display value as percentage
    )

    # Define the update function that will be called when the slider value changes
    def update_image_display(change):
        with output_image_widget: # Direct output to this widget
            output_image_widget.clear_output(wait=True) # Clear previous image
            
            blend_factor = change['new'] # Get the new slider value (0.0 to 1.0)
            
            # Perform the image blending
            blended_array = (1 - blend_factor) * img1 + blend_factor * img2
            
            # Convert the blended NumPy array back to a PIL Image
            pil_blended_img = pilImage.fromarray(blended_array.astype(np.uint8))
            
            # Convert PIL Image to bytes in PNG format for display in Jupyter
            img_byte_arr = io.BytesIO()
            pil_blended_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0) # Rewind to the beginning of the BytesIO object
            
            # Display the image
            display(Image(data=img_byte_arr.read()))

    # Link the slider's value changes to the update function
    blend_slider.observe(update_image_display, names='value')

    # Initial display of the image when the widget is first created
    # Call the update function once with the initial slider value
    update_image_display({'new': blend_slider.value})

    # Return a VBox (vertical box) containing the slider and the image output
    # This VBox is the interactive widget that will be displayed in Jupyter
    return widgets.VBox([blend_slider, output_image_widget])


def plot_folium_from_VEDA_STAC(
    tiles_url_template: str,
    center_coords: list,
    zoom_level: int = 6,
    rescale: tuple = (0, 1),
    colormap_name: str = "viridis",
    layer_name: str = "VEDA Data",
    date: str = None,
    colorbar_caption: str = "Value",
    attribution: str = "VEDA",
    tile_name: str = None,
    opacity: float = 0.8,
    width: str = "100%", 
    height: str = "500px",
    capitalize_cmap: bool = False
) -> folium.Map:
    """
    Create a Folium map displaying VEDA STAC data with a colorbar and title.
    
    Parameters
    ----------
    tiles_url_template : str
        The tile URL template from VEDA STAC (with {z}, {x}, {y} placeholders)
    center_coords : list
        [latitude, longitude] for map center
    zoom_level : int, optional
        Initial zoom level (default 6)
    rescale : tuple, optional
        (vmin, vmax) values for data scaling (default (0, 1))
    colormap_name : str, optional
        Name of the colormap (default "viridis")
    layer_name : str, optional
        Display name for the layer and title (default "VEDA Data")
    date : str, optional
        Date string for the title (e.g., '2022-05-11T00:00:00Z')
    colorbar_caption : str, optional
        Caption for the colorbar legend (default "Value")
    attribution : str, optional
        Attribution text for the tiles (default "VEDA")
    tile_name : str, optional
        Name for the tile layer in layer control (defaults to layer_name)
    opacity : float, optional
        Layer opacity (default 0.8)
    width : str, optional
        Map width (default "100%")
    height : str, optional
        Map height (default "500px")
    capitalize_cmap : bool, optional
        Whether to apply alternating capitalization to colormap name (default False)
        
    Returns
    -------
    folium.Map
        The configured Folium map object
    """
    # Apply colormap name transformation if requested
    if capitalize_cmap:
        cmap_name = "".join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(colormap_name)
        )
    else:
        cmap_name = colormap_name
    
    # Use layer_name for tile_name if not provided
    if tile_name is None:
        tile_name = layer_name
    
    # Extract rescale values
    vmin_val, vmax_val = rescale
    
    # Initialize the Folium Map
    m = folium.Map(
        location=center_coords,
        zoom_start=zoom_level,
        width=width,
        height=height,
        control_scale=True,
        crs="EPSG3857"
    )
    
    # Add the Tile Layer to the Map
    folium.TileLayer(
        tiles=tiles_url_template,
        attr=attribution,
        name=tile_name,
        overlay=True,
        control=True,
        tms=False,
        opacity=opacity
    ).add_to(m)
    
    # Add Layer Control
    folium.LayerControl().add_to(m)
    
    # Add Colorbar (Legend)
    steps = 10
    mpl_cmap = plt.get_cmap(cmap_name)
    colors = [mpl_cmap(i / (steps - 1)) for i in range(steps)]
    
    legend = LinearColormap(
        colors=colors,
        vmin=vmin_val,
        vmax=vmax_val,
        caption=colorbar_caption
    ).to_step(steps)
    
    legend.add_to(m)
    
    # Add Dynamic Title if date is provided
    if date:
        try:
            formatted_date = pd.to_datetime(date).strftime('%B %d, %Y')
        except Exception:
            formatted_date = str(date)
        
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
            {layer_name} — {formatted_date}
          </div>
        """
        m.get_root().html.add_child(Element(title_html))
    
    return m

