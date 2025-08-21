import folium
from folium import Map, Element
from folium.raster_layers import ImageOverlay
from folium.plugins import FloatImage, SideBySideLayers
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import io
import base64
import pandas as pd
from typing import Tuple
from branca.colormap import LinearColormap
import imageio.v2 as imageio
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from IPython.display import Image, display
from PIL import Image as pilImage
import ipywidgets as widgets


def plot_folium_from_xarray(dataset, day_select, bbox, var_name_for_title, flipud, matplot_ramp, zoom_level, save_tif=False, tif_filename=None, crs=None, opacity=0.8, basemap_style='cartodb-positron'):
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
    basemap_style : str, optional
        Basemap style to use (default 'cartodb-positron').
        See get_available_basemaps() for options.
        
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
        zoom_start=zoom_level,
        tiles=None  # We'll add basemap separately
    )
    
    # Add basemap
    try:
        add_basemap_to_map(m, basemap_style)
    except Exception as e:
        print(f"Warning: Could not add basemap '{basemap_style}': {e}. Continuing without basemap.")
    
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


def get_available_basemaps() -> dict:
    """
    Get a dictionary of available basemap styles and their descriptions.
    
    Returns
    -------
    dict
        Dictionary mapping basemap style names to their descriptions
    """
    return {
        'openstreetmap': 'OpenStreetMap standard tiles',
        'cartodb-positron': 'Light gray CartoDB basemap (subtle, good for data visualization)',
        'cartodb-dark': 'Dark CartoDB basemap (good for bright data)',
        'esri-satellite': 'ESRI satellite imagery without labels',
        'esri-satellite-labels': 'ESRI satellite imagery with place labels overlay',
        None: 'No basemap (transparent background)'
    }


def add_basemap_to_map(m: folium.Map, basemap_style: str) -> None:
    """
    Add a basemap layer to a Folium map with error checking.
    
    Parameters
    ----------
    m : folium.Map
        The Folium map object to add the basemap to
    basemap_style : str
        The style of basemap to add. Options:
        - 'openstreetmap': OpenStreetMap standard tiles
        - 'cartodb-positron': Light gray CartoDB basemap
        - 'cartodb-dark': Dark CartoDB basemap
        - 'esri-satellite': ESRI satellite imagery
        - 'esri-satellite-labels': ESRI satellite with place labels overlay
        - None: No basemap added
        
    Raises
    ------
    ValueError
        If an invalid basemap_style is provided
    TypeError
        If m is not a folium.Map object
        
    Returns
    -------
    None
        Modifies the map in place
    """
    # Type checking
    if not isinstance(m, (folium.Map, Map)):
        raise TypeError(f"Expected folium.Map object, got {type(m).__name__}")
    
    # Skip if no basemap requested
    if basemap_style is None or basemap_style == "":
        return
    
    # Validate basemap_style
    valid_styles = {
        'openstreetmap', 'cartodb-positron', 'cartodb-dark', 
        'esri-satellite', 'esri-satellite-labels'
    }
    
    if not isinstance(basemap_style, str):
        raise TypeError(f"basemap_style must be a string, got {type(basemap_style).__name__}")
    
    if basemap_style not in valid_styles:
        raise ValueError(
            f"Invalid basemap_style '{basemap_style}'. "
            f"Valid options are: {', '.join(sorted(valid_styles))}, or None"
        )
    
    # Add the appropriate basemap
    try:
        if basemap_style == 'openstreetmap':
            folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
            
        elif basemap_style == 'cartodb-positron':
            folium.TileLayer('cartodbpositron', name='CartoDB Positron').add_to(m)
            
        elif basemap_style == 'cartodb-dark':
            folium.TileLayer('cartodbdark_matter', name='CartoDB Dark').add_to(m)
            
        elif basemap_style == 'esri-satellite':
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='ESRI Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            
        elif basemap_style == 'esri-satellite-labels':
            # Add satellite imagery
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='ESRI Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            # Add reference overlay with cities, towns, and street labels
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Reference_Overlay/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Place Labels',
                overlay=True,
                control=True,
                show=True
            ).add_to(m)
            
    except Exception as e:
        raise RuntimeError(f"Failed to add basemap '{basemap_style}': {str(e)}")


def add_custom_html_legend(m: folium.Map, custom_colors: list, colorbar_caption: str, 
                           position: str = "top", top_offset: int = 50) -> None:
    """
    Add a custom HTML legend with discrete color categories to a Folium map.
    
    Parameters
    ----------
    m : folium.Map
        The Folium map object to add the legend to
    custom_colors : list
        List of dictionaries with 'color', 'label', and optionally 'value' keys.
        Example: [{"color": "#add8e6", "label": "EF0", "value": 0}, ...]
    colorbar_caption : str
        Caption/title for the legend (e.g., "EF Rating")
    position : str, optional
        Position of the legend. Options: "top", "bottom", "left", "right" (default "top")
    top_offset : int, optional
        Pixels from top when position="top" (default 50)
        
    Returns
    -------
    None
        Modifies the map in place by adding the HTML legend
        
    Raises
    ------
    ValueError
        If custom_colors is empty or invalid format
    TypeError
        If m is not a folium.Map object
    """
    from branca.element import Template, MacroElement
    
    # Type checking
    if not isinstance(m, (folium.Map, Map)):
        raise TypeError(f"Expected folium.Map object, got {type(m).__name__}")
    
    # Validate custom_colors
    if not custom_colors:
        raise ValueError("custom_colors list cannot be empty")
    
    if not isinstance(custom_colors, list):
        raise TypeError(f"custom_colors must be a list, got {type(custom_colors).__name__}")
    
    # Validate each color entry
    for i, cat in enumerate(custom_colors):
        if not isinstance(cat, dict):
            raise TypeError(f"custom_colors[{i}] must be a dict, got {type(cat).__name__}")
        if 'color' not in cat or 'label' not in cat:
            raise ValueError(f"custom_colors[{i}] must have 'color' and 'label' keys")
    
    # Set position styles based on position parameter
    if position == "top":
        position_style = f"""
            position: fixed; 
            top: {top_offset}px; 
            left: 50%;
            transform: translateX(-50%);
        """
    elif position == "bottom":
        position_style = """
            position: fixed; 
            bottom: 50px; 
            left: 50%;
            transform: translateX(-50%);
        """
    elif position == "left":
        position_style = """
            position: fixed; 
            top: 50%; 
            left: 20px;
            transform: translateY(-50%);
        """
    elif position == "right":
        position_style = """
            position: fixed; 
            top: 50%; 
            right: 20px;
            transform: translateY(-50%);
        """
    else:
        raise ValueError(f"Invalid position '{position}'. Must be 'top', 'bottom', 'left', or 'right'")
    
    # Build the HTML legend
    legend_html = '''
    {% macro html(this, kwargs) %}
    <div style="
        ''' + position_style + '''
        width: auto;
        max-width: 90%;
        height: auto; 
        background-color: white; 
        border: 2px solid grey; 
        z-index: 9999; 
        font-size: 14px;
        padding: 8px 15px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        border-radius: 4px;
        ">
        <p style="margin: 0 0 8px 0; font-weight: bold; text-align: center; font-size: 16px;">''' + colorbar_caption + '''</p>
        <div style="display: flex; justify-content: center; align-items: center; gap: 15px; flex-wrap: wrap;">
    '''
    
    for cat in custom_colors:
        legend_html += f'''
            <div style="display: flex; align-items: center;">
                <span style="background-color: {cat['color']}; 
                             display: inline-block; 
                             width: 25px; 
                             height: 20px; 
                             margin-right: 5px;
                             border: 1px solid #333;"></span>
                <span style="font-weight: 600;">{cat['label']}</span>
            </div>
        '''
    
    legend_html += '''
        </div>
    </div>
    {% endmacro %}
    '''
    
    # Create and add the legend to the map
    custom_legend = MacroElement()
    custom_legend._template = Template(legend_html)
    m.get_root().add_child(custom_legend)


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
    custom_colors: list = None,
    layer_name: str = "VEDA Data",
    date: str = None,
    colorbar_caption: str = "Value",
    attribution: str = "VEDA",
    tile_name: str = None,
    opacity: float = 0.8,
    width: str = "100%", 
    height: str = "500px",
    capitalize_cmap: bool = False,
    remove_default_legend: bool = False,
    basemap_style: str = "cartodb-positron"
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
    custom_colors : list, optional
        List of dicts with 'value', 'color', and 'label' keys for categorical data legend.
        When provided, ONLY the HTML legend will be shown (no LinearColormap).
        Example: [{"value": 0, "color": "#add8e6", "label": "EF0"}, ...]
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
    remove_default_legend : bool, optional
        Whether to remove the default LinearColormap colorbar (default False).
        Note: Ignored when custom_colors is provided (HTML legend used instead)
    basemap_style : str, optional
        Basemap style to use. Options: 'openstreetmap', 'cartodb-positron', 'cartodb-dark', 
        'esri-satellite', 'esri-satellite-labels', None (default 'cartodb-positron')
        
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
        crs="EPSG3857",
        tiles=None  # We'll add tiles separately
    )
    
    # Add basemap
    try:
        add_basemap_to_map(m, basemap_style)
    except Exception as e:
        print(f"Warning: Could not add basemap '{basemap_style}': {e}. Continuing without basemap.")
    
    # Add the VEDA data Tile Layer to the Map
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
    
    # Handle colorbar/legend with clear, mutually exclusive logic
    if custom_colors:
        # For categorical data, ONLY use HTML legend (no LinearColormap)
        # HTML legend is added later in the function via add_custom_html_legend()
        pass  # Don't add LinearColormap for categorical data
        
    elif not remove_default_legend:
        # For continuous data, add LinearColormap
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
    # else: remove_default_legend=True and no custom_colors = no legend at all
    
    # Add custom HTML legend if provided
    if custom_colors:
        try:
            add_custom_html_legend(m, custom_colors, colorbar_caption, position="top", top_offset=50)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not add custom legend: {e}")
    
    # Add Dynamic Title if date is provided
    if date:
        # Handle various date formats
        if '(' in str(date) and ')' in str(date):
            # Handle cases like "(March-May 2024)"
            formatted_date = str(date)
        else:
            # Convert standard date formats
            formatted_date = pd.to_datetime(date).strftime('%B %d, %Y')
        
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


def plot_folium_SidebySide_layer_from_VEDA_STAC(
    tiles_url_left: str,
    tiles_url_right: str,
    center_coords: list,
    zoom_level: int = 14,
    title: str = "Side-by-Side Comparison",
    label_left: str = "Left Layer",
    label_right: str = "Right Layer",
    layer_name_left: str = "Left Layer",
    layer_name_right: str = "Right Layer",
    opacity: float = 0.8,
    basemap_style: str = 'esri-satellite-labels',
    height: str = "800px",
    width: str = "100%",
    colormap_left: str = None,
    colormap_right: str = None,
    rescale_left: tuple = None,
    rescale_right: tuple = None,
    units_left: str = None,
    units_right: str = None
) -> folium.Map:
    """
    Create a Folium map with side-by-side layer comparison using a draggable slider.
    
    Uses the Leaflet leaflet-side-by-side plugin to create an interactive comparison
    between two tile layers. The user can drag a vertical slider to reveal more of
    either layer.
    
    Parameters
    ----------
    tiles_url_left : str
        Tile URL template for the left layer (with {z}, {x}, {y} placeholders)
    tiles_url_right : str
        Tile URL template for the right layer (with {z}, {x}, {y} placeholders)
    center_coords : list
        [latitude, longitude] for map center
    zoom_level : int, optional
        Initial zoom level (default 14)
    title : str, optional
        Main title displayed at top of map (default "Side-by-Side Comparison")
    label_left : str, optional
        Label for left layer, e.g., "Reflectivity (-10 to 50 dBZ)" (default "Left Layer")
    label_right : str, optional
        Label for right layer, e.g., "Velocity (-75 to 75 m/s)" (default "Right Layer")
    layer_name_left : str, optional
        Name for left layer in layer control (default "Left Layer")
    layer_name_right : str, optional
        Name for right layer in layer control (default "Right Layer")
    opacity : float, optional
        Opacity for both layers, between 0 and 1 (default 0.8)
    basemap_style : str, optional
        Basemap style to use. Options: 'openstreetmap', 'cartodb-positron', 
        'cartodb-dark', 'esri-satellite', 'esri-satellite-labels', None 
        (default 'esri-satellite-labels')
    height : str, optional
        Map height as CSS string (default "800px")
    width : str, optional
        Map width as CSS string (default "100%")
    colormap_left : str, optional
        Name of matplotlib colormap for left layer (e.g., 'turbo')
    colormap_right : str, optional
        Name of matplotlib colormap for right layer (e.g., 'seismic')
    rescale_left : tuple, optional
        (vmin, vmax) values for left layer colorbar
    rescale_right : tuple, optional
        (vmin, vmax) values for right layer colorbar
    units_left : str, optional
        Units for left colorbar (e.g., 'dBZ')
    units_right : str, optional
        Units for right colorbar (e.g., 'm/s')
        
    Returns
    -------
    folium.Map
        The configured Folium map object with side-by-side layers
        
    Examples
    --------
    >>> # Create a comparison of two radar products
    >>> m = create_side_by_side_map(
    ...     tiles_url_left="https://example.com/reflectivity/{z}/{x}/{y}.png",
    ...     tiles_url_right="https://example.com/velocity/{z}/{x}/{y}.png",
    ...     center_coords=[41.668, -95.372],
    ...     zoom_level=14,
    ...     title="DOW7 Radar Comparison",
    ...     label_left="← Reflectivity (dBZ)",
    ...     label_right="Velocity (m/s) →"
    ... )
    >>> m
    
    Notes
    -----
    - Both tile layers must be added to the map before being passed to SideBySideLayers
    - The slider divides the view vertically; left layer shows on left, right layer on right
    - Basemap is added beneath both comparison layers for context
    - HTML elements are used for title, labels, and description positioning
    """
    
    # Create the base map
    m = folium.Map(
        location=center_coords,
        zoom_start=zoom_level,
        control_scale=True,
        width=width,
        height=height,
        tiles=None  # We'll add basemap separately
    )
    
    # Add basemap if specified
    if basemap_style:
        try:
            add_basemap_to_map(m, basemap_style)
        except Exception as e:
            print(f"Warning: Could not add basemap '{basemap_style}': {e}. Continuing without basemap.")
    
    # Create left layer
    layer_left = folium.TileLayer(
        tiles=tiles_url_left,
        attr=layer_name_left,
        name=layer_name_left,
        overlay=True,
        control=True,
        opacity=opacity,
        tms=False
    )
    
    # Create right layer
    layer_right = folium.TileLayer(
        tiles=tiles_url_right,
        attr=layer_name_right,
        name=layer_name_right,
        overlay=True,
        control=True,
        opacity=opacity,
        tms=False
    )
    
    # IMPORTANT: Add layers to map BEFORE using in SideBySideLayers
    # This is required by the plugin
    layer_left.add_to(m)
    layer_right.add_to(m)
    
    # Create and add the SideBySideLayers plugin
    sbs = SideBySideLayers(
        layer_left=layer_left,
        layer_right=layer_right
    )
    sbs.add_to(m)
    
    # Add layer control for basemap options
    folium.LayerControl().add_to(m)
    
    # Add title to the map
    title_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        font-size: 18px;
        font-weight: bold;
        background: rgba(255,255,255,0.9);
        padding: 8px 12px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    ">
        {title}
    </div>
    """
    m.get_root().html.add_child(Element(title_html))
    
    # Helper function to generate HTML colorbar
    def generate_html_colorbar(colormap_name, vmin, vmax, units=None):
        if not colormap_name or vmin is None or vmax is None:
            return ""
        
        # Get matplotlib colormap
        try:
            cmap = plt.get_cmap(colormap_name)
        except:
            return ""
        
        # Generate gradient CSS
        n_stops = 10
        gradient_stops = []
        for i in range(n_stops):
            ratio = i / (n_stops - 1)
            rgba = cmap(ratio)
            color = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 1)"
            gradient_stops.append(f"{color} {ratio*100}%")
        
        gradient_css = f"linear-gradient(to right, {', '.join(gradient_stops)})"
        
        # Format units string
        units_str = f" {units}" if units else ""
        
        return f"""
        <div style="margin-top: 8px; background: rgba(255,255,255,0.9); padding: 4px; border-radius: 3px;">
            <div style="
                width: 150px;
                height: 20px;
                background: {gradient_css};
                border: 1px solid #ccc;
                margin: 0 auto;
            "></div>
            <div style="display: flex; justify-content: space-between; width: 150px; font-size: 11px; margin-top: 2px;">
                <span>{vmin}{units_str}</span>
                <span>{vmax}{units_str}</span>
            </div>
        </div>
        """
    
    # Generate colorbars HTML if needed
    left_colorbar_html = ""
    right_colorbar_html = ""
    
    if colormap_left and rescale_left:
        left_colorbar_html = generate_html_colorbar(
            colormap_left, rescale_left[0], rescale_left[1], units_left
        )
    
    if colormap_right and rescale_right:
        right_colorbar_html = generate_html_colorbar(
            colormap_right, rescale_right[0], rescale_right[1], units_right
        )
    
    # Add labels for left and right panels with optional colorbars
    labels_html = f"""
    <div style="
        position: fixed;
        top: 60px;
        left: 25%;
        transform: translateX(-50%);
        z-index: 1000;
        font-size: 14px;
        font-weight: bold;
        background: rgba(255,255,255,0.9);
        padding: 5px 10px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    ">
        <div style="text-align: center;">{label_left}</div>
        {left_colorbar_html}
    </div>
    <div style="
        position: fixed;
        top: 60px;
        right: 25%;
        transform: translateX(50%);
        z-index: 1000;
        font-size: 14px;
        font-weight: bold;
        background: rgba(255,255,255,0.9);
        padding: 5px 10px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    ">
        <div style="text-align: center;">{label_right}</div>
        {right_colorbar_html}
    </div>
    """
    m.get_root().html.add_child(Element(labels_html))
    
    # Add description text at the bottom if provided
    description_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        font-size: 13px;
        background: rgba(255,255,255,0.9);
        padding: 8px 12px;
        border-radius: 4px;
        max-width: 600px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    ">
        Drag the slider to compare rendered tiles. 
    </div>
    """
    m.get_root().html.add_child(Element(description_html))
    
    return m

