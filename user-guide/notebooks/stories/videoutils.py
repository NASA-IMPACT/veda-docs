from IPython.display import HTML

def display_gif_from_url(url, height = 700, width=800):
    """
    Display a GIF image from a URL, centered in a Jupyter Notebook cell.

    Parameters
    ----------
    url : str
        The direct URL to the GIF file.
    width : int, optional
        Width of the displayed GIF in pixels. Default is 800.

    Returns
    -------
    IPython.display.HTML
        An HTML object that, when displayed, shows the centered GIF in the notebook.
    
    Raises
    ------
    ValueError
        If the URL is not provided or is not a string.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("You must provide a valid URL string for the GIF.")
    
    return HTML(f"""
    <div align="center">
        <img src="{url}" eight="{height}" width="{width}">
    </div>
    """)


def create_youtube_embed(iframe_width=560, iframe_height=315, src=""):
    """
    Create a centered HTML iframe for a YouTube video to display in a Jupyter Notebook.

    Parameters
    ----------
    iframe_width : int
        Width of the iframe in pixels (default 560).
    iframe_height : int
        Height of the iframe in pixels (default 315).
    src : str
        The full YouTube embed URL (e.g., "https://www.youtube.com/embed/VIDEO_ID").

    Returns
    -------
    IPython.display.HTML
        An HTML object containing the centered iframe for display in a notebook.
    
    Raises
    ------
    ValueError
        If the src is empty or not a valid embed link.
    """
    if not src:
        raise ValueError("You must provide a 'src' URL.")

    # Validate that the src has an embed link format
    if "embed/" not in src:
        raise ValueError("The 'src' URL must contain an 'embed/' link to a YouTube video.")
    
    html_output = HTML(f"""
    <div align="center">
        <iframe width="{iframe_width}" height="{iframe_height}" src="{src}" frameborder="0" allowfullscreen></iframe>
    </div>
    """)
    
    return html_output