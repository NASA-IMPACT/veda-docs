#!/usr/bin/env python3
import requests
import json
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def get_raster_tilejson(
    raster_api_url: str,
    collection_id: str,
    item: str,
    rescale: Tuple[float, float],
    assets: str = "cog_default",
    color_formula: str = "gamma+r+1.05",
    colormap_name: str = "rdbu_r",
) -> Dict[str, Any]:
    """
    Fetch the TileJSON configuration for a raster item from a STAC‐based Raster API.

    This returns the TileJSON document, including tile URL templates and styling metadata,
    for the given collection and item, applying the specified asset, color formula, colormap,
    and rescale range.

    Args:
        raster_api_url: Base URL of the Raster API (e.g. "https://example.com/api/raster").
        collection_id: Identifier of the collection containing the raster (e.g. "nldas-3").
        item_id: Identifier of the specific item within the collection.
        rescale: A tuple of (min, max) values to rescale the raster data.
        assets: Asset key to request (default "cog_default").
        color_formula: Color formula string to apply (default "gamma+r+1.05").
        colormap_name: Name of the colormap to use (default "rdbu_r").

    Returns:
        A dict representing the TileJSON JSON response.

    Raises:
        requests.HTTPError: If the HTTP request returns a bad status code.
    """
    # Build endpoint URL without worrying about trailing slashes
    response = requests.get(
        f"{raster_api_url.rstrip('/')}/collections/{collection_id}"
        f"/items/{item['id']}/tilejson.json?"
        f"&assets={assets}"
        f"&color_formula={color_formula}&colormap_name={colormap_name}"
        f"&rescale={rescale[0]},{rescale[1]}",
    )

    response.raise_for_status()
    return response.json()



def get_collection(
    stac_api_url: str,
    collection_name: str,
) -> Dict[str, Any]:
    """
    Fetch a STAC Collection by its name.

    Args:
        stac_api_url: Base URL of the STAC API (e.g. "https://example.com/api/stac").
        collection_name: Identifier of the collection to retrieve.

    Returns:
        A dict containing the STAC Collection JSON.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    url = f"{stac_api_url.rstrip('/')}/collections/{collection_name}"
    logger.debug("Requesting STAC collection from %s", url)
    response = requests.get(url)
    response.raise_for_status()
    collection = response.json()
    logger.info("Retrieved collection %s (title: %s)", collection_name, collection.get("title"))
    return collection


def get_collection_items(
    stac_api_url: str,
    collection_name: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    List Items within a given STAC Collection.

    Args:
        stac_api_url: Base URL of the STAC API.
        collection_name: Identifier of the collection whose items to list.
        limit: Maximum number of items to return (default is 100).

    Returns:
        A list of feature dicts (`"features"`) from the STAC Collection Items endpoint.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
        KeyError: If the expected `"features"` key is missing in the response.
    """
    url = f"{stac_api_url.rstrip('/')}/collections/{collection_name}/items"
    params = {"limit": limit}
    logger.debug("Requesting items for collection %s with params %s", collection_name, params)
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    items = data.get("features")
    if items is None:
        raise KeyError(f"No 'features' key in response: {data.keys()}")
    logger.info("Retrieved %d items from collection %s", len(items), collection_name)
    return items


def search_stac_features(
    stac_api_url: str,
    collection_id: str,
    date: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Search a STAC API for features in a collection matching a specific datetime.

    Args:
        stac_api_url: Base URL of the STAC API.
        collection_id: Identifier of the collection to search.
        date: Date string in YYYY-MM-DD format; will match items at midnight UTC.
        limit: Maximum number of features to return (default is 100).

    Returns:
        A list of feature dicts from the STAC `/search` endpoint.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
        KeyError: If the expected `"features"` key is missing in the response.
    """
    url = f"{stac_api_url.rstrip('/')}/search"
    payload = {
        "collections": [collection_id],
        "query": {
            "datetime": {
                "eq": f"{date}T00:00:00"
            }
        },
        "limit": limit,
    }
    logger.debug("POSTing STAC search to %s with payload %s", url, payload)
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    items = data.get("features")
    if items is None:
        raise KeyError(f"No 'features' key in search response: {data.keys()}")
    logger.info(
        "Found %d features in collection %s on date %s",
        len(items), collection_id, date
    )
    return items


def get_raster_band_statistics(
    item: Dict[str, Any],
    asset_key: str = "cog_default",
    band_index: int = 0,
) -> Dict[str, Any]:
    """
    Retrieve the 'statistics' dict for a specific raster band from a STAC Item asset.

    This digs into item['assets'][asset_key]['raster:bands'][band_index]['statistics']
    and returns that dictionary.

    Args:
        item: A STAC Item JSON object (as dict) containing an 'assets' section.
        asset_key: The key under item['assets'] that holds the raster (default "cog_default").
        band_index: Index of the band in the asset's 'raster:bands' list (default 0).

    Returns:
        A dict of statistics (e.g. min, max, mean, std) for the requested raster band.

    Raises:
        KeyError: If the asset_key or any expected sub-key is missing.
        IndexError: If band_index is out of range for the 'raster:bands' list.
    """
    try:
        stats = item["assets"][asset_key]["raster:bands"][band_index]["statistics"]
    except KeyError as e:
        logger.error("Missing key in item assets path: %s", e)
        raise
    except IndexError as e:
        logger.error(
            "Band index %d out of range for asset '%s'", band_index, asset_key
        )
        raise

    logger.debug(
        "Retrieved statistics for asset '%s', band %d: %s",
        asset_key,
        band_index,
        stats,
    )
    return stats

def return_render_information(collection):
    # grab the dashboard render block
    dashboard_rend = collection["renders"]["dashboard"]
    
    bands      = dashboard_rend["bidx"]           
    asset_keys = dashboard_rend["assets"]         
    (vmin, vmax), = dashboard_rend["rescale"]    
    cmap_name  = dashboard_rend["colormap_name"]
    return dashboard_rend, bands, asset_keys, vmin, vmax, cmap_name
