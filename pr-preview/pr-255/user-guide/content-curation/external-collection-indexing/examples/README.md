# External Collection Configuration Examples

This directory contains example configuration files for indexing external datasets in VEDA using both Titiler-CMR and pyarc2stac.

## Titiler-CMR Examples

### GPM Precipitation (gpm-precipitation-config.json)

A complete STAC collection configuration for the GPM IMERG Final Daily precipitation dataset:

- **Dataset**: GPM Level 3 IMERG Final Daily 10 x 10 km
- **CMR Concept ID**: C2723754864-GES_DISC
- **Format**: NetCDF
- **Backend**: xarray
- **Temporal Coverage**: 1998-present (daily)
- **Spatial Resolution**: 0.1° × 0.1°

This example demonstrates:
- Multi-variable configuration with different visualization settings
- Proper backend selection for NetCDF data
- Dimension and coordinate system definitions
- Rendering configuration for precipitation data
- Metadata structure following STAC conventions

## ArcGIS Server Examples

### ESI MapServer (esi-mapserver-config.json)

SERVIR Global Evaporative Stress Index 4-week MapServer service:

- **Service**: Global ESI 4-Week MapServer
- **Format**: Styled map layers
- **Temporal Coverage**: 2001-present (weekly)
- **Spatial Resolution**: 5 km
- **Update Frequency**: Weekly

This example demonstrates:
- Multi-layer MapServer configuration
- WMS integration for visualization
- Time-enabled service handling
- Layer-based rendering configuration

### Soil Moisture ImageServer (soil-moisture-imageserver-config.json)

NASA disasters test soil moisture percentile ImageServer:

- **Service**: LIS VSM Percentile 10cm ImageServer
- **Format**: Multidimensional raster data
- **Temporal Coverage**: Daily time series
- **Spatial Coverage**: Continental United States
- **Variables**: Soil moisture percentiles

This example demonstrates:
- Datacube extension for multidimensional data
- Variable and dimension definitions
- Temporal data handling
- ImageServer-specific configuration

### Climate Projections FeatureServer (climate-projections-featureserver-config.json)

Climate resilience and adaptation projections FeatureServer:

- **Service**: CMRA Climate and Coastal Inundation Projections
- **Format**: Vector features (polygons)
- **Coverage**: Counties, tracts, and tribal areas
- **Data Type**: Climate projection and vulnerability data

This example demonstrates:
- Multi-layer FeatureServer configuration
- Timeless data handling
- Feature layer metadata
- Administrative boundary data structure

## Using These Examples

1. **Copy Configuration**: Start with an example configuration closest to your dataset
2. **Update Metadata**: Modify collection ID, concept ID, and descriptive fields
3. **Adjust Dimensions**: Update spatial and temporal extents for your dataset
4. **Configure Variables**: Add/remove variables based on your dataset structure
5. **Set Rendering**: Customize visualization parameters (colormaps, rescaling)
6. **Test Configuration**: Validate using Titiler-CMR endpoints before deployment

## Testing Configurations

### Testing Titiler-CMR Configurations

Before deploying to VEDA, test your configuration using the Titiler-CMR API:

```bash
# Test tile generation
curl "https://staging.openveda.cloud/api/titiler-cmr/WebMercatorQuad/tilejson.json?concept_id=YOUR_CONCEPT_ID&datetime=2024-01-15&backend=xarray&variable=your_variable"

# Test info endpoint
curl "https://staging.openveda.cloud/api/titiler-cmr/info?concept_id=YOUR_CONCEPT_ID&datetime=2024-01-15&backend=xarray"
```

### Testing ArcGIS Server Configurations

For ArcGIS Server integrations, test using pyarc2stac:

```python
from pyarc2stac.ArcReader import ArcReader

# Test service accessibility
service_url = "https://your-arcgis-server.com/rest/services/YourService/ImageServer"
reader = ArcReader(server_url=service_url)

# Generate and validate collection
collection = reader.generate_stac()
collection.validate()

print(f"✅ Generated collection: {collection.id}")
```

## Contributing Examples

To contribute additional examples:

1. Create a new JSON configuration file
2. Follow the naming convention: `{dataset-name}-config.json`
3. Include a brief description in this README
4. Test the configuration thoroughly
5. Submit a pull request

For more detailed information, see the [Titiler-CMR Integration Guide](../titiler-cmr.qmd).