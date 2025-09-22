# External Collection Configuration Examples

This directory contains example configuration files for indexing external datasets in VEDA using Titiler-CMR.

## Available Examples

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

## Using These Examples

1. **Copy Configuration**: Start with an example configuration closest to your dataset
2. **Update Metadata**: Modify collection ID, concept ID, and descriptive fields
3. **Adjust Dimensions**: Update spatial and temporal extents for your dataset
4. **Configure Variables**: Add/remove variables based on your dataset structure
5. **Set Rendering**: Customize visualization parameters (colormaps, rescaling)
6. **Test Configuration**: Validate using Titiler-CMR endpoints before deployment

## Testing Configurations

Before deploying to VEDA, test your configuration using the Titiler-CMR API:

```bash
# Test tile generation
curl "https://staging.openveda.cloud/api/titiler-cmr/WebMercatorQuad/tilejson.json?concept_id=YOUR_CONCEPT_ID&datetime=2024-01-15&backend=xarray&variable=your_variable"

# Test info endpoint
curl "https://staging.openveda.cloud/api/titiler-cmr/info?concept_id=YOUR_CONCEPT_ID&datetime=2024-01-15&backend=xarray"
```

## Contributing Examples

To contribute additional examples:

1. Create a new JSON configuration file
2. Follow the naming convention: `{dataset-name}-config.json`
3. Include a brief description in this README
4. Test the configuration thoroughly
5. Submit a pull request

For more detailed information, see the [Titiler-CMR Integration Guide](../titiler-cmr.qmd).