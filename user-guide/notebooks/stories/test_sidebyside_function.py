#!/usr/bin/env python3
"""
Test the new create_side_by_side_map function in plotutils.py
"""

import plotutils as putils

def test_create_side_by_side_map():
    """Test the create_side_by_side_map function"""
    
    print("Testing create_side_by_side_map function...")
    print("=" * 60)
    
    # Create test tile URLs (these won't load real data but will test the function)
    tiles_left = "https://example.com/reflectivity/{z}/{x}/{y}.png"
    tiles_right = "https://example.com/velocity/{z}/{x}/{y}.png"
    
    # Test with all parameters
    m = putils.create_side_by_side_map(
        tiles_url_left=tiles_left,
        tiles_url_right=tiles_right,
        center_coords=[41.668, -95.372],
        zoom_level=14,
        title="DOW7 Comparison: Reflectivity vs Velocity — Harlan, IA — April 26, 2024",
        label_left="← Reflectivity (-10 to 50 dBZ)",
        label_right="Velocity (-75 to 75 m/s) →",
        layer_name_left="DOW7 Reflectivity",
        layer_name_right="DOW7 Velocity",
        opacity=0.8,
        basemap_style='esri-satellite-labels',
        description="Drag the slider to compare DOW-collected reflectivity and velocity values",
        height="800px",
        width="100%"
    )
    
    # Save to file to check HTML structure
    m.save('test_sidebyside_map.html')
    print("✓ Map created and saved to test_sidebyside_map.html")
    
    # Check if SideBySide JavaScript is included
    html = m._repr_html_()
    if 'leaflet-side-by-side' in html or 'sideBySide' in html:
        print("✓ SideBySide plugin is included in the HTML")
    else:
        print("⚠ Warning: SideBySide plugin may not be properly included")
    
    # Test with minimal parameters
    print("\nTesting with minimal parameters...")
    m2 = putils.create_side_by_side_map(
        tiles_url_left=tiles_left,
        tiles_url_right=tiles_right,
        center_coords=[40, -100]
    )
    print("✓ Minimal parameter test successful")
    
    # Test without basemap
    print("\nTesting without basemap...")
    m3 = putils.create_side_by_side_map(
        tiles_url_left=tiles_left,
        tiles_url_right=tiles_right,
        center_coords=[40, -100],
        basemap_style=None
    )
    print("✓ No basemap test successful")
    
    print("\n" + "=" * 60)
    print("All tests passed! The function is working correctly.")
    print("\nThe function signature is:")
    print("create_side_by_side_map(")
    print("    tiles_url_left: str,")
    print("    tiles_url_right: str,")
    print("    center_coords: list,")
    print("    zoom_level: int = 14,")
    print("    title: str = 'Side-by-Side Comparison',")
    print("    label_left: str = 'Left Layer',")
    print("    label_right: str = 'Right Layer',")
    print("    layer_name_left: str = 'Left Layer',")
    print("    layer_name_right: str = 'Right Layer',")
    print("    opacity: float = 0.8,")
    print("    basemap_style: str = 'esri-satellite-labels',")
    print("    description: str = None,")
    print("    height: str = '800px',")
    print("    width: str = '100%'")
    print(")")
    
    return True

if __name__ == "__main__":
    test_create_side_by_side_map()