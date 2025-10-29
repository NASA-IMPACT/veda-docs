import folium
from folium.plugins import SideBySideLayers

# Create map
m = folium.Map(location=[40, -100], zoom_start=4)

# Create two tile layers
layer_left = folium.TileLayer(
    tiles='openstreetmap',
    name='OpenStreetMap'
)

layer_right = folium.TileLayer(
    tiles='cartodbpositron',
    name='CartoDB'
)

# IMPORTANT: Add layers to map FIRST
layer_left.add_to(m)
layer_right.add_to(m)

# Then create SideBySideLayers
sbs = SideBySideLayers(layer_left, layer_right)
sbs.add_to(m)

# Save and check
m.save('test_sbs.html')
print("Map saved to test_sbs.html")

# Check if the SideBySide JavaScript is included
html = m._repr_html_()
if 'leaflet-side-by-side' in html or 'sideBySide' in html:
    print("✓ SideBySide plugin is included in the HTML")
else:
    print("✗ SideBySide plugin NOT found in HTML")
