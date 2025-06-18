restaurants = [
    {"name": "The Independent", "lat":50.8261 , "lon":-0.1273 , "desc": "Great selection of curated pints", "category": "Legendary"},
    {"name": "The Jolly Brewer", "lat":50.8390 , "lon":-0.1334 , "desc": "Great selection of curated pints", "category": "Legendary"},
    {"name": "The Hole in the Wall", "lat":50.8232 , "lon":-0.1492 , "desc": "Great selection of curated pints", "category": "Legendary"},
]

import folium

# Category colours
category_colors = {
    "Legendary": "green",
    "Deece": "yellow",
    "Bad": "red"
}

# Create a map centered around Brighton
map_brighton = folium.Map(
    location=[50.8225, -0.1372],
    zoom_start=14,
    tiles='CartoDB positron',
)

# Available - OpenStreetMap, Stamen Terrain, Stamen Toner, Stamen Watercolour, CartoDB positron, CartoDB dark_matter

# Add your restaurants to the map
for restaurant in restaurants:

        # Build HTML popup content with fixed width via inline CSS
    html = f"""
    <div style="width: 280px;">
        <h4 style="margin-bottom:5px;">{restaurant['name']}</h4>
        <p style="margin-top:0;">{restaurant['desc']}</p>
        <small><i>Category: {restaurant['category']}</i></small>
    </div>
    """    
    popup = folium.Popup(html, max_width=300)

    icon_colour = category_colors.get(restaurant["category"], "gray")  # default to gray if unknown

    folium.Marker(
        location = [restaurant["lat"], restaurant["lon"]],
        popup=popup,
        icon=folium.Icon(color=icon_colour, icon="beer", prefix="fa"),
    ).add_to(map_brighton)

# Save the map to an HTML file
map_brighton.save("brighton_pubs.html")
