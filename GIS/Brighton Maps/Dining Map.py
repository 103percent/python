restaurants = [
    {"name": "Food for Friends", "lat": 50.8225, "lon": -0.1372, "desc": "Vegetarian fine dining.", "category": "Vegetarian"},
    {"name": "The Chilli Pickle", "lat": 50.8261, "lon": -0.1397, "desc": "Indian street food.", "category": "Indian"},
    {"name": "Bincho Yakitori", "lat": 50.8239, "lon": -0.1477, "desc": "Japanese skewers & sake.", "category": "Japanese"},
]

import folium

# Category colours
category_colors = {
    "Vegetarian": "green",
    "Indian": "orange",
    "Japanese": "blue"
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
    colour = category_colors.get(restaurant["category"], "gray")  # default to gray if unknown
    folium.Marker(
        location = [restaurant["lat"], restaurant["lon"]],
        popup=f"<b>{restaurant['name']}</b><br>{restaurant['desc']}",
        icon=folium.Icon(color=colour, icon="cutlery", prefix="fa"),
        max_width=2000,  # Adjust as needed
        min_width=1000
    ).add_to(map_brighton)

# Save the map to an HTML file
map_brighton.save("brighton_restaurants.html")
