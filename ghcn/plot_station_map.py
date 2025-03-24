import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define the file path
file_path = "/local1/storage1/jml559/ne-winter-rain/ghcn/logfile_d_1960to2024_90complete_list.txt"

# Initialize lists for coordinates
lats = []
lons = []
lats_100 = []
lons_100 = []
pct_99_arr = [1.49, 1.2, 1.42, 0.99, 1.38, 1.49, 1.36, 1.42, 1.4, 1.34, 
            1.54, 0.95, 1.02, 0.99, 0.78, 0.8, 1.4, 0.85, 1.26, 1.2, 
            1.03, 0.83, 0.96, 1.85, 1.34, 1.38, 1.45, 0.86]

# Regular expression to extract coordinates from "Coords: [lon, lat]"
coord_pattern = re.compile(r"Coords: \[([-+]?\d*\.\d+),\s*([-+]?\d*\.\d+)\]")
completeness_pattern = re.compile(r"Completeness:\s*(\d+\.\d+)%")

# Read the file and extract coordinates
with open(file_path, "r") as file:
    lines = file.readlines()[7397:]  # Skip the first 7397 lines

    for line in lines:
        coord_match = coord_pattern.search(line)
        completeness_match = completeness_pattern.search(line)

        if coord_match and completeness_match:
            completeness = float(completeness_match.group(1))

            if completeness == 100.00:
                lon_100, lat_100 = map(float, coord_match.groups())
                lats_100.append(lat_100)
                lons_100.append(lon_100)

            else:
                lon, lat = map(float, coord_match.groups())
                lats.append(lat)
                lons.append(lon)

# Create the map
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent([-81, -67, 37, 50])  # Set the domain

# Add features
ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.5)
ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.LAKES, facecolor="lightblue")
ax.add_feature(cfeature.BORDERS, linestyle=":")

# Plot the station locations
sc = ax.scatter(lons_100, lats_100, c=pct_99_arr[:-1], cmap="YlGnBu", s=40, transform=ccrs.PlateCarree())
ax.scatter(-76.44905, 42.44915, c=pct_99_arr[-1], cmap="YlGnBu", marker='*', s=40, transform=ccrs.PlateCarree())
#ax.scatter(lons, lats, color="black", s=10, transform=ccrs.PlateCarree(), label=">90% complete stations")

cbar = plt.colorbar(sc, ax=ax, orientation='vertical', label="Precip (in)")

# Show the plot
#plt.legend()
plt.title("99th percentile daily precipitation, GHCN-D (1960-2024)")
plt.show()
plt.savefig("station_map_new.pdf")

#print(lons_100)
#print(lats_100)
