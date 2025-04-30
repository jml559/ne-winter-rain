import re
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

### PLOT OF STATION PRECIP ON 2023-12-18 (INDIVIDUAL STATIONS + INTERPOLATED)
""" file_path = "/local1/storage1/jml559/ne-winter-rain/ghcn/extreme_events_20231218.txt"
obs_extr = [] # stations with extreme precip
obs_non = [] # stations with non-extreme precip

with open(file_path, "r") as file:
    text = file.read()

split_text = text.split("Did not meet extreme criteria:")
extreme_text = split_text[0]
pattern = r"lat ([\d\.-]+) and lon ([\d\.-]+) with ([\d\.]+) in"
matches_extr = re.findall(pattern, extreme_text)
obs_extr = [(float(lat), float(lon), float(precip)) for lat, lon, precip in matches_extr]

nonextreme_text = split_text[1]
matches_nonextreme = re.findall(pattern, nonextreme_text)
obs_non = [(float(lat), float(lon), float(precip)) for lat, lon, precip in matches_nonextreme]

lat_extr = [pt[0] for pt in obs_extr]
lon_extr = [pt[1] for pt in obs_extr]
precip_extr = [pt[2] for pt in obs_extr]
lat_non = [pt[0] for pt in obs_non]
lon_non = [pt[1] for pt in obs_non]
precip_non = [pt[2] for pt in obs_non]

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent([-81, -67, 37, 50])

ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.5)
ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.LAKES, facecolor="lightblue")
ax.add_feature(cfeature.BORDERS, linestyle=":")

ax.scatter(lon_extr, lat_extr, c=precip_extr, cmap="YlGnBu", s=40, 
    edgecolor="black", vmin=0, vmax=4, transform=ccrs.PlateCarree())
sc = ax.scatter(lon_non, lat_non, c=precip_non, cmap="YlGnBu", s=40, 
    vmin=0, vmax=4, transform=ccrs.PlateCarree())
cbar = plt.colorbar(sc, ax=ax, extend="max", orientation='vertical', label="Precip (in)")

ax.set_title("Precipitation Totals on 2023-12-18")
fig.savefig("./figures/station_precip_20231218.pdf") """


### PLOT OF STATION 99TH PERCENTILE
# Initialize lists for coordinates
file_path = "/local1/storage1/jml559/ne-winter-rain/ghcn/logfile_d_1960to2024_90complete_list.txt"

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
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={"projection": ccrs.PlateCarree()})
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

ax.set_title("99th percentile daily precipitation, \nGHCN-D (DJF 1960-2025)")
fig.tight_layout()
fig.savefig("../figures/station_map_new.pdf")
print("done")
