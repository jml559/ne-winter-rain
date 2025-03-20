import json
import re
import numpy as np
import netCDF4 as nc
from datetime import datetime

# Input and output file names
path = "/local1/storage1/jml559/ne-winter-rain/ghcn/"
input_file = path + "logfile_d_1960to2024_final_raw.txt"
output_file = "daily_1960to2024_"

# Initialize storage dictionaries
stations = {}
time_series = set()

# Read and parse the input file
with open(input_file, "r") as f:
    for line in f:
        match = re.search(r"Station (US\w+):", line)
        if match:
            station_id = match.group(1)
            continue
        try:
            data = json.loads(line.strip())
            meta = data["meta"]
            ll = meta["ll"]
            precip_data = data["data"]
            
            if station_id not in stations:
                stations[station_id] = {
                    "lat": ll[1],
                    "lon": ll[0],
                    "dates": [],
                    "precip": []
                }
            
            for date, value in precip_data:
                time_series.add(date)
                if value == "T":
                    value = 0.0  # Convert Trace to 0.0
                elif value == "M":
                    value = np.nan  # Convert Missing to NaN
                else:
                    value = float(value)
                stations[station_id]["dates"].append(date)
                stations[station_id]["precip"].append(value)
        except json.JSONDecodeError:
            continue

# Sort the time series to create a consistent time axis
time_series = sorted(time_series)
time_index = {date: i for i, date in enumerate(time_series)}
num_times = len(time_series)
num_stations = len(stations)

# Create netCDF file
dataset = nc.Dataset(output_file, "w", format="NETCDF4")
dataset.createDimension("time", num_times)
dataset.createDimension("station", num_stations)

# Create variables
time_var = dataset.createVariable("time", "i4", ("time",))
time_var.units = "days since 1960-01-01"
time_var.calendar = "gregorian"
time_var[:] = nc.date2num([datetime.strptime(d, "%Y-%m-%d") for d in time_series], units=time_var.units, calendar=time_var.calendar)

station_var = dataset.createVariable("station", "S10", ("station",))
lat_var = dataset.createVariable("lat", "f4", ("station",))
lon_var = dataset.createVariable("lon", "f4", ("station",))
precip_var = dataset.createVariable("precip", "f4", ("station", "time"), fill_value=np.nan)
precip_var.units = "mm"

# Store station metadata and precipitation data
station_ids = list(stations.keys())
station_var[:] = np.array(station_ids, dtype="S10")
lat_var[:] = np.array([stations[sid]["lat"] for sid in station_ids])
lon_var[:] = np.array([stations[sid]["lon"] for sid in station_ids])

precip_array = np.full((num_stations, num_times), np.nan, dtype=np.float32)
for i, sid in enumerate(station_ids):
    for date, value in zip(stations[sid]["dates"], stations[sid]["precip"]):
        precip_array[i, time_index[date]] = value

precip_var[:, :] = precip_array

dataset.close()
print(f"NetCDF file saved as {output_file}")
