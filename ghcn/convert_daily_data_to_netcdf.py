import json
import re
import numpy as np
import netCDF4 as nc
from datetime import datetime
import pygeode as pyg

# Note that Kingston, Spring Grove, and Ithaca stations are missing a few values.
# Remaining stations have fully complete data.
station_ids_arr = ["USW00014764", "USW00014745", "USW00093730", "USW00014607", "USW00014734",
    "USW00014739", "USW00093720", "USW00093721", "USW00013781", "USW00014732",
    "USW00094728", "USW00004725", "USW00014735", "USW00014733", "USW00014768",
    "USW00094725", "USW00013739", "USC00306196", "USC00368379", "USW00014778",
    "USW00014777", "USW00014742", "USW00094823", "USC00374266", "USW00014737",
    "USW00094702", "USW00014740", "USC00304174"]

# Input and output file names
path = "/local1/storage1/jml559/ne-winter-rain/ghcn/"
input_file = path + "logfile_d_1960to2024_final_raw.txt"
output_file = "ghcn_daily_1960to2024.nc"

# Initialize storage dictionaries
stations = {}
time_series = set()
station_id = None  # Ensure station_id is always initialized

# Read and parse the input file
with open(input_file, "r") as f:
    for line in f:
        match = re.search(r"Station (US\w+):", line)
        if match:
            station_id = match.group(1)
            continue 

        if station_id is None:
            print(f"Warning: Data encountered before any station header: {line.strip()}")
            continue

        # Preprocess JSON: Convert single quotes to double quotes (if applicable)
        cleaned_line = line.strip().replace("'", '"')  # Converts ' to " (only works in simple cases)

        try:
            data = json.loads(cleaned_line)  # Attempt to load JSON
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e} in line: {line.strip()}")
            continue  # Skip bad lines instead of crashing

        try:
            meta = data["meta"]
            ll = meta["ll"]
            precip_data = data["data"]

            if station_id not in stations:
                stations[station_id] = {
                    "lat": ll[1],
                    "lon": ll[0],
                    "sid": station_id,
                    "dates": [],
                    "precip": []
                }

            for date, value in precip_data:
                if not re.match(r"\d{4}-\d{2}-\d{2}", date):
                    print(f"Skipping invalid date format: {date}")
                    continue

                time_series.add(date)

                # Convert values
                if value == "T":
                    value = 0.0  # Convert Trace to 0.0
                elif value == "M" or value == "S" or value.endswith('A'):
                    value = np.nan  # Treat 'S' and values ending in 'A' as missing
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        print(f"Could not convert value to float: {value}")
                        continue

                stations[station_id]["dates"].append(date)
                stations[station_id]["precip"].append(value)

        except KeyError as e:
            print(f"Missing expected key {e} in JSON: {data}")
            continue
    #print("Stored station IDs:", list(stations.keys()))

# Ensure time_series is not empty
if not time_series:
    raise ValueError("Error: No valid dates extracted. Check input file format.")

# Sort time series for consistent time axis
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
time_var[:] = nc.date2num([datetime.strptime(d, "%Y-%m-%d") for d in time_series], 
                          units=time_var.units, calendar=time_var.calendar)
 
max_sid_length = max(len(sid) for sid in station_ids_arr)
dataset.createDimension('sid_len', max_sid_length)
station_var = dataset.createVariable("sid", 'S11', ('station', 'sid_len'))

lat_var = dataset.createVariable("lat", "f4", ("station",))
lon_var = dataset.createVariable("lon", "f4", ("station",))
precip_var = dataset.createVariable("precip", "f4", ("station", "time"), fill_value=np.nan)
precip_var.units = "mm"

# Store station metadata and precipitation data
station_ids = list(stations.keys())
lat_var[:] = np.array([stations[sid]["lat"] for sid in station_ids])
lon_var[:] = np.array([stations[sid]["lon"] for sid in station_ids])
#station_var[:] = np.array(station_ids)
station_ids_padded = np.array([sid.ljust(max_sid_length) for sid in station_ids], dtype='S' + str(max_sid_length))
station_var[:] = nc.stringtochar(station_ids_padded)
#print(np.array([stations[sid]["sid"] for sid in station_ids]))
#print(np.array(station_ids))
#print(station_var[:]) 
#print(lon_var[:])
#print(np.array([stations[sid]["lon"] for sid in station_ids]))
#station_var[:] = np.array([sid.encode("ascii") for sid in station_ids])

precip_array = np.full((num_stations, num_times), np.nan, dtype=np.float32)
for i, sid in enumerate(station_ids):
    for date, value in zip(stations[sid]["dates"], stations[sid]["precip"]):
        precip_array[i, time_index[date]] = value

precip_var[:, :] = precip_array

dataset.close()
print(f"NetCDF file saved as {output_file}")

## open the file
ds = pyg.open(f"{output_file}")
station_ids = [''.join(sid.astype(str)) for sid in ds.sid[:]]
print(station_ids)