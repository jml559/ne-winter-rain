import pygeode as pyg
import numpy as np
import pandas as pd
import requests
from collections import defaultdict, Counter

ds = pyg.open("ghcn_daily_1960to2024.nc")
lat = np.round(ds.lat[:],3)
lon = np.round(ds.lon[:],3)
station_ids = [''.join(sid.astype(str)) for sid in ds.sid[:]] # get list of station IDs
precip_data = ds.precip[:].T # shape (days, stations), to get time series per station

# list of date strings 
all_dates = pd.date_range(start="1960-01-01", end="2024-12-31", freq="D")
date_strings = all_dates[all_dates.month.isin([1, 2, 12])].strftime("%Y-%m-%d").tolist()

# make a list of the top 20 dates with highest area-wide rainfall 
""" precip_sum = ds.precip.sum("station")

total_precip = precip_sum[:]
index = ds.time[:]
sorted_indices = total_precip.argsort()[::-1]
top_20_indices = sorted_indices[:20]

# Print top 20 dates and their total precip values
for idx in top_20_indices:
    print(f"{index}: {total_precip[idx]:.2f} in") """

# get dates of extreme events (outputted to extreme_events_list.txt)
"""
So far I have this code. I have computed the station-specific 90th percentile
precipitation. All days with precip amounts exceeding pct_90 qualifies as an 
extreme event. Note that len(precip_data[:,0]) = len(date_strings) = 5867, 
and the precip data is ordered chronologically. 

1) I want my output to have the following format, where the station and sid is listed, 
followed by a list of corresponding dates and the precip totals for extreme events:

Station USW00014764: # f"{station_ids[0]}" 
1960-01-03: 1.27 in 
1963-02-02: 2.36 in 
etc. (repeating for all stations)

2) Next, I would like to count the occurrences of extreme event by date. Rank from 
most numnber of extremes to least number of extremes. 
1983-12-13: 28 stations
1990-01-06: 26 stations
etc. (repeating for all dates with at least 1 extreme event)
"""

""" station_extreme_events = defaultdict(list) 
date_extreme_counts = Counter()
pct_99_list = []

for i, station_id in enumerate(station_ids):
    pct_99 = np.nanpercentile(precip_data[:,i], 99)
    pct_99_list.append(round(pct_99,2))

    for j, (date, precip) in enumerate(zip(date_strings, precip_data[:, i])):
        if precip > pct_99:  # Check if it qualifies as an extreme event
            station_extreme_events[station_id].append(f"{date}: {precip:.2f} in")
            date_extreme_counts[date] += 1  # Count occurrences

for i, (station_id, events) in enumerate(station_extreme_events.items()):
    print(f"Station {station_id} with lat {lat[i]:.3f} and lon {lon[i]:.3f}")
    for event in events:
        print(event)
    print()

print("Extreme event occurrences ranked by date:")
for date, count in date_extreme_counts.most_common():
    print(f"{date}: {count} stations")

print(pct_99_list)"""

# scatterplot of stations along with their 99pct 
"""station_extreme_events = defaultdict(list) 
date_extreme_counts = Counter()

for i, station_id in enumerate(station_ids):
    pct_99 = np.nanpercentile(precip_data[:,i], 99)"""







# hours
"""def get_extreme_hrs(sid):
    for sdate in station_extreme_events[sid]:
        sdate = sdate.split(":")[0].strip() 
        #edate = pd.to_datetime(sdate) + pd.Timedelta(days=1)  # next day
        #edate = edate.strftime("%Y-%m-%d")   

        input_dict = {
            "sid": sid,  
            "sdate": sdate,
            "edate": sdate, 
            "elems": [{"vX":5}], # hourly precip code
        }

        req = requests.post('http://data.nrcc.rcc-acis.org/StnData', json = input_dict)
        data = req.json()
        print(data)

for id in station_ids:
    get_extreme_hrs(id)"""

"""
Now I have a list of extreme days for each station stored in a text file, 
extreme_events_list_days.txt. Sample output:

Station USW00014764:
1960-02-19: 2.75 in
1962-12-06: 3.13 in
1965-02-25: 3.21 in
etc.

Station USW00014745:
1960-01-03: 1.23 in
1962-12-06: 1.66 in
1965-02-25: 1.29 in
etc.

Add code for the following. I have already started with def get_extreme_hrs(sid).
For each site:
    For each date:
        Gather the hourly precip data for that date. 

Note: dates_per_station is a list of station-specific extreme dates that is stored 
in station_extreme_events[station_id]. 
"""