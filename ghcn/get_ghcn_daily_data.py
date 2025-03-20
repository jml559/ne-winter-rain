import requests
import re 
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

sids = ["USW00014764", "USW00014745", "USW00093730", "USW00014607", "USW00014734",
    "USW00014739", "USW00093720", "USW00093721", "USW00013781", "USW00014732",
    "USW00094728", "USW00004725", "USW00014735", "USW00014733", "USW00014768",
    "USW00094725", "USW00013739", "USC00306196", "USC00368379", "USW00014778",
    "USW00014777", "USW00014742", "USW00094823", "USC00374266", "USW00014737",
    "USW00094702", "USW00014740", "USC00304174"]

# counts the number of complete stations in a given time interval
# with at least (100*threshold)% data availability 
def count_complete_stations(start_year, end_year, threshold):
    date_ranges = [] # stores time intervals Jan 1-Feb 28/29, and Dec 1-Dec 31

    # appends start and end dates to date_ranges array
    for year in range(start_year, end_year + 1):
        date_ranges.append((f"{year}-01-01", f"{year}-02-28"))
        date_ranges.append((f"{year}-12-01", f"{year}-12-31"))

    # configure end dates for each (sdate,edate) period requested
    for i, (sdate, edate) in enumerate(date_ranges):
        end_date = datetime.strptime(edate, "%Y-%m-%d")
        if end_date.month == 2 and end_date.year % 4 == 0: # leap year
            date_ranges[i] = (sdate, edate.replace("28", "29"))

    # compute total days in overall time period 
    total_djf_days = sum((datetime.strptime(edate, "%Y-%m-%d") - 
        datetime.strptime(sdate, "%Y-%m-%d")).days + 1 for sdate, edate in date_ranges)

    station_valid_days_tracker = defaultdict(dict)
    station_metadata = {}

    # Read UIDs from the text file
    uids = []
    with open('logfile_d_1960to2024_final_list.txt', 'r') as file:
        for line in file:
            match = re.search(r'UID: (\d+)', line)
            if match:
                uids.append(match.group(1))
    print(uids)
    
    for sdate, edate in date_ranges:
        input_dict = {
            "sids": uids,  # Filter stations by UIDs,
            "sdate": sdate,
            "edate": edate,
            "elems": [{"name": "pcpn"}],  # daily precip code
        }

        req = requests.post('http://data.nrcc.rcc-acis.org/MultiStnData', json = input_dict)
        data = req.json()
        print(f"\n This is data starting {sdate} ending {edate}")
        print(str(data["data"]).replace("{'meta':", "\n{'meta':"))

        for station in data["data"]:
            #print(station)
            station_id = station["meta"]["uid"]
            station_name = station["meta"]["name"]
            station_coords = station["meta"].get("ll", "NA")
            station_values = np.array(station["data"]).flatten()
            station_valid_days = sum(1 for day in station_values if day not in ['', 'M'])

            if station_id not in station_metadata:
                station_metadata[station_id] = {
                    "name": station_name,
                    "coords": station_coords,
                } # collect name and coordinates of station, avoid duplicates

            station_valid_days_tracker[station_id][(sdate, edate)] = station_valid_days

    total_valid_days = {
        station_id: sum(date_range.values())  # Sum over all time intervals
        for station_id, date_range in station_valid_days_tracker.items()
    }

    print("\nStation Completeness (days):")
    for station_id, days in total_valid_days.items(): 
        print(f"{station_id}: {days} of {total_djf_days}") 

    print("\nSufficiently complete stations:")
    sufficient_stations = 0
    for station_id, hrs in total_valid_days.items():
        completeness = hrs / total_djf_days
        if completeness >= threshold:
            sufficient_stations += 1
            metadata = station_metadata[station_id]
            print(f"{metadata['name']}, UID: {station_id}, "
                  f"Coords: {metadata['coords']}, Completeness: {completeness:.2%}")

    return sufficient_stations

syr = 1960
eyr = 2024
print(f"\nComplete stations in {syr}-{eyr}: {count_complete_stations(syr, eyr, 0.9)}")
 
def get_extreme_events(station_ids, start_year, end_year):
    date_ranges = [] # stores time intervals Jan 1-Feb 28/29, and Dec 1-Dec 31

    # appends start and end dates to date_ranges array
    for year in range(start_year, end_year + 1):
        date_ranges.append((f"{year}-01-01", f"{year}-02-28"))
        date_ranges.append((f"{year}-12-01", f"{year}-12-31"))

    # configure end dates for each (sdate,edate) period requested
    for i, (sdate, edate) in enumerate(date_ranges):
        end_date = datetime.strptime(edate, "%Y-%m-%d")
        if end_date.month == 2 and end_date.year % 4 == 0: # leap year
            date_ranges[i] = (sdate, edate.replace("28", "29"))

    extreme_events = {}

    for station_id in station_ids:
        all_station_values = []  # Store all valid data across date ranges

        for sdate, edate in date_ranges:
            input_dict = {
                "sid": f"{station_id}",
                "sdate": sdate,
                "edate": edate,
                "elems": [{"name": "pcpn"}], # daily precip code
            }

            req = requests.post('http://data.nrcc.rcc-acis.org/StnData', json=input_dict)
            data = req.json()

            if "data" not in data or not data["data"]:
                print(f"No data for station {station_id} in range {sdate}-{edate}, skipping.")
                continue
    
            # extract (date, precip) pairs
            station_values = [(entry[0], entry[1]) for entry in data["data"] if entry[1] not in ["", "M"]]
            all_station_values.extend(station_values)

            if not station_values:
                print(f"No valid precipitation data for station {station_id}, skipping.")
                continue # Skip if no valid data

        rainfall_values = np.array([
            0.0 if value[1] == 'T' # trace precip is converted to 0
            else float(value[1]) # convert numerical values to float 
            for value in all_station_values
        ]) # change if needed

        if len(rainfall_values) < 0.9*5867:  # 90% completeness
            print(f"Not enough valid data for station {station_id}, skipping.")
            continue

        threshold = np.percentile(rainfall_values, 99)

        extreme_days = [(station_values[i][0], rainfall)  # Keep original date from station_values
            for i, rainfall in enumerate(rainfall_values) if rainfall >= threshold]
        extreme_events[station_id] = extreme_days
    
    return extreme_events

"""syr, eyr = 1960, 2024
sufficiently_complete_stations = ["9434", "9443", "9488", "9495", "9503", "9518", "9527", "9535"]  # Example UIDs
extreme_events_data = get_extreme_events(sufficiently_complete_stations, syr, eyr)

for station_id, events in extreme_events_data.items():
    print(f"\nExtreme events for station {station_id}:")
    for date, rainfall in events:
        print(f"Date: {date}, Rainfall: {rainfall:.2f} inches") """

# get station_ids from the text file

def inspect_site_metadata(start_year, end_year, uid):
    date_ranges = []

    # appends start and end dates to date_ranges array
    for year in range(start_year, end_year + 1):
        date_ranges.append((f"{year}-01-01", f"{year}-02-28"))
        date_ranges.append((f"{year}-12-01", f"{year}-12-31"))

    # configure end dates for each (sdate,edate) period requested
    for i, (sdate, edate) in enumerate(date_ranges):
        end_date = datetime.strptime(edate, "%Y-%m-%d")
        if end_date.month == 2 and end_date.year % 4 == 0: # leap year
            date_ranges[i] = (sdate, edate.replace("28", "29"))

    # compute total days in overall time period 
    total_djf_days = sum((datetime.strptime(edate, "%Y-%m-%d") - 
        datetime.strptime(sdate, "%Y-%m-%d")).days + 1 for sdate, edate in date_ranges)

    station_valid_days_tracker = defaultdict(dict)

    for sdate, edate in date_ranges: 
        input_dict = {
            "sid": uid,
            "sdate": sdate,
            "edate": edate,
            "elems": [{"name": "pcpn"}],  # Daily precip code
        }

        req = requests.post('http://data.nrcc.rcc-acis.org/StnData', json=input_dict)
        data = req.json()

        print(f"\n This is data starting {sdate} ending {edate}")
        print(data)
           
        if "data" not in data or not data["data"]:
            print(f"No data available for station {uid} in the period {sdate} to {edate}.")
            continue

        print(f"\nInspecting data for station {uid} from {sdate} to {edate}.")

        station_values = np.array(data["data"]).flatten()
        station_valid_days = sum(1 for day in station_values if day not in ['', 'M'])
        station_valid_days_tracker[(sdate, edate)] = station_valid_days

    # Compute total valid days for the station
    total_valid_days = sum(station_valid_days_tracker.values()) 

    print(f"\nMetadata for Station {uid}:")
    print(f"Total Valid Days: {total_valid_days} of {total_djf_days}")
    print(f"Completeness: {total_valid_days / total_djf_days:.2%}")

"""syr, eyr = 1960, 2024
uid = "32682"  # Target station
inspect_site_metadata(syr, eyr, uid)"""

# then go back to the problematic station (9434)