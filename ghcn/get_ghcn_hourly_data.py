import requests
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

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

    # compute total hours in overall time period 
    total_djf_hours = 24 * sum((datetime.strptime(edate, "%Y-%m-%d") - 
        datetime.strptime(sdate, "%Y-%m-%d")).days + 1 for sdate, edate in date_ranges)

    station_valid_hours_tracker = defaultdict(dict)
    station_metadata = {}

    for sdate, edate in date_ranges:
        input_dict = {
            "state": "ME,VT,NH,MA,CT,RI,NY,NJ,PA,MD,DE,DC",
            "sdate": sdate,
            "edate": edate,
            "elems": [{"vX":5}], # hourly precip code
        }
        #print(sdate)
        #print(edate)
        #print(input_dict)

        req = requests.post('http://data.nrcc.rcc-acis.org/MultiStnData', json = input_dict)
        data = req.json()
        print(f"\n This is data starting {sdate} ending {edate}")
        #print(data["data"]) # remove ["data"], just print data alone?

        for station in data["data"]: 
            station_id = station["meta"]["uid"]
            station_name = station["meta"]["name"]
            station_coords = station["meta"]["ll"]
            station_values = np.array(station["data"]).flatten()
            #print(station_id)
            #print(len(station_values))
            #print(station_name)
            #print(station_values)
            #valid_hours = sum(1 for day in station_values for hour in day if hour not in ["M", ""])
            station_valid_hours = sum(1 for item in station_values if item not in ['', 'M'])
            #print(f"Station {station_id} has {station_valid_hours} valid hours from {sdate}.")

            if station_id not in station_metadata:
                station_metadata[station_id] = {
                    "name": station_name,
                    "coords": station_coords,
                } # collect name and coordinates of station, avoid duplicates
    
            """if station_id not in station_valid_hours_tracker:
                station_valid_hours_tracker[station_id] = 0  """

            """station_valid_hours_tracker[station_id][(sdate,edate)] = {
                "valid_hours": station_valid_hours 
            }"""
            #print(f"{station_id} is zero for start date {sdate}") 

            # Store valid_hours separately for each date range:
            station_valid_hours_tracker[station_id][(sdate, edate)] = station_valid_hours

    #total_valid_hours = sum(station_valid_hours_tracker[station_id].values())
    total_valid_hours = {
        station_id: sum(date_range_hours.values())  # Sum over all time intervals
        for station_id, date_range_hours in station_valid_hours_tracker.items()
    }
    #print(f"{station_id} has {total_valid_hours}")

    #total_valid_hours_per_station = {
        #station_id: sum(date_range_hours.values())
        #for station_id, date_range_hours in station_valid_hours_tracker.items()
    #}       
    #print(data["data"]) 
    # for each station, associate with a date (eventually?) 

    print("\nStation Completeness (hours):")
    for station_id, hrs in total_valid_hours.items(): # include total_valid_hours in for loop?
        print(f"{station_id}: {hrs} of {total_djf_hours}") 

    print("\nSufficiently complete stations:")
    sufficient_stations = 0
    for station_id, hrs in total_valid_hours.items():
        completeness = hrs / total_djf_hours
        if completeness >= threshold:
            sufficient_stations += 1
            metadata = station_metadata[station_id]
            print(f"{metadata['name']}, UID: {station_id}, "
                  f"Coords: {metadata['coords']}, Completeness: {completeness:.2%}")

    return sufficient_stations

#print(f"Complete stations in 1960-2024: {count_complete_stations(1960, 2024, 0.8)}")
#print(f"Complete stations in 1970-2024: {count_complete_stations(1970, 2024, 0.8)}")

syr = 2000
eyr = 2024
print(f"\nComplete stations in {syr}-{eyr}: {count_complete_stations(syr, eyr, 0.8)}") 

""" syr = 2000
eyr = 2024
print(f"\nComplete stations in {syr}-{eyr}: {count_complete_stations(syr, eyr, 0.8)}") """
