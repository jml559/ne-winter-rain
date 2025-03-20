import requests
from datetime import datetime, timedelta
import time

start_time = time.time()

#uids = ['9443','17568','17710','9613','17832','9053','9189','31771','29634','18343','18345','18571','18747','18795','18846','19003','95','18518','15690','16121','29616','29699','15920','16366','15953','4224','29569','18667']
sids = ["USW00014764", "USW00014745", "USW00093730", "USW00014607", "USW00014734",
    "USW00014739", "USW00093720", "USW00093721", "USW00013781", "USW00014732",
    "USW00094728", "USW00004725", "USW00014735", "USW00014733", "USW00014768",
    "USW00094725", "USW00013739", "USC00306196", "USC00368379", "USW00014778",
    "USW00014777", "USW00014742", "USW00094823", "USC00374266", "USW00014737",
    "USW00094702", "USW00014740", "USC00304174"]

def print_raw_data(start_year, end_year):
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

    # for each site, for each date range, process the metadata 
    for sid in sids:
        #print("\n")
        print(f"Station {sid}:")
        for sdate, edate in date_ranges:
            input_dict = {
                "sid": sid,  
                "sdate": sdate,
                "edate": edate,
                "elems": [{"name": "pcpn"}],  # daily precip code
            }

            req = requests.post('http://data.nrcc.rcc-acis.org/StnData', json = input_dict)
            data = req.json()
            print(data)
            #print("\n".join([str(data)[i:i+100] for i in range(0, len(str(data)), 100)])) # indent every 100 characters
            #print("\n")

syr = 1960
eyr = 2024
print_raw_data(syr, eyr)

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime:.4f} seconds")




# list out SIDs (starting USC/USW)
# make a list of SIDs (list of strings) NOT UIDs

# although below code still did not work, I am using a method
# where I am just running a loop
"""input_dict = {
    "sids": sids,  # Filter stations by UIDs,
    "sdate": "1960-01-01",
    "edate": "1960-02-28",
    "elems": [{"name": "pcpn"}],  # daily precip code
}

req = requests.post('http://data.nrcc.rcc-acis.org/StnData', json = input_dict)
data = req.json()
print(data) """