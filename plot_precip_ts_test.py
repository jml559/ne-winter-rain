import numpy as np
import matplotlib.pyplot as plt
import requests 
import json 

def plot_precip_ts(sid, sdate, edate, out_file): # title, out_file
    input_dict = {"sid":sid,"sdate":sdate,"edate":edate,"elems":[{"vX":5,"units":"mm"}]}
    req = requests.post("https://data.nrcc.rcc-acis.org/StnData", json = input_dict)
    json_data = req.json()

    precip_data = json_data['data']

    precip_sdate = []
    precip_edate = []

    for entry in precip_data:
        if entry[0] == sdate:
            precip_sdate = entry[1]
        elif entry[0] == edate:
            precip_edate = entry[1]

    precip_data = precip_sdate[-5:] + precip_edate[:14]
    precip_data = [float(p) for p in precip_data]
    times = ["{:02d}".format(i) for i in range(19)]
    title = "Precipitation Time Series for site " + sid

    fig, ax = plt.subplots()
    print(times)
    print(precip_data)
    ax.bar(times, precip_data)  
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Precipitation (mm)')
    ax.set_title(title)
    plt.savefig("./figures/" + out_file)

# edate must be sdate + 1
#plot_precip_ts("KBWI", "2023-12-17", "2023-12-18", "121823_KBWI_precip_ts.pdf")
plot_precip_ts("KCDW", "2023-12-17", "2023-12-18", "121823_KCDW_precip_ts.pdf")
plot_precip_ts("KPVD", "2023-12-17", "2023-12-18", "121823_KPVD_precip_ts.pdf")
plot_precip_ts("KLEW", "2023-12-17", "2023-12-18", "121823_KLEW_precip_ts.pdf")








""">>> input_dict = {"sid":"KBWI","sdate":"2024-01-08","edate":"2024-01-10","elems":[{"vX":5}]}
>>> req = requests.post("https://data.nrcc.rcc-acis.org/StnData", json = input_dict)
>>> data = req.json()
>>> data
{'meta': {'uid': 31771, 'll': [-76.68408, 39.17329], 'sids': ['93721 1', '180465 2', 'BWI 3', 'BAL 3', '72406 4', 'KBWI 5', 'USW00093721 6', 'BWI 7', 'BAL 7', 'USW00093721 32'], 'state': 'MD', 'elev': 138.0, 'name': 'BALTIMORE-WASHINGTON INTERNATIONAL AIRPORT'}, 'data': [['2024-01-08', ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00']], ['2024-01-09', ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.05', '0.10', '0.13', '0.09', '0.05', '0.15', '0.35', '0.24', '0.14', '0.54', '0.40', '0.26', '0.11', '0.00', '0.01']], ['2024-01-10', ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00']]]}
>>> """

# {"sid":"KPVD","sdate":"2024-01-08","edate":"2024-01-10","elems":[{"vX":5,"units":"mm"},{"vX":23}]}