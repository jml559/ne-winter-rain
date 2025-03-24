import cdsapi
import calendar
import sys, os
import time

start_time = time.time()
path = "/local1/storage1/jml559/ne-winter-rain/era5/"

""" start_year = 1940
end_year = 2024

months = [1, 2, 11, 12]

for yr in range(start_year, end_year + 1):
    for mn in months:
        if mn in [1, 2] and yr == start_year: # skip Jan, Feb start_year
            continue
        if mn in [11, 12] and yr == end_year: # skip Nov, Dec end_year
            continue 

        dy = calendar.monthrange(yr, mn)[1] # Determine the last day of the month
        dt = "%04d-%02d-01/to/%04d-%02d-%02d" % (yr, mn, yr, mn, dy)
        fn = path + "plevels_1h_%04d%02d_T_w.grib" % (yr, mn) ## change filename variables

        if os.path.exists(fn): continue

        days = [f"{day:02d}" for day in range(1, dy + 1)]

        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'grib',
                'variable': [
                    'temperature', 'vertical_velocity',
                ], # 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                # 'geopotential', 'relative_humidity',
                'pressure_level': [
                    '1', '2', '3',
                    '5', '7', '10',
                    '20', '30', '50',
                    '70', '100', '125',
                    '150', '175', '200',
                    '225', '250', '300',
                    '350', '400', '450',
                    '500', '550', '600',
                    '650', '700', '750',
                    '775', '800', '825',
                    '850', '875', '900',
                    '925', '950', '975',
                    '1000',
                ],
                'year': str(yr),
                'month': f"{mn:02d}",
                'day': days,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    50, -130, 5,
                    -55,
                ],
            },
            fn)  """

"""c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': [
            'geopotential', 'specific_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
        ],
        'pressure_level': ['300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000',
        ],
        'year': '1979',
        'month': '01',
        'day': [
            '12', '13', '14',
            '15', '16', '17',
            '18', '19', '20',
            '21', 
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            50, -130, 5,
            -55,
        ],
    },
    path+'plevels_ending_19790121.grib')  """
# 6 vars x 37 levels x 24 h x 31 days = 165168 items > 60000 item limit

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"
    ],
    "year": ["1979"],
    "month": ["01"],
    "day": [
        "12", "13", "14",
        "15", "16", "17",
        "18", "19", "20",
        "21"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "pressure_level": [
        "300", "350", "400",
        "450", "500", "550",
        "600", "650", "700",
        "750", "775", "800",
        "825", "850", "875",
        "900", "925", "950",
        "975", "1000"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [50, -130, 5, -55]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

print(time.time() - start_time)



# eventually, will need to find a way to combine contents of GRIB files