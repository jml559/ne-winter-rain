""" import cdsapi
import calendar
import sys, os

c = cdsapi.Client()

yr = int(sys.argv[1])

for mn in [1, 2, 11, 12]: # want NDJF to collect trajectories from late Nov
    dy = calendar.monthrange(yr, mn)[1]

    dt = "%04d-%02d-01/to/%04d-%02d-%02d" % (yr, mn, yr, mn, dy)
    fn = "sfc_analysis_1h_%04d%02d.grib" % (yr, mn)

    if os.path.exists(fn): continue

    d = {


    }

#c.retrieve("reanalysis-era5-single-levels", d, fn)

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'boundary_layer_height', 'convective_available_potential_energy',
            'friction_velocity', 'geopotential', 'sea_surface_temperature',
            'surface_latent_heat_flux', 'surface_pressure', 'surface_sensible_heat_flux',
            'surface_solar_radiation_downwards', 'total_cloud_cover', 'total_precipitation',
        ],
        'year': '2024',
        'month': '01',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10',
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
    'download.grib') 
    
    ### change """

# runtime <1 min for 10 days
# file size 379MB with download speed ~11MB/s


import cdsapi
import calendar
import sys, os

path = "/local1/storage1/jml559/ne-winter-rain/era5/"
c = cdsapi.Client()

start_year = 1940
end_year = 2024

months = [1, 2, 11, 12]

for yr in range(start_year, end_year + 1):
    for mn in months:
        if mn in [1, 2] and yr == start_year: # skip Jan, Feb 1940
            continue
        if mn in [11, 12] and yr == end_year: # skip Nov, Dec 2024
            continue

        dy = calendar.monthrange(yr, mn)[1] # Determine the last day of the month
        dt = "%04d-%02d-01/to/%04d-%02d-%02d" % (yr, mn, yr, mn, dy)
        fn = path + "sfc_analysis_1h_%04d%02d.grib" % (yr, mn)

        if os.path.exists(fn): continue

        days = [f"{day:02d}" for day in range(1, dy + 1)]

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'grib',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                    '2m_temperature', 'boundary_layer_height', 'convective_available_potential_energy',
                    'friction_velocity', 'geopotential', 'sea_surface_temperature',
                    'surface_latent_heat_flux', 'surface_pressure', 'surface_sensible_heat_flux',
                    'surface_solar_radiation_downwards', 'total_cloud_cover', 'total_precipitation',
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
            fn)