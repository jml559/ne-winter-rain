# code modified from data.py by Kara Hartig 
import pygeode as pyg
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import xarray as xr
import pandas as pd
import cftime
from datetime import datetime
import calendar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import imageio.v3 as iio

class TrajectoryFile:
    def __init__(self, filepath):
        '''
        Parameters
        ----------
        filepath:  string
            file path to HYSPLIT trajectory file
        '''
        # open the .traj file
        file = open(filepath, 'r')

        # Header 1
        #    number of meteorological grids used
        header_1 = file.readline()
        header_1 = (header_1.strip()).split()
        ngrids = int(header_1[0])

        # read in list of grids used
        h1_columns = ['model', 'year', 'month', 'day', 'hour', 'fhour']
        h1_dtypes = ['str', 'int32', 'int32', 'int32', 'int32', 'int32']

        # loop over each grid
        grids_list = []
        for i in range(ngrids):
            line = file.readline().strip().split()
            grids_list.append(line)
        grids_df = pd.DataFrame(grids_list, columns=h1_columns)
        self.grids = grids_df.astype(dict(zip(h1_columns, h1_dtypes)))

        # Header 2
        #    col 0: number of different trajectories in file
        #    col 1: direction of trajectory calculation (FORWARD, BACKWARD)
        #    col 2: vertical motion calculation method (OMEGA, THETA, ...)
        header_2 = file.readline()
        header_2 = (header_2.strip()).split()
        ntraj = int(header_2[0])          # number of trajectories
        direction = header_2[1]           # direction of trajectories
        vert_motion = header_2[2]         # vertical motion calculation method
        self.ntraj = ntraj
        self.direction = direction

        # read in list of trajectories
        h2_columns = ['year', 'month', 'day', 'hour', 'lat', 'lon', 'height']
        h2_dtypes = ['int32', 'int32', 'int32',
                     'int32', 'float32', 'float32', 'float32']
        
        # loop over each trajectory
        traj_start_list = []
        for i in range(ntraj):
            line = file.readline().strip().split()
            traj_start_list.append(line)
        traj_df = pd.DataFrame(traj_start_list, columns=h2_columns)
        self.traj_start = traj_df.astype(dict(zip(h2_columns, h2_dtypes)))
        
        # Format end_time string 
        first_line = traj_start_list[0]

        year = str(first_line[0])
        month = str(int(first_line[1]))   
        day = str(int(first_line[2]))   
        hour = f"{int(first_line[3]):02}"   

        end_time = f"{month}-{day}-{year[-2:]} {hour} UTC"
        self.end_time = end_time

        # Header 3
        #    col 0 - number (n) of diagnostic output variables
        #    col 1+ - label identification of each of n variables (PRESSURE,
        #             AIR_TEMP, ...)
        header_3 = file.readline()
        header_3 = header_3.strip().split()
        nvars = int(header_3[0])  # number of diagnostic variables
        self.diag_var_names = header_3[1:]

        file.close()

        # skip over header; length depends on number of grids and trajectories
        traj_skiprow = 1 + ngrids + 1 + ntraj + 1

        # set up column names, dtype, widths
        traj_columns = ['traj #', 'grid #', 'year', 'month', 'day', 'hour',
                        'minute', 'fhour', 'traj age', 'lat', 'lon', 'height (m)', 
                        'pressure (hPa)', 'precip (mm)', 'specific humidity (g/kg)']
        traj_dtypes = {'traj #': int, 'grid #': int, 'year': int, 'month': int, 'day': int, 'hour': int,
                       'minute': int, 'fhour': int, 'traj age': int, 'lat': float, 'lon': float, 
                       'height (m)': float, 'pressure (hPa)': float, 'precip (mm)': float, 
                       'specific humidity (g/kg)': float}
        col_widths = [6, 6, 6, 6, 6, 6, 6, 6, 8, 9, 9, 9, 9, 9]
        for var in self.diag_var_names:
            col_widths.append(9)
            traj_columns.append(var)
            traj_dtypes[var] = float

        # read in file in fixed-width format
        trajectories = pd.read_fwf(filepath, widths=col_widths, names=traj_columns, dtype=traj_dtypes,
                           skiprows=traj_skiprow).set_index(['traj #', 'traj age'])
        trajectories.sort_index(inplace=True)

        # remove columns that have become indices before changing dtype
        del traj_dtypes['traj #']
        del traj_dtypes['traj age']
        trajectories = trajectories.astype(traj_dtypes)

        # new column: datetime string
        def traj_datetime(row):
            return '00{:02.0f}-{:02.0f}-{:02.0f} {:02.0f}:{:02.0f}:00'.format(row['year'], row['month'], row['day'], row['hour'], row['minute'])
        trajectories['datetime'] = trajectories.apply(traj_datetime, axis=1)

        # new column: cftime Datetime objects
        def traj_cftimedate(row):
            return cftime.DatetimeNoLeap(row['year'], row['month'], row['day'], row['hour'], row['minute'])
        trajectories['cftime date'] = trajectories.apply(traj_cftimedate, axis=1)

        # new column: ordinal time (days since 0001-01-01 00:00:00)
        # min_time = cftime.date2num(cftime.datetime(7 + winter_idx, 12, 1),
        # time_object.units, calendar=time_object.calendar)
        def traj_numtime(row):
            return cftime.date2num(row['cftime date'], units='days since 0001-01-01 00:00:00', calendar='Gregorian')
        trajectories['ordinal time'] = trajectories.apply(traj_numtime, axis=1)

        # Store trajectories in increments of 1 hour and 3 hours
        # default self.data will be every 3 hours to match CAM output frequency
        self.data_1h = trajectories

    def get_trajectory(self, trajectory_number, hourly_interval=None, age_interval=None):
        '''
        Return data from every hourly_interval hours or age_interval age for a
        single trajectory

        Must choose between hourly_interval and age_interval, as they are
        mutually exclusive methods for selecting trajectory data

        Parameters
        ----------
        trajectory_number: int
            which trajectory to retrieve
        hourly_interval: int or float
            retrieve trajectory values every hourly_interval hours, based on
            'hour' data column
        age_interval: int or float
            retrieve trajectory values every age_interval hours, based on
            'traj age' index

        Returns
        -------
        pandas DataFrame of trajectory data every X hours, where X is set by
        either hourly_interval or age_interval
        '''

        single_trajectory = self.data_1h.loc[trajectory_number]

        if (hourly_interval is not None) and (age_interval is None):
            interval_col = single_trajectory['hour']
            interval = hourly_interval
        elif (hourly_interval is None) and (age_interval is not None):
            interval_col = single_trajectory.index
            interval = age_interval
        else:
            raise ValueError("Must provide interval for retrieval either by hour-of-day or by trajectory age in hours")
        
        iseveryxhours = interval_col % interval == 0
        return single_trajectory[iseveryxhours]

    def col2da(self, trajectory_number, data_column, include_coords=None):
        '''
        Convert any trajectory data column into an xarray.DataArray with
        additional coordinate(s) given by other column(s)

        Dimension of resulting DataArray will be highest-order index in 
        trajectory data, self.data. If additional columns are requested in
        include_coords, they will share the same dimension.

        Parameters
        ----------
        trajectory_number: int
            number corresponding to specific trajectory of interest
            Trajectory data is retrieved with self.data.loc[trajectory_number]
        data_column: string
            key of data column to be converted to DataArray
        include_coords: string or list of strings
            keys for other columns to be included as additional coordinates in
            the DataArray produced
            Default is None: no additional coordinates

        Returns
        -------
        column_da: xarray.DataArray 
        trajectory data as a DataArray. The indexing dimension is the
            highest-order index in TrajectoryFile.data, and additional
            coordinates can be added with include_coords
        '''
         
        trajectory = self.data.loc[trajectory_number]
        column_da = xr.DataArray.from_series(trajectory[data_column])

        # Retrieve name of dimension
        if len(column_da.dims) == 1:
            indexer_name = column_da.dims[0]
        else:
            raise ValueError("Trajectory data has too many dimensions: {}.\
                Expecting only 1 indexer like 'traj age' on a single\
                trajectory".format(column_da.dims))

        # Add extra columns as coordinates
        if isinstance(include_coords, str):
            include_coords = (include_coords, ) # make string an iterable

        if include_coords is not None:
            for column in include_coords:
                column_da = column_da.assign_coords({column: (indexer_name, trajectory[column])})
        return column_da

    # f = TrajectoryFile(path)
    # print(f.grids)
    # print(f.ntraj)
    # print(f.direction) # etc. # need to initialize the object w/parameters  

    def plot_trajectories(self, location, out_file):
        # create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), subplot_kw={'projection': ccrs.PlateCarree()})
        #self.data_1h['datetime'] = pd.to_datetime(self.data_1h['datetime'], format='%y-%m-%d %H:%M:%S')
        #end_time = self.data_1h.iloc[0]['datetime'].strftime('%-m-%-d-%y %H UTC')
        fig.suptitle(f'Back trajectories ending {self.end_time} at ' 
            + location + ', 5 days', fontweight="bold") # automate also back traj time (__-day back traj)

        for ax in [ax1, ax2]:
            ax.set_extent([-110, -55, 15, 50], crs=ccrs.PlateCarree()) # -130, -55, 5, 50
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAKES, alpha=0.5)

        """for traj_num in range(traj_file.ntraj):
            trajectory = traj_file.get_trajectory(traj_num + 1)

            lats = trajectory['lat'].values
            lons = trajectory['lon'].values
            start_height = trajectory.loc[(traj_num + 1, 0), 'height (m)']

            # Plotting trajectories
            ax.plot(lons, lats, transform=ccrs.PlateCarree())

            # Adding markers at 24-hour intervals
            for idx, traj_age in enumerate(trajectory.index.get_level_values('traj age')):
                if abs(traj_age) % 24 == 0 and traj_age != 0:
                    ax.plot(lons[idx], lats[idx], marker='o', markersize=5, color=color, transform=ccrs.PlateCarree())  """ 
        
        cmap_1 = cm.get_cmap('plasma_r')
        cmap_2 = cm.get_cmap('viridis')

        # Plot each trajectory
        for traj_num in range(1, self.ntraj + 1):
            trajectory = self.data_1h.loc[traj_num]

            lats = trajectory['lat'].values
            lons = trajectory['lon'].values
            heights = trajectory['height (m)'].values
            sp_hum = trajectory['specific humidity (g/kg)'].values
            
            # Plot based on trajectory height z(x(t),y(t),t)
            norm_1 = plt.Normalize(0, 6000)
            for i in range(len(lats)):
                ax1.plot(lons[i:i+2], lats[i:i+2], color=cmap_1(norm_1(heights[i])), linewidth=1, transform=ccrs.PlateCarree())

                # adding markers at 24-h intervals
                traj_age = trajectory.index.get_level_values('traj age')[i] 
                if abs(traj_age) % 24 == 0 and traj_age != 0:
                    ax1.plot(lons[i], lats[i], marker='o', markersize=5, color=cmap_1(norm_1(heights[i])), transform=ccrs.PlateCarree())
                #if traj_age == 0:
                    #ax.plot(lons[i], lats[i], marker='*', markersize=7, color=cmap(norm(heights[i])), transform=ccrs.PlateCarree())"""

            # Plot based on specific humidity q(x(t),y(t),t)
            norm_2 = plt.Normalize(0, 14)
            for i in range(len(lats)):
                ax2.plot(lons[i:i+2], lats[i:i+2], color=cmap_2(norm_2(sp_hum[i])), linewidth=1, transform=ccrs.PlateCarree())

                # adding markers at 24-h intervals
                traj_age = trajectory.index.get_level_values('traj age')[i]
                if abs(traj_age) % 24 == 0 and traj_age != 0:
                    ax2.plot(lons[i], lats[i], marker='o', markersize=5, color=cmap_2(norm_2(sp_hum[i])), transform=ccrs.PlateCarree())

            # Color code based on starting height h(t = 0)
            #start_height = trajectory.loc[(traj_num, 0), 'height (m)']
            """ start_height = trajectory.loc[trajectory.index.get_level_values('traj age') == 0, 'height (m)'].values[0]
            if start_height == 1000:
                color = 'green'
            elif start_height == 2000:
                color = 'blue' """

            # Add custom legend
            """legend_elements = [
                Line2D([0], [0], color='green', lw=2, label='1000 m'),
                Line2D([0], [0], color='blue', lw=2, label='2000 m')
            ]"""

            #ax.legend(handles=legend_elements, title='final m ASL', title_fontsize='large', loc='lower left')
            #ax.plot(trajectory['lon'], trajectory['lat'], color=cmap(norm(heights[i])), linewidth=1) # color=color

            # Adding markers at 24-h intervals
            """for i, traj_age in enumerate(trajectory.index.get_level_values('traj age')):
                ax.plot(trajectory['lon'], trajectory['lat'], color=cmap(norm(heights[i])), linewidth=1, transform=ccrs.PlateCarree())
                if abs(traj_age) % 24 == 0 and traj_age != 0:
                    ax.plot(lons[i], lats[i], marker='o', markersize=5, color=cmap(norm(heights[i])), transform=ccrs.PlateCarree())
                if traj_age == 0:
                    ax.plot(lons[i], lats[i], marker='*', markersize=7, color=cmap(norm(heights[i])), transform=ccrs.PlateCarree())"""
        
        #ax.set_title('Back trajectories ending 1-10-24 12 UTC') # automate the date string and also back traj time (__-day back traj)
        #plt.show()

        # Legends and plotting
        sm1 = plt.cm.ScalarMappable(cmap=cmap_1, norm=norm_1)
        cbar1 = plt.colorbar(sm1, ax=ax1, orientation='vertical', extend='max', fraction=0.035, pad=0.03)
        cbar1.set_ticks([0, 1000, 2000, 3000, 4000, 5000, 6000])

        sm2 = plt.cm.ScalarMappable(cmap=cmap_2, norm=norm_2)
        cbar2 = plt.colorbar(sm2, ax=ax2, orientation='vertical', extend='max', fraction=0.035, pad=0.03)
        cbar2.set_ticks([0, 2, 4, 6, 8, 10, 12, 14])

        ax1.set_title('Height (m)')
        ax2.set_title('Specific Humidity (g/kg)')
        
        fig.subplots_adjust(top=0.85, hspace=0.01, left=0.03, right=0.97)
        fig.savefig("./figures/" + out_file)
        #plt.show()
    
    # plots omega using two methods
    # Method 1: estimate w via reanalysis-derived values at nearest p surface and gridpoint 
    # Method 2: coarse estimation of omega using traj pressure output 
    # (self, out_file, case_number, title, lat, lon)
    def plot_omega_gridplots(self, ds_era5, title, out_file): 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

        traj_ages = np.arange(-18,1)
        traj_nums = np.arange(1,31) ### bookmark - figure out how nums are treated
        pressure_levels = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        omega_grid1 = np.full((len(traj_nums), len(traj_ages)), np.nan)
        height_zero_grid = np.zeros((len(traj_nums), len(traj_ages)), dtype=bool) # is height zero?
        #omega_grid2 = np.zeros(len(traj_nums), len(traj_ages)) # can ignore for now

        for traj_num in range(1, self.ntraj + 1):
            try:
                trajectory = self.data_1h.loc[traj_num]
            except KeyError:
                continue  # Skip missing trajectory numbers
             
            #numbers = trajectory['traj #'].values
            lats = trajectory['lat'].values
            lons = trajectory['lon'].values
            heights = trajectory['height (m)'].values
            pressures = trajectory['pressure (hPa)'].values # ignore this for now
            ages = trajectory['traj age'].values
            dates = trajectory['datetime'].values

            #for i in range(len(lats)):
                #ages = trajectory.index.get_level_values('traj age')[i]
            
            ### Method 1
            for lat, lon, height, age, date in zip(lats, lons, heights, ages, dates):
                date2 = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")  
                date3 = date2.replace(year = date2.year + 2000)
                date_str = date3.strftime("%H:%M:%S %d %b %Y")
                
                # Find two adjacent pressure levels for interpolation
                upper_level = None
                lower_level = None

                for i in range(len(pressure_levels) - 1): # also, what if sfc p < 1000 hPa?
                    upper_level_height = ds_era5.z(latitude=lat, longitude=lon, time=date_str, level=pressure_levels[i]) 
                    lower_level_height = ds_era5.z(latitude=lat, longitude=lon, time=date_str, level=pressure_levels[i + 1]) 

                    if lower_level_height <= height <= upper_level_height:
                        upper_level = pressure_levels[i]
                        lower_level = pressure_levels[i + 1]
                        break

                if upper_level and lower_level:
                    upper_omega = ds_era5.w(latitude=lat, longitude=lon, time=date_str, level=upper_level)
                    lower_omega = ds_era5.w(latitude=lat, longitude=lon, time=date_str, level=lower_level)

                    upper_level_height = ds_era5.z(latitude=lat, longitude=lon, time=date_str, level=upper_level)
                    lower_level_height = ds_era5.z(latitude=lat, longitude=lon, time=date_str, level=lower_level)
                    
                    upper_omega = upper_omega[0,0,0,0]
                    lower_omega = lower_omega[0,0,0,0]
                    upper_level_height = upper_level_height[0,0,0,0]
                    lower_level_height = lower_level_height[0,0,0,0]

                    # Linear interpolation
                    omega_interp = np.interp(height, [lower_level_height, upper_level_height], [lower_omega, upper_omega])
                    omega_grid1[traj_num - 1, age + 18] = omega_interp
                else:
                    omega_grid1[traj_num - 1, age + 18] = np.nan

                if height == 0: # keep track whether height is zero
                    height_zero_grid[traj_num - 1, age + 18] = True
                    continue

        X, Y = np.meshgrid(traj_ages, traj_nums) 
        cmap = cm.get_cmap('bwr')
        norm1 = plt.Normalize(np.nanmin(omega_grid1), -np.nanmin(omega_grid1)) # -6, 6?

        # check the following code
        c = ax1.pcolormesh(X, Y, omega_grid1, cmap=cmap, norm=norm1, shading='auto')
        fig.colorbar(c, ax=ax1, label='Omega (Pa/s)')
        ax1.set_xlabel('Trajectory age (hours)', fontsize=12)
        ax1.set_ylabel('Trajectory number', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold') # include ending location and end time
        ax1.set_yticks(traj_nums)
        ax1.set_xticks(traj_ages)
        
        # Annotate asterisks for zero height values
        for i in range(len(traj_nums)):
            for j in range(len(traj_ages)):
                if height_zero_grid[i, j]:
                    ax1.text(traj_ages[j], traj_nums[i], '*', ha='center', va='center', color='black')

        plt.savefig("./figures/" + out_file)

        ### double check how traj # is handled above

    # plots a gridplot "sounding" time series q(z,t) at a fixed point 
    def plot_sphum_gridplot(self, out_file, case_number, title, lat, lon):
        times = ["{:02d}".format(i) for i in range(19)]
        heights = np.arange(200, 6001, 200)
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        sphum_grid = np.zeros((len(heights), len(times)))
        path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"

        # Loop over each hour and populate the grid
        for i, time in enumerate(times):
            case_num = case_number + i 
            fn = path + f"traj_{case_num}.traj" 
            traj_file = TrajectoryFile(fn)

            # Extract relevant data for the rows after the header at time 0 in the .traj file
            for j, height in enumerate(heights):
                specific_humidity = traj_file.data_1h[(traj_file.data_1h['lat'] == lat) 
                    & (traj_file.data_1h['lon'] == lon)
                    & (traj_file.data_1h['height (m)'] == height)]['specific humidity (g/kg)'].values
                sphum_grid[j, i] = specific_humidity[0] # traj_file.data_1h['traj age'] == 0

        X, Y = np.meshgrid(range(len(times)), heights)
    
        cmap = cm.get_cmap('viridis')
        norm = plt.Normalize(0, 14)
        
        c = ax.pcolormesh(X, Y, sphum_grid, cmap=cmap, norm=norm, shading='auto')
        cbar = fig.colorbar(c, ax=ax, extend='max')
        cbar.set_label('Specific humidity (g/kg)', fontsize=12)

        ax.set_xticks(np.arange(len(times)))
        ax.set_xticklabels(times)
        #ax.set_yticks(np.arange(6))
        ax.set_yticklabels(np.arange(0, 7000, 1000))
        ax.set_xlabel('Time (UTC)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold') # 'Specific Humidity Grid Plot for Baltimore (39.2 N, 76.5 W)'

        plt.savefig("./figures/" + out_file)

    def plot_time_series(self, out_file, case_number, title, lat, lon):  
        times = ["{:02d}".format(i) for i in range(19)]
        fig, ax = plt.subplots()
        precip_ts = np.zeros(len(times))

        path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"
         
        for i, time in enumerate(times):
            case_num = case_number + i 
            fn = path + f"traj_{case_num}.traj" 
            traj_file = TrajectoryFile(fn)

            precip = traj_file.data_1h[(traj_file.data_1h['lat'] == lat) 
                    & (traj_file.data_1h['lon'] == lon)
                    & (traj_file.data_1h['height (m)'] == 200)]['precip (mm)'].values
            precip_ts[i] = precip

        ax.bar(times, precip_ts)
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel('Precipitation (mm)')
        ax.set_title(title)

        plt.savefig("./figures/" + out_file)

# might be useful to plot time series (ht vs time) of trajectories?

# Example with test case + animate files
"""path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"
case_nums = range(1021, 1034) ### change as needed
hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", 
    "13", "14", "15", "16", "17", "18"]"""

""" for case_nums, hours in zip(case_nums, hours): # loop across multiple case_nums and hours jointly
    out_file = f"{case_nums}_121823_{hours}_trajmaps.png"
    fn = path + f"traj_{case_nums}.traj"
    traj_file = TrajectoryFile(fn) 
    traj_file.plot_trajectories("Lewiston, ME (KLEW)", out_file) ### 
    print("Completed " + out_file) """

""" fn = []
for case_nums, hours in zip(case_nums, hours):
    filepath = f"./figures/{case_nums}_121823_{hours}_trajmaps.png"
    fn.append(filepath)
images = []

for filename in fn:
    images.append(iio.imread(filename))
    print("Completed " + filename)
iio.imwrite('./figures/121823_lewiston.gif', images, duration = 3000, loop = 0) ### """

path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"
case_num = 1021
hour = "00"
out_file = f"{case_num}_011024_{hour}_omega_gridplot.png"
fn = path + f"traj_{case_num}.traj"
traj_file = TrajectoryFile(fn) 
traj_file.plot_omega_gridplots(pyg.open("./era5/plevels_jan2024_test_2.nc"), 
    "Omega gridplot following trajectories \nending at Baltimore 00 UTC", out_file) ### 
print("Completed " + out_file)

# plot gridplot
"""path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"
traj_file = TrajectoryFile(path + "traj_1021.traj") 
traj_file.plot_sphum_gridplot("011024_baltimore_gridplot.pdf")"""

""" path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"
traj_file = TrajectoryFile(path + "traj_1121.traj") 
traj_file.plot_sphum_gridplot("121823_lewiston_gridplot.pdf", 1121, 
    'Specific Humidity Grid Plot for Lewiston (44.1 N, 70.3 W)', 44.1, -70.3) """

# plot precip time series
""" path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"
traj_file = TrajectoryFile(path + "traj_1021.traj") 
traj_file.plot_time_series("011024_baltimore_precip_ts.pdf") """

""" path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"
traj_file = TrajectoryFile(path + "traj_1041.traj") 
traj_file.plot_time_series("011024_providence_precip_ts.pdf", 1041, 
    "Precipitation Time Series for Providence (41.8 N, 71.5 W)", 41.8, -71.5) """
















# precip time series seems suspect - unrealistically high every hour? 
# also some missing hrly data possible - please recheck log files
# for precip, best to use observed station data, instead of HYSPLIT estimates
# but do send the plots to Peter and Dr. D, but explain what is going on 

# Baltimore time series based on 
# https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=DMH&data=p01m&year1=2024&month1=1&day1=9&year2=2024&month2=1&day2=11&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3&report_type=4
# Baltimore (DMH from MD ASOS)
# 8.4, 15, 8.1, 3.8, 0 rest of day (from 00 to 12 UTC)
# Providence (PVD from RI ASOS)
# 5.1, 1.5, 3.6, 0, 7.4, 8.6, 9.7, 11.9, 10.7, 22.6, 7.1, 1.3, 1.0, 1.0
