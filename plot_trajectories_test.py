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
                           skiprows=traj_skiprow)
        #print(list(trajectories.keys()))
        trajectories = trajectories.set_index(['traj #', 'traj age'])
        trajectories.sort_index(inplace=True)
        #print(trajectories.index)
        
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
            + location + ', 3 days', fontweight="bold") # automate also back traj time (__-day back traj)

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
    
    # plots Lagrangian omega(x,y,p,t) using two methods
    # Method 1: estimate w via reanalysis-derived values at nearest p surface and gridpoint 
    # Method 2: coarse estimation of omega using traj pressure output 
    # (self, out_file, case_number, title, lat, lon)
    def plot_omega_lagr_gridplots(self, ds_era5, title, out_file):  
        #print(ds_era5.z)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)) # 10, 3.5

        traj_ages = np.arange(-18,1)
        traj_nums = np.arange(1,31) 
        pressure_levels = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        omega_grid1 = np.full((len(traj_nums), len(traj_ages)), np.nan)
        height_zero_grid = np.full((len(traj_nums), len(traj_ages)), False, dtype=bool) # is height zero?
        omega_grid2 = np.full((len(traj_nums), len(traj_ages)), np.nan) 

        for traj_num in range(1, self.ntraj + 1): # loop through each trajectory 
            try:
                trajectory = self.data_1h.loc[traj_num]
            except KeyError:
                continue  # Skip missing trajectory numbers
            
            i = -19 # start at age -18 h ... to 0 h 
            lats = trajectory['lat'].values[i:]
            lons = trajectory['lon'].values[i:]
            heights = trajectory['height (m)'].values[i:]
            #print(min(heights))
            #print(heights)
            pressures = trajectory['pressure (hPa)'].values[i:] 
            #print(pressures)
            pres_init = trajectory['pressure (hPa)'].values[i-1]
            ages = trajectory.index.get_level_values('traj age')
            ages = ages[i:] 
            dates = trajectory['datetime'].values[i:]
             
            for lat, lon, height, pres, age, date in zip(lats, lons, heights, pressures, ages, dates): 
                # loop through each pt in a trajectory 
                date2 = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")  
                date3 = date2.replace(year = date2.year + 2000)
                date_str = date3.strftime("%H:%M %d %b %Y") # :%S
                
                ### Method 1: estimate w via reanalysis-derived values at nearest p surface and gridpoint
                # Find two adjacent pressure levels for interpolation
                e5_heights = ds_era5.z(latitude=lat, longitude=lon, time=date_str).squeeze()[:]
                e5_heights = e5_heights/9.8 # get height (in m) instead of geopotential
                e5_omega = ds_era5.w(latitude=lat, longitude=lon, time=date_str).squeeze()[:]

                """if age == 0: # test/debug 
                    print(e5_heights)
                    print(e5_omega)""" 

                for i in range(1, len(pressure_levels)): 
                    #upper_level_height = ds_era5.z(latitude=lat, longitude=lon, time=date_str, level=pressure_levels[i])[:]
                    #lower_level_height = ds_era5.z(latitude=lat, longitude=lon, time=date_str, level=pressure_levels[i + 1])[:] 
                    if height > e5_heights[i]:
                        upper_level_height = e5_heights[i-1]
                        lower_level_height = e5_heights[i] 
                        upper_omega = e5_omega[i-1]
                        lower_omega = e5_omega[i]
                        break

                if height < e5_heights[-1]: # interpolate omega between 1000 hPa height and zero, assuming w(z = 0) = 0
                    upper_level_height = e5_heights[-1]
                    lower_level_height = 0
                    upper_omega = e5_omega[-1]
                    lower_omega = 0

                # linear interpolation
                omega_interp = np.interp(height, [lower_level_height, upper_level_height], [lower_omega, upper_omega])
                omega_grid1[traj_num - 1, age + 18] = omega_interp

                #if age == 0: # test/debug 
                    #print("Traj num " + str(traj_num) + " has height " + str(height))
                    #print("Traj num " + str(traj_num) + " has LLH " + str(lower_level_height) + " has height " + 
                        #str(height) + " has ULH " + str(upper_level_height))
                    #print("Traj num " + str(traj_num) + " has upper level " + str(upper_level_height) +
                        #" m and lower level " + str(lower_level_height) + " m")
            
                ### Method 2: coarse estimation of omega using traj pressure output only 
                # for time = i, take the difference between p(t = i) and p(t = (i-1)), convert to Pa, divide by 3600 s
                if age == -18: 
                    omega_grid2[traj_num - 1, age + 18] = (100*(pres - pres_init))/3600
                else:
                    omega_grid2[traj_num - 1, age + 18] = (100*(pres - pressures[age + 17]))/3600

                if height == 0: # keep track whether height is zero
                    height_zero_grid[traj_num - 1, age + 18] = True
                    omega_grid1[traj_num - 1, age + 18] = np.nan
                    omega_grid2[traj_num - 1, age + 18] = np.nan
                    continue

            #print(np.min(height)) ### double check code/unreliable heights
            #print(height_zero_grid)

        X, Y = np.meshgrid(traj_ages, traj_nums) 
        cmap = cm.get_cmap('bwr')
        #norm1 = plt.Normalize(np.nanmin(omega_grid1), -np.nanmin(omega_grid1)) # -6, 6?
        norm = plt.Normalize(-6, 6) # -3, 3 or -6, 6

        # plotting code
        c1 = ax1.pcolormesh(X, Y, omega_grid1, cmap=cmap, norm=norm)
        fig.colorbar(c1, ax=ax1, extend="both")
        ax1.set_xlabel('Trajectory age (hours)', fontsize=12)
        ax1.set_ylabel('Trajectory number', fontsize=12)
        #ax1.set_title(title, fontsize=14, fontweight='bold') # include ending location and end time
        ax1.set_title("(1) Reanalysis-interpolated omega")
        ax1.set_yticks(traj_nums[::2])
        ax1.set_xticks(traj_ages[::2])

        c2 = ax2.pcolormesh(X, Y, omega_grid2, cmap=cmap, norm=norm)
        fig.colorbar(c2, ax=ax2, extend="both")
        ax2.set_xlabel('Trajectory age (hours)', fontsize=12)
        ax2.set_ylabel('Trajectory number', fontsize=12)
        #ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.set_title("(2) Trajectory file coarse estimate")
        ax2.set_yticks(traj_nums[::2])
        ax2.set_xticks(traj_ages[::2])

        fig.suptitle(title, fontsize=14, fontweight='bold')

        #print(omega_grid1[:,18])
        #print(np.sum(heights == 0.0))
        #print(np.size(heights))
        #print(np.sum(height_zero_grid == True))

        # Annotate zeros for zero height values  
        for i in range(len(traj_nums)):
            for j in range(len(traj_ages)):
                if height_zero_grid[i, j] == True:
                    ax1.text(traj_ages[j], traj_nums[i], '0', ha='center', va='center', color='black')
                    ax2.text(traj_ages[j], traj_nums[i], '0', ha='center', va='center', color='black')

        plt.savefig("./figures/" + out_file)
        
        ### double check reasonableness of plots 
    
    # plots specific humidity using two methods in a Lagrangian sense 
    # Method 1: reanalysis-interpolated specific humidity (multiply era5 values by 1000 to get g/kg)
    # Method 2: trajectory file output 
    def plot_sphum_lagr_gridplots(self, ds_era5, title, out_file):  
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7)) 

        traj_ages = np.arange(-18,1)
        traj_nums = np.arange(1,31) 
        pressure_levels = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        sphum_grid1 = np.full((len(traj_nums), len(traj_ages)), np.nan)
        height_zero_grid = np.full((len(traj_nums), len(traj_ages)), False, dtype=bool) # is height zero?
        sphum_grid2 = np.full((len(traj_nums), len(traj_ages)), np.nan) 

        for traj_num in range(1, self.ntraj + 1): # loop through each trajectory 
            try:
                trajectory = self.data_1h.loc[traj_num]
            except KeyError:
                continue  # Skip missing trajectory numbers
            
            i = -19 # start at age -18 h ... to 0 h 
            lats = trajectory['lat'].values[i:]
            lons = trajectory['lon'].values[i:]
            heights = trajectory['height (m)'].values[i:]
            #print(heights)
            #pressures = trajectory['pressure (hPa)'].values[i:] 
            sphums = trajectory['specific humidity (g/kg)'].values[i:]
            ages = trajectory.index.get_level_values('traj age')
            ages = ages[i:] 
            dates = trajectory['datetime'].values[i:]
             
            for lat, lon, height, sphum, age, date in zip(lats, lons, heights, sphums, ages, dates): 
                # loop through each pt in a trajectory 
                date2 = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")  
                date3 = date2.replace(year = date2.year + 2000)
                date_str = date3.strftime("%H:%M %d %b %Y")

                # Find two adjacent pressure levels for interpolation
                e5_heights = ds_era5.z(latitude=lat, longitude=lon, time=date_str).squeeze()[:]
                e5_heights = e5_heights/9.8 # get height (in m) instead of geopotential
                e5_sphums = 1000*ds_era5.q(latitude=lat, longitude=lon, time=date_str).squeeze()[:] # level?
                
                """if age == -18 and traj_num == 1:
                    print(e5_heights)
                    print(e5_sphums)
                    print(lat)
                    print(lon)
                    print(date_str)
                    print(ds_era5.z(latitude=lat, longitude=lon, time=date_str))
                    print(ds_era5.q(latitude=lat, longitude=lon, time=date_str))"""

                for i in range(1, len(pressure_levels)):  
                    if height > e5_heights[i]:
                        upper_level_height = e5_heights[i-1]
                        lower_level_height = e5_heights[i] 
                        upper_q = e5_sphums[i-1]
                        lower_q = e5_sphums[i]
                        break

                if height < e5_heights[-1]: # interpolate q between 1000 hPa height and zero 
                    upper_level_height = e5_heights[-1]
                    lower_level_height = 0 
                    upper_q = e5_sphums[-1]
                    lower_q = e5_sphums[-1] + ((e5_sphums[-1] - e5_sphums[-2])/(e5_heights[-1] - e5_heights[-2]))*e5_heights[-1]

                """if age == -18 and traj_num == 1:
                    print(height)
                    print(i, pressure_levels[i])
                    print(upper_level_height)
                    print(lower_level_height)
                    print(sphum)
                    print(upper_q)
                    print(lower_q)"""

                # linear interpolation
                q_interp = np.interp(height, [lower_level_height, upper_level_height], [lower_q, upper_q])
                sphum_grid1[traj_num - 1, age + 18] = q_interp

                # Method 2
                sphum_grid2[traj_num - 1, age + 18] = sphum   

                if height == 0: # keep track whether height is zero
                    height_zero_grid[traj_num - 1, age + 18] = True
                    continue

                """for e5_sphum in e5_sphums:
                    sphum_grid1[traj_num - 1, age + 18] = e5_sphum
                sphum_grid2[traj_num - 1, age + 18] = sphum"""

        X, Y = np.meshgrid(traj_ages, traj_nums) 
        cmap = cm.get_cmap('viridis')
        norm = plt.Normalize(0, 14)

        # plotting code
        c1 = ax1.pcolormesh(X, Y, sphum_grid1, cmap=cmap, norm=norm)
        fig.colorbar(c1, ax=ax1, extend="max")
        ax1.set_xlabel('Trajectory age (hours)', fontsize=12)
        ax1.set_ylabel('Trajectory number', fontsize=12)
        ax1.set_title("(1) Reanalysis-interpolated $q$")
        ax1.set_yticks(traj_nums[::2])
        ax1.set_xticks(traj_ages[::2])

        c2 = ax2.pcolormesh(X, Y, sphum_grid2, cmap=cmap, norm=norm)
        fig.colorbar(c2, ax=ax2, extend="max")
        ax2.set_xlabel('Trajectory age (hours)', fontsize=12)
        ax2.set_ylabel('Trajectory number', fontsize=12)
        ax2.set_title("(2) Trajectory file output $q$")
        ax2.set_yticks(traj_nums[::2])
        ax2.set_xticks(traj_ages[::2])
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Annotate zeros for zero height values  
        for i in range(len(traj_nums)):
            for j in range(len(traj_ages)):
                if height_zero_grid[i, j] == True:
                    ax1.text(traj_ages[j], traj_nums[i], '0', ha='center', va='center', color='black')
                    ax2.text(traj_ages[j], traj_nums[i], '0', ha='center', va='center', color='black')

        plt.savefig("./figures/" + out_file)

    # plots a gridplot "sounding" time series q(z,t) at a fixed point (Eulerian)
    def plot_sphum_eul_gridplot(self, out_file, case_number, title, lat, lon):
        times = ["{:02d}".format(i) for i in range(19)]
        heights = np.arange(200, 6001, 200)
    
        fig, ax = plt.subplots(figsize=(10, 8)) # 10, 8
    
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

    # plots ERA-5 profile, ARL profile, and trajectory profile
    def plot_init_sphum_profiles(self, ds_era5, arl_sphums, out_file): 
        fig, ax = plt.subplots()

        era5_plevels = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        arl_plevels = era5_plevels

        lat = self.data_1h.loc[1]['lat'].values[-1]
        lon = self.data_1h.loc[1]['lon'].values[-1]
        date = self.data_1h.loc[1]['datetime'].values[-1]
        date2 = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        date3 = date2.replace(year = date2.year + 2000)
        date_str = date3.strftime("%H:%M %d %b %Y")

        traj_plevels = np.zeros(30)
        traj_sphums = np.zeros(30)
        for i in range(1, self.ntraj + 1): 
            traj_plevels[i-1] = self.data_1h.loc[i]['pressure (hPa)'].values[-1] # get final p level at each trajectory 
            traj_sphums[i-1] = self.data_1h.loc[i]['specific humidity (g/kg)'].values[-1]
        
        era5_sphums = 1000*ds_era5.q(latitude=lat, longitude=lon, time=date_str).squeeze()[:]
        # arl_sphums will be provided by the user 

        print(traj_plevels)
        print(traj_sphums)

        ax.plot(era5_sphums, era5_plevels, color="blue", marker="o", label="ERA-5")
        ax.plot(arl_sphums, arl_plevels, color="green", 
            linestyle=(0,(5,10)), marker="o", markerfacecolor="none", label="ARL profile")
        ax.plot(traj_sphums, traj_plevels, color="orange", linestyle="dashed", marker="x",
            label="HYSPLIT traj file \n" + r"$\bf{" + "omega (0)" + "}$") # omega (0) or diverge (5)
        ax.set_xlabel("Specific humidity (g/kg)", fontsize=12)
        ax.set_ylabel("Pressure (hPa)", fontsize=12)
        ax.yaxis.set_inverted(True)
        
        ax.set_title("Specific humidity profiles at lat " + str(lat) + ", lon " + str(lon) 
            + "\n" + date_str, fontsize=14, fontweight='bold') 
        ax.legend()

        plt.savefig("./figures/" + out_file)

    # plot ERA-5 profile, ARL profile, and trajectory variable/value *at a single point*
    # age is age of trajectory (hrs befor initialization, use NEGATIVE INDEXING)
    # traj_num is the number of the trajectory
    # date_str parameter must be in format "%H:%M %d %b %Y", e.g., "00:00 10 Jan 2024"
    def plot_point_sphum_profiles(self, ds_era5, age, traj_num, arl_sphums, out_file): 
        fig, ax = plt.subplots()

        era5_plevels = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        arl_plevels = era5_plevels 

        lat = self.data_1h.loc[traj_num]['lat'].values[age-1]
        lon = self.data_1h.loc[traj_num]['lon'].values[age-1]
        date = self.data_1h.loc[traj_num]['datetime'].values[age-1]
        date2 = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        date3 = date2.replace(year = date2.year + 2000)
        date_str = date3.strftime("%H:%M %d %b %Y")

        traj_plevel = self.data_1h.loc[traj_num]['pressure (hPa)'].values[age-1]
        traj_sphum = self.data_1h.loc[traj_num]['specific humidity (g/kg)'].values[age-1]

        era5_sphums = 1000*ds_era5.q(latitude=lat, longitude=lon, time=date_str).squeeze()[:]
        # arl_sphums will be provided by the user 

        ax.plot(era5_sphums, era5_plevels, color="blue", marker="o", label="ERA-5")
        ax.plot(arl_sphums, arl_plevels, color="green", 
            linestyle=(0,(5,10)), marker="o", markerfacecolor="none", label="ARL profile")
        ax.plot(traj_sphum, traj_plevel, color="orange", linestyle="dashed", marker="X",
            label="HYSPLIT traj file \n" + r"$\bf{" + "diverge (5)" + "}$") # omega (0) or diverge (5)
        ax.set_xlabel("Specific humidity (g/kg)", fontsize=12)
        ax.set_ylabel("Pressure (hPa)", fontsize=12)
        ax.yaxis.set_inverted(True)
        
        ax.set_title("Specific humidity profiles at lat " + str(lat) + ", lon " + str(lon) 
            + "\n" + date_str, fontsize=14, fontweight='bold') 
        ax.legend()

        plt.savefig("./figures/" + out_file)

    

# might be useful to plot time series (ht vs time) of trajectories?

# Example with test case + animate files
path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"
case_nums = range(1321, 1334) ### change as needed # 1021, 1034 and 1221, 1234
hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

for case_num, hour in zip(case_nums, hours): # loop across multiple case_nums and hours jointly
    out_file = f"{case_num}_011024_{hour}_trajmaps.png"
    #out_file = f"{case_num}_011024_{hour}_omega_lagr_gridplot.png"
    #out_file = f"{case_num}_011024_{hour}_sphum_lagr_gridplot.png"
    fn = path + f"traj_{case_num}.traj"
    traj_file = TrajectoryFile(fn) 
    #traj_file.plot_sphum_lagr_gridplots(pyg.open("./era5/plevels_jan2024_test_2.nc"), 
        #"Specific humidity gridplots (g/kg) following trajectories \nending at Baltimore " + f"Jan 10 2024 {hour} UTC", out_file)
    traj_file.plot_trajectories("Baltimore, MD", out_file) ### 
    #traj_file.plot_omega_lagr_gridplots(pyg.open("./era5/plevels_jan2024_test_2.nc"), 
        #"Omega gridplots (Pa/s) following trajectories ending at Baltimore " + f"Jan 10 2024 {hour} UTC", out_file) ### 
    print("Completed " + out_file)

""" fn = []
for case_num, hour in zip(case_nums, hours):
    #filepath = f"./figures/{case_num}_121823_{hour}_trajmaps.png"
    #filepath = f"./figures/{case_num}_011024_{hour}_omega_gridplot.png"
    #filepath = f"./figures/{case_num}_011024_{hour}_sphum_lagr_gridplot.png"
    filepath = f"./figures/{case_num}_011024_{hour}_trajmaps.png"
    #filepath = f"./figures/{case_num}_011024_{hour}_omega_lagr_gridplot.png"
    fn.append(filepath)
images = []

for filename in fn:
    images.append(iio.imread(filename))
    print("Completed " + filename)
iio.imwrite('./figures/011024_baltimore_trajmaps_vert5.gif', images, duration = 3000, loop = 0) ### """

"""path = "/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working/trajectories/"
case_num = 1021 ###
hour = "00" ###
out_file = f"{case_num}_011024_{hour}_sphum_lagr_gridplot.png"
fn = path + f"traj_{case_num}.traj"
traj_file = TrajectoryFile(fn) 
#traj_file.plot_trajectories("Baltimore", f"{case_num}_011024_{hour}_trajmaps.png")
#print("Completed " + f"{case_num}_011024_{hour}_trajmaps.png")
traj_file.plot_sphum_lagr_gridplots(pyg.open("./era5/plevels_jan2024_test_2.nc"), 
    "Specific humidity gridplots (g/kg) following trajectories \nending at Baltimore " + f"Jan 10 2024 {hour} UTC", out_file) ### 
print("Completed " + out_file) """

#def plot_sphum_lagr_gridplot(self, ds_era5, title, out_file)

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

# plot initial specific humidity profiles
"""traj_file = TrajectoryFile(path + "traj_1021.traj") ##
arl_sphums = [0.309, 0.699, 1.208, 1.897, 2.573, 3.412, 4.341, 5.103, 5.652, 6.440,
            6.847, 7.508, 8.122, 8.510, 8.747, 8.807, 8.843, 8.841, 8.736, 8.345] ## 1021, 1221
arl_sphums = [0.0225, 0.0523, 0.0471, 0.113, 0.190, 0.118, 0.192, 1.357, 3.528, 3.767,
            3.451, 3.482, 4.229, 5.614, 7.239, 8.599, 9.499, 9.769, 9.409, 9.111] ## 1026, 1226
# get the corresponding profile (in order of 300-1000 hPa)
outfile = "1021_011024_00_init_sphum.png" ##
traj_file.plot_init_sphum_profiles(pyg.open("./era5/plevels_jan2024_test_2.nc"), arl_sphums,
    outfile)"""

# plot point specific humifity profiles
"""traj_file = TrajectoryFile(path + "traj_1224.traj")
arl_sphums = [0.383, 0.763, 0.907, 1.939, 1.496, 1.869, 3.068, 3.898, 4.533, 5.913,
            6.843, 7.640, 8.938, 9.341, 9.631, 9.830, 9.953, 10.216, 10.323, 10.728]
outfile = "1224_011024_03_age-8_traj2_sphum.png" ##
traj_file.plot_point_sphum_profiles(pyg.open("./era5/plevels_jan2024_test_2.nc"), 
    -8, 2, arl_sphums, outfile) # age, traj_num """


# 37.997  -76.889












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
