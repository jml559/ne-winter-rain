import time as timer
import numpy as np
import cftime
from datetime import timedelta
import os
import pickle

### Makes a control file from ARL file. 

# Directories + such
#case_dir = '/glade/u/home/kara/cao_project/ssp585_2300/sample_coldtail_v1' ### Change - just the era5 directory?
arl_data_dir = '/local1/storage1/jml559/ne-winter-rain/arl'

control_output_path = '/local1/storage1/HYSPLIT/hysplit.v5.3.0_UbuntuOS20.04.6LTS_public/working' # will contain CONTROL files
#all_events_file = os.path.join(case_dir, 'events_500.pkl')
hysplit_traj_dir = os.path.join(control_output_path, "trajectories")
traj_heights = np.arange(200, 6001, 200)  # in meters
backtrack_time = -3*24  # in hours

"""if not os.path.exists(control_output_path):
    os.makedirs(control_output_path)"""

# remove event and/or unique_ID
def make_CONTROL_local(unique_ID, t, traj_heights, track_time, control_dir, traj_dir, arl_path): # getting rid of event for now
    '''
    MODIFIED:
        event_ID (int) -> unique_ID (string, appended to .traj and CONTROL file names)
        data_dir -> arl_path (dir and filename, split with os.path.split())
        output_dir -> control_dir 
        backtrack_time (pos int, appended with '-') -> track_time (signed int, can be forward or back trajectory)
        remove case_name; no longer used
        
    '''
    # Set up file paths
    arl_dir, arl_filename = os.path.split(arl_path)
    arl_dir = os.path.join(arl_dir, '') # add trailing slash if not already there
    traj_dir = os.path.join(traj_dir, '') # add trailing slash if not already there

    control_path = os.path.join(control_dir, 'CONTROL.' + unique_ID)
    if not os.path.exists(control_dir):
        os.makedirs(control_dir)
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    # Write CONTROL file ### bookmark
    with open(control_path, 'w') as f:
        """t = cftime.num2date(event['time'], 'days since 0001-01-01 00:00:00', calendar='gregorian') ### may need leap years eventually
        lat = event['lat']
        if event['lon'] > 180:
            # HYSPLIT requires longitude on a -180 to 180 scale
            lon = event['lon'] - 360
        else:
            lon = event['lon']"""
        
        # test case time, lat, lon:
        #t = cftime.DatetimeGregorian(2024, 1, 10, 12, 0, 0) 
        lat = [39.2] # [39.2, 40.9, 41.7, 44.1]
        lon = [-76.5] #[-76.7, -74.3, -71.4, -70.3]
        
        # Start time:
        f.write('{:02d} {:02d} {:02d} {:02d}\n'.format(t.year, t.month, t.day, t.hour))
        # Number of starting positions:
        n_start = len(traj_heights) * len(lat) * len(lon)
        f.write('{:d}\n'.format(n_start)) 
        # Starting positions:
        for ht in traj_heights:
            for i in range(len(lat)):
                f.write('{:.1f} {:.1f} {:.1f}\n'.format(lat[i], lon[i], ht))
        # Duration of trajectory in hours:
        f.write('{:d}\n'.format(track_time))
        # Vertical motion option:
        f.write('0\n') # 0 to use data's vertical velocity fields
        # Top of model:
        f.write('15000.0\n')  # in meters above ground level; trajectories terminate when they reach this level
        # Number of input files:
        f.write('1\n')
        # Input file path:
        f.write(arl_dir + '\n') ### arl_dir should just be arl_data_dir?
        # Input file name:
        f.write(arl_filename + '\n')
        # Output trajectory file path:
        f.write(traj_dir + '\n')
        # Output trajectory file name:
        f.write('traj_{}.traj\n'.format(unique_ID))

# Load events Dataframe
#with open(all_events_file, "rb") as input_file:
    #all_events = pickle.load(input_file)

# Make CONTROL files for HYSPLIT
tic = timer.perf_counter()
print('\nGenerating CONTROL files for HYSPLIT:')
print('  ', control_output_path)

""" for idx in all_events.index:
    event = all_events.loc[idx]
    t = cftime.num2date(event['time'], 'days since 0001-01-01 00:00:00', calendar='noleap')
    start_year = t.year
    if t.month < 4:
        start_year = t.year - 1
    arl_path = os.path.join(arl_data_dir, '_{:02d}{:02d}.arl'.format(start_year, start_year + 1))
    unique_ID = 'event{:03d}'.format(idx)
    make_CONTROL_local(event, unique_ID, traj_heights, backtrack_time, control_output_path, hysplit_traj_dir, arl_path) """

#t = cftime.num2date(event['time'], 'days since 0001-01-01 00:00:00', calendar='gregorian')

"""start_year = t.year
if t.month < 4:
    start_year = t.year - 1"""

#arl_path = os.path.join(arl_data_dir, '_{:02d}{:02d}.arl'.format(start_year, start_year + 1))
arl_path = os.path.join(arl_data_dir, 'jan2024_test_new.ARL') # 'dec2023_test.ARL'

# looping through IDs and times  
unique_IDs = range(1221, 1234)
start_time = cftime.DatetimeGregorian(2024, 1, 10, 0, 0, 0)  

for i, unique_ID in enumerate(unique_IDs):
    t = start_time + timedelta(hours=i)
    make_CONTROL_local(str(unique_ID), t, traj_heights, backtrack_time, control_output_path, hysplit_traj_dir, arl_path)

toc = timer.perf_counter()
print('Time elapsed: {:.2f} sec'.format(toc - tic))

print('Finished!')