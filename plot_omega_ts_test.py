import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
#from matplotlib.colors import DivergingNorm
import pygeode as pyg

era5_file = "./era5/plevels_jan2024_test_2.nc" # convert to netCDF first
# "./era5/plevels_dec2023_test.nc"
ds = pyg.open(era5_file)

# make gridplot of omega(p,t) at a point (units: Pa s^-1) from reanalysis data 
def plot_omega_ts(lat, lon, sdate, edate, title, out_file):
    omega = ds.w(lat = lat, lon = lon, l_level = np.arange(400,1001,50), time=(sdate,edate)) 
    omega_grid = omega[:,::-1,0,0].transpose() # time, level at fixed lat, lon
    #print(omega_grid.shape)
    print(omega_grid[:,0]) # omega at 0Z

    omega_paxis = omega.level[:]
    omega_taxis = omega.time[:]

    """omega_paxis = ds.w(lat = lat, lon = lon).level[:]
    remove_plevels = [775, 825, 875, 925, 975]
    omega_paxis_new = [value for value in omega_paxis if value not in remove_plevels]

    omega_taxis = ds.w(lat = lat, lon = lon, time = (sdate, edate)).time # sdate, edate are strings
    omega_taxis_new = omega_taxis[0:19] # adjust as needed

    fig, ax = plt.subplots(figsize=(10, 8))
    omega_grid = np.zeros((len(omega_paxis_new), len(omega_taxis_new)))

    for t in omega_taxis_new:
        for p in omega_paxis_new:
            p_index = int((-1/50)*(p-1000))
            t_index = int(t - omega_taxis_new[0])
            omega_grid[p_index, t_index] = omega(level = p, time = t)[0,0,0,0]"""
    #print(omega_grid)
    #print(np.min(omega_grid[:,:]))

    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(range(len(omega_taxis)), omega_paxis)

    cmap = cm.get_cmap('bwr')
    #norm = plt.Normalize(-2,2)
    norm = plt.Normalize(np.min(omega_grid[:,:]), -np.min(omega_grid[:,:])) # adjust later 
    #norm = DivergingNorm(vmin=-np.max(omega_grid), vmax=np.max(omega_grid), vcenter=0)

    c = ax.pcolormesh(X, Y, omega_grid, cmap=cmap, norm=norm, shading='auto')
    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label("Omega (Pa/s)", fontsize=12)

    utc_times = ["{:02d}".format(i) for i in range(24)] # adjust as needed
    ax.set_xticks(np.arange(len(utc_times)))
    ax.set_xticklabels(utc_times)
    #print(omega_paxis_new)
    ax.set_yticklabels(np.linspace(1100,300,9).astype(int))
    ax.set_xlabel('Time (UTC)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pressure (hPa)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.savefig("./figures/" + out_file)
    
sdate = "10 Jan 2024"
edate = "11 Jan 2024"
"""plot_omega_ts(39.2, -76.7, sdate, edate, 
    "Omega Grid Plot for Baltimore/KBWI (39.2 N, 76.7 W)", "121823_baltimore_omega.pdf")
plot_omega_ts(40.9, -74.3, sdate, edate,
    "Omega Grid Plot for Caldwell/KCDW (40.9 N, 74.3 W)", "121823_caldwell_omega.pdf")
plot_omega_ts(41.7, -71.4, sdate, edate,
    "Omega Grid Plot for Providence/KPVD (41.7 N, 71.4 W)", "121823_providence_omega.pdf")
plot_omega_ts(44.1, -70.3, sdate, edate,
    "Omega Grid Plot for Lewiston/KLEW (40.9 N, 70.3 W)", "121823_lewiston_omega.pdf")"""
plot_omega_ts(39.2, -76.5, sdate, edate, "Omega Grid Plot for Baltimore (39.2 N, 76.5 W)", "011024_baltimore_omega.pdf")
#plot_omega_ts(41.8, -71.5, sdate, edate, "Omega Grid Plot for Providence (41.8 N, 71.5 W)", "011024_providence_omega.pdf")





# 39.2 N, 76.5 W
# time series?
# gridplot?