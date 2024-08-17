# Convert ERA5 GRIB data to ARL data 
# code from user yaqiang, https://hysplitbbs.arl.noaa.gov/viewtopic.php?t=1778

import os
import mipylib.dataset.midata

print("test")

#---- Set data folder
datadir = r''
#---- Set output data file
outfn = os.path.join(datadir, 'test_era5_grib_diff.arl')
#---- Read a GRIB data file
infn3d = addfile('{}/ERA5_2017.Aug22.3dpl.grib'.format(datadir))
infn2d = addfile('{}/ERA5_2017.Aug22.2dpl.all.grib  '.format(datadir))

print('GRIB data file has been opened...')
#---- Set output ARL data file
arlf = addfile(outfn, 'c', dtype='arl')

#---- Set variable and level list
#---- Variable names in ERA5 data file
gvar2d = ['Surface_pressure_surface','2_metre_temperature_surface','10_metre_U_wind_component_surface',\
    '10_metre_V_wind_component_surface','Boundary_layer_height_surface','Convective_available_potential_energy_surface',\
    'Instantaneous_eastward_turbulent_surface_stress_surface','Instantaneous_northward_turbulent_surface_stress_surface']

gvar3d = ['Geopotential_isobaric','Temperature_isobaric','Vertical_velocity_isobaric',\
    'U_component_of_wind_isobaric','V_component_of_wind_isobaric','Relative_humidity_isobaric']

#---- Corresponding variable names in ARL data file
avar2d = ['PRSS','T02M','U10M','V10M','PBLH','CAPE','UMOF','VMOF']
avar3d = ['HGTS','TEMP','WWND','UWND','VWND','RELH']
#--- Add DIFF fields - difference between the original data and the packed data
avar3d_diff = list(avar3d)
avar3d_diff.append('DIFW')

#---- Set parameters of ARL data file
gv = infn3d['Geopotential_isobaric']
nx = gv.dimlen(gv.ndim - 1)
ny = gv.dimlen(gv.ndim - 2)
levels = gv.dimvalue(gv.ndim - 3)[::-1]
nz = len(levels)
arlf.setlevels(levels)
arlf.set2dvar(avar2d)
for l in levels:
    arlf.set3dvar(avar3d_diff)
arlf.setx(gv.dimvalue(gv.ndim - 1))
arlf.sety(gv.dimvalue(gv.ndim - 2))

#---- Write ARL data file
tNum = infn3d.timenum()
fhour = 0
for t in range(0, tNum):
    print('Time index: ' + str(t))
    atime = infn3d.gettime(t)
    print(atime.strftime('%Y-%m-%d %H:00'))
    dhead = arlf.getdatahead(infn3d.proj, 'RSMC', 2, fhour)
    #Pre-write index record without checksum - will be over-write latter
    arlf.writeindexrec(atime, dhead)
    #Checksum list
    ksumlist = []
    # Write 2d variables
    ksums = []
    for avname,gvname in zip(avar2d, gvar2d):
        gdata = infn2d[gvname][t,:,:]
        if avname == 'PRSS':
            gdata = gdata * 0.01
        ksum = arlf.writedatarec(atime, 0, avname, fhour, 99, gdata)
        ksums.append(ksum)
    ksumlist.append(ksums)
    # Write 3d variables
    for lidx in range(0, nz):
        ksums = []
        llidx = nz - lidx - 1
        print(lidx,llidx)        
        for avname,gvname in zip(avar3d, gvar3d):
            gdata = infn3d[gvname][t,llidx,:,:]            
            if avname == 'WWND':
                gdata = gdata * 0.01
                difw = arlf.diff_origin_pack(gdata)
            elif avname == 'SPHU':
                gdata = gdata * 1000.
            elif avname == 'HGTS':
                gdata = gdata / 9.80665
            ksum = arlf.writedatarec(atime, lidx + 1, avname, fhour, 99, gdata)
            ksums.append(ksum)
        ksum = arlf.writedatarec(atime, lidx + 1, 'DIFW', fhour, 99, difw)
        ksums.append(ksum)
        ksumlist.append(ksums)
    #Re-write index record with checksum
    arlf.writeindexrec(atime, dhead, ksumlist)
    fhour += 1
arlf.close()
print('Finished!')