import xarray as xr

fn1 = "/local1/storage1/jml559/ne-winter-rain/era5/plevels_1h_194011_T_w.grib"
fn2 = "/local1/storage1/jml559/ne-winter-rain/era5/plevels_1h_194011_z_RH.grib"

ds1 = xr.open_dataset(fn1,engine='cfgrib',backend_kwargs={'indexpath': ''})
ds2 = xr.open_dataset(fn2,engine='cfgrib',backend_kwargs={'indexpath': ''})

print(ds1)
print(ds2)