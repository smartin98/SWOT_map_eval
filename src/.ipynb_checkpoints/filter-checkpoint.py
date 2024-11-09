import numpy as np
import xarray as xr
from astropy.convolution import convolve_fft, Gaussian2DKernel



def haversine(lon0, lat0, lon1, lat1):
    r=6371e3
    dphi = np.deg2rad(lat1) - np.deg2rad(lat0)
    dlambda = np.deg2rad(lon1) - np.deg2rad(lon0)
    phi0 = np.deg2rad(lat0)
    phi1 = np.deg2rad(lat1)
    return 2 * r * np.arcsin(np.sqrt((1 - np.cos(dphi) + np.cos(phi0) * np.cos(phi1) * (1 - np.cos(dlambda))) / 2))

def calculate_along_track_dist(ds):
    return haversine(ds['longitude'].values, ds['latitude'].values, np.roll(ds['longitude'].values, 1, axis = 0), np.roll(ds['latitude'].values, 1, axis = 0))

def along_track_filter(data, scale, filt_type = 'high_pass', filt_vars = 'ssha'):
    
    gauss = Gaussian2DKernel(x_stddev = scale/2e3)
    
    data.ds['separation_along_track'] = (['num_lines', 'num_pixels'], calculate_along_track_dist(data.ds))
    break_points = (data.ds['separation_along_track'] > 2.1e3)
    
    n_lines = data.ds['num_lines'].values.shape[0]
    
    line_breaks = np.where(break_points.isel(num_pixels = 0, drop = True))[0]
    line_starts = [i + 1 for i in line_breaks if i < n_lines - 1]
    line_ends = [i for i in line_breaks[1:] if i < n_lines]
    line_starts = line_starts[:len(line_ends)]
    
    
    for var in filt_vars:
        print('filtering '+ var)
        data.ds[var + '_filtered'] = (data.ds.dims, np.full(data.ds[var].values.shape, np.nan))
        for segment in range(len(line_starts)):
            print(f'filtering segment {segment}/{len(line_starts)}')
        
            if filt_type == 'high_pass':
                data.ds[var + '_filtered'][line_starts[segment]:line_ends[segment], :] = data.ds[var][line_starts[segment]:line_ends[segment], :].values - convolve_fft(data.ds[var][line_starts[segment]:line_ends[segment], :].values, gauss)
            elif filt_type == 'low_pass':
                data.ds[var + '_filtered'][line_starts[segment]:line_ends[segment], :] = convolve_fft(data.ds[var][line_starts[segment]:line_ends[segment], :].values, gauss)
            else:
                raise ValueError("filt_type should be high_pass or low_pass")

    return data
