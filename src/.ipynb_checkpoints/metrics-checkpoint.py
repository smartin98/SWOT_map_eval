import xarray as xr
import numpy as np
import scipy
import pyinterp

def spatial_aggregate(data, single_vars, pairwise_vars, lon_bin_size, lat_bin_size, lon_bounds = (0, 360), lat_bounds = (-90, 90)):
    
    N_lon = int((lon_bounds[1] - lon_bounds[0]) / lon_bin_size) + 1
    N_lat = int((lat_bounds[1] - lat_bounds[0]) / lat_bin_size) + 1
    
    lon_bins = np.linspace(lon_bounds[0], lon_bounds[1], N_lon)
    lat_bins = np.linspace(lat_bounds[0], lat_bounds[1], N_lat)
    
    data_binned = {}
    binning = pyinterp.Binning2D(pyinterp.Axis(lon_bins, is_circle=True), pyinterp.Axis(lat_bins))
    
    for var in single_vars:
        print('Binning: ' + var)
        binning.push(data['longitude'], data['latitude'], data[var])
        data_binned[var] = {'sum': binning.variable('sum'), 'count': binning.variable('count')}
        binning.clear()
        binning.push(data['longitude'], data['latitude'], data[var]**2)
        data_binned[var]['sum_squares'] = binning.variable('sum')
        binning.clear()
        
    for var1, var2 in pairwise_vars:
        print('Binning diff between: ' + var1 + ' and ' + var2)
        binning.push(data['longitude'], data['latitude'], data[var1] - data[var2])
        data_binned[var1 + '_diff'] = {'sum': binning.variable('sum'), 'count': binning.variable('count')}
        binning.clear()
        binning.push(data['longitude'], data['latitude'], (data[var1] - data[var2])**2)
        data_binned[var1 + '_diff']['sum_squares'] = binning.variable('sum')
        binning.clear()
        
    for v, var in enumerate(single_vars):
        
        for s in ['sum', 'sum_squares', 'count']:
            da = xr.DataArray(data_binned[var][s].T, dims=('latitude', 'longitude'), coords=[lat_bins, lon_bins])
            if (v == 0) & (s == 'sum'):
                # initialize xr dataset on first var
                ds = xr.Dataset({var + '_' + s: da})
            
            else:
                ds[var + '_' + s] = da
                
    for var1, var2 in pairwise_vars: 
        for s in ['sum', 'sum_squares', 'count']:
            da = xr.DataArray(data_binned[var1 + '_diff'][s].T, dims=('latitude', 'longitude'), coords=[lat_bins, lon_bins])
            ds[var1 + '_diff_' + s] = da
            
    return ds
        

def calculate_stats(data, single_vars, pairwise_vars):
    keep_vars = []
    
    for var in single_vars:
        try:
            data[var + '_variance'] = data[var + '_sum_squares'] / data[var + '_count']
            keep_vars.append(var + '_variance')
        except:
            print("Variable not in data") 
        
    for var, _ in pairwise_vars:
        try:
            data[var + '_mse'] = data[var + '_diff_sum_squares'] / data[var + '_diff_count']
            keep_vars.append(var + '_mse')
        except:
            print("Variable not in data") 
            
        data[var + '_R2'] = 1 - data[var + '_diff_sum_squares']/data[var + '_sum_squares']
        keep_vars.append(var + '_R2')
        
    return data[keep_vars]

def calc_aggregate_stats(data, single_vars, pairwise_vars, lon_bin_size, lat_bin_size, lon_bounds = (0, 360), lat_bounds = (-90, 90)):
    agg = spatial_aggregate(data, single_vars, pairwise_vars, lon_bin_size, lat_bin_size, lon_bounds, lat_bounds)
    stats = calculate_stats(agg, single_vars, pairwise_vars)
    
    return agg, stats
        
    
    