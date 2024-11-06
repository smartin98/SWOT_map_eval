import numpy as np
import xarray as xr
import cmocean
from src.loaders import *
from src.colocate import *
from src.metrics import *
import os
import pandas as pd
import datetime
import pyinterp
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--start', type = str, help = 'start date for stats calculation')
parser.add_argument('--end', type = str, help = 'end date for stats calculation')
parser.add_argument('--lon_bin_size', type = float, help = 'size of longitude bins')
parser.add_argument('--lat_bin_size', type = float, help = 'size of latitude bins')
parser.add_argument('--time_bin_size', type = int, help = 'size of time bins in days')
parser.add_argument('--lon_min', type = float, help = 'minimum longitude [0-360] for calculating stats on subdomain')
parser.add_argument('--lon_max', type = float, help = 'maximum longitude [0-360] for calculating stats on subdomain')
parser.add_argument('--lat_min', type = float, help = 'minimum latitude for calculating stats on subdomain')
parser.add_argument('--lat_max', type = float, help = 'maximum latitude for calculating stats on subdomain')
parser.add_argument('--swot_dir', type = str, help = 'path to directory containing SWOT L3 data')
parser.add_argument('--map_dir', type = str, help = 'path to directory containing mapped L4 data')
parser.add_argument('--output_dir', type = str, help = 'path to directory to save results')
parser.add_argument('--output_name', type = str, help = 'filename in which to save results')

args = parser.parse_args()

if args.start is None:
    print('start not specified, defaulting to 2023-03-28...')
    start_date = '2023-03-28'
else:
    start_date = args.start
    if np.datetime64(start_date) < np.datetime64('2023-03-28'):
        raise ValueError("start must be no earlier than 2023-03-28")
    
if args.end is None:
    print('end not specified, defaulting to 2024-09-16...')
    end_date = '2024-09-16'
else:
    end_date = args.end
    if np.datetime64(end_date) > np.datetime64('2024-09-16'):
        raise ValueError("end must be no later than 2024-09-16")
    
if (args.lon_min is None) or (args.lon_max is None) or (args.lat_min is None) or (args.lat_max is None):
    print('At least one of [lon_min, lon_max, lat_min, lat_max] not specified, defaulting to global computation...')
    subsetting = False
    lon_bounds = (0, 360)
    lat_bounds = (-90, 90)
else:
    subsetting = True
    lon_bounds = (args.lon_min, args.lon_max)
    lat_bounds = (args.lat_min, args.lat_max)

if args.lon_bin_size is None:
    print('lon_bin_size not specified, defaulting to 1 degree...')
    lon_bin_size = 1
else:
    lon_bin_size = args.lon_bin_size

if args.lat_bin_size is None:
    print('lat_bin_size not specified, defaulting to 1 degree...')
    lat_bin_size = 1
else:
    lat_bin_size = args.lat_bin_size
    
if args.time_bin_size is None:
    print('time_bin_size not specified, defaulting to 1 day...')
    time_bin_size = 1
else:
    time_bin_size = args.time_bin_size
    
if args.swot_dir is None:
    print('swot_dir not specified, defaulting to /dat1/smart1n/SWOT/data/SWOT_L3_LR_SSH_EXPERT_1.0.2/')
    swot_dir = '/dat1/smart1n/SWOT/data/SWOT_L3_LR_SSH_EXPERT_1.0.2/'
else:
    swot_dir = args.swot_dir

if args.map_dir is None:
    print('map_dir not specified, defaulting to /dat1/smart1n/NeurOST_SSH-SST/')
    map_dir = '/dat1/smart1n/NeurOST_SSH-SST/'
else:
    map_dir = args.map_dir
    
if args.output_dir is None:
    print('output_dir not specified, defaulting to ./results/')
    output_dir = './results/'
else:
    output_dir = args.output_dir
    if output_dir[-1] != '/':
        output_dir = output_dir + '/'

if args.output_name is None:
    n_outputs = len(os.listdir(output_dir))
    print('output_name not specified, defaulting to ' + f'rmse_ssh{n_outputs}.nc')
    output_name = f'rmse_ssh{n_outputs}.nc'
else:
    output_name = args.output_name
    if '.nc' not in output_name:
        output_name = output_name + '.nc'


output_path = output_dir + output_name


start_dt = np.datetime64(start_date)
end_dt = np.datetime64(end_date)

num_days = (((end_dt - start_dt)//time_bin_size).astype('timedelta64[D]') + 1).astype('int')

date_array = np.array([(start_dt + np.timedelta64(n * time_bin_size, 'D')).astype('datetime64[s]') for n in range(num_days)]) #np.arange(start_dt, end_dt + np.timedelta64(1, 'D')).astype('datetime64[s]')

N_lon = int((lon_bounds[1] - lon_bounds[0]) / lon_bin_size) + 1
N_lat = int((lat_bounds[1] - lat_bounds[0]) / lat_bin_size) + 1

lon_bins = np.linspace(lon_bounds[0], lon_bounds[1], N_lon)
lat_bins = np.linspace(lat_bounds[0], lat_bounds[1], N_lat)

ds_results = xr.Dataset(
    {
        "ssha_sum": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "ssha_sum_squares": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "ssha_count": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "sla_map_sum": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "sla_map_sum_squares": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "sla_map_count": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "ssha_diff_sum": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "ssha_diff_sum_squares": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "ssha_diff_count": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "sla_map_variance": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "ssha_variance": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "ssha_mse": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
        "ssha_R2": (["time", "lat", "lon"], np.full((len(date_array), N_lat, N_lon), np.nan)),
    },
    coords={
        "time": date_array,
        "lat": lat_bins,
        "lon": lon_bins
    }
)

for i, t in enumerate(date_array):
    start = t
    end = t + np.timedelta64(time_bin_size, 'D')

    print(f'Processing: {start} to {end}')
    swot_data = SWOT_L3_Dataset(swot_dir, start, end, file_prefix = 'SWOT_L3_LR_SSH_Expert_XXX_YYY_')
    map_data = Map_L4_Dataset(map_dir, start, end)
    interp = interp_L4_to_L3(map_data, swot_data)
    
    del swot_data, map_data
    
    if subsetting:
        interp = interp.subset(lon_min = lon_min, lon_max = lon_max, lat_min = lat_min, lat_max = lat_max)
    
    
    if interp is not None:
        agg, stats = calc_aggregate_stats(data = interp.ds, 
                                          single_vars = ['sla_map', 'ssha'], 
                                          pairwise_vars = [['ssha', 'sla_map']], 
                                          lon_bin_size = 1, 
                                          lat_bin_size = 1, 
                                          lon_bounds = lon_bounds,
                                          lat_bounds = lat_bounds,
                                         )
        
        for var in agg.data_vars:
            ds_results[var][i,:,:] = agg[var].values
        
        for var in stats.data_vars:
            ds_results[var][i,:,:] = stats[var].values
        
ds_results.to_netcdf(output_path)
        