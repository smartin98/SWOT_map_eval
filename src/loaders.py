import numpy as np
import xarray as xr
import datetime
import os
import pandas as pd

def convert_string_to_npdatetime(string):
    datetime_obj = pd.to_datetime(string, format="%Y%m%dT%H%M%S", errors='coerce', yearfirst=True, utc=True)
    return np.datetime64(datetime_obj)

def find_swot_start_end(files, file_prefix = 'SWOT_L3_LR_SSH_Basic_XXX_YYY_'):
    start_dates = [convert_string_to_npdatetime(f[len(file_prefix):len(file_prefix)+15]) for f in files]
    
    end_dates = [convert_string_to_npdatetime(f[len(file_prefix)+16:len(file_prefix)+16+15]) for f in files]
    
    return start_dates, end_dates

class SWOT_L3_Dataset:
    
    """
    A dataset for storing SWOT L3 data from all granules between two datetimes, concatenated alon the num_lines axis.

    Attributes:
        data_dir (str): The directory containing SWOT L3 files.
        window_start (np.datetime64): Start datetime of the window to load data from
        window_end (np.datetime64): End datetime of the window to load data from
        
        
    Methods:
        
    """
    
    def __init__(self, datadir, window_start, window_end, keep_vars = ['ssha', 'mdt', 'time', 'quality_flag'], file_prefix = 'SWOT_L3_LR_SSH_Expert_XXX_YYY_'):
        
        files = sorted(os.listdir(datadir))
        files = [f for f in files if '.nc' in f]
        start_dates, end_dates = find_swot_start_end(files, file_prefix)
        
        files_load = []
        
        for i, f in enumerate(files):
            if end_dates[i]>window_start and start_dates[i]<window_end:
                files_load.append(f)
            elif start_dates[i]>window_start and start_dates[i]<window_end:
                files_load.append(f)
            elif end_dates[i]>window_start and end_dates[i]<window_end:
                files_load.append(f)
                
        paths_load = [datadir +'/' + f for f in files_load]
        paths_load = sorted(paths_load)
        
        num_lines_global = 0
        datasets = []
        for i, f in enumerate(paths_load):
            ds = xr.open_dataset(f)
            # mask = ((ds.num_lines == ds['i_num_line']) & (ds.num_pixels == ds['i_num_pixel'])).sum(dim='num_nadir').astype('bool')#.plot()
            # ds = ds.where(~mask)
            ds = ds.drop(['i_num_line', 'i_num_pixel'])
            time_expanded = xr.DataArray(
                                        np.repeat(ds.time.values[:, np.newaxis], ds['num_pixels'].values.shape[0], -1),
                                        dims=('num_lines', 'num_pixels'),
                                        coords={'num_lines': ds.num_lines, 'num_pixels': ds.num_pixels}
                                    )

            # Assign the expanded time dimension to the dataset
            ds['time'] = time_expanded
            ds['num_lines'] = ds['num_lines'] + num_lines_global
            num_lines_global += ds['num_lines'].shape[0]
            datasets.append(ds[keep_vars])
            
        self.ds = xr.concat(datasets, dim = 'num_lines')
        
    def subset(self, lon_min = None, lon_max = None, lat_min = None, lat_max = None, time_min = None, time_max = None):
        # longitude masking
        if (lon_min is not None) or (lon_max is not None):
            mask_lon = ((self.ds['longitude'] > lon_min) & (self.ds['longitude'] < lon_max)).astype('bool')
        else:
            mask_lon = xr.DataArray(True, dims=self.ds.dims, coords={'longitude': self.ds['longitude'], 'latitude': self.ds['latitude']})
        
        # latitude masking
        if (lat_min is not None) or (lat_max is not None):
            mask_lat = ((self.ds['latitude'] > lat_min) & (self.ds['latitude'] < lat_max)).astype('bool')
        else:
            mask_lat = xr.DataArray(True, dims=self.ds.dims, coords={'longitude': self.ds['longitude'], 'latitude': self.ds['latitude']})
            
        # time masking
        if (time_min is not None) or (time_max is not None):
            mask_time = ((self.ds['time'] > time_min) & (self.ds['time'] < time_max)).astype('bool')
        else:
            mask_time = xr.DataArray(True, dims=self.ds.dims, coords={'longitude': self.ds['longitude'], 'latitude': self.ds['latitude']})
            
        # total mask by combining all 3:
        
        total_mask = mask_lon & mask_lat & mask_time
        
        return self.ds.where(total_mask)