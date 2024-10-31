import numpy as np
import xarray as xr
import datetime
import os
import pandas as pd

def convert_string_to_npdatetime(string):
    datetime_obj = pd.to_datetime(string, format="%Y%m%dT%H%M%S", errors='coerce', yearfirst=True, utc=True)
    return np.datetime64(datetime_obj)

def find_swot_start_end(data_dir, file_prefix = 'SWOT_L3_LR_SSH_Basic_XXX_YYY_'):
    files = os.listdir(data_dir)
    
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
    
    def __init__(self, datadir, window_start, window_end):
        
        files = sorted(os.listdir(data_dir))
        start_dates, end_dates = find_swot_start_end(data_dir)
        
        files_load = []
        
        for i, f in enumerate(files):
            if end_dates[i]>window_start and start_dates[i]<window_end:
                files_load.append(f)
            elif start_dates[i]>window_start and start_dates[i]<window_end:
                files_load.append(f)
            elif end_dates[i]>window_start and end_dates[i]<window_end:
                files_load.append(f)
                
        paths_load = [data_dir +'/' + f for f in files_load]
        paths_load = sorted(paths_load)
        
        num_lines_global = 0
        datasets = []
        for i, f in enumerate(paths_load):
            ds = xr.open_dataset(f)
            ds['num_lines'] = ds['num_lines'] + num_lines_global
            num_lines_global += ds['num_lines'].shape[0]
            datasets.append(ds)
            
        self.dataset = xr.concat(datasets, dim = 'num_lines')
        
        