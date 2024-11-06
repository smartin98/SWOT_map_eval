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

start_date = '2023-07-26'
end_date = '2023-11-15'

# Convert to datetime64 objects
start_dt = np.datetime64(start_date)
end_dt = np.datetime64(end_date)

# Calculate the number of days between the two dates
num_days = (end_dt - start_dt).astype('timedelta64[D]') + 1

# Create an array of dates with the desired spacing
date_array = np.arange(start_dt, end_dt + np.timedelta64(1, 'D'))

print(date_array.astype('datetime64[s]'))