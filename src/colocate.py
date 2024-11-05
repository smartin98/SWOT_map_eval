import xarray as xr

def interp_L4_to_L3(L4_Dataset, L3_Dataset, method = 'linear'):
    """Interpolate L4 Mapped dataset onto a L3 SWOT dataset.

    L4_Dataset (src.loaders.Map_L4_Dataset): The dataset containing Mapped L4 data.
    L3_Dataset (src.loaders.SWOT_L3_Dataset): The dataset containing SWOT L3 data.
    method (str): The interpolation method from available xr.interp() options (default: linear)
    """
    
    result = L3_Dataset.clone()
    
    ds_l3 = L3_Dataset.ds
    ds_l4 = L4_Dataset.ds
    
    obs_lon, obs_lat, obs_time = ds_l3.longitude, ds_l3.latitude, ds_l3.time

    interpolated_data = {}
    for var_name in ds_l4.data_vars:
        interpolated_data[var_name] = ds_l4[var_name].interp(
            longitude = obs_lon, latitude = obs_lat, time = obs_time, method = method 
        )

    interp_result =  xr.Dataset(
      data_vars=interpolated_data,
      coords=ds_l3.coords,
      attrs=ds_l3.attrs,
    ).reset_coords()
    
    exclude = ["time", "latitude", "longitude"]
    interp_result = interp_result.drop(exclude)
    
    mask = ~ds_l3['ssha'].isnull()
    
    for var in interp_result.data_vars: 
        #if var not in exclude:
        interp_result[var] = interp_result[var].where(mask)
    
    result.add_vars(data = [interp_result[var_name] for var_name in interp_result.data_vars], var_names = [var_name + '_map' for var_name in interp_result.data_vars])
    
    mask_l4_nan = ~ds_l3[var_names[0] + '_map'].isnull()
    for var in ds_l3.data_vars:
        result.ds[var] = ds_l3[var].where(mask_l4_nan)
    
    return result
    
    
    