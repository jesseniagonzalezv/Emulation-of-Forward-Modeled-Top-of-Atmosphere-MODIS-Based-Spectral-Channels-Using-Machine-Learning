
import xarray as xr
import pandas as pd
import os
import numpy as np
import math
import argparse

def convert_fractional_day_to_time(time_varible: float) -> (int, int, int, int, int, int):
    """
    Parameters:
    time_varible (float64) -- Given a time value as a fractional day in the format YYYYMMDD.fractionalday. For example time = convert_fractional_day_to_time(ds.time.values)

    Returns:
    formatted_datetimes (string) -- '%Y%m%dT%H%M' representing the specified date and calculated time (year, month, day, hours, minutes).
    """
    # Split the input into date and fractional day parts
    date, fractional_day = divmod(time_varible, 1)

    # Convert the date part to an integer and then to a string for processing
    date_str = str(int(date))

    # Extract year, month, and day from the date string
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # Convert fractional day to hours, minutes, and seconds
    hours = fractional_day * 24
    minutes = (hours - int(hours)) * 60
    seconds = (minutes - int(minutes)) * 60

    # Round to the nearest whole numbers
    hours = math.floor(hours)
    minutes = math.floor(minutes)
    seconds = round(seconds)

    # If seconds are 60, increment minutes by one and reset seconds to zero
    if seconds == 60:
        minutes += 1
        seconds = 0

    # If minutes are 60, increment hours by one and reset minutes to zero
    if minutes == 60:
        hours += 1
        minutes = 0
    
    time_data = pd.Timestamp(year=year, month=month, day=day, hour=hours, minute=minutes, second=seconds)

    # formatted_datetimes = time_data.strftime('%Y-%m-%dT%H:%M')
    formatted_datetimes = time_data.strftime('%Y%m%dT%H%M')

    # print(formatted_datetimes)

    return formatted_datetimes


def get_save_lwp_Nd_max(ds, output_folder_path, name_file):
    print(" ---------- Calculating LWP and Nd_max ---------- ")

    if (ds['clc'] > 2).any():
        print("Converting 'clc' values from percentage to fraction.")
        # Convert 'clc' from percent to fraction where 'clc' is greater than 1
        ds['clc'] = ds['clc'].where(ds['clc'] <= 1, ds['clc'] / 100)
        ds['clc'] =  ds['clc'] / 100
    else:
        print("No 'clc' values in percent. No conversion necessary to fraction.")

    T_c = np.float64(ds.ta) - 273.15
    esat_2013 = 0.611 * np.exp((17.3 * T_c) / (T_c + 237.3)) * 1000.0
    # esat_2013 = np.ma.masked_array(esat_2013,  esat_2013 == 0) ## check it!!!!!!!!
    pres = np.ma.masked_array(ds.pres, ds.pres == 0)  ## check it!!!!!!!!
    qs_2013 = 0.622 * (esat_2013 / pres)  # this is diffent compared with Alexandre code
    r_2013 = ds.hus / (1 - ds.hus)
    RH_2013 = 100 * (r_2013 / qs_2013)
    pv_2013 = (esat_2013 * RH_2013) / 100.0
    pd_2013 = ds.pres - pv_2013
    rho_2013 = (pd_2013 / (287.058 * ds.ta)) + (pv_2013 / (461.495 * ds.ta))  # nana
    cdnc_2013_cm = (rho_2013 * ds.qnc) / 1000000  # convert to cm^-3

    Nd_max = np.nanmax(cdnc_2013_cm, axis=0)
    ds["Nd_max"] = (['lat', 'lon'], Nd_max)  # this is an array
    ds.Nd_max.attrs['units'] = "cm-3"
    ds.Nd_max.attrs['standard_name'] = "Nd_max"
    ds.Nd_max.attrs['long_name'] = "Cloud dropler number maximun"

    ds["Nd"] = cdnc_2013_cm  # thi is a xarray.DataArray
    ds.Nd.attrs['units'] = "cm-3"
    ds.Nd.attrs['standard_name'] = "Nd"
    ds.Nd.attrs['long_name'] = "Cloud dropler number in each layer"

    ds["time"] = ds.time.values  # thi is a xarray.DataArray

    ds = ds.assign(lwp=ds.clwvi * 1000)
    ds.lwp.attrs['units'] = "gm-2"
    ds.lwp.attrs['standard_name'] = "LWP"
    ds.lwp.attrs['long_name'] = "Liquid water path"


    variable_calculated = ["lwp", "Nd_max"]
    variable_3D = ["pres", "ta", "hus",  "clc"]
    variables_total = variable_3D  + variable_calculated + ["time"]

    ds = ds.get(variables_total)

    
    if ds['time'].dtype == np.dtype('datetime64[ns]'):
        print("The 'time' variable is correctly type datetime64[ns]")
        # time_data = pd.to_datetime(ds['time'].values) # time is only a variable
        # formatted_time_data = time_data.strftime('%Y%m%dT%H%M')

    else:
        # Handle the case where 'time' is not datetime64[ns]
        print("The 'time' variable is not of type datetime64[ns] is proleptic_gregorian")
        formatted_time_data = convert_fractional_day_to_time(ds.time.values)
        ds['time'] = pd.to_datetime(formatted_time_data)
        print("New formatted datetimes:", ds['time'].dtype)

#     print("Date %Y%m%T%H%M:", formatted_time_data)
       
#     path_icon_netCDF = f"{output_folder_path}/test_icon_{formatted_time_data}_lwp_Nd.nc"


    # Extract the file name without extension
    name_file = os.path.splitext(os.path.basename(name_file))[0]
    # name_output = os.path.splitext(fname)[0] + "_lwp_Nd.nc"

    path_icon_netCDF = f"{output_folder_path}/{name_file}_lwp_Ndmax.nc"
    ds.to_netcdf(path_icon_netCDF)  # 

    ds.close
    # return ds
    if os.path.exists(path_icon_netCDF):
        print(f"File saved successfully at {path_icon_netCDF}")

    else:
        print(f"Failed to save the file at {path_icon_netCDF}")
        
        

def read_get_1sample(icon_path, names_variables):
    """
     Rename variables
    """
    name_needed = ['ta', 'hus', 'pres', 'qnc', 'clc', 'clwvi']

    ds = xr.open_dataset(icon_path)

    # Create a dictionary to map old variable names to new variable names
    variable_map = dict(zip(names_variables, name_needed))

    # Rename the variables in the dataset
    ds_renamed = ds.rename(variable_map)

    if 'time' in ds_renamed.dims:
        # If 'time' exists, select the first time index
        ds_1sample = ds_renamed.isel(time=0)

        # Delete dimension 'time'
        ds_1sample = ds_1sample.reset_coords('time')
        # , drop=True)
        print("Processed: Time dimension found, selected the first timestep and delete the time dimension and keep time as variable.")
    else:
        print("No action taken: correctly find time variable.")
        ds_1sample = ds_renamed
        
    return ds_1sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the netCDF to predict the 36 spectral channel and plot them and RGB")
    arg = parser.add_argument
    arg('--path_icon_netCDF', type=str, default="/work/bb1036/rttov_share/dataset_ICON/icon.nc",
        help='File wich has the netcdf file to be processed.')
    arg('--results_output_path', type=str, default="/work/bb1036/b381362/output/results_3days_cleaning_test_T10/files_needed",
        help='Directory for saving the files')
    arg('--names_variables', nargs=6, type=str, default=['ta', 'hus', 'pres', 'qnc', 'clc', 'clwvi'],
        help="If the variable names do not match the expected ['ta', 'hus', 'pres', 'qnc', 'clc', 'clwvi'], specify the actual names in the file in this corresponding order.")

    args = parser.parse_args()


    ds_1sample = read_get_1sample(icon_path=args.path_icon_netCDF,
                                  names_variables=args.names_variables)
    
    get_save_lwp_Nd_max(ds=ds_1sample,
                        output_folder_path=args.results_output_path,
                        name_file=args.path_icon_netCDF)