import joblib
import pickle as pk
import tensorflow as tf
import sys
import xarray as xr
import pandas as pd
import os
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature
import re

from skimage import exposure
from datetime import datetime
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from create_variables_emulator import convert_fractional_day_to_time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from pickle import load


def from_xarray_icon_to_dataframe(input_ds, variable_3d):
    """
    Convert an xarray Dataset from ICON (ICOsahedral Non-hydrostatic) model to a pandas DataFrame.

    The function specifically transforms 3D variables to separate 2D variables for each height level.
    After manipulating the data structure, the Dataset is stacked over lat and lon dimensions,
    and then converted to a DataFrame.

    Parameters:
       -- input_ds (xarray.Dataset): Input xarray Dataset with ICON model data. It should include
                                   3D variables such as "pres", "ta", "hus", etc., and dimensions
                                   like 'height', 'lat', and 'lon'.
                                   

    Returns:
       -- df (pandas dataframe): A DataFrame representation of the input Dataset with the 3D variables split into individual 2D variables based on height levels and stacked over 'lat' and 'lon' dimensions.


    """
    variable_3d = variable_3d

    for var in variable_3d:
        for i in range(len(input_ds['height'])):
            input_ds[var + '_height_' + str(i)] = input_ds[var].sel(height=input_ds['height'][i])

    # Drop the old 4D variables (time, ch, lat, lon)
    input_ds = input_ds.drop(variable_3d)
    data_at_specific_times = input_ds.drop_dims('height')
    # Stack 'lat' and 'lon' into a single dimension
    ds_stacked = data_at_specific_times.stack(lat_lon=('lat', 'lon'))
    # ds_stacked

    # Now convert to DataFrame
    df = ds_stacked.to_dataframe()  # 6gb

    input_ds.close()
    ds_stacked.close()

    return df


def apply_pca_transform_icon(df_icon_testing, pca, scaler_x):
    """
    Applies a previously fitted PCA transformation to a given test dataset.

    Parameters:
    df_icon_testing (dataframe): The test dataframe to apply scaler and PCA.
    pca (sklearn.decomposition.PCA): The PCA object that was previously fitted on the training data.
    scaler_x (StandardScaler): it has the scaler to apply to the features

    Returns:
    df_icon_pca_test (pandas.DataFrame): The transformed test dataset with 'Nd_max', 'lwp', and principal components.
    """

    print(" \n ---------------- Scaling features x --------")
    if isinstance(scaler_x, (MinMaxScaler, StandardScaler)):
        print("Scaling features")
        df_icon_testing = pd.DataFrame(scaler_x.transform(df_icon_testing), columns=df_icon_testing.columns)

    x_test_sub = df_icon_testing.drop(['Nd_max', 'lwp', 'lat', 'lon'], axis=1)
    print(" ---------------- Applying PCA in x--------")
    # Use the same PCA object to transform the test data
    principal_components_test = pca.transform(x_test_sub)
    # Create a new DataFrame with principal components
    df_pca_test = pd.DataFrame(data=principal_components_test,
                               columns=['pc_' + str(i + 1) for i in range(principal_components_test.shape[1])])

    # Reset the index of the original DataFrame (with 'Nd_max', 'lwp')
    df_icon_not_pca = df_icon_testing[['Nd_max', 'lwp']].reset_index(drop=True)
    # Concatenate 'Nd_max' and 'lwp' from the original DataFrame (with index reset) with the PCA DataFrame
    df_icon_pca_test = pd.concat([df_icon_not_pca, df_pca_test], axis=1)  
    
    return df_icon_pca_test  



def load_scaler_pca_all(path_data, name_extra):
    """
    Load the PCA and scalers for a given model from the specified path.

    Parameters:
    - path_data (str): Directory path where the PCA and scaler files are stored.
    - name_extra (str): Name of the files

    Returns:
    - pca_x: Loaded PCA for x.
    - scaler_x: Loaded scaler for x. If not found, returns None.
    - scaler_y: Loaded scaler for y. If not found, returns None.
    """
    try:
        pca_x = pk.load(open('{}/pca_x_{}.npy'.format(path_data, name_extra), 'rb'))
    except FileNotFoundError:
        print("The file pca_x was not found. Continue with the ejecution without using it.")
        
    try:
        scaler_x = load(open('{}/scaler_x_{}.npy'.format(path_data, name_extra), 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_x was not found. Continue with the ejecution using None as scaler.")
        scaler_x = None
    try:
        scaler_y = load(open('{}/scaler_y_{}.npy'.format(path_data, name_extra), 'rb'))  # , allow_pickle=True)
    except FileNotFoundError:
        print("The file scaler_y was not found. Continue with the ejecution without using it.")
        scaler_y = None
    return pca_x, scaler_x, scaler_y


def rmse_2(y_true, y_pred):
# def rmse(y_true, y_pred):
    # return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    # Calculate RMSE for each output
    rmse_per_output = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=0))
    
    # Aggregate RMSEs, here we use the mean RMSE across all outputs
    mean_rmse = tf.reduce_mean(rmse_per_output)

    return mean_rmse
    

# RMSE metric
def r2_score_2(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())


def load_model(path, name_model):
    """
    Load a pre-trained machine learning model from the specified directory.

    Parameters:
    - path_output (str): Directory path where the model's `.joblib` file is stored.
    - name_model (str): Name of the model file (without the `.joblib` extension).

    Returns:
    - model_loaded: Loaded machine learning model.
    """
    custom_objects = {"r2_score_2": r2_score_2, "rmse_2": rmse_2}

    model_path = os.path.join(path, name_model + ".h5")

    # Check if the path exists
    if os.path.exists(model_path):
        model_loaded = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Check if it's a Scikit-learn model
    elif os.path.exists(os.path.join(path, name_model + ".joblib")):
          model_path = os.path.join(path, name_model + ".joblib")
          model_loaded = joblib.load(model_path)
    else:
        raise ValueError("Model file not found!")
    print(model_path)
    
    return model_loaded


def from2to3d(x, lat_ds, lon_ds):
    """
    Convert a 2D array to a 3D array, primarily designed for reshaping image data.

    The function assumes that the 2D array has a shape (HxW, CH), where HxW
    are the dimensions of an image and CH is the number of channels or
    principal components of the image. The function reshapes this array
    into a 3D array with shape (CH, H, W).

    Arguments:
        x (numpy.ndarray): 2D array of shape (HxW, CH).
        lat_ds (int): Size of height dimension (H) in the reshaped 3D array (Y axis).
        lon_ds (int): Size of width dimension (W) in the reshaped 3D array (X axis).

    Returns:
        numpy.ndarray: Reshaped 3D array of shape (CH, H, W).
    """
    columns = np.shape(x)[1]
    x_3d = np.zeros((columns, len(lat_ds), len(lon_ds)))

    for i in range(columns):
        x_3d[i, :, :] = x[:, i].reshape(-1, len(lon_ds))

    return x_3d


def get_features(path_icon_nc, pca_x, scaler_x, variable_2d, variable_3d):

    test_input_ds = xr.open_dataset(path_icon_nc)

    # Check if the 'time' variable is a datetime64[ns] dtype
    if test_input_ds['time'].dtype == np.dtype('datetime64[ns]'):

        time_data = test_input_ds['time'].values
        # Display the formatted datetimes
        print("The 'time' variable is correctly type datetime64[ns]:", time_data)
    else:
        # Handle the case where 'time' is not datetime64[ns]
        print("The 'time' variable is not of type datetime64[ns] is proleptic_gregorian")

        time_data = convert_fractional_day_to_time(test_input_ds.time.values)
        time_data = pd.to_datetime(time_data)

    variable = variable_2d + variable_3d
    test_input_ds = test_input_ds[variable].sel(
        height=slice(40, None))  # lower that this usually the values are zero or doesn't change to much

    test_input_ds['lat'] = test_input_ds['lat'].astype(np.float32)
    test_input_ds['lon'] = test_input_ds['lon'].astype(np.float32)
    lat_test_input_ds = test_input_ds.lat.values
    lon_test_input_ds = test_input_ds.lon.values



    input_test = test_input_ds.expand_dims({"time": [time_data]})

    df_icon_testing = from_xarray_icon_to_dataframe(input_ds=input_test, variable_3d=variable_3d)
    num_feature_columns = len(df_icon_testing.columns) 

    df_icon_testing.reset_index(drop=True, inplace=True)


    df_icon_testing.loc[(df_icon_testing['lwp'] == 0) & (df_icon_testing['Nd_max'] != 0), 'Nd_max'] = 0

    df_icon_testing.loc[(df_icon_testing['Nd_max'] == 0) & (df_icon_testing['lwp'] != 0), 'lwp'] = 0
    

    df_icon_pca_test_features = apply_pca_transform_icon(df_icon_testing=df_icon_testing,
                                                         pca=pca_x,
                                                         scaler_x=scaler_x)

    index_names_drop = None
    return df_icon_pca_test_features, lat_test_input_ds, lon_test_input_ds, time_data, index_names_drop 


def predict_and_save_xrarray(df_icon_pca_test_features, index_names_drop, latitude, longitude, time_data, type_model, scaler_y, model, path_output): 
    """
    Prepare xarray Dataset containing the predicted and target image values after inverting the scaling transformation.

    This function takes the predictions made by a model and the target (ground truth) values,
    and then reverts any scaling transformations applied to these data (either standard scaling or MinMax scaling).
    The predictions and target values are then packed into an xarray Dataset for easier handling and interpretation.

    Arguments:
    - scaler_y (StandardScaler or MinMaxScaler): The scaler object used to scale the target values.
    - test_ref_rad_ds_labels (xarray.DataArray): An xarray DataArray containing target values (ground truth) for reference.
    - df_test_labels_ref (pandas.DataFrame): DataFrame containing test labels.
    - test_predictions_ref_rad (numpy.ndarray): Array containing the model predictions.

    Returns:
    - xarray.Dataset: A Dataset containing two DataArrays: 'ch_prediction' and 'ch_target', representing the
      predicted and target values, respectively.

    Notes:
    The function assumes that the channel indices provided (channels_idx) are 0-based. However,
    when storing them in the xarray Dataset attributes, they are incremented by 1 to reflect 1-based channel numbering.
    """
                      
                      
    x = df_icon_pca_test_features
    if isinstance(model, Sequential):  # If the model is a Keras model
        pred = model.predict(x, verbose=0)
    else:  # If the model is a scikit-learn model or other type
        pred = model.predict(x)
        
    # pred = np.maximum(pred, 0)
    # print("Flatted shape along lat long", np.shape(pred))
                  
    ch_ds_img = ['1: 0.645', '2: 0.856', '3: 0.466', '4: 0.554', '5: 1.241', '6: 1.628',
                 '7: 2.113', '8: 0.412', '9: 0.442', '10: 0.487', '11: 0.530',
                 '12: 0.547', '13: 0.666', '14: 0.678', '15: 0.747', '16: 0.867',
                 '17: 0.904', '18: 0.936', '19: 0.935', '20: 3.777', '21: 3.981',
                 '22: 3.971', '23: 4.061', '24: 4.448', '25: 4.526', '26: 1.382',
                 '27: 6.783', '28: 7.344', '29: 8.550', '30: 9.720', '31: 11.017',
                 '32: 12.036', '33: 13.362', '34: 13.683', '35: 13.923', '36: 14.213']
   
    lat_ds_img = latitude
    lon_ds_img = longitude

    if isinstance(scaler_y, StandardScaler):
        # Get means and variances for these channels
        means = scaler_y.mean_
        variances = scaler_y.var_

        # Revert the standard scaling
        test_predictions_ref_rad_no_scaler = pred * np.sqrt(variances) + means

    # print(np.shape(test_predictions_ref_rad_no_scaler))
    # test_predictions_ref_rad_no_scaler[index_names_drop] = np.nan

    # test_predictions_ref_rad_no_scaler[index_names_drop, :] = np.nan

    img_predicted_3d = from2to3d(x=test_predictions_ref_rad_no_scaler,
                                 lat_ds=lat_ds_img,
                                 lon_ds=lon_ds_img)


    ch_m_pred = xr.DataArray(img_predicted_3d.astype(np.float64), dims=['chan', 'lat', 'lon'],
                             coords=[ch_ds_img, lat_ds_img, lon_ds_img])  # range(1, np.shape(img_predicted_3d)[0]+1),
  
    print(f" ======== 3D shape pred {np.shape(ch_m_pred)}, lat {np.shape(lat_ds_img)}, lon {np.shape(lon_ds_img)}")
          
# # Count how many values are less than 0
#     negative_values_count = (xr_output.ref_rad_prediction < 0).sum().item()
    
    xr_output = xr.Dataset(dict(ref_rad_prediction=ch_m_pred, time=time_data))
    xr_output["ref_rad_prediction"] = xr_output.ref_rad_prediction.where(xr_output.ref_rad_prediction >= 0, 0)

    # Assigning the attributes
    xr_output.attrs['ref_units'] = '-'
    xr_output.attrs['rad_units'] = 'W/m2/sr/um'
    xr_output.attrs['ref_bands_idx'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26])
    xr_output.attrs['rad_bands_idx'] = np.array([20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
    xr_output.attrs['ref_long_name'] = 'All-sky upwelling reflectance at TOA'
    xr_output.attrs['rad_long_name'] = 'All-sky upwelling radiance at TOA'
    xr_output.attrs['name_variable'] = 'ref1-19_26_rad20-25_27_36'
    xr_output.attrs['chan_index'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])

    # time_data
    # print(xr_output.attrs)
    # time_data = datetime.strptime(time_data, '%Y-%m-%dT%H:%M') 
    # formatted_time_data = time_data.strftime('%Y%m%dT%H%M')

    time_data = pd.to_datetime(time_data)
    formatted_time_data = time_data.strftime('%Y%m%dT%H%M')
        
    try:
        file_path = f"{path_output}/prediction_channels_{type_model}_{formatted_time_data}_{xr_output.name_variable}.nc"
        xr_output.to_netcdf(file_path, 'w')
        print(f"Predictions file saved successfully at {file_path}")
    except Exception as e:
        print(f"Failed to save the file: {str(e)}")
    finally:
        xr_output.close()

    return file_path


def extract_model_type(filename):
    # Use a regular expression to find the model type (NN, CNN, RF)
    match = re.search(r'(NN|CNN|RF)', filename)
    
    if match:
        return match.group(0)
    else:
        return "Unknown Model"


def read_predict_save_test(path_icon_netCDF, path_pca_scaler_model, results_output_path, name_file_model="ref1-19_26_rad20-25_27_36_NN11_k_fold_0"):
    """
    Read data from a specified netCDF file, apply scaler, PCA and machine learning prediction, and save the results in a netCDF file.
    
    This function is specifically tailored for handling atmospheric data variables in a 2D and 3D format, processing them through a pre-trained model to predict spectral channels (Reflectances: 1-19, 26 channel and Radiances 20-25, 27-36). 
   

    Parameters:
    - path_icon_netCDF (str): Path to the input netCDF file containing atmospheric data (it should have a time variable).
    - path_pca_scaler_model (str): Directory path where the PCA scaler model is stored.
    - results_output_path (str): Path where the prediction results will be saved.

    Notes:
    - The function is configured to use a fixed variables and with the names format names for the scaler, pca and the name of the model:
    - variable_2d (list): List of 2D variable names to be processed.
    - variable_3d (list): List of 3D variable names to be processed.
    - format_name_pca_scaler (str): Naming format for the PCA and scaler to be used.
    - name_file_model (str): Name of the machine learning model file for making predictions.

    Returns:
    - None. The function directly writes the output to a netCDF file located at the `results_output_path`.
    """
    print(" ---------- Read NetCDF file and predict spectral channels ---------- ")

    variable_time = ["time"]
    variable_2d = ["Nd_max", "lwp"]
    variable_3d = ["pres", "ta", "hus", "clc"]
    format_name_pca_scaler = "all_channels_k_fold_0"
    name_file_model = name_file_model
    # "ref1-19_26_rad20-25_27_36_NN11_k_fold_0"

    pca_x, scaler_x, scaler_y = load_scaler_pca_all(path_data=path_pca_scaler_model,
                                                    name_extra=f"{format_name_pca_scaler}")

    model = load_model(path=path_pca_scaler_model,
                       name_model=name_file_model)
    
    df_icon_pca_test_features, lat_test_input_ds, lon_test_input_ds, time_data, index_names_drop = get_features(path_icon_netCDF, pca_x, scaler_x, variable_2d, variable_3d)


    type_model = extract_model_type(name_file_model)
    print(f"---------- Predicting with model {type_model}----------------") 

    path_pred_netCDF = predict_and_save_xrarray(df_icon_pca_test_features=df_icon_pca_test_features,
                                         index_names_drop=index_names_drop,
                                         latitude=lat_test_input_ds,
                                         longitude=lon_test_input_ds,
                                         time_data=time_data,
                                                type_model=type_model,
                                         scaler_y=scaler_y,
                                         model=model,
                                         path_output=results_output_path)
    

    return path_pred_netCDF


def plot_specific_channel(i, ds_model1, name_model, prediction_name,
                                      all_imgs, axes, fig, plt, formatted_date, cmap='Spectral'):
    """
     Plot distribution of the channel
    """
    if all_imgs == "False":
        fig, axes = plt.subplots(1, 1, figsize=(4, 40))
        axli = axes.ravel()

    else:
        fig = fig
        axli = axes


    titles_font_size = 16.5
    subtitles_font_size = 16
    labels_font_size = 14  
    # 15.5
    axes_font_size = 13
    # 14
    
    fig.suptitle(f"{formatted_date}\nChannel {ds_model1.chan[i].values}µm", fontsize=titles_font_size, weight='bold')

    ax_idx = 0
    im = ds_model1[prediction_name][i].plot(ax=axli[ax_idx], cmap=cmap, add_colorbar=False) 
    axli[ax_idx].add_feature(ccrs.cartopy.feature.BORDERS, color='black', linewidth=1)
    axli[ax_idx].add_feature(ccrs.cartopy.feature.COASTLINE, color='black', linewidth=1)

    gl = axli[ax_idx].gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    axli[ax_idx].set_xlabel('')
#     -----------------------------------------------------------------------------------
    axli[ax_idx].set_ylabel(f"Prediction ({name_model})", fontsize=subtitles_font_size)
# -----------------------------------------------------------------------------------
    axli[ax_idx-1].set_title('')
    axli[ax_idx].set_title('')

    # Reflectance colorbar
    if ds_model1.chan_index[i] in ds_model1.ref_bands_idx:
        cbar = fig.colorbar(im, location='bottom')
        # cbar = fig.colorbar(im, location='right', ax=axli[1:3])
        cbar.set_label(rf'Reflectance at TOA ({ds_model1.attrs.get("ref_units")})', fontsize=axes_font_size)

    # Radiancia colorbar
    elif ds_model1.chan_index[i] in ds_model1.rad_bands_idx:
        cbar = fig.colorbar(im, location='bottom')
        cbar.set_label(rf'Radiance at TOA (W/$m^2$/sr/µm)', fontsize=axes_font_size)
    cbar.ax.tick_params(labelsize=axes_font_size)


    
def plot_spectral_channels(path_pred_nc, cmap, path_output, channel_list=None):
    print(f" \n ------------- Plot spectral channels ------------- ")
    
    
    name_model = path_pred_nc.split("/")[-1].split("_")[2]
    file_name_part = path_pred_nc.split('/')[-1]  
    unique_name = file_name_part.split('_prediction_')[0].split('.nc')[0]
    



    ds = xr.open_dataset(path_pred_nc)

    time_data = pd.to_datetime(ds.time.values)
    # date_time_plot = datetime.strftime(time_data, '%Y-%m-%dT%H:%M')
    date_time_plot = time_data.strftime('%d %B %Y, %I:%M %p')

    # formatted_date_time = time_data.strftime('%Y%m%dT%H%M')
   

    if channel_list:
        ds = xr.open_dataset(path_pred_nc)

        # Find the positions of these values
      
        print(" ------ channel_list", channel_list)
        positions = [idx for idx, value in enumerate(ds.chan_index) if value in channel_list]

        ds_filter = ds.isel(chan=positions)

        # Filtra el array original para mantener solo los índices deseados
        filtered_chan_index = ds.chan_index[np.isin(ds.chan_index, channel_list)]
        ds_filter.attrs['chan_index'] = filtered_chan_index
        pred_ds = ds_filter
        
        name_file = f'{path_output}/distribution_{unique_name}_{channel_list}ch.png'
            
    
    else:
        
        pred_ds = ds
        name_file = f'{path_output}/distribution_{unique_name}_all_channels.png'


    
    n_imgs = pred_ds['ref_rad_prediction'].shape[0]
    # fig = plt.figure(constrained_layout=True, facecolor='white', figsize=(18, 4*n_imgs))
    fig = plt.figure(constrained_layout=True, facecolor='white', figsize=(4, 4*n_imgs))

    if n_imgs > 1:
        sfigs = fig.subfigures(n_imgs,1)

        sfig = sfigs.ravel()
    else:
        sfig = [fig]

    for i, sf in enumerate(sfig):
        axs_geo = [sf.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())]
        axs = axs_geo
        
        plot_specific_channel(i=i,
                              ds_model1=pred_ds,
                              name_model=name_model,
                              prediction_name='ref_rad_prediction',
                              axes=axs,
                              fig=sfig[i],
                              all_imgs='True',
                              plt=plt,
                              cmap=cmap,
                              formatted_date=date_time_plot)


    fig.savefig(name_file) 
    # print(f"-------------------------------- Image save in {name_file}")
    # Check if the file was actually created
    if os.path.exists(name_file):
        print(f"Spectral channels file was successfully saved at: {name_file}")
    else:
        print(f"Error: The spectral channels file was not saved at: {name_file}")

        
def enhance_visibility(image, method_rgb='inverse_gamma', gamma_value=2.2):

    if method_rgb == 'inverse_gamma':
        # Apply gamma correction
        gamma_corrected = np.power(image, 1/gamma_value)
        return (gamma_corrected * 255).astype('uint8')
    
    elif method_rgb == 'histogram':
        # Apply histogram equalization
        equalized = exposure.equalize_hist(image)
        return (equalized * 255).astype('uint8')
    
    elif method_rgb == 'contrast_stretch':
        # Apply contrast stretching
        p2, p98 = np.percentile(image, (2, 98))
        stretched = exposure.rescale_intensity(image, in_range=(p2, p98))
        return stretched
    
    
def create_rgb_image(red_channel, green_channel, blue_channel, 
                                        method_rgb, gamma_value=2.2):
    """Create an RGB image from specified channels in the dataset."""
    print(f" \n ------------- Plot True color (RGB) ------------- ")
    red = red_channel.values
    green = green_channel.values
    blue = blue_channel.values
    
    # nan_mask = np.isnan(red)
    nan_mask = np.isnan(red) | np.isnan(green) | np.isnan(blue)

    rgb = np.stack([red, green, blue], axis=-1)
    # rgb = np.clip(rgb, 0, 1)
    enhanced_image = enhance_visibility(image=rgb, 
                                        method_rgb=method_rgb, 
                                        gamma_value=gamma_value)
 

    return enhanced_image
    # return rgb



def plot_rgb_with_projection_target_prediction(rgb_prediction, rgb_target, lat, lon, title_pred, title_target, save_path=None):
    """Plot two RGB images with lat/lon labels."""
    projection = ccrs.PlateCarree()  # Definir la proyección aquí

    fig, ax = plt.subplots(1, 2, figsize=(20, 7), subplot_kw={'projection': projection})
    # Configurar los ejes para mostrar líneas de latitud/longitud y costas
    for a in ax:
        a.coastlines(resolution='10m')
        a.add_feature(cfeature.BORDERS)  # Agrega bordes de países
        a.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=projection)

        # Configurar formateadores y marcas de latitud y longitud
        gl = a.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        # Ajustar el tamaño de fuente de las etiquetas de latitud y longitud
        gl.xlabel_style = {'size': 28, 'color': 'black'}
        gl.ylabel_style = {'size': 28, 'color': 'black'}

    # Plot RGB Prediction
    ax[0].imshow(rgb_prediction, origin='lower', extent=[lon.min(), lon.max(), lat.min(), lat.max()], transform=projection)
    ax[0].set_title(title_pred, fontsize=28)

    # Plot RGB Target
    ax[1].imshow(rgb_target, origin='lower', extent=[lon.min(), lon.max(), lat.min(), lat.max()], transform=projection)
    ax[1].set_title(title_target, fontsize=28)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if os.path.exists(save_path):
        print(f" --------------------- Image saved en:  {save_path}")
    else:
        print(f" Imagen True Color not saved")



def plot_single_rgb_with_projection(rgb_data, lat, lon, title, projection=ccrs.PlateCarree(), save_path=None):
    """Plot a single RGB image with lat/lon labels."""
    titles_font_size = 16.5
    axes_font_size = 13
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS) 
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=projection)

    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    ax.imshow(rgb_data, origin='lower', extent=[lon.min(), lon.max(), lat.min(), lat.max()], transform=projection)
    ax.set_title(title, fontsize=titles_font_size)
    fig.tight_layout()

    if save_path:
        file_output = f"{save_path}"
        plt.savefig(file_output, dpi=300)
        print(f" --------------------- Image saved en: {file_output}")


def process_and_plot_rgb(path, method_rgb, gamma_value=2.2, indx_rgb=[0, 3, 2], save_path=None):
    """
    Procesa y grafica imágenes RGB(channel 1,4,3) a partir de un archivo NetCDF.

    Parámetros:
    path (str): Ruta al archivo NetCDF que contiene los datos MODIS.
    indx_rgb (list): Índices de los canales para los colores RGB.
    
    Estructura del Dataset:
    - Dimensiones: 'chan', 'lat', 'lon'
    - Variables:
        - 'ch_prediction': Datos de la predicción con dimensiones (chan, lat, lon).
        - 'ch_target': Datos objetivo con dimensiones (chan, lat, lon).
    - 'chan' representa los canales o bandas y no es opcional.
    - Las variables deben estar alineadas con las coordenadas de latitud y longitud.

    La función extrae los canales especificados en indx_rgb para crear y graficar imágenes RGB.
    Si solo están disponibles los datos de predicción, solo se grafica la predicción.
    """
    ds = xr.open_dataset(path)
    date_str = ds.time.values
    # .item()  # '%Y-%m-%dT%H:%M'
 
    file_name_part = path.split('/')[-1]  
    unique_name = file_name_part.split('_channels_')[1].split('.nc')[0]
    file_path = f'{save_path}/rgb_{unique_name}_method_rgb_{method_rgb}.png'    
      
    time_data = pd.to_datetime(ds.time.values)
    # date_time_plot = time_data.strftime('%d %B %Y, %I:%M %p')
    date_time_plot = time_data.strftime('%d %B %Y, %H:%M UTC')

    print(date_time_plot)

    if 'ref_rad_prediction' in ds:
        prediction = ds.ref_rad_prediction
        rgb_prediction = create_rgb_image(red_channel=prediction[indx_rgb[0]],
                                          green_channel=prediction[indx_rgb[1]],
                                          blue_channel=prediction[indx_rgb[2]],
                                          method_rgb=method_rgb,
                                          gamma_value=gamma_value)
        
        
        lat = ds.lat
        lon = ds.lon
        plot_single_rgb_with_projection(rgb_data=rgb_prediction, 
                                        lat=lat, 
                                        lon=lon, 
                                        title=f'RGB Prediction ({date_time_plot})',
                                        save_path=file_path)

    if 'ref_rad_target' in ds:
        print(" -------------------- ref_rad_target ------------------------------------ ")
        target = ds.ref_rad_target
        rgb_target = create_rgb_image(red_channel=target[indx_rgb[0]],
                                      green_channel=target[indx_rgb[1]],
                                      blue_channel=target[indx_rgb[2]],
                                          method_rgb=method_rgb,
                                          gamma_value=gamma_value)

        
        
        print(" -------------- plot_rgb_with_projection_target_prediction ----------")

        plot_rgb_with_projection_target_prediction(rgb_prediction=rgb_prediction, 
                                                   rgb_target=rgb_target, 
                                                   lat=lat, 
                                                   lon=lon, 
                                                   title_pred=f'RGB Prediction ({date_time_plot})', 
                                                   title_target=f'RGB Target ({date_time_plot})', 
                                                   save_path=file_path)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the netCDF to predict the 36 spectral channel and plot them and RGB")
    arg = parser.add_argument
    arg('--path_pca_scaler_model', type=str, default="output/results_3days_cleaning_test_T10/files_needed", 
        help='Directory which has the pca, scaler and model')
    arg('--path_icon_netCDF', type=str, default="rttov_share/dataset_ICON/icon_germany_20130502_T10_lwp_Nd.nc",
        help='File wich has the netcdf file to be processed.')
    arg('--results_output_path', type=str, default="output/results_3days_cleaning_test_T10/files_needed",
        help='Directory for saving the files')
    arg('--name_file_model', type=str, default="ref1-19_26_rad20-25_27_36_NN11_k_fold_0",
        help='name of the model saved')
    arg('--channel_list', nargs='+', type=int, default=None, 
        help='List of channels 1-36 to plot for example only [8] or plot the different channels as the same time [8, 22] or choose None to plot all the channel')   
    arg('--method_rgb', type=str, default="inverse_gamma",
        help="This is the method to use to improve the quality of the rgb image. Options are: 'inverse_gamma', 'histogram' (for areas with water this is a good option), 'contrast_stretch', 'adaptive_histogram'")
    arg('--gamma_value', type=float, default=2.2,
        help="value of the gamma to apply")
    args = parser.parse_args()
    method_rgb =args.method_rgb
    gamma_value =args.gamma_value

      

    path_pred_netCDF = read_predict_save_test(path_icon_netCDF=args.path_icon_netCDF, 
                                              path_pca_scaler_model=args.path_pca_scaler_model, 
                                              results_output_path=args.results_output_path,
                                             name_file_model=args.name_file_model)



    
    plot_spectral_channels(path_pred_nc=path_pred_netCDF,
                           cmap='coolwarm',
                           channel_list=args.channel_list,
                           path_output=args.results_output_path)


    process_and_plot_rgb(path=path_pred_netCDF,
                         indx_rgb=[0, 3, 2],
                         save_path=args.results_output_path,
                         method_rgb=args.method_rgb,
                         gamma_value=args.gamma_value)
    
    
