#!/bin/bash

# Set environment
# source ~/.bashrc
module load pytorch

#  ---------------------------------------------------------------------------------------
# 1. **Step 1: Preparation of the NetCDF** - This step prepares the NetCDF file containing the variables needed for the emulator.
# 2. **Step 2: Calculation and Visualization** - It calculates the predictions of the 36 spectral channels (Reflectances: 1-19, 26 channel and Radiances 20-25, 27-36), saves them into another NetCDF file, and generates corresponding plots in PNG format, including true color.
 # ---------------------------------------------------------------------------------------
# #SBATCH -o $results_output_path/log_predict_spectrals.txt

# ----------------------------------------------------------------------------------------------------------- 
# --------------------------------------------- MODIFY PATH ------ ------------------------------------------ 
# ----------------------------------------------------------------------------------------------------------- 
# --------------------------  Set the output directory for the results -------------------------------------
results_output_path="output/results_emulator"

# ----------------------------------------------------- END MODIFY ------------------------------------------- 
# ------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------- STEP 1 ------------------------------------------- 
echo " --------------------------------------------------------------------------------------------------- "
echo " --------------------------------------------------------------------------------------------------- "
echo " -------------- Create the variables needed for the machine learning model ------------------------- "
echo " --------------------------------------------------------------------------------------------------- "
echo " --------------------------------------------------------------------------------------------------- "

# ----------------------------------------------------------------------------------------------------------- 
# -------------------------------------- MODIFY PATH AND VARIABLES ------------------------------------------ 
# ----------------------------------------------------------------------------------------------------------- 
# --------------------------  Define the base path for the netCDF files ------------------------------------- 
# Replace these with the correct file paths as per your dataset
# Define the base path for the netCDF files
base_path="/work/bb1036/rttov_share/dataset_ICON/"
name_file="icon_germany_20130502_T10.nc"

path_icon_netCDF=$base_path/$name_file

# If the variable names in your files differ from the expected names, list them here.
# Replace the placeholders with the actual names used in your files.
# names_variables="temp qv pres qnc clc tqc_dia"  # Example custom names
names_variables="ta hus pres qnc clc clwvi"  # Default expected names
# ----------------------------------------------------- END MODIFY ------------------------------------------- 
# ------------------------------------------------------------------------------------------------------------

# Execute the Python script to create variables for the emulator
python ../src/create_variables_emulator.py --path_icon_netCDF $path_icon_netCDF \
                                           --names_variables $names_variables \
                                           --results_output_path $results_output_path \
                                           &> $results_output_path/log_creation_variables_emulator.txt  

# Log file containing detailed output information
echo " ---- check log in $results_output_path/log_creation_variables_emulator.txt  ----"


# ```
# This code will generate a new file with the format of:
#   $results_output_path/"${name_base}_lwp_Ndmax.nc" ("$results_output_path/icon_germany_20130502_T10_lwp_Nd.nc")
#   # name base is define as the $name_file without the ".nc" extension and we add the new suffix "_lwp_Ndmax.nc"
#   name_base="${name_file%.nc}"
# ```
  
  
# -------------------------------------------------- STEP 2 ----------------------------------------------
echo " --------------------------------------------------------------------------------------------------- "
echo " --------------------------------------------------------------------------------------------------- "
echo " -------------- Get predictions in a netCDF and generate plots all spectral and RGB ---------------- "
echo " --------------------------------------------------------------------------------------------------- "
echo " --------------------------------------------------------------------------------------------------- "

# Specify the directory containing the PCA, scaler, and model files
path_pca_scaler_model="../model"

# Remove the ".nc" extension and add the new suffix "_lwp_Ndmax.nc"
# Path to the input NetCDF file obtained with the step one
name_base="${name_file%.nc}"
new_name="${name_base}_lwp_Ndmax.nc"
path_icon_emulator_netCDF=$results_output_path/$new_name

# Choose the method for RGB visualization (either "inverse_gamma" or "histogram")
method_rgb="inverse_gamma"
# method_rgb="histogram"
# gamma_value=22e-1

# Set the gamma value for gamma correction
gamma_value=2.2
name_file_model="NN_model"
python ../src/get_prediction.py --path_pca_scaler_model $path_pca_scaler_model \
                                --path_icon_netCDF $path_icon_emulator_netCDF \
                                --results_output_path $results_output_path \
                                --name_file_model $name_file_model \
                                --method_rgb $method_rgb \
                                --gamma_value $gamma_value \
                                &> $results_output_path/log_get_prediction_and_rgb_plot_all_channel.txt  

# Log file containing detailed output information
echo " ---- check log in $results_output_path/
log_get_prediction_and_rgb_plot_all_channel.txt  ----"



# --------------------------- OPTIONAL ---------------------------
# If it only require to plot one of the spectral channels

# channel_list="8"
# python ../src/get_prediction.py --path_pca_scaler_model $path_pca_scaler_model \
#                          --path_icon_netCDF $path_icon_emulator_netCDF \
#                          --channel_list $channel_list \
#                          --results_output_path $results_output_path &> $results_output_path/log_get_prediction_and_rgb_plot_$channel_list.txt  
