#!/bin/bash

# Set environment
source ~/.bashrc
module load pytorch 

prepro_type="StandardScaler_StandardScaler"
n_splits=5
batch_size=1024
lr=1e-4
data_type="ref1-19_26_rad20-25_27_36"
channel_number_list={1..36}
n_epochs=60


path_dataframes_pca_scaler="dataframes_kfolds/3days_cleaning_test_T10_zeroboth/${prepro_type}" 

pca_include_nd_lwp="no_nd_lwp" 

variable_2d="Nd_max lwp"


pca_scaled="False"

joined_variable="${variable_2d// /}"
path_models="output/pytorch_results_3days_cleaning_test_T10_zeroboth_${joined_variable}/${prepro_type}" 

# Create the directory and any necessary parent directories
mkdir -p "$path_models"




type_model="NN" 

date="training"
#date="all_testing"

hr="-"
path_model_file="$path_models/ref1-19_26_rad20-25_27_36_NN_k_fold_0.pth"

# ---------------------------- Cross validation k-folds ----------------------------
# for fold_num in $(seq 0 $((n_splits - 1))); do
for fold_num in 0; do

# Submit the job using sbatch
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=${date}_${type_model}_${hr}
#SBATCH --partition=gpu
##SBATCH --account=
#SBATCH --nodes=1 
#SBATCH --gpus=2                   # Specify number of GPUs needed for the job
#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=${path_models}/log_${data_type}_${type_model}_${prepro_type}_kfold_${fold_num}_%j.log


python pytorch_training_emulator.py --fold-num $fold_num \
                           --channel_number_list $channel_number_list \
                           --type-model $type_model \
                           --lr $lr \
                           --data-type $data_type \
                           --n-epochs $n_epochs \
                           --batch-size $batch_size \
                           --path_dataframes_pca_scaler $path_dataframes_pca_scaler \
                           --variable_2d $variable_2d\
                           --path_models $path_models 




        
python pytorch_get_metrics.py --date $date \
                              --hr $hr \
                              --fold-num $fold_num \
                              --type-model $type_model \
                              --path_dataframes_pca_scaler $path_dataframes_pca_scaler \
                              --variable_2d $variable_2d\
                              --path_model_file $path_model_file \
                              --results_output_path $path_models

EOF

    done
