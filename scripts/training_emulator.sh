#!/bin/bash

# Set environment
source ~/.bashrc
module load pytorch #python3

prepro_type="StandardScaler_StandardScaler"
n_splits=5
batch_size=1024
lr=1e-4
data_type="ref1-19_26_rad20-25_27_36"
channel_number_list={1..36}
n_epochs=60


path_dataframes_pca_scaler="dataframes_kfolds/3days_cleaning_test_T10_zeroboth_only_clouds/${prepro_type}" 
path_models="output/results_3days_cleaning_test_T10_zeroboth_only_clouds/${prepro_type}" 
pca_include_nd_lwp="no_nd_lwp" 
# variable_2d="Nd_max lwp"
variable_2d="lwp"
pca_scaled="False"


# Create the directory and any necessary parent directories
mkdir -p "$path_models"



# I will keep the next models
# type_model="NN11"
# type_model="NN4"
type_model="RF2"
# type_model="CNN_op2"
# type_model="RF3"

# ---------------------------- Cross validation k-folds ----------------------------
# for fold_num in $(seq 0 $((n_splits - 1))); do
for fold_num in 0; do
# 
# Submit the job using sbatch
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=${type_model}k${fold_num} 
#SBATCH --partition=compute
##SBATCH --account=
#SBATCH --nodes=1 #16 #16 #16 #16 #5
#SBATCH --cpus-per-task=64 # 32 CNN Request four CPUs per task
## SBATCH --nodes=<n>
#SBATCH --time=08:00:00  #8 
#SBATCH --mail-type=ALL
##SBATCH -o models-ML.o%j
##SBATCH --error=models-ML%j.log
#SBATCH -o ${path_models}/log_${data_type}_${type_model}_${prepro_type}_kfold_${fold_num}%j.txt

python training_emulator.py --fold-num $fold_num \
                           --channel_number_list $channel_number_list \
                           --type-model $type_model \
                           --lr $lr \
                           --data-type $data_type \
                           --n-epochs $n_epochs \
                           --batch-size $batch_size \
                           --path_dataframes_pca_scaler $path_dataframes_pca_scaler \
                           --path_models $path_models 


EOF
    done
