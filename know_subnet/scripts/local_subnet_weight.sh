#!/bin/bash

################################################################################
# Experiment Parameters
################################################################################
project_name="subnet_weights_wordnet"
# project_name="subnet_weights_conceptnet"
lm_name="gpt2"
gpu_setting="single_gpu"
ablation_name="no_expression_targetkg"
top_k_layers=6

seed_list="84 1318 735"

if [[ "$project_name" == *"wordnet"* ]]; then
    kg_type="wordnet"
    kg_list="statement.n.01 location.n.01 building.n.01 change.n.01 magnitude.n.01 representation.n.02 communication.n.02"
elif [[ "$project_name" == *"conceptnet"* ]]; then
    kg_type="conceptnet"
    kg_list="sun swimming fruit"
else
    echo "Project name must contain either 'wordnet' or 'conceptnet'"
    exit 1
fi

NFS_MOUNT="../.." # alternatively, absolute path to dir where you cloned the repo
WORK_DIR="${NFS_MOUNT}/know-subnet-clean/know_subnet"

################################################################################
# Train & test loop
################################################################################
echo "Starting mask training experiments with:"
echo "    lm=$lm_name, kg_type=$kg_type, kgs=$kg_list, seeds=$seed_list, ablation=$ablation_name"

for kg_name in $kg_list; do
    for seed in $seed_list; do
        exper_name=$"lm=$lm_name-kg=$kg_name-seed=$seed-ablation=$ablation_name"
        exper_date=$(date +"%Y_%m_%d_%H:%M:%S")

        echo "Exper name: $exper_name"
        echo "Date: $exper_date"
        
        ########################################################################
        # Training experiment launch
        ########################################################################
        accelerate launch \
            --config_file="$WORK_DIR/configs/accelerate_$gpu_setting.yaml" \
            subnet_train.py  \
            --kg_type="$kg_type" \
            --targetkg_name="$kg_name" \
            --seed="$seed" \
            --exper_name="$exper_name" \
            --project_name="$project_name" \
            --subnet_config_file="$WORK_DIR/configs/ablation_$ablation_name.json" \
            --date="$exper_date" \
            --lm="$lm_name" \
            --top_k_layers=$top_k_layers \
            --nfs_dir="$WORK_DIR" \
            --params="1,1"

        ########################################################################
        # Post-training evaluation
        ########################################################################
        accelerate launch \
            --config_file="$WORK_DIR/configs/accelerate_$gpu_setting.yaml" \
            subnet_test.py  \
            --exper_name="$exper_name" \
            --project_name="$project_name" \
            --date="$exper_date" \
            --nfs_dir="$WORK_DIR" \
            --is_random_baseline="False"
        
        done
    done

################################################################################
# Main result aggregation
################################################################################
echo "All mask training experiments are completed."
echo "Gathering main results..."

python subnet_gather_results.py \
    --project_name="$project_name" \
    --nfs_dir="$WORK_DIR" \
    --is_random_baseline="False" \
    --expected_num_experiments=21

################################################################################
# Random baseline evaluation at the same sparsity
################################################################################
echo "Running random baseline experiments..."
for kg_name in $kg_list; do
    for seed in $seed_list; do
        accelerate launch \
            --config_file="$WORK_DIR/configs/accelerate_$gpu_setting.yaml" \
            subnet_test.py  \
            --exper_name="$exper_name" \
            --project_name="$project_name" \
            --date="" \
            --nfs_dir="$WORK_DIR" \
            --is_random_baseline="True" \
            --random_baseline_seed="$seed" \
            --random_baseline_kg="$kg_name"
        done
    done

################################################################################
# Random baseline aggregation
################################################################################
echo "Gathering random baseline results..."
python subnet_gather_results.py \
    --project_name="$project_name" \
    --nfs_dir="$WORK_DIR" \
    --is_random_baseline="True" \
    --expected_num_experiments=21
