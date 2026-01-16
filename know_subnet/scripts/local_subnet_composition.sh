#!/bin/bash

################################################################################
# Composition experiment parameters
################################################################################
project_name="subnet_weights_wordnet"
gpu_setting="single_gpu"
composition_patterns="intersection union floral"
seed_list="84 735 1318"
kg_list="representation.n.02 location.n.01 communication.n.02"
composition_kg_list="representation.n.02,location.n.01,communication.n.02"
composition_success_file="success_best_ckpnts-random=False.csv"

NFS_MOUNT="../.." # alternatively, absolute path to dir where you cloned the repo
WORK_DIR="${NFS_MOUNT}/know-subnet-clean/know_subnet"
OUTPUT_DIR="${WORK_DIR}/logs/seedvar"

mkdir -p "$OUTPUT_DIR"

################################################################################
# Seed composition: 3 seeds for the same KG
################################################################################
echo "Running seed composition experiments..."
for kg_name in $kg_list; do
    for pattern in $composition_patterns; do
        echo "Seed composition: kg=$kg_name, pattern=$pattern"

        accelerate launch \
            --config_file="$WORK_DIR/configs/accelerate_$gpu_setting.yaml" \
            subnet_test.py \
            --project_name="$project_name" \
            --nfs_dir="$WORK_DIR" \
            --is_composition="True" \
            --composition_mode="seed" \
            --composition_pattern="$pattern" \
            --composition_kg="$kg_name" \
            --composition_success_file="$composition_success_file"

        compose_project="${project_name}_compose_seed"
        compose_exper="composition-seed-${pattern}"
        compose_dir="$WORK_DIR/logs/${compose_project}/logs/${compose_exper}-date=composition"
        res_file="${compose_dir}/res-${compose_exper}-date=composition-random=False.csv"
        out_file="${OUTPUT_DIR}/seedvar-${kg_name}-${pattern}-${project_name}.csv"

        if [[ -f "$res_file" ]]; then
            cp "$res_file" "$out_file"
        else
            echo "Missing result file: $res_file"
            exit 1
        fi
    done
done

################################################################################
# KG composition: 3 KGs for the same seed
################################################################################
echo "Running KG composition experiments..."
for seed in $seed_list; do
    for pattern in $composition_patterns; do
        echo "KG composition: seed=$seed, pattern=$pattern"

        accelerate launch \
            --config_file="$WORK_DIR/configs/accelerate_$gpu_setting.yaml" \
            subnet_test.py \
            --project_name="$project_name" \
            --nfs_dir="$WORK_DIR" \
            --is_composition="True" \
            --composition_mode="kg" \
            --composition_pattern="$pattern" \
            --composition_kg_list="$composition_kg_list" \
            --composition_seed="$seed" \
            --composition_success_file="$composition_success_file"

        compose_project="${project_name}_compose_kg"
        compose_exper="composition-kg-${pattern}"
        compose_dir="$WORK_DIR/logs/${compose_project}/logs/${compose_exper}-date=composition"
        res_file="${compose_dir}/res-${compose_exper}-date=composition-random=False.csv"
        out_file="${OUTPUT_DIR}/kgvar-${seed}-${pattern}-${project_name}.csv"

        if [[ -f "$res_file" ]]; then
            cp "$res_file" "$out_file"
        else
            echo "Missing result file: $res_file"
            exit 1
        fi
    done
done

################################################################################
# Aggregate composition results by pattern
################################################################################
pattern_count=$(echo "$composition_patterns" | wc -w)
kg_count=$(echo "$kg_list" | wc -w)
seed_count=$(echo "$seed_list" | wc -w)

seed_expected=$((pattern_count * kg_count))
kg_expected=$((pattern_count * seed_count))

echo "Aggregating seed composition results (expected experiments: $seed_expected)..."
python subnet_gather_results.py \
    --project_name="${project_name}_compose_seed" \
    --nfs_dir="$WORK_DIR" \
    --is_composition="True" \
    --group_by_key="pattern" \
    --expected_num_experiments="$seed_expected"

echo "Aggregating KG composition results (expected experiments: $kg_expected)..."
python subnet_gather_results.py \
    --project_name="${project_name}_compose_kg" \
    --nfs_dir="$WORK_DIR" \
    --is_composition="True" \
    --group_by_key="pattern" \
    --expected_num_experiments="$kg_expected"