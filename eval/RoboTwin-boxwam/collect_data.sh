#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}
extra_args=("${@:4}")

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config "${extra_args[@]}"

data_root="${SAVE_PATH:-data}"
prev=""
for arg in "${extra_args[@]}"; do
    if [[ "$arg" == "--collect-failure" ]]; then
        data_root="${data_root}_fail"
    elif [[ "$arg" == "--target-noise-std" ]]; then
        prev="--target-noise-std"
    elif [[ "$prev" == "--target-noise-std" ]]; then
        if [[ "$arg" != "0" && "$arg" != "0.0" ]]; then
            data_root="${data_root}_fail"
        fi
        prev=""
    fi
done
rm -rf ${data_root}/${task_name}/${task_config}/.cache
