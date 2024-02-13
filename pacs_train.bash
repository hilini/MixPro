#!/bin/bash

# Set the number of GPUs you want to use
num_gpus=7 # 예: 4개의 GPU 사용

# Set the configuration file, imagenet path, batch size, output directory, and job tag
config_file="configs/deit/deit_tiny_patch14_mask56_224_alpha1.yaml"  # 예: 설정 파일 경로
#config_file="configs/vae/vae.yaml"  # 예: 설정 파일 경로

pacs_path="data/pacs"  # 예: ImageNet 데이터셋 경로
batch_size=128  # 예: 각 GPU에 대한 배치 크기
output_directory="pacs_results"  # 예: 결과를 저장할 디렉터리
#output_directory="vae_results"  # 예: 결과를 저장할 디렉터리
job_tag="my_job"  # 예: 작업에 대한 태그

# Execute the Python command with the specified arguments
python -m torch.distributed.launch --nproc_per_node $num_gpus --master_port 12345 main.py \
--cfg $config_file --data-path $pacs_path --batch-size $batch_size --output $output_directory --tag $job_tag
 
 python -m torch.distributed.launch --nproc_per_node 1 --master_port 14  main.py \
--cfg configs/deit/deit_tiny_patch14_mask56_224_alpha1.yaml --data-path /data/pacs --batch-size 128 --output output --tag 1004