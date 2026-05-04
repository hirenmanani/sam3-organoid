#!/bin/bash
export PATH="/home/hmanani/miniconda3/bin:$PATH"
source /home/hmanani/miniconda3/etc/profile.d/conda.sh
conda activate sam3
cd /home/hmanani/sam3
/home/hmanani/miniconda3/envs/sam3/bin/python sam3/train/train.py -c configs/organoid/organoid_finetune_gpu
