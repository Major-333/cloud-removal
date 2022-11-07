#!/bin/bash

# this example uses a single node (`NUM_NODES=1`) w/ 4 GPUs (`NUM_GPUS_PER_NODE=4`)
export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
export CUDA_VISIBLE_DEVICES="6,7"
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
# master address
export MASTER_ADDR="localhost"
export MASTER_PORT="9996"
# wandb group name
export WANDB_GROUP=$(date "+%Y%m%dT%H%M%S")

# launch your script w/ `torch.distributed.launch`
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --use_env gan_train.py