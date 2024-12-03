#!/bin/bash

GPU=${1}
CONFIG=${2}
ARGS=${@:3}

torchrun --standalone --nproc_per_node ${GPU} scripts/train_magicdrive.py \
     ${CONFIG} ${ARGS}
