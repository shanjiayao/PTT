#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

cd tools

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train_tracking.py --launcher pytorch ${PY_ARGS}
