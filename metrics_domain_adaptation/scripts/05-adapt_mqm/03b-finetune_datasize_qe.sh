
#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#!/usr/bin/bash

# 
# Finetune MQM models (referenceless) on top of DA with various BIO_COUNT size.
# 

mkdir -p logs/
mkdir -p ${ADAPTATION_ROOT}/data/experiments/finetune_datasize_qe/

BIO_COUNT=$1
SEED=$2

get_seeded_random() {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

head -n 1 "${ADAPTATION_ROOT}/data/mqm/bio/dev/pudding_train.csv" > "${ADAPTATION_ROOT}/data/experiments/finetune_datasize_qe/count${BIO_COUNT}_seed${SEED}.csv"
tail -n +2 ${ADAPTATION_ROOT}/data/mqm/bio/dev/pudding_train.csv \
    | shuf -n $BIO_COUNT --random-source=<(get_seeded_random $SEED) \
    >> "${ADAPTATION_ROOT}/data/experiments/finetune_datasize_qe/count${BIO_COUNT}_seed${SEED}.csv"

echo "Running with ${BIO_COUNT} sentences and seed ${SEED} on GPU:${CUDA_VISIBLE_DEVICES}"

comet-train \
    --load_from_checkpoint ${ADAPTATION_ROOT}/models/trained/mqm/qe/checkpoints/epoch=0-*.ckpt \
    --train-data "${ADAPTATION_ROOT}/data/experiments/finetune_datasize_qe/count${BIO_COUNT}_seed${SEED}.csv" \
    --validation-data "${ADAPTATION_ROOT}/data/mqm/bio/dev/pudding_eval.csv" \
    --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/finetune_datasize_qe/count${BIO_COUNT}_seed${SEED}/" \
    --cfg metrics_domain_adaptation/configs/cometqe_mqm_xep_keep1/main.yaml \
    > logs/train_finetune_datasize_qe_count${BIO_COUNT}_seed${SEED}.log
