
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
# Train mixed General + Bio (subsampled, upsampled) MQM models on top of DA for 1 epoch.
# 

mkdir -p logs/
mkdir -p ${ADAPTATION_ROOT}/data/experiments/scratch_datasize_1ep/

BIO_COUNT=$1
BIO_UPSAMPLE=$2
SEED=$3
TMP_FILE=$(mktemp)

get_seeded_random() {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

# start with general train data which also have the header
cat "${ADAPTATION_ROOT}/data/mqm/general/train/pudding_train.csv" > "${ADAPTATION_ROOT}/data/experiments/scratch_datasize_1ep/count${BIO_COUNT}_up${BIO_UPSAMPLE}_seed${SEED}.csv"

# get the Bio subset
tail -n +2 ${ADAPTATION_ROOT}/data/mqm/bio/dev/all.csv \
    | shuf -n $BIO_COUNT --random-source=<(get_seeded_random $SEED) \
    >> $TMP_FILE
    
# Do the upsampling
for ((i=0; i<${BIO_UPSAMPLE}; i++)); do
    cat $TMP_FILE >> "${ADAPTATION_ROOT}/data/experiments/scratch_datasize_1ep/count${BIO_COUNT}_up${BIO_UPSAMPLE}_seed${SEED}.csv";
done
rm $TMP_FILE

echo "Running with up=${BIO_UPSAMPLE}, count=${BIO_COUNT} bio sentences and seed ${SEED} on GPU:${CUDA_VISIBLE_DEVICES}"

comet-train \
    --load_from_checkpoint ${ADAPTATION_ROOT}/models/trained/da/main/checkpoints/*.ckpt \
    --train-data "${ADAPTATION_ROOT}/data/experiments/scratch_datasize_1ep/count${BIO_COUNT}_up${BIO_UPSAMPLE}_seed${SEED}.csv" \
    --validation-data "${ADAPTATION_ROOT}/data/mqm/general/train/pudding_eval.csv" \
    --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/scratch_datasize_1ep/count${BIO_COUNT}_up${BIO_UPSAMPLE}_seed${SEED}/" \
    --cfg metrics_domain_adaptation/configs/comet_mqm_1ep/main.yaml \
    > logs/train_scratch_datasize_1ep_count${BIO_COUNT}_up${BIO_UPSAMPLE}_seed${SEED}.log