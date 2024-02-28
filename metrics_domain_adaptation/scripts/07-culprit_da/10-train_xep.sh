
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

#!/usr/bin/env python3

# 
# Train DA and DA+MQM for possibly unlimited epochs.
# 

# train DA for unlimited epochs and save all
comet-train \
    --cfg metrics_domain_adaptation/configs/comet_da_xep/main.yaml \
    --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/da_xep/wmt22" \
    --train-data "${ADAPTATION_ROOT}/data/da/general/train.csv" \
    --validation-data "${ADAPTATION_ROOT}/data/da/general/micro_fake.csv" \
> logs/train_main_da_xep.log;

# create COMET model from xlmr (0ep)
comet-train \
    --cfg metrics_domain_adaptation/configs/comet_da_xep/main.yaml \
    --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/da_xep/0ep" \
    --train-data "${ADAPTATION_ROOT}/data/da/general/micro_fake.csv" \
    --validation-data "${ADAPTATION_ROOT}/data/da/general/micro_fake.csv" \
> logs/train_main_da_0ep.log;

function train_mqm_xep() {
    DA_CHECKPOINT=$1
    DA_EP_NAME=$2

    echo "Launching ${DA_EP_NAME} (da ${DA_CHECKPOINT}) on cuda:${CUDA_VISIBLE_DEVICES}"
    if [[ "$DA_CHECKPOINT" == "xlmr" ]]; then
        # train without DA model
        comet-train \
            --cfg metrics_domain_adaptation/configs/comet_mqm_xep/main.yaml \
            --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/mqm/from_da0/" \
            --train-data "${ADAPTATION_ROOT}/data/mqm/general/train/all.csv" \
            --validation-data "${ADAPTATION_ROOT}/data/mqm/general/test/micro_fake.csv" \
        > logs/train_mqm_from_da${DA_EP_NAME}.log;
    else
        comet-train \
            --load_from_checkpoint $DA_CHECKPOINT \
            --cfg metrics_domain_adaptation/configs/comet_mqm_xep/main.yaml \
            --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/mqm/from_da${DA_EP_NAME}/" \
            --train-data "${ADAPTATION_ROOT}/data/mqm/general/train/all.csv" \
            --validation-data "${ADAPTATION_ROOT}/data/mqm/general/test/micro_fake.csv" \
        > logs/train_mqm_from_da${DA_EP_NAME}.log;
    fi
}

train_mqm_xep "xlmr" 0;
train_mqm_xep ${ADAPTATION_ROOT}/models/trained/da_xep/checkpoints/epoch=0-*.ckpt 1;
train_mqm_xep ${ADAPTATION_ROOT}/models/trained/da_xep/checkpoints/epoch=1-*.ckpt 2;
train_mqm_xep ${ADAPTATION_ROOT}/models/trained/da_xep/checkpoints/epoch=3-*.ckpt 4;
train_mqm_xep ${ADAPTATION_ROOT}/models/trained/da_xep/checkpoints/epoch=7-*.ckpt 8;