
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
# Train DA and DA+MQM models based on finetuned XLMR.
# 


# train DA on adapted XLMR
comet-train \
    --pretrained_model "${ADAPTATION_ROOT}/models/trained/xlmr-large/XLM;bio lr_scheduler_type='cosine';warmup_steps=10000;learning_rate=5e-7/checkpoint-40000/" \
    --cfg metrics_domain_adaptation/configs/comet_da_1ep/main.yaml \
    --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/da/main/checkpoints_bioxlmr" \
    --train-data "${ADAPTATION_ROOT}/data/da/general/train.csv" \
    --validation-data "${ADAPTATION_ROOT}/data/da/general/test.csv" \
> logs/train_main_da_bioxlmr.log;

comet-train \
    --pretrained_model "${ADAPTATION_ROOT}/models/trained/xlmr-large/XLM;general lr_scheduler_type='cosine';warmup_steps=10000;learning_rate=5e-7/checkpoint-40000/" \
    --cfg metrics_domain_adaptation/configs/comet_da_1ep/main.yaml \
    --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/da/main/checkpoints_generalxlmr" \
    --train-data "${ADAPTATION_ROOT}/data/da/general/train.csv" \
    --validation-data "${ADAPTATION_ROOT}/data/da/general/test.csv" \
> logs/train_main_da_bioxlmr.log;

# train MQM on top of DA on top of adapted XLMR
for DOMAIN in "bio" "general"; do
    comet-train \
        --load_from_checkpoint ${ADAPTATION_ROOT}/models/trained/da/main/checkpoints_${DOMAIN}xlmr/epoch=0-*.ckpt \
        --cfg metrics_domain_adaptation/configs/comet_mqm_1ep/main.yaml \
        --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/mqm/main/checkpoints_${DOMAIN}xlmr" \
        --train-data "${ADAPTATION_ROOT}/data/mqm/general/train/all.csv" \
        --validation-data "${ADAPTATION_ROOT}/data/mqm/general/test/micro_fake.csv" \
    > logs/train_main_mqm_${DOMAIN}xlmr.log;
done;

# eval trained COMET
for DOMAIN in "bio" "general"; do
    ./metrics_domain_adaptation/run_metric.py \
        --metric comet-ours \
        --model-path ${ADAPTATION_ROOT}/models/trained/mqm/main/checkpoints_${DOMAIN}xlmr/*.ckpt \
    2>1 | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_adaptlm_comet.jsonl;
done