
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

#!/usr/bin/env bash

# 
# Train MQM on DA which was trained on subsampled data.
# 

function launch_train() {
    TRIMODE=$1
    PERC=$2

    echo "Launching trimode_${TRIMODE}_${PERC}p on cuda:$CUDA_VISIBLE_DEVICES"

    nohup comet-train \
        --load_from_checkpoint ${ADAPTATION_ROOT}/models/trained/da/culprit_da/trimode_${TRIMODE}_${PERC}p/epoch=0-*.ckpt \
        --cfg metrics_domain_adaptation/configs/comet_mqm_1ep/main.yaml \
        --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/mqm/main/checkpoints_culprit/trimode_${TRIMODE}_${PERC}p" \
        --train-data "${ADAPTATION_ROOT}/data/mqm/general/train/all.csv" \
        --validation-data "${ADAPTATION_ROOT}/data/mqm/general/test/micro_fake.csv" \
    > logs/train_main_mqm_culprit_${TRIMODE}_${PERC}p.log &
}

launch_train "none" "00"
launch_train "none" "10"
launch_train "none" "15"
launch_train "none" "20"
launch_train "none" "25"
launch_train "none" "50"
launch_train "none" "75"
launch_train "none" "100"

launch_train "auth" "25"
launch_train "auth" "50"
launch_train "auth" "75"
launch_train "auth" "100"
launch_train "both" "25"
launch_train "both" "50"
launch_train "both" "75"
launch_train "both" "100"
