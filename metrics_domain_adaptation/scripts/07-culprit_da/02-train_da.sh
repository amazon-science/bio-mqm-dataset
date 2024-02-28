
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
# Train DA on subsampled data.
# 

function launch_train() {
    TRIMODE=$1
    PERC=$2

    TRAIN_SIZE=$(wc -l < "${ADAPTATION_ROOT}/data/experiments/da_culprit/train_trimode_${TRIMODE}_${PERC}p.csv")

    echo "Launching trimode_${TRIMODE}_${PERC}p on cuda:$CUDA_VISIBLE_DEVICES which is ${TRAIN_SIZE} lines"

    nohup comet-train \
        --cfg metrics_domain_adaptation/configs/comet_da_1ep/main.yaml \
        --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/da/culprit_da/trimode_${TRIMODE}_${PERC}p" \
        --train-data "${ADAPTATION_ROOT}/data/experiments/da_culprit/train_trimode_${TRIMODE}_${PERC}p.csv" \
        --validation-data "${ADAPTATION_ROOT}/data/da/general/micro_fake.csv" \
    > logs/train_main_da_culprit_${TRIMODE}_${PERC}p.log &
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