
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
# Train MQM models without a specific language (or the reverse or any direction).
# 

function launch_train() {
    NOLANGS=$1
    echo "Launching $NOLANGS on cuda:$CUDA_VISIBLE_DEVICES"

    comet-train \
        --cfg metrics_domain_adaptation/configs/comet_mqm_xep_keep1/main.yaml \
        --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/lang_bias/no_$NOLANGS" \
        --train-data "${ADAPTATION_ROOT}/data/experiments/lang_bias/no_$NOLANGS.csv" \
        --validation-data "${ADAPTATION_ROOT}/data/mqm/bio/dev/pudding_eval.csv" \
    > logs/train_mqm_lang_bias_${NOLANGS}.log;
}

launch_train "ende"
launch_train "deen"
launch_train "deen_ende" # this is reversed because the langs are alphabetically sorted
launch_train "enru"
launch_train "ruen"
launch_train "enru_ruen"
launch_train "enzh"
launch_train "zhen"
launch_train "enzh_zhen"
launch_train "enfr"
launch_train "fren"
launch_train "enfr_fren"
launch_train "enes"
launch_train "esen"
launch_train "enes_esen"
launch_train "bren"