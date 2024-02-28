
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
# Evaluate infinite DA and DA+MQM.
# 

function eval_mqm_xep() {
    # this is very sloppy organization
    MQM_EP_NAME=$1
    DA_EP_NAME=$2

    if [[ "$DA_EP_NAME-$MQM_EP_NAME" == "0-0" ]]; then
        echo "CASE 0-0"
        ls "${ADAPTATION_ROOT}/models/trained/da_xep/checkpoints/xlm-roberta-large.ckpt";
        nohup ./metrics_domain_adaptation/run_metric.py 2> nohup.out \
            --metric comet-ours --model-path ${ADAPTATION_ROOT}/models/trained/da_xep/checkpoints/xlm-roberta-large.ckpt \
            | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_dax_mqmx.jsonl &
    elif [[ "$MQM_EP_NAME" == "0" ]]; then
        echo "CASE x-0"
        # fallback to DA model
        # DA is off by one
        DA_EP_NAME=$((DA_EP_NAME-1))

        ls ${ADAPTATION_ROOT}/models/trained/da_xep/checkpoints/epoch\=${DA_EP_NAME}-*.ckpt;
        nohup ./metrics_domain_adaptation/run_metric.py 2> nohup.out \
            --metric comet-ours --model-path ${ADAPTATION_ROOT}/models/trained/da_xep/checkpoints/epoch\=${DA_EP_NAME}-*.ckpt \
            | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_dax_mqmx.jsonl &
    else
        echo "CASE x-x"
        # MQM is off by one, DA is correct
        MQM_EP_NAME=$((MQM_EP_NAME-1))

        ls ${ADAPTATION_ROOT}/models/trained/mqm/from_da${DA_EP_NAME}/epoch\=${MQM_EP_NAME}-*.ckpt;
        nohup ./metrics_domain_adaptation/run_metric.py 2> nohup.out \
            --metric comet-ours --model-path ${ADAPTATION_ROOT}/models/trained/mqm/from_da${DA_EP_NAME}/epoch\=${MQM_EP_NAME}-*.ckpt \
            | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_dax_mqmx.jsonl &
    fi
}

# MQM 0 DA x
eval_mqm_xep 0 0;
eval_mqm_xep 0 1;
eval_mqm_xep 0 2;
eval_mqm_xep 0 4;
eval_mqm_xep 0 8;
# MQM 1 DA x
eval_mqm_xep 1 0;
eval_mqm_xep 1 1;
eval_mqm_xep 1 2;
eval_mqm_xep 1 4;
eval_mqm_xep 1 8;
# MQM 2 DA x
eval_mqm_xep 2 0;
eval_mqm_xep 2 1;
eval_mqm_xep 2 2;
eval_mqm_xep 2 4;
eval_mqm_xep 2 8;
# MQM 4 DA x
eval_mqm_xep 4 0;
eval_mqm_xep 4 1;
eval_mqm_xep 4 2;
eval_mqm_xep 4 4;
eval_mqm_xep 4 8;
# MQM 8 DA x
eval_mqm_xep 8 0;
eval_mqm_xep 8 1;
eval_mqm_xep 8 2;
eval_mqm_xep 8 4;
eval_mqm_xep 8 8;
# MQM 16 DA x
eval_mqm_xep 16 0;
eval_mqm_xep 16 1;
eval_mqm_xep 16 2;
eval_mqm_xep 16 4;
eval_mqm_xep 16 8;