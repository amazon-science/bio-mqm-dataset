
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
# Eval MQM models without a specific language (or the reverse or any direction).
# 

function launch_eval() {
    NOLANGS=$1
    echo "Launching $NOLANGS on cuda:$CUDA_VISIBLE_DEVICES"

    ./metrics_domain_adaptation/run_metric.py --metric comet-ours \
        --model-path ${ADAPTATION_ROOT}/models/trained/lang_bias/no_$NOLANGS/*.ckpt \
        --domain bio --langs "total" \
    | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_bias.jsonl;
}

launch_eval "ende"
launch_eval "enru"
launch_eval "enzh"
launch_eval "enes"
launch_eval "enfr"
launch_eval "deen"
launch_eval "ruen"
launch_eval "zhen"
launch_eval "esen"
launch_eval "fren"
launch_eval "deen_ende"
launch_eval "enru_ruen"
launch_eval "enzh_zhen"
launch_eval "enes_esen"
launch_eval "enfr_fren"
launch_eval "bren"

# mode 0 and baselines

./metrics_domain_adaptation/run_metric.py --metric comet-ours \
    --model-path ${ADAPTATION_ROOT}/models/trained/mqm/main/checkpoints/*.ckpt \
    --domain bio --langs "total" \
| grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_bias.jsonl;

./metrics_domain_adaptation/run_metric.py --metric comet-ours \
    --model-path ${ADAPTATION_ROOT}/models/trained/da/main/checkpoints/*.ckpt \
    --domain bio --langs "total" \
| grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_bias.jsonl;

./metrics_domain_adaptation/run_metric.py --metric comet-ours \
    --model-path ${ADAPTATION_ROOT}/models/trained/finetune_datasize/count5500_seed0/*.ckpt \
    --domain bio --langs "total" \
| grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_bias.jsonl;

./metrics_domain_adaptation/run_metric.py --metric bleu \
    --domain bio --langs "total" \
| grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_bias.jsonl;

./metrics_domain_adaptation/run_metric.py --metric chrf \
    --domain bio --langs "total" \
| grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_bias.jsonl;

./metrics_domain_adaptation/run_metric.py --metric ter \
    --domain bio --langs "total" \
| grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_bias.jsonl;

