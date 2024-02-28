
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
# Evaluate downstream PRISM based on finetuned models
# 

function eval_prism_src() {
    LANGS=$1
    MODEL=$2
    CUDA_VISIBLE_DEVICES=$3 nohup ./metrics_domain_adaptation/run_metric.py \
        --metric "prism2-src"  --langs $LANGS \
        --model-name $MODEL 2>>nohup.out \
            | grep JSON | cut -c 6- >> "${ADAPTATION_ROOT}/computed/metrics_prism2_finetuned.jsonl" &
}

function eval_prism_ref() {
    LANGS=$1
    MODEL=$2
    CUDA_VISIBLE_DEVICES=$3 nohup ./metrics_domain_adaptation/run_metric.py \
        --metric "prism2-ref"  --langs $LANGS \
        --model-name $MODEL 2>>nohup.out \
            | grep JSON | cut -c 6- >> "${ADAPTATION_ROOT}/computed/metrics_prism2_finetuned.jsonl" &
}

# running
eval_prism_src "en-de" "/sdf/vzouhar/models/trained/nllb/ende/1.3B_bio_lr1e-6_bs4_melone/checkpoint-22500/" 1
eval_prism_src "en-ru" "/sdf/vzouhar/models/trained/nllb/enru/1.3B_bio_lr1e-6_bs4_melone/checkpoint-21000/" 3
eval_prism_src "zh-en" "/sdf/vzouhar/models/trained/nllb/zhen/1.3B_bio_lr1e-7_bs4_melone/checkpoint-4714/" 5
eval_prism_ref "en-de" "/sdf/vzouhar/models/trained/nllb/ende/1.3B_bio_lr1e-6_bs4_melone/checkpoint-22500/" 6
eval_prism_ref "en-ru" "/sdf/vzouhar/models/trained/nllb/enru/1.3B_bio_lr1e-6_bs4_melone/checkpoint-21000/" 0
eval_prism_ref "zh-en" "/sdf/vzouhar/models/trained/nllb/zhen/1.3B_bio_lr1e-7_bs4_melone/checkpoint-4714/" 3
eval_prism_src "en-de" "facebook/nllb-200-3.3B" 1
eval_prism_src "en-ru" "facebook/nllb-200-3.3B" 5
eval_prism_src "zh-en" "facebook/nllb-200-3.3B" 6
eval_prism_ref "en-de" "facebook/nllb-200-3.3B" 5
eval_prism_ref "en-ru" "facebook/nllb-200-3.3B" 1
eval_prism_ref "zh-en" "facebook/nllb-200-3.3B" 0

# finished
eval_prism_src "en-de" "facebook/nllb-200-1.3B" 0
eval_prism_src "en-ru" "facebook/nllb-200-1.3B" 1
eval_prism_src "zh-en" "facebook/nllb-200-1.3B" 2
eval_prism_ref "en-de" "facebook/nllb-200-1.3B" 3
eval_prism_ref "en-ru" "facebook/nllb-200-1.3B" 6
eval_prism_ref "zh-en" "facebook/nllb-200-1.3B" 0
eval_prism_src "en-de" "facebook/nllb-200-distilled-600M" 0
eval_prism_src "en-ru" "facebook/nllb-200-distilled-600M" 1
eval_prism_src "zh-en" "facebook/nllb-200-distilled-600M" 2
eval_prism_ref "en-de" "facebook/nllb-200-distilled-600M" 3
eval_prism_ref "en-ru" "facebook/nllb-200-distilled-600M" 4
eval_prism_ref "zh-en" "facebook/nllb-200-distilled-600M" 5
eval_prism_src "en-de" "${ADAPTATION_ROOT}/models/trained/nllb/ende/600M_bio_lr1e-6_bs4/checkpoint-105000/" 0
eval_prism_src "en-ru" "${ADAPTATION_ROOT}/models/trained/nllb/enru/600M_bio_lr1e-6_bs4/checkpoint-105000/" 1
eval_prism_src "zh-en" "${ADAPTATION_ROOT}/models/trained/nllb/zhen/600M_bio_lr1e-7_bs16/checkpoint-9428/" 2
eval_prism_ref "en-de" "${ADAPTATION_ROOT}/models/trained/nllb/ende/600M_bio_lr1e-6_bs4/checkpoint-105000/" 3
eval_prism_ref "en-ru" "${ADAPTATION_ROOT}/models/trained/nllb/enru/600M_bio_lr1e-6_bs4/checkpoint-105000/" 6
eval_prism_ref "zh-en" "${ADAPTATION_ROOT}/models/trained/nllb/zhen/600M_bio_lr1e-7_bs16/checkpoint-9428/" 7

# test
# MODEL="${ADAPTATION_ROOT}/models/trained/nllb/ende/600M_bio_lr1e-6_bs4/checkpoint-105000/"
# ./metrics_domain_adaptation/run_metric.py --count 200 --domain bio --metric "prism2-src" --langs en-de --model-name $MODEL 2>>nohup.out | grep JSON | cut -c 6-

# MODEL="facebook/nllb-200-distilled-600M"
# ./metrics_domain_adaptation/run_metric.py --count 200 --domain bio --metric "prism2-src" --langs en-de --model-name $MODEL 2>>nohup.out | grep JSON | cut -c 6-