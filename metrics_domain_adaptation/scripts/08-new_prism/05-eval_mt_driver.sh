
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
# Launch evaluation of MT performance of finetuned models.
# 

function eval_mt() {
    LANGS=$1
    MODEL=$2
    CUDA_VISIBLE_DEVICES=$3 nohup ./metrics_domain_adaptation/scripts/08-new_prism/04-eval_mt.py \
        --model $MODEL --langs $LANGS --domain "bio" 2>>nohup.out \
        | grep JSON | cut -c 6- >> "${ADAPTATION_ROOT}/computed/mtperf_finetuned.jsonl" &

    CUDA_VISIBLE_DEVICES=$4 nohup ./metrics_domain_adaptation/scripts/08-new_prism/04-eval_mt.py \
        --model $MODEL --langs $LANGS --domain "general" 2>>nohup.out \
        | grep JSON | cut -c 6- >> "${ADAPTATION_ROOT}/computed/mtperf_finetuned.jsonl" &
}


function eval_mt_slow() {
    LANGS=$1
    MODEL=$2
    CUDA_VISIBLE_DEVICES=$3 nohup ./metrics_domain_adaptation/scripts/08-new_prism/04-eval_mt.py \
        --model $MODEL --langs $LANGS --domain "bio" --batch-size 1 2>>nohup.out \
        | grep JSON | cut -c 6- >> "${ADAPTATION_ROOT}/computed/mtperf_finetuned.jsonl" &

    CUDA_VISIBLE_DEVICES=$4 nohup ./metrics_domain_adaptation/scripts/08-new_prism/04-eval_mt.py \
        --model $MODEL --langs $LANGS --domain "general" --batch-size 1 2>>nohup.out \
        | grep JSON | cut -c 6- >> "${ADAPTATION_ROOT}/computed/mtperf_finetuned.jsonl" &
}

# finished
eval_mt "en-de" "facebook/nllb-200-distilled-600M" 0 1
eval_mt "en-ru" "facebook/nllb-200-distilled-600M" 2 3
eval_mt "zh-en" "facebook/nllb-200-distilled-600M" 0 1
eval_mt "en-de" "facebook/nllb-200-1.3B" 2 3
eval_mt "en-ru" "facebook/nllb-200-1.3B" 4 5
eval_mt "zh-en" "facebook/nllb-200-1.3B" 6 7
eval_mt "en-de" "${ADAPTATION_ROOT}/models/trained/nllb/ende/600M_bio_lr1e-6_bs4/checkpoint-105000/" 0 1
eval_mt "en-ru" "${ADAPTATION_ROOT}/models/trained/nllb/enru/600M_bio_lr1e-6_bs4/checkpoint-105000/" 2 3
eval_mt "zh-en" "${ADAPTATION_ROOT}/models/trained/nllb/zhen/600M_bio_lr1e-7_bs16/checkpoint-9428/" 4 5
eval_mt "en-de" "/sdf/vzouhar/models/trained/nllb/ende/1.3B_bio_lr1e-6_bs4_melone/checkpoint-22500/" 0 1
eval_mt "en-ru" "/sdf/vzouhar/models/trained/nllb/enru/1.3B_bio_lr1e-6_bs4_melone/checkpoint-21000/" 2 3
eval_mt "zh-en" "/sdf/vzouhar/models/trained/nllb/zhen/1.3B_bio_lr1e-7_bs4_melone/checkpoint-4714/" 4 5
eval_mt_slow "en-de" "facebook/nllb-200-3.3B" 6 7
eval_mt_slow "en-ru" "facebook/nllb-200-3.3B" 0 2
eval_mt_slow "zh-en" "facebook/nllb-200-3.3B" 0 4
eval_mt "en-de" "facebook/nllb-200-1.3B" 0 1
eval_mt "en-ru" "facebook/nllb-200-1.3B" 2 3
eval_mt "zh-en" "facebook/nllb-200-1.3B" 4 5