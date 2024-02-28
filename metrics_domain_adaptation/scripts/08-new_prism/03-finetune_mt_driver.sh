
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
# Launch finetuning of NLLB MT models
#

function train_mt_nllb_600m() {
    LANGS=$2
    LANGSX=${LANGS//-/}
    DOMAIN=$3
    BATCH_SIZE=$4
    LEARNING_RATE=$5
    LEARNING_RATEX=${LEARNING_RATE//-/m}
    echo "Launching" $LANGS $LANGSX $BATCH_SIZE $LEARNING_RATE $LEARNING_RATEX

    CUDA_VISIBLE_DEVICES=$1 nohup ./metrics_domain_adaptation/scripts/08-new_prism/02-finetune_mt.py \
    --langs $LANGS \
    --domain $DOMAIN \
    --model "facebook/nllb-200-distilled-600M" \
    --batch-size $BATCH_SIZE \
    --wandb-name "lr1e$LEARNING_RATE, bs$BATCH_SIZE" \
    --learning-rate 1e$LEARNING_RATE \
    > logs/600m_bio_${LANGSX}_bs${BATCH_SIZE}_lr1e${LEARNING_RATEX}.log &
}

train_mt_nllb_600m 0 "en-ru" "bio" 4 -6
train_mt_nllb_600m 1 "en-de" "bio" 4 -6
train_mt_nllb_600m 2 "zh-en" "bio" 16 -7
train_mt_nllb_600m 3 "en-ru" "general" 4 -6
train_mt_nllb_600m 4 "en-de" "general" 4 -6
train_mt_nllb_600m 5 "zh-en" "general" 16 -7
train_mt_nllb_600m 6 "zh-en" "bio" 4 -6
train_mt_nllb_600m 7 "zh-en" "general" 4 -6

function train_mt_nllb_1b() {
    LANGS=$2
    LANGSX=${LANGS//-/}
    DOMAIN=$3
    BATCH_SIZE=$4
    LEARNING_RATE=$5
    LEARNING_RATEX=${LEARNING_RATE//-/m}
    echo "Launching" $LANGS $LANGSX $BATCH_SIZE $LEARNING_RATE $LEARNING_RATEX

    CUDA_VISIBLE_DEVICES=$1 nohup ./metrics_domain_adaptation/scripts/08-new_prism/02-finetune_mt.py \
    --langs $LANGS \
    --domain $DOMAIN \
    --model "facebook/nllb-200-1.3B" \
    --batch-size $BATCH_SIZE \
    --save-path "/sdf/vzouhar/models/trained/nllb/" \
    --wandb-name "lr1e$LEARNING_RATE, bs$BATCH_SIZE, melone" \
    --learning-rate 1e$LEARNING_RATE \
    > logs/1b_bio_${LANGSX}_bs${BATCH_SIZE}_lr1e${LEARNING_RATEX}.log &
}

# running on melone
train_mt_nllb_1b 0 "en-ru" "bio" 4 -6
train_mt_nllb_1b 1 "en-de" "bio" 4 -6
train_mt_nllb_1b 2 "zh-en" "bio" 4 -6
train_mt_nllb_1b 3 "en-ru" "bio" 4 -7
train_mt_nllb_1b 4 "en-de" "bio" 4 -7
train_mt_nllb_1b 5 "zh-en" "bio" 4 -7
train_mt_nllb_1b 6 "en-ru" "bio" 4 -5
train_mt_nllb_1b 7 "en-de" "bio" 4 -5
train_mt_nllb_1b 8 "zh-en" "bio" 4 -5



function train_mt_nllb_3b() {
    LANGS=$2
    LANGSX=${LANGS//-/}
    DOMAIN=$3
    BATCH_SIZE=$4
    LEARNING_RATE=$5
    LEARNING_RATEX=${LEARNING_RATE//-/m}
    echo "Launching" $LANGS $LANGSX $BATCH_SIZE $LEARNING_RATE $LEARNING_RATEX

    CUDA_VISIBLE_DEVICES=$1 nohup ./metrics_domain_adaptation/scripts/08-new_prism/02-finetune_mt.py \
    --langs $LANGS \
    --domain $DOMAIN \
    --model "facebook/nllb-200-3.3B" \
    --batch-size $BATCH_SIZE \
    --save-path "/sdf/vzouhar/models/trained/nllb/" \
    --wandb-name "lr1e$LEARNING_RATE, bs$BATCH_SIZE, melone" \
    --learning-rate 1e$LEARNING_RATE \
    --gradient-accumulation-steps 4 \
    > logs/3b_bio_${LANGSX}_bs${BATCH_SIZE}_lr1e${LEARNING_RATEX}.log &
}

train_mt_nllb_3b 0 "en-ru" "bio" 1 -6
train_mt_nllb_3b 1 "en-de" "bio" 1 -6
train_mt_nllb_3b 2 "zh-en" "bio" 1 -6
train_mt_nllb_3b 3 "en-ru" "bio" 1 -7
train_mt_nllb_3b 4 "en-de" "bio" 1 -7
train_mt_nllb_3b 5 "zh-en" "bio" 1 -7


# make all nodes download the model
python3 -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
# AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); \
# AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M');
# AutoTokenizer.from_pretrained('facebook/nllb-200-1.3B'); \
# AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-1.3B');
AutoTokenizer.from_pretrained('facebook/nllb-200-3.3B'); \
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-3.3B'); \
"
