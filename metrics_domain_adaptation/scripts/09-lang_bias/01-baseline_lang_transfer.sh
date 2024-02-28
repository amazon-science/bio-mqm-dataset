
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
# Compute performance on individual languages.
# 

LANGS=$1

./metrics_domain_adaptation/run_metric.py --metric comet-ours --langs $LANGS --domain bio \
    --model-path ${ADAPTATION_ROOT}/models/trained/da/main/checkpoints/epoch\=0-*.ckpt \
    | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_transfer.jsonl;
./metrics_domain_adaptation/run_metric.py --metric comet-ours --langs $LANGS --domain bio \
    --model-path ${ADAPTATION_ROOT}/models/trained/mqm/main/checkpoints/epoch\=0-*.ckpt \
    | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_transfer.jsonl;
./metrics_domain_adaptation/run_metric.py --metric comet-ours --langs $LANGS --domain bio \
    --model-path ${ADAPTATION_ROOT}/models/trained/finetune_datasize/count5500_seed0/*.ckpt \
    | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_transfer.jsonl;
./metrics_domain_adaptation/run_metric.py --metric comet-ours --langs $LANGS --domain bio \
    --model-path ${ADAPTATION_ROOT}/models/trained/scratch_datasize_1ep/count5500_up8_seed0/*.ckpt \
    | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_lang_transfer.jsonl;