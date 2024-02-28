
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
# Generate scores for each sample by our models.
# 

./metrics_domain_adaptation/run_metric.py --metric comet-ours \
    --model-path ${ADAPTATION_ROOT}/models/trained/finetune_datasize/count5500_seed0/*.ckpt \
    --save-scores-path ${ADAPTATION_ROOT}/computed/scores_ft.jsonl;

./metrics_domain_adaptation/run_metric.py --metric comet-ours \
    --model-path ${ADAPTATION_ROOT}/models/trained/mqm/main/checkpoints/*.ckpt \
    --save-scores-path ${ADAPTATION_ROOT}/computed/scores_base.jsonl;
