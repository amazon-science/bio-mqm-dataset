
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
# Train DA for one epoch.
# 

# use "fake" dev data to skip validation inference because we train for one epoch in any case
head -n 10 "${ADAPTATION_ROOT}/data/da/general/train.csv" > "${ADAPTATION_ROOT}/data/da/general/micro_fake.csv"

nohup comet-train \
    --cfg metrics_domain_adaptation/configs/comet_da_1ep/main.yaml \
    --model-checkpoint-dirpath "${ADAPTATION_ROOT}/models/trained/da/main/checkpoints" \
    --train-data "${ADAPTATION_ROOT}/data/da/general/train.csv" \
    --validation-data "${ADAPTATION_ROOT}/data/da/general/micro_fake.csv" \
> logs/train_main_da.log &