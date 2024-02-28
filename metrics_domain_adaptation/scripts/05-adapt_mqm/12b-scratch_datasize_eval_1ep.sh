
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
# Evaluate mixed General + Bio (subsampled, upsampled) MQM models.
# 

BIO_COUNT=$1
BIO_UPSAMPLE=$2
SEED=$3

# run eval for both the first and the second epoch
for f in ${ADAPTATION_ROOT}/models/trained/scratch_datasize_1ep/count${BIO_COUNT}_up${BIO_UPSAMPLE}_seed${SEED}/*.ckpt; do
  echo "Running $f on gpu:${CUDA_VISIBLE_DEVICES}" 1>&2;
  ./metrics_domain_adaptation/run_metric.py --metric comet-ours --model-path "$f" | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_scratch_datasize_1ep.jsonl;
done


# CUDA_VISIBLE_DEVICES=0 nohup ./metrics_domain_adaptation/scripts/05-adapt_mqm/12b-scratch_datasize_eval_1ep.sh "5500" "6" "0" &
# CUDA_VISIBLE_DEVICES=1 nohup ./metrics_domain_adaptation/scripts/05-adapt_mqm/12b-scratch_datasize_eval_1ep.sh "5500" "10" "0" &
# CUDA_VISIBLE_DEVICES=2 nohup ./metrics_domain_adaptation/scripts/05-adapt_mqm/12b-scratch_datasize_eval_1ep.sh "5500" "12" "0" &
# CUDA_VISIBLE_DEVICES=3 nohup ./metrics_domain_adaptation/scripts/05-adapt_mqm/12b-scratch_datasize_eval_1ep.sh "5500" "14" "0" &
