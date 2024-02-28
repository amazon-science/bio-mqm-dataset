
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
# Eval MQM which was trained on DA with subsampled data.
# 

for f in ${ADAPTATION_ROOT}/models/trained/mqm/main/checkpoints_culprit/trimode_auth_*/epoch=0-*.ckpt; do
    echo "Evaluating $f";
    ./metrics_domain_adaptation/run_metric.py --metric comet-ours --model-path "$f" | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_da_culprit_mqm.jsonl;
done

for f in ${ADAPTATION_ROOT}/models/trained/mqm/main/checkpoints_culprit/trimode_both_*/epoch=0-*.ckpt; do
    echo "Evaluating $f";
    ./metrics_domain_adaptation/run_metric.py --metric comet-ours --model-path "$f" | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_da_culprit_mqm.jsonl;
done

for f in ${ADAPTATION_ROOT}/models/trained/mqm/main/checkpoints_culprit/trimode_none_*/epoch=0-*.ckpt; do
    echo "Evaluating $f";
    ./metrics_domain_adaptation/run_metric.py --metric comet-ours --model-path "$f" | grep JSON | cut -c 6- >> ${ADAPTATION_ROOT}/computed/metrics_da_culprit_mqm.jsonl;
done