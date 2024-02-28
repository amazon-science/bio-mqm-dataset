
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
# Run all the baseline metrics.
# 

OUTFILE="computed/metrics_base.jsonl"

echo -n "This will override $OUTFILE. Press [ENTER] or [Ctrl+C]"
read

rm -f $OUTFILE

for DOMAIN in "general" "bio"; do
for METRIC in "bleu" "chrf" "ter" "meteor" \
    "comet" "comet-qe" "comet-da" "cometinho" "cometinho-da" \
    "unite-mup" "bleurt" "bertscore-xlmr" "bartscore" \
    "prism-ref" "prism-src"; do
    echo "### Running $METRIC on $DOMAIN";
    ./metrics_domain_adaptation/run_metric.py \
        --metric $METRIC --domain $DOMAIN --langs all \
        | grep JSON! \
        | cut -c 6- \
        >> $OUTFILE
done;
done;

# remove malformed prism-src and prism-ref bytes
# sed 's/\x0//g' -i ${ADAPTATION_ROOT}/computed/metrics_base.jsonl
