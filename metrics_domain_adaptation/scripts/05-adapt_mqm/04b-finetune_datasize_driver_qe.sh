
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
# Launch finetuning training (referenceless).
# 

SEED=$1

for BIO_COUNT in 50 100 500 1000 2000 4000 5500; do
  echo "Launching $BIO_COUNT";
  ./metrics_domain_adaptation/scripts/05-adapt_mqm/03b-finetune_datasize_qe.sh $BIO_COUNT $SEED;
done