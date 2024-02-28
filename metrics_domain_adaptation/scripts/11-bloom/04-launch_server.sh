
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

cd ~/transformers-bloom-inference/

TOKENIZERS_PARALLELISM=false \
MODEL_NAME=bigscience/bloomz-mt \
MODEL_CLASS=AutoModelForCausalLM \
DEPLOYMENT_FRAMEWORK=hf_accelerate \
DTYPE=int8 \
MAX_INPUT_LENGTH=2048 \
MAX_BATCH_SIZE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nohup gunicorn -t 0 -w 1 -b 127.0.0.1:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s' > server.log &


# or interactivelly
# python3 -m inference_server.cli --model_name $MODEL_NAME --model_class AutoModelForCausalLM --dtype int8 --deployment_framework hf_accelerate --generate_kwargs '{"max_new_tokens": 100, "do_sample": true}'