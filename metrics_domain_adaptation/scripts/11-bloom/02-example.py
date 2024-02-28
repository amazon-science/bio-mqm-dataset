
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

import json
from sagemaker import ModelPackage
from sagemaker import get_execution_role
import sagemaker as sage
import boto3
import ai21

sagemaker_session = sage.Session()
runtime_sm_client = boto3.client("runtime.sagemaker")
ENDPOINT_NAME_LONG = "jumpstart-console-infer-huggingface-tex-2023-08-30-22-31-11-036"

response = ai21.Completion.execute(
    sm_endpoint=ENDPOINT_NAME_LONG,
    text_inputs="To be, or",
    maxTokens=4,
    temperature=0,
    numResults=1
)

print(response['completions'][0]['data']['text'])
