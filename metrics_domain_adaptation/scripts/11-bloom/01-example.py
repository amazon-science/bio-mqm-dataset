
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

import json
import boto3
from botocore.config import Config


def query_endpoint(data, endpoint_name):
    my_config = Config(
        region_name='us-west-2',
    )
    client = boto3.client("runtime.sagemaker", config=my_config)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(data).encode('utf-8')
    )
    return response


def parse_response(query_response):
    model_predictions = json.loads(query_response["Body"].read())
    return model_predictions


ENDPOINT = "jumpstart-console-infer-huggingface-tex-2023-08-30-22-31-11-036"
PROMPT_REF = 'Score the following machine translation from {source_lang} to {target_lang} with respect to the human reference on a continuous scale from 0 to 100 that starts with "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{source_lang} source: "{src}"\n{target_lang} human reference: "{ref}"\n{target_lang} machine translation: "{tgt}"\nScore (0-100):\n'


def main():
    query_response = query_endpoint(
        {
            "text_inputs": PROMPT_REF.format(
                source_lang="German", target_lang="English",
                src="Ich esse gerne Pizza.",
                tgt="I like to eat pizza.",
                ref="I like to eat pizza."
            ),
            "max_length": 1,
            # "num_return_sequences": 1,
            # "top_k": 250,
            # "top_p": 0.95,
            # "do_sample": True,
            # "num_beams": 5,
        },
        endpoint_name=ENDPOINT
    )
    generated_text = parse_response(query_response)
    print(generated_text)


if __name__ == "__main__":
    main()
