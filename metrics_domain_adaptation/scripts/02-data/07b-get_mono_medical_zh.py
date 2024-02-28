
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

import os
import json
from tqdm import tqdm
from datasets import load_dataset
import mosestokenizer
from metrics_domain_adaptation import utils

os.makedirs(f"{utils.ROOT}/data/mono/bio", exist_ok=True)

splitter = mosestokenizer.MosesSentenceSplitter(lang="zh")
dataset = load_dataset("shibing624/medical", "pretrain")

data = []
for line in tqdm(dataset["train"]):
    data += splitter([line["text"]])

print("total", len(data))
data = set(data)
print("unique", len(data))


with open(f"{utils.ROOT}/data/mono/bio/zh.jsonl", "w") as f:
    for line in data:
        f.write(json.dumps({"zh": line}, ensure_ascii=False)+'\n')
