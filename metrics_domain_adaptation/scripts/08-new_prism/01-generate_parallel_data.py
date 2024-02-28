
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
# Generate parallel data for NLLB finetuning.
#

import random
import json
import os
import tqdm
from metrics_domain_adaptation import utils

os.makedirs(f"{utils.ROOT}/data/experiments/nllb/", exist_ok=True)

for domain, count_i in [("general", 0), ("bio", 1)]:
    random.seed(0)

    for langs, count in tqdm.tqdm([
        ("en-de", (300_000, 30_000)),
        ("en-ru", (300_000, 30_000)),
        ("en-zh", (300_000, 30_000)),
    ]):
        count = count[count_i]
        lang1, lang2 = langs.split("-")
        data_parallel = [
            json.loads(x)
            for x in open(f"{utils.ROOT}/data/parallel/{domain}/{langs}.jsonl", "r")
        ][:count]

        # save parallel data
        random.Random(0).shuffle(data_parallel)
        langs_tmp = langs.replace("-", "").replace("enzh", "zhen")
        with open(f"{utils.ROOT}/data/experiments/nllb/{domain}_{langs_tmp}.jsonl", "w") as f:
            for line in data_parallel:
                f.write(json.dumps(line, ensure_ascii=False)+"\n")
