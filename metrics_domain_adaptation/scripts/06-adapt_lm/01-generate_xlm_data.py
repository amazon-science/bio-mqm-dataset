
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
# Generate TLM, MLM and XLM (TLM+MLM) data for XLMR finetuning.
#

import random
import json
import os
import csv
import tqdm
from metrics_domain_adaptation import utils

EXP_DIR = f"{utils.ROOT}/data/experiments/lm/"

os.makedirs(EXP_DIR, exist_ok=True)

data_train_paralell = []
data_train_mono = []
for domain, count_i in [("general", 0), ("bio", 1)]:
    data_train_all = []

    random.seed(0)

    for lang1, count in tqdm.tqdm([
        ("en", (800_000, 300_000)),
        ("de", (300_000, 100_000)),
        ("ru", (300_000, 100_000)),
        ("zh", (300_000, 100_000)),
    ]):
        count = count[count_i]
        data_mono = [
            json.loads(x)
            for x in open(f"{utils.ROOT}/data/mono/{domain}/{lang1}.jsonl", "r")
        ]
        data_mono = [
            x[lang1]
            for x in random.sample(data_mono, k=count)
        ]
        data_train_all += data_mono
        data_train_mono += data_mono

    for langs, count in tqdm.tqdm([
        ("en-de", (300_000, 25_000)),
        ("en-ru", (300_000, 25_000)),
        ("en-zh", (300_000, 25_000)),
    ]):
        count = count[count_i]
        lang1, lang2 = langs.split("-")
        data_parallel = [
            json.loads(x)
            for x in open(f"{utils.ROOT}/data/parallel/{domain}/{langs}.jsonl", "r")
        ]
        data_parallel = [
            # randomly flip language direction
            x[lang1] + "[/s]" + x[lang2]
            if random.choice([True, False]) else
            x[lang2] + "[/s]" + x[lang1]
            for x in random.sample(data_parallel, k=min(count, len(data_parallel)))
        ]
        data_train_all += data_parallel
        data_train_paralell += data_parallel

    # save TLM data
    random.Random(0).shuffle(data_train_paralell)
    with open(f"{EXP_DIR}/{domain}_tlm_train.txt", "w") as f:
        f.write("\n".join(data_train_paralell))

    # save MLM data
    random.Random(0).shuffle(data_train_mono)
    with open(f"{EXP_DIR}/{domain}_mlm_train.txt", "w") as f:
        f.write("\n".join(data_train_mono))

    random.Random(0).shuffle(data_train_all)
    with open(f"{EXP_DIR}/{domain}_xlm_train.txt", "w") as f:
        f.write("\n".join(data_train_all))

    print(len(data_train_all), "training sentences for", domain)

    data_test = list(csv.DictReader(
        open(f"{utils.ROOT}/data/mqm/{domain}/test/all.csv", "r")
    ))
    data_test_tlm = [x["src"] + "[/s]" + x["mt"] for x in data_test]
    data_test_mlm = (
        [x["src"] for x in data_test] + [x["mt"] for x in data_test]
    )
    data_test_all = data_test_tlm + data_test_mlm

    with open(f"{EXP_DIR}/{domain}_xlm_test_mqm.txt", "w") as f:
        f.write("\n".join(data_test_all))

    with open(f"{EXP_DIR}/{domain}_mlm_test_mqm.txt", "w") as f:
        f.write("\n".join(data_test_mlm))

    with open(f"{EXP_DIR}/{domain}_tlm_test_mqm.txt", "w") as f:
        f.write("\n".join(data_test_tlm))

    print(len(data_test_all), "test sentences for", domain)
