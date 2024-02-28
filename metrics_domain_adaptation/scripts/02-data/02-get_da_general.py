
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
# Downloads publicly available DA data which are already zscored.
#

import collections
from datasets import load_dataset
from metrics_domain_adaptation import utils
import os
import csv
import json

os.makedirs(f"{utils.ROOT}/data/da/general", exist_ok=True)

dataset = load_dataset("RicardoRei/wmt-da-human-evaluation")
data_all = []
open_files = {}


def get_file(langs):
    if langs not in open_files:
        open_files[langs] = open(
            f"{utils.ROOT}/data/da/general/{langs}.jsonl",
            "w"
        )
    return open_files[langs]


for line in dataset["train"]:
    line_new = {
        "src": line["src"],
        "tgt": line["mt"],
        "ref": line["ref"],
        "score": line["score"],
        "domain": line["domain"],
        "year": line["year"],
        "langs": line["lp"],
    }
    data_all.append(line_new)
    get_file(line['lp']).write(json.dumps(line_new, ensure_ascii=False) + "\n")

data_train = []
data_test = []

for line in data_all:
    if line["year"] >= 2022:
        data_test.append(line)
    else:
        data_train.append(line)

with open(f"{utils.ROOT}/data/da/general/train.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,annotators,domain\n")
    f = csv.writer(f)
    f.writerows([
        (
            line["langs"], line["src"], line["tgt"],
            line["ref"], line["score"], "", "", line["domain"],
        )
        for line in data_train
    ])
with open(f"{utils.ROOT}/data/da/general/test.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,annotators,domain\n")
    f = csv.writer(f)
    f.writerows([
        (
            line["langs"], line["src"], line["tgt"],
            line["ref"], line["score"], "", "", line["domain"],
        )
        for line in data_test
    ])


def format_for_table(data):
    data = collections.Counter([x["langs"] for x in data])
    data = [(k, f"{v//1000:.0f}k") for k, v in data.items()]
    # sort by the other lang
    data.sort(key=lambda x: x[0].replace("en", "").replace("-", ""))
    for langs, v in data:
        lang1, lang2 = langs.split("-")
        lang1 = lang1.capitalize()
        lang2 = lang2.capitalize()
        print(f"{lang1}-{lang2} & {v} \\\\")


print("Test")
format_for_table(data_test)
print("Train")
format_for_table(data_train)
