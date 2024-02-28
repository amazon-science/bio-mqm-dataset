
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
# Split Bio MQM data and produce a table for LaTeX.
#

import random
from metrics_domain_adaptation import utils
import collections
import json
import csv
import os

os.makedirs(f"{utils.ROOT}/data/mqm/bio/dev", exist_ok=True)
os.makedirs(f"{utils.ROOT}/data/mqm/bio/test", exist_ok=True)

TEST_SIZE = {
    "br-en": 700,
    "en-de": 2600,
    "de-en": 2400,
    "en-es": 1100,
    "es-en": 1000,
    "en-ru": 800,
    "ru-en": 1300,
    "en-fr": 1200,
    "fr-en": 1100,
    "zh-en": 2800,
    "en-zh": 3800,
}

if os.path.exists("data/doc_id_splits.json"):
    print(
        "The splits file already exists (doc ids).\n"
        "If you're creating new data, make sure to "
        "`git add --force data/doc_id_splits.json`.\n"
        "[Enter] to continue, [Ctrl+C] to cancel.",
        end=""
    )
    input()


def print_format_line(langs, len_data_test, len_data_dev):
    len_data = len_data_test + len_data_dev
    lang1, lang2 = langs.split("-")
    lang1 = lang1.capitalize()
    lang2 = lang2.capitalize()
    # for LaTeX
    # ({{\\small {len_data_test/len_data*100:.0f}\\% }})
    print(
        f"{lang1}-{lang2} & "
        f"{len_data_test} & "
        f"{len_data_dev} & {len_data} & 0 \\\\"
    )


all_data_dev = []
all_data_test = []
doc_id_splits = {}

for langs in TEST_SIZE.keys():
    data = utils.load_data(kind="mqm", domain="bio", langs=langs, split="")

    # data is already zscored
    data_src = collections.defaultdict(list)
    for line in data:
        data_src[line["doc"]].append(line)

    data_test = []
    data_dev = []
    data_src = list(data_src.values())
    random.Random(0).shuffle(data_src)
    for group in data_src:
        if len(data_test) < TEST_SIZE[langs]:
            data_test += group
        else:
            data_dev += group

    doc_id_splits[langs] = {
        "test": sorted(list({x["doc"] for x in data_test}), key=lambda x: int(x.lstrip("doc"))),
        "dev": sorted(list({x["doc"] for x in data_dev}), key=lambda x: int(x.lstrip("doc"))),
    }

    with open(f"{utils.ROOT}/data/mqm/bio/dev/{langs}.jsonl", "w") as f:
        f.write("\n".join([
            json.dumps(line, ensure_ascii=False)
            for line in data_dev
        ]))
    with open(f"{utils.ROOT}/data/mqm/bio/test/{langs}.jsonl", "w") as f:
        f.write("\n".join([
            json.dumps(line, ensure_ascii=False)
            for line in data_test
        ]))
    print_format_line(langs, len(data_test), len(data_dev))

    all_data_dev += data_dev
    all_data_test += data_test

# manual JSON dumping with pretty indent
with open(f"data/doc_id_splits.json", "w") as f:
    f.write("{\n")
    for langs_i, langs in enumerate(TEST_SIZE.keys()):
        f.write(f'  "{langs}": {{\n')
        f.write(
            f'    "test": ["' + '", "'.join(doc_id_splits[langs]["test"]) + '"],\n')
        f.write(
            f'    "dev": ["' + '", "'.join(doc_id_splits[langs]["dev"]) + '"]\n')
        f.write(f'  }}{"," if langs_i != len(TEST_SIZE)-1 else ""}\n')
    f.write("}")

with open(f"{utils.ROOT}/data/mqm/bio/test/all.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,annotators,domain\n")
    f = csv.writer(f)
    f.writerows([
        ("", line["src"], line["tgt"],
         line["ref"], line["score"], "", "", "bio")
        for line in all_data_test
    ])
with open(f"{utils.ROOT}/data/mqm/bio/dev/all.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,annotators,domain\n")
    f = csv.writer(f)
    f.writerows([
        ("", line["src"], line["tgt"],
         line["ref"], line["score"], "", "", "bio")
        for line in all_data_dev
    ])

print(len(all_data_dev), len(all_data_test))
random.Random(0).shuffle(all_data_dev)
with open(f"{utils.ROOT}/data/mqm/bio/dev/pudding_train.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,annotators,domain\n")
    f = csv.writer(f)
    f.writerows([
        ("", line["src"], line["tgt"],
         line["ref"], line["score"], "", "", "bio")
        for line in all_data_dev[512:]
    ])
with open(f"{utils.ROOT}/data/mqm/bio/dev/pudding_eval.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,annotators,domain\n")
    f = csv.writer(f)
    f.writerows([
        ("", line["src"], line["tgt"],
         line["ref"], line["score"], "", "", "bio")
        for line in all_data_dev[:512]
    ])
