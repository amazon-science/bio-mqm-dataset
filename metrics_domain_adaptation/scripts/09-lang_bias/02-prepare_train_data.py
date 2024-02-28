
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
# Prepare training data for MQM models trained without specific language.
#

import os
import copy
from metrics_domain_adaptation import utils
import csv

os.makedirs(f"{utils.ROOT}/data/experiments/lang_bias/", exist_ok=True)

LANGS = utils.LANGS
LANGS_UNIQUE = [
    "br-en", "de-en", "es-en", "fr-en", "ru-en", "zh-en",
]

data = []
for langs in LANGS:
    data_local = utils.load_data("mqm", "bio", langs, split="dev")
    for line in data_local:
        line["langs"] = langs
    data += data_local


def filter_out_data_and_save(no_langs: set):
    # deep copy to make sure
    data_local = copy.deepcopy(data)
    data_local = [x for x in data_local if x["langs"] not in no_langs]
    no_langs = [x.replace("-", "") for x in no_langs]
    no_langs.sort()
    path = f"{utils.ROOT}/data/experiments/lang_bias/no_{'_'.join(no_langs)}.csv"
    with open(path, "w") as f:
        f.write("lp,src,mt,ref,score,system,annotators,domain\n")
        f = csv.writer(f)
        f.writerows([
            (
                line["langs"], line["src"], line["tgt"],
                line["ref"], line["score"], "", "", "bio",
            )
            for line in data_local
        ])


# mode A: L - {l1, rev(l1)}
for langs1 in LANGS_UNIQUE:
    langs1_r = utils.rev_langs(langs1)
    filter_out_data_and_save({langs1, langs1_r})

# mode A & B: L - {l1} , L - {rev(l1)}
for langs1 in LANGS:
    filter_out_data_and_save({langs1})
