
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
# Generate subsampled DA data.
#

import csv
import os
import random
from metrics_domain_adaptation import utils

os.makedirs(f"{utils.ROOT}/data/experiments/da_culprit/", exist_ok=True)

data_orig = list(csv.DictReader(
    open(f"{utils.ROOT}/data/da/general/train.csv", "r"))
)


def generate_culprit_subset(proportion, trimode):
    if trimode == "none":
        data_local = data_orig
    elif trimode == "auth":
        data_local = [
            x for x in data_orig
            if x["lp"] in {"en-de", "en-ru", "zh-en"}
        ]
    elif trimode == "both":
        data_local = [
            x for x in data_orig
            if x["lp"] in {
                "en-de", "en-ru", "zh-en",
                "de-en", "ru-en", "en-zh",
            }
        ]
    else:
        raise Exception("Unknown trimode")

    random.seed(0)
    data_new_size = max(
        10,
        int(len(data_local)*proportion)
    )
    data_new = random.sample(data_local, k=data_new_size)

    trimode_fname = f"train_trimode_{trimode}_{proportion*100:02.0f}p"
    print(trimode_fname, len(data_new))

    with open(f"{utils.ROOT}/data/experiments/da_culprit/{trimode_fname}.csv", "w") as f:
        f.write("lp,src,mt,ref,score,system,annotators,domain\n")
        f = csv.writer(f)
        f.writerows([
            (
                line["lp"], line["src"], line["mt"],
                line["ref"], line["score"], "", "", line["domain"],
            )
            for line in data_new
        ])


generate_culprit_subset(0.00, "none")
generate_culprit_subset(0.10, "none")
generate_culprit_subset(0.15, "none")
generate_culprit_subset(0.20, "none")
generate_culprit_subset(0.25, "none")
generate_culprit_subset(0.50, "none")
generate_culprit_subset(0.75, "none")
generate_culprit_subset(1.00, "none")

generate_culprit_subset(0.25, "auth")
generate_culprit_subset(0.50, "auth")
generate_culprit_subset(0.75, "auth")
generate_culprit_subset(1.00, "auth")

generate_culprit_subset(0.25, "both")
generate_culprit_subset(0.50, "both")
generate_culprit_subset(0.75, "both")
generate_culprit_subset(1.00, "both")
