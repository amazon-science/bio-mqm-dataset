
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
# Per-language performance with projected performance based on SM.
# The table is not used in the paper but the stored jsonl file is neede for the figure.
#

import collections
import json
import numpy as np
from metrics_domain_adaptation import utils

data_mt = [
    utils.permissive_jsonl(x) for x
    in open(f"{utils.ROOT}/computed/mtperf_finetuned.jsonl", "r")
]
data_prism = [
    utils.permissive_jsonl(x) for x
    in open(f"{utils.ROOT}/computed/metrics_prism2_finetuned.jsonl", "r")
]


def path_to_model(model):
    model = model.lower()
    if "600m" in model:
        model_name = "NLLB-600M"
    elif "1.3b" in model:
        model_name = "NLLB-1.3B"
    elif "3.3b" in model:
        model_name = "NLLB-3.3B"
    elif "opus" in model:
        model_name = "OPUS"
    else:
        raise Exception(f"Unknown model name {model}")

    if "bio" in model:
        model_target = "bio"
    elif "general" in model:
        model_target = "general"
    elif "facebook" in model or "helsinki" in model:
        model_target = "base"
    else:
        raise Exception(f"Unknown model target {model}")

    return model_name, model_target


data_all = collections.defaultdict(dict)
for line in data_mt:
    model_name, model_target = path_to_model(line["model"])
    data_all[(
        model_name, model_target, line["domain"], "bleu"
    )][line["langs"]] = line["eval_bleu"]
    data_all[(
        model_name, model_target, line["domain"], "chrf"
    )][line["langs"]] = line["eval_chrf"]

for line in data_prism:
    model_name, model_target = path_to_model(line["args"]["model_name"])
    data_all[(
        model_name, model_target, line["domain"], line["metric"]
    )][line["langs"]] = line["tau"]

data_to_dump = {}


def retrieve(key):
    x = list(data_all[key].values())
    if len(x) > 0 and len(x) != 3:
        print(key, data_all[key])
    if not x:
        return "-"
    else:
        x = np.average(np.abs(x))
        data_to_dump["|".join(key)] = x
        if x < 1:
            return f"{x:.3f}"
        else:
            return f"{x:.1f}"


print(
    "",
    "Model",
    "Finetuning",
    "`BLEU` Bio",
    "`BLEU` Gen",
    "`-src` Bio",
    "`-src` Gen.",
    "`-ref` Bio",
    "`-ref` Gen",
    "",
    sep="|"
)
print("|" + "-|"*8)
for model_name in ["NLLB-600M", "NLLB-1.3B", "NLLB-3.3B"]:
    for model_target in ["base", "general", "bio"]:
        print(
            "",
            model_name,
            model_target,
            retrieve((model_name, model_target, "bio", "bleu")),
            retrieve((model_name, model_target, "general", "bleu")),
            retrieve((model_name, model_target, "bio", "prism2-src")),
            retrieve((model_name, model_target, "general", "prism2-src")),
            retrieve((model_name, model_target, "bio", "prism2-ref")),
            retrieve((model_name, model_target, "general", "prism2-ref")),
            "",
            sep="|"
        )

# dump data for 21-
json.dump(
    data_to_dump,
    open(f"{utils.ROOT}/computed/prism_finetuning_precomputed.json", "w")
)
