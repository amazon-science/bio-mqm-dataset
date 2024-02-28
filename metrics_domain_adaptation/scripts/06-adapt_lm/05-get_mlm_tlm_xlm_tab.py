
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
# Get BERTScore ablation table.
#

import collections
import json
import numpy as np
from metrics_domain_adaptation import utils

data_bertscore = [
    json.loads(x) for x in
    open(f"{utils.ROOT}/computed/metrics_adaptlm_bertscore.jsonl", "r")
]

data_ppl = [
    json.loads(x) for x in
    open(f"{utils.ROOT}/computed/xlmr_large_ppl.jsonl", "r")
]

data_ppl_clean = {}
for line in data_ppl:
    if line["model"] == "xlm-roberta-large":
        model_name = "-"
    elif "MLM" in line["model"]:
        model_name = "MLM"
    elif "TLM" in line["model"]:
        model_name = "TLM"
    elif "XLM" in line["model"]:
        model_name = "XLM"
    else:
        model_name = "XLM"

    if "general" in line["model"]:
        lm_target = "general"
    elif "bio" in line["model"]:
        lm_target = "bio"
    else:
        lm_target = "-"

    data_ppl_clean[
        (model_name, line["domain"], lm_target, line["xlm_type"])
    ] = line["ppl"]

data_bertscore_clean = collections.defaultdict(list)
for line in data_bertscore:
    if line["args"]["model_path"] == "xlm-roberta-large":
        model_name = "-"
    elif "MLM" in line["args"]["model_path"]:
        model_name = "MLM"
    elif "TLM" in line["args"]["model_path"]:
        model_name = "TLM"
    elif "XLM" in line["args"]["model_path"]:
        model_name = "XLM"
    else:
        model_name = "XLM"

    if "general" in line["args"]["model_path"]:
        lm_target = "general"
    elif "bio" in line["args"]["model_path"]:
        lm_target = "bio"
    else:
        lm_target = "-"

    data_bertscore_clean[
        (model_name, line["domain"], lm_target)
    ].append(line["tau"])

for model_name in ["-", "XLM", "MLM", "TLM"]:
    for lm_target in ["general", "bio"] if model_name != "-" else ["-"]:
        tau_gen = np.average(
            data_bertscore_clean[(model_name, "general", lm_target)]
        )
        tau_bio = np.average(
            data_bertscore_clean[(model_name, "bio", lm_target)]
        )
        ppl_mlm_gen = data_ppl_clean[(model_name, "general", lm_target, "mlm")]
        ppl_tlm_gen = data_ppl_clean[(model_name, "general", lm_target, "tlm")]
        ppl_mlm_bio = data_ppl_clean[(model_name, "bio", lm_target, "mlm")]
        ppl_tlm_bio = data_ppl_clean[(model_name, "bio", lm_target, "tlm")]

        print(
            model_name, lm_target,
            f"{tau_gen:.3f}",
            f"$^{{{ppl_mlm_gen:.2f}}}_{{{ppl_tlm_gen:.2f}}}$",
            f"{tau_bio:.3f}",
            f"$^{{{ppl_mlm_bio:.2f}}}_{{{ppl_tlm_bio:.2f}}}$",
            sep=" & ", end="\\\\\n"
        )
