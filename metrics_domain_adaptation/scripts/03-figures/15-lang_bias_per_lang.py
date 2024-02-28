
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
#

import json
import numpy as np
from metrics_domain_adaptation import utils

data = [
    json.loads(x)
    for x
    in open(f"{utils.ROOT}/computed/metrics_lang_bias.jsonl", "r")
]


def is_mqm_c(x):
    return (
        x["metric"] == "comet-ours" and
        (
            (
                "/no_" +
                utils.rev_langs(x["langs"]).replace("-", "") +
                "_" + x["langs"].replace("-", "") + "/"
            ) in x["args"]["model_path"]
            or
            (
                "/no_" +
                x["langs"].replace("-", "") +
                "_" + utils.rev_langs(x["langs"]).replace("-", "") + "/"
            ) in x["args"]["model_path"]
        )
    )


def is_mqm_b(x):
    return (
        x["metric"] == "comet-ours" and
        (
            "/no_" + utils.rev_langs(x["langs"]).replace("-", "") + "/"
        ) in x["args"]["model_path"]
    )


def is_mqm_a(x):
    return (
        x["metric"] == "comet-ours" and
        (
            "/no_" + x["langs"].replace("-", "") + "/"
        ) in x["args"]["model_path"]
    )


MODEL_TO_NAME = {
    "bleu": "BLEU",
    "ter": "TER",
    "chrf": "ChrF",
    "da": "DA",
    "mqm": "MQM",
    "mqm_mode_a": "MQM - $\\{l_1\\}$",
    "mqm_mode_b": "MQM - $\\{\\bar{l_1}\\}$",
    "mqm_mode_c": "MQM - $\\{l_1, \\bar{l_1}\\}$",
}
xticks_i = []
xticks_s = []
SCATTER_KWARGS_MQM = dict(
    s=40, alpha=0.7, linewidth=0, color="black"
)
SCATTER_KWARGS_SM = dict(
    s=40, alpha=0.7, linewidth=0,
)

# custom langs ordering detached from utils.LANGS
LANGS = [
    "br-en", "de-en", "en-de", "es-en", "en-es",
    "fr-en", "en-fr", "ru-en", "en-ru", "en-zh",
    "zh-en",
]

data_table = {}
for langs_i, langs in enumerate(LANGS):
    langs_rev_f = utils.rev_langs(langs).replace("-", "")
    langs_f = langs.replace("-", "")
    data_inlang = [x for x in data if x["langs"] == langs]
    sm_inlang = np.average([
        x["tau"] for x in data_inlang if x["metric"] in {"bleu", "ter", "chrf"}
    ])

    tau_bleu = [abs(x["tau"]) for x in data_inlang if x["metric"] == "bleu"][0]
    tau_chrf = [abs(x["tau"]) for x in data_inlang if x["metric"] == "chrf"][0]
    tau_ter = [abs(x["tau"]) for x in data_inlang if x["metric"] == "ter"][0]
    tau_mqmbioc = (
        [x["tau"] for x in data_inlang if is_mqm_c(x)][0]
        if langs != "br-en" else ""
    )
    tau_mqmbiob = (
        [x["tau"] for x in data_inlang if is_mqm_b(x)][0]
        if langs != "br-en" else ""
    )

    tau_mqmbioa = [x["tau"] for x in data_inlang if is_mqm_a(x)][0]
    tau_mqmbio = [
        x["tau"] for x in data_inlang
        if x["metric"] == "comet-ours" and f"/finetune_datasize/" in x["args"]["model_path"]
    ][0]
    tau_mqm = [
        x["tau"] for x in data_inlang
        if x["metric"] == "comet-ours" and f"/mqm/" in x["args"]["model_path"]
    ][0]

    data_table[langs] = {
        "mqm_noft": tau_mqm,
        "mqm_0": tau_mqmbio,
        "mqm_a": tau_mqmbioa,
        "mqm_b": tau_mqmbiob,
        "mqm_c": tau_mqmbioc,
        "sm": np.average([tau_bleu, tau_chrf, tau_ter]),
    }

print(r"\bf Model")
for langs in LANGS:
    print(
        " & "
        r"\bf",
        utils.pretty_langs(langs),
    )
print(r"& $\Delta$ \\ \midrule")

print(r"MQM")
for langs in LANGS:
    print(
        " &",
        f'{data_table[langs]["mqm_noft"]:.3f}',
        end=" ",
    )
# delta
print(
    " &",
    f'{np.average([data_table[langs]["mqm_0"]-data_table[langs]["mqm_noft"] for langs in utils.LANGS if langs != "br-en"]):.3f}'
)
print(r"\\")
print(r"MQM+Bio")
for langs in LANGS:
    print(
        " &",
        f'{data_table[langs]["mqm_0"]:.3f}',
        end=" ",
    )
# delta
print(
    " &",
    f'{np.average([data_table[langs]["mqm_0"]-data_table[langs]["mqm_0"] for langs in utils.LANGS if langs != "br-en"]):.3f}'
)
print(r"\\")
print(r"MQM+Bio $\setminus \{l_1{\rightarrow}l_2\}$")
for langs in LANGS:
    print(
        " &",
        f'{data_table[langs]["mqm_a"]:.3f}',
        end=" ",
    )
# delta
print(" & -",)
print(r"\\")
print(r"MQM+Bio $\setminus \{l_2{\rightarrow}l_1\}$")
for langs in LANGS:
    print(
        " &",
        f'{data_table[langs]["mqm_b"]:.3f}' if langs != "br-en" else "-",
        end=" ",
    )

# delta
print(
    " &",
    f'{np.average([data_table[langs]["mqm_0"]-data_table[langs]["mqm_b"] for langs in utils.LANGS if langs != "br-en"]):.3f}'
)
print(r"\\")
print(
    r"MQM+Bio $\setminus \{l_1{\rightarrow}l_2, l_2{\rightarrow}l_1\}$ \hspace{-4mm}"
)
for langs in LANGS:
    print(
        " &",
        f'{data_table[langs]["mqm_c"]:.3f}' if langs != "br-en" else "-",
        end=" ",
    )

# delta
print(
    " &",
    f'{np.average([data_table[langs]["mqm_0"]-data_table[langs]["mqm_c"] for langs in utils.LANGS if langs != "br-en"]):.3f}'
)
print(r"\\")
print(r"String-Matching")
for langs in LANGS:
    print(
        " &",
        f'{data_table[langs]["sm"]:.3f}',
        end=" ",
    )

# delta
print(
    " &",
    f'{np.average([data_table[langs]["mqm_0"]-data_table[langs]["sm"] for langs in utils.LANGS if langs != "br-en"]):.3f}'
)
print(r"\\")
