
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
# Infinite DA and DA+MQM training as opposed to just one epoch each.
#

import collections
import json
import numpy as np
import re
import argparse
from metrics_domain_adaptation import utils

RE_SIGNATURE = re.compile(r".*count(\d+)_up(\d+)_seed(\d+)/.*")

args = argparse.ArgumentParser()
args.add_argument("--count", type=int, default=5500)
args.add_argument("-em", "--epoch-mode", default="1")
args = args.parse_args()

data = [
    json.loads(x)
    for x
    in open(f"{utils.ROOT}/computed/metrics_dax_mqmx.jsonl", "r")
]
data = [x for x in data if x]


def get_da_xep_mqmx_name(x):
    # hotfix with old naming
    x = x.replace("daX", "da_xep")
    if x.endswith("xlm-roberta-large.ckpt"):
        return (0, 0)
    elif matches := re.match(r".*/(da_xep|da_xep)/.*/epoch=(\d+)-.*\.ckpt", x):
        return (int(matches.group(2))+1, 0)
    elif matches := re.match(r".*/from_da(\d+)/epoch=(\d+)-.*\.ckpt", x):
        return (int(matches.group(1)), int(matches.group(2))+1)
    else:
        raise Exception(f"Unable to parse {x}")


data_groupped = collections.defaultdict(list)
for line in data:
    da_epoch, mqm_epoch = get_da_xep_mqmx_name(line["args"]["model_path"])
    data_groupped[(da_epoch, mqm_epoch)].append(line)

data_clean = {}
for (da_epoch, mqm_epoch), lines in data_groupped.items():
    data_clean[(da_epoch, mqm_epoch)] = (
        np.average([abs(x["tau"]) for x in lines if x["domain"] == "general"]),
        np.average([abs(x["tau"]) for x in lines if x["domain"] == "bio"]),
    )

print("(da_epoch, mqm_epoch), (tau_gen, tau_bio)")
print("max bias", max(list(data_clean.items()), key=lambda x: x[1][0]-x[1][1]))
print("min bias", min(list(data_clean.items()), key=lambda x: x[1][0]-x[1][1]))

# take the second element, not the first, to get nice coloring
min_bio = sorted(list(data_clean.values()), key=lambda x: x[1])[1][1]
max_bio = sorted(list(data_clean.values()), key=lambda x: -x[1])[1][1]
min_gen = sorted(list(data_clean.values()), key=lambda x: x[0])[1][0]
max_gen = sorted(list(data_clean.values()), key=lambda x: -x[0])[1][0]


EPOCH_TO_I = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
    8: 4,
}

print("%" + "="*10 + " BEGIN TABLE " + "="*10)


def format_cell(value, norm_min, norm_max):
    value_color = (value-norm_min)/(norm_max-norm_min)*50
    value_color = max(value_color, 0)
    return (
        f"\\cellcolor{{black!{value_color:.2f}}} " +
        f"{value:.3f}"
    )


# first general
print(
    r"\textbf{\large Test:General}\hspace{-2cm} & \multicolumn{6}{c}{MQM epochs} \\")
print(
    r"\parbox[t]{2mm}{\multirow{6}{*}{\rotatebox[origin=c]{90}{DA epochs\hspace{5mm}}}}"
)
print(
    "", "",
    *[str(x) for x in EPOCH_TO_I.keys()],
    sep=" & ", end="\\\\\n"
)
print(r"\hline")
for da_epoch in list(EPOCH_TO_I.keys()):
    print("&" + str(da_epoch) + " & ", end="\n")
    line_out = []
    for mqm_epoch in EPOCH_TO_I.keys():
        line_out.append(format_cell(
            data_clean[(da_epoch, mqm_epoch)][0],
            min_gen, max_gen
        ))
        if mqm_epoch == 1 and da_epoch == 1:
            line_out[-1] += r" \zerowidthsymbol{\alpha}"
        if mqm_epoch == 8 and da_epoch == 8:
            line_out[-1] += r" \zerowidthsymbol{\beta}"
    print(" & ".join(line_out) + "\\\\")

print(r"\\[-0.5em]")
# second bio
print(
    r"\textbf{\large Test:Bio}\hspace{-1.5cm} & \multicolumn{6}{c}{MQM epochs} \\")
print(
    r"\parbox[t]{2mm}{\multirow{6}{*}{\rotatebox[origin=c]{90}{DA epochs\hspace{5mm}}}}"
)
print(
    "", "",
    *[str(x) for x in EPOCH_TO_I.keys()],
    sep=" & ", end="\\\\\n"
)
print(r"\hline")
for da_epoch in list(EPOCH_TO_I.keys()):
    print("&" + str(da_epoch) + " & ", end="\n")
    line_out = []
    for mqm_epoch in EPOCH_TO_I.keys():
        line_out.append(format_cell(
            data_clean[(da_epoch, mqm_epoch)][1],
            min_bio, max_bio
        ))
        if mqm_epoch == 1 and da_epoch == 1:
            line_out[-1] += r" \zerowidthsymbol{\alpha}"
        if mqm_epoch == 8 and da_epoch == 8:
            line_out[-1] += r" \zerowidthsymbol{\beta}"
    print(" & ".join(line_out))

print("%" + "="*10 + " END TABLE " + "="*10)
