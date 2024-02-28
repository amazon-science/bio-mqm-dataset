
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
# Performance of (General+Bio) mixed models for a particular Bio count.
#

import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import metrics_domain_adaptation.fig_utils as fig_utils
from metrics_domain_adaptation import utils
import re
import argparse

RE_SIGNATURE = re.compile(r".*count(\d+)_up(\d+)_seed(\d+)/.*")

args = argparse.ArgumentParser()
args.add_argument("--count", type=int, default=5500)
args.add_argument("-em", "--epoch-mode", default="1")
args = args.parse_args()

data = [
    json.loads(x)
    for x
    in open(f"{utils.ROOT}/computed/metrics_scratch_datasize_{args.epoch_mode}ep.jsonl", "r")
]
data = [x for x in data if x]

data_groupped = collections.defaultdict(lambda: collections.defaultdict(list))
for line in data:
    matches = RE_SIGNATURE.match(line["args"]["model_path"])
    count = int(matches.group(1))
    upsample = int(matches.group(2))
    seed = int(matches.group(3))
    if count == args.count and upsample != 0:
        # aggregate across seeds
        data_groupped[upsample][seed].append(line)
    if count == 0 or upsample == 0:
        data_groupped[0][seed].append(line)

data_clean = []
for upsample, upsample_dict in data_groupped.items():
    data_local = []
    for seed, lines in upsample_dict.items():
        data_local.append((
            np.average([x["tau"] for x in lines if x["domain"] == "general"]),
            np.average([x["tau"] for x in lines if x["domain"] == "bio"]),
        ))
    data_clean.append((
        upsample,
        utils.get_mean_inta_intb([x[0] for x in data_local]),
        utils.get_mean_inta_intb([x[1] for x in data_local]),
    ))

plt.figure(figsize=(3.5, 2))

# sort by count size
data_clean.sort(key=lambda x: x[0])

best_k = None

xticks = []
xtick_labels = []
X_OFFSET = 0.35
BAR_KWARGS = dict(
    width=X_OFFSET,
    linewidth=1,
    edgecolor="black",
)


def diff_to_dy(diff):
    diff /= 4
    diff = float(np.sign(diff))*max(0.005, abs(diff))
    return diff


for line_i, (upsample, pack_gen, pack_bio) in enumerate(data_clean):
    tau_bio, inta_bio, intb_bio = pack_bio
    tau_gen, inta_gen, intb_gen = pack_gen

    xticks.append(line_i)
    xtick_labels.append(str(upsample)+r"{\tiny$\times$}")

    # precise values
    plt.text(
        x=line_i-X_OFFSET*0.4, y=0.272,
        s=f"{tau_gen:.3f}".replace("0.", "."),
        ha="center", va="bottom", rotation=90,
        fontsize=7, color="white"
    )
    plt.text(
        x=line_i+X_OFFSET*0.6, y=0.272,
        s=f"{tau_bio:.3f}".replace("0.", "."),
        ha="center", va="bottom", rotation=90,
        fontsize=7
    )

    plt.bar(
        [line_i-X_OFFSET/2],
        [tau_gen],
        color=fig_utils.COLORS_DOMAIN["general"],
        **BAR_KWARGS
    )
    plt.bar(
        [line_i+X_OFFSET/2],
        [tau_bio],
        color=fig_utils.COLORS_DOMAIN["bio"],
        **BAR_KWARGS
    )

    if (
        best_k is None or
        best_k["bio"][0] + best_k["general"][0] < tau_bio + tau_gen
    ):
        best_k = {
            "bio": pack_bio,
            "general": pack_gen,
            "k": upsample,
            "count": args.count,
        }

# plot "legend"
plt.text(
    x=0.75-0.26, y=0.95,
    s="Test:General",
    clip_on=False,
    transform=plt.gca().transAxes,
    fontsize=8,
    va="center", ha="left"
)
plt.text(
    x=0.75+0.05, y=0.95,
    s="Test:Bio",
    clip_on=False,
    transform=plt.gca().transAxes,
    fontsize=8,
    va="center", ha="left"
)
plt.scatter(
    x=[0.75-0.26-0.04], y=[0.95],
    marker="s",
    edgecolor="black",
    color=fig_utils.COLORS_DOMAIN["general"],
    s=50,
    clip_on=False,
    transform=plt.gca().transAxes,
)
plt.scatter(
    x=[0.75+0.02], y=[0.95],
    marker="s",
    edgecolor="black",
    color=fig_utils.COLORS_DOMAIN["bio"],
    s=50,
    clip_on=False,
    transform=plt.gca().transAxes,
)


plt.ylim(0.27, 0.345)
plt.yticks(
    np.linspace(0.28, 0.34, 5),
    [0.28, "", "", "", 0.34],
)
# hack to move the label
ax = plt.gca()
ax.xaxis.set_label_coords(0.6, -plt.ylim()[0]+0.01)

# make a gap at the bottom
ax.spines["bottom"].set_position(('data', plt.ylim()[0]-0.004))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# bottom cloud
plt.scatter(
    np.linspace(-0.5, 9.5, 20),
    [plt.ylim()[0]-0.0025]*20,
    s=150,
    marker="o",
    edgecolor="#000",
    color="white",
    linewidth=0.7,
    linestyle="-"
)


count_name = str(args.count).replace("5500", "6000").replace("000", "k")

print("JSON!"+json.dumps(best_k))

plt.ylabel(r"Kendall's $\tau$", labelpad=-10)
plt.xlabel(r"General + $k\,{\times}$ Bio")
plt.xticks(xticks, xtick_labels)

plt.tight_layout(pad=0.1)
fig_utils.save(
    f"adapt_mqm_scratch_datasize_count{count_name}"
)
plt.show()
