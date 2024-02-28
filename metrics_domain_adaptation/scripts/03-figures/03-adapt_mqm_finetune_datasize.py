
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
# Performance of Bio finetuned models.
#

import collections
import matplotlib.pyplot as plt
import numpy as np
import metrics_domain_adaptation.fig_utils as fig_utils
from metrics_domain_adaptation import utils
import json
import re
import argparse

RE_SIGNATURE = re.compile(r".*count(.+)_seed(\d+)/.*")

args = argparse.ArgumentParser()
args.add_argument("--qe", action="store_true")
args = args.parse_args()

data = [
    json.loads(x)
    for x in open(f"{utils.ROOT}/computed/metrics_finetune_datasize{'_qe' if args.qe else ''}.jsonl", "r")
]

data_groupped = collections.defaultdict(list)
for line in data:
    if "count" not in line["args"]["model_path"]:
        seed = 0
        count_size = 0
    else:
        match = RE_SIGNATURE.match(line["args"]["model_path"])
        seed = int(match.group(2))
        count_size = int(
            match.group(1)
            .replace("5500", "6000")
            .replace("DA", "-1")
        )

    data_groupped[(count_size, seed)].append(line)

# collapse multiple seeds together
count_to_perf = collections.defaultdict(list)
for (count_size, seed), lines in data_groupped.items():
    tau_bio = np.average([x["tau"] for x in lines if x["domain"] == "bio"])
    tau_gen = np.average([x["tau"] for x in lines if x["domain"] == "general"])
    count_to_perf[count_size].append((tau_bio, tau_gen))

data_clean = []
for count_size, taus in count_to_perf.items():
    taus_bio = [tau_bio for tau_bio, tau_gen in taus]
    taus_gen = [tau_gen for tau_bio, tau_gen in taus]
    data_clean.append((
        count_size,
        utils.get_mean_inta_intb(taus_bio),
        utils.get_mean_inta_intb(taus_gen),
    ))

plt.figure(figsize=(3.5, 2))

# sort by count size
data_clean.sort(key=lambda x: x[0])

best_avg = 0
xticks = []

X_OFFSET = 0.35
BAR_KWARGS = dict(
    width=X_OFFSET,
    linewidth=1,
    edgecolor="black",
)

for line_i, (count_size, pack_bio, pack_gen) in enumerate(data_clean):
    tau_bio, inta_bio, intb_bio = pack_bio
    tau_gen, inta_gen, intb_gen = pack_gen

    # print DA
    if count_size == -1:
        print({"bio": tau_bio, "general": tau_gen, "k": -1, "count": -1})
    elif count_size == 0:
        print({"bio": tau_bio, "general": tau_gen, "k": 0, "count": 0})

    # separate DA and DA+MQM from the rest
    if line_i <= 1:
        left_side = True
        line_i -= 0.5
    else:
        left_side = False

    xticks.append((line_i, count_size))

    # precise values
    plt.text(
        x=line_i-X_OFFSET*0.4, y=0.205 if args.qe else 0.272,
        s=f"{tau_gen:.3f}".replace("0.", "."),
        ha="center", va="bottom", rotation=90,
        fontsize=7, color="white"
    )
    plt.text(
        x=line_i+X_OFFSET*0.6, y=0.205 if args.qe else 0.272,
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

    best_avg = max(best_avg, (tau_bio+tau_gen)/2)

print(f"best avg {best_avg:.3f}")

# plot "legend"
plt.text(
    x=0.75-0.26, y=0.95,
    s="Test:WMT",
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

plt.ylabel(r"Kendall's $\tau$", labelpad=-10)
plt.xlabel("Fine-tuning Bio$_{\leq s}$ data size",)
if args.qe:
    plt.ylim(0.2, 0.30)
    plt.yticks(
        np.linspace(0.2, 0.3, 5),
        [0.20, "", "", "", 0.30],
    )
else:
    plt.ylim(0.27, 0.345)
    plt.yticks(
        np.linspace(0.28, 0.34, 5),
        [0.28, "", "", "", 0.34],
    )
# hack to move the label
ax = plt.gca()
ax.xaxis.set_label_coords(0.6, -plt.ylim()[0]+0.01)


ax.spines["bottom"].set_position(('data', plt.ylim()[0]-0.004))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# bottom clouds
plt.scatter(
    np.linspace(-1, 8.5, 23),
    [plt.ylim()[0]-0.0025]*23,
    s=150,
    marker="o",
    edgecolor="#000",
    color="white",
    linewidth=0.7,
    linestyle="-"
)

XTICK_OVERRIDE = {
    -1: "\\text{\\footnotesize DA}",
    0: "\\text{\\footnotesize DA+}" + "\n\\text{\\footnotesize MQM}",
}

plt.xticks(
    [i for i, x in xticks],
    [
        XTICK_OVERRIDE[x] if x in XTICK_OVERRIDE else
        str(x).replace("000", "k")
        for i, x in xticks
    ]
)

plt.tight_layout(pad=0.1)
fig_utils.save(f"adapt_mqm_finetune_datasize{'_qe' if args.qe else ''}")
plt.show()
