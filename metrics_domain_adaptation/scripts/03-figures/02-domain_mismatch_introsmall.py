
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
# Introductory figure which overviews bias in different metrics.
#

import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import metrics_domain_adaptation.fig_utils as fig_utils
from metrics_domain_adaptation import utils
import argparse

args = argparse.ArgumentParser()
args.add_argument("--langs", default=None)
args = args.parse_args()

data_raw = [
    json.loads(x)
    for x in open(f"{utils.ROOT}/computed/metrics_base.jsonl", "r")
]
if args.langs is not None:
    data_raw = [
        x for x in data_raw if x["langs"] == args.langs
    ]
data = collections.defaultdict(list)

for line in data_raw:
    data[(line["metric"], line["domain"])].append(line["tau"])

data = {k: np.average(np.abs(v)) for k, v in data.items()}

METRICS_SM = ["bleu", "chrf", "ter"]
METRICS_N_PROPER = [
    "comet", "comet-qe", "comet-da", "cometinho",
    "comet22-da", "unite-mup", "bleurt"
]
METRICS_N_ALG = ["prism-ref", "prism-src", "bertscore-xlmr"]

METRICS = {
    "COMET\nBLEURT\nUniTE": METRICS_N_PROPER,
    "BERTScore\nPRISM": METRICS_N_ALG,
    "ChrF\nBLEU\nTER": METRICS_SM,
}

fig = plt.figure(figsize=(2, 1.3))
ax = plt.gca()

ANNOTATION_KWARGS = dict(
    va="center", ha="left",
    fontsize=8, color="#555",
    linespacing=1.2,
    arrowprops=dict(
        width=0.2,
        headwidth=3,
        headlength=2.5,
        color="#777",
    ),
)

X_OFFSET = 0.25
BAR_KWARGS = dict(
    width=X_OFFSET,
    linewidth=1.1,
    edgecolor="black",
)

for metric_group_i, (metric_group_name, metric_group) in enumerate(METRICS.items()):
    tau_gen_group = []
    tau_bio_group = []
    for metric in metric_group:
        tau_gen_group.append(data[(metric, "general")])
        tau_bio_group.append(data[(metric, "bio")])

    tau_gen = np.average(tau_gen_group)
    tau_bio = np.average(tau_bio_group)
    plt.bar(
        [metric_group_i-X_OFFSET/2],
        [tau_gen],
        color=fig_utils.COLORS_DOMAIN["general"],
        **BAR_KWARGS
    )
    plt.bar(
        [metric_group_i+X_OFFSET/2],
        [tau_bio],
        color=fig_utils.COLORS_DOMAIN["bio"],
        **BAR_KWARGS
    )

    plt.text(
        metric_group_i+0.01-X_OFFSET/2,
        y=0.01,
        ha="center", va="bottom",
        # s="General" if metric_group_i != 2 else "Gen.",
        s="WMT",
        rotation=90, fontsize=7,
        color="white",
    )
    plt.text(
        metric_group_i+0.01+X_OFFSET/2,
        y=0.01,
        ha="center", va="bottom",
        s="Bio",
        rotation=90, fontsize=7,
    )

plt.xlim(-0.5, 2.4)
plt.ylim(0.0, 0.3)
plt.yticks([0.0, 0.30])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# add some more slack at the top of the plot
plt.xticks(
    range(len(METRICS)),
    METRICS.keys(),
    fontsize=7,
)

plt.ylabel(
    r"Kendall's $\tau$",
    labelpad=-10,
    fontsize=7,
)

plt.tight_layout(pad=0.1)
fig_utils.save("domain_mismatch_small")
plt.show()
