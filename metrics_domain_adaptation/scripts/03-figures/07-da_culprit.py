
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
# Results of subsampled DA training.
#

import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import metrics_domain_adaptation.fig_utils as fig_utils
from metrics_domain_adaptation import utils
import re
import argparse

RE_PARSER = re.compile(r".*trimode_([a-z]+)_(\d+)p.*")

args = argparse.ArgumentParser()
args.add_argument("--mode", default="mqm")
args = args.parse_args()

data_agg = collections.defaultdict(list)
data_raw = [
    json.loads(x)
    for x in open(f"{utils.ROOT}/computed/metrics_da_culprit_{args.mode}.jsonl", "r")
]

for line in data_raw:
    match = RE_PARSER.match(line["args"]["model_path"])
    strategy = match.group(1)
    prop = int(match.group(2))
    data_agg[(
        strategy, prop, line["domain"]
    )].append(line["tau"])
data_agg = {
    k: np.average(np.abs(v)) for k, v in data_agg.items()
}

for domain in ["bio", "general"]:
    data_agg[("auth", 0, domain)] = data_agg[("none", 0, domain)]
    data_agg[("both", 0, domain)] = data_agg[("none", 0, domain)]

fig = plt.figure(figsize=(2.8, 1.5))

STRATEGY_COLOR = {
    "none": "black",
    "auth": fig_utils.COLORS[2],
    "both": fig_utils.COLORS[4],
}
STRATEGY_TO_FULL_SIZE = {
    "none": 1172587,
    "auth": 361130,
    "both": 667609,
}

PROPS = [0, 10, 15, 20, 25, 50, 75, 100]
for strategy in ["none", "auth", "both"]:
    for domain in ["bio", "general"]:
        data_local = [
            (
                prop/100 * STRATEGY_TO_FULL_SIZE[strategy],
                data_agg[(strategy, prop, domain)]
            )
            for prop in PROPS
            if (strategy, prop, domain) in data_agg
        ]
        for point_prev, point_now in zip(data_local, data_local[1:]):
            plt.plot(
                [point_prev[0], point_now[0]],
                [point_prev[1], point_now[1]],
                linestyle="--" if domain == "bio" else "-",
                marker=".",
                color=STRATEGY_COLOR[strategy],
            )

XTICKS = [
    0,
    0.3*1e6,
    0.6*1e6,
    0.9*1e6,
    1.2*1e6,
]
plt.xticks(
    XTICKS,
    [f"{x/1_000_000:.1f}M" for x in XTICKS]
)

plt.ylabel(r"Kendall's $\tau$", labelpad=-10)
if args.mode == "mqm":
    plt.xlabel("DA {\\tiny (subset)} + MQM", fontsize=8, loc="center")
elif args.mode == "da":
    plt.xlabel("DA only", fontsize=8, loc="center")


plt.ylim(0.23, 0.331)
plt.yticks(
    ticks=np.linspace(0.23, 0.33, 5),
    labels=["0.23", "", "", "", "0.33"]
)

plt.text(
    x=0.93, y=0.8,
    s="Test:General",
    ha="right", va="center",
    color="#777",
    fontsize=8,
    transform=fig.transFigure
)
plt.text(
    x=0.93, y=0.4,
    s="Test:Bio",
    ha="right", va="center",
    color="#777",
    fontsize=8,
    transform=fig.transFigure
)

ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_position(('data', plt.ylim()[0]-0.008))


plt.tight_layout(
    pad=0.1,
)
fig_utils.save(f"da_culprit_{args.mode}")
plt.show()
