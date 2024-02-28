
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
# Distribution of model v true scores.
#

import collections
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import metrics_domain_adaptation.fig_utils as fig_utils
from metrics_domain_adaptation import utils
import re

RE_SIGNATURE = re.compile(r".*count(\d+)_up(\d+)_seed(\d+)/.*")

data_base = [
    json.loads(x)
    for x in open(f"{utils.ROOT}/computed/scores_base.jsonl", "r")
]
data_ft = [
    json.loads(x)
    for x in open(f"{utils.ROOT}/computed/scores_ft.jsonl", "r")
]
data_clean = collections.defaultdict(list)
for line in data_base:
    data_clean[("base", line["domain"])].append(line)
for line in data_ft:
    data_clean[("ft", line["domain"])].append(line)

fig = plt.figure(figsize=(2.8, 2.1))

LIMIT = -1.7

ax_i = 0
for model in ["base", "ft"]:
    for domain in ["general", "bio"]:
        ax_i += 1
        ax = plt.subplot(2, 2, ax_i)
        random.seed(0)

        data_local = random.sample(data_clean[(model, domain)], k=6000)
        data_local = [
            x for x in data_local
            if x["score"] >= LIMIT and x["model_score"] >= LIMIT
        ]
        data_local.sort(key=lambda x: x["score"])
        Xs = [x["score"] for x in data_local]
        Ys = [x["model_score"] for x in data_local]

        data_local_small = random.sample(data_local, k=5000)
        Xs_small = [x["score"] for x in data_local_small]
        Ys_small = [x["model_score"] for x in data_local_small]

        ax.scatter(
            x=Xs_small, y=Ys_small,
            marker=".",
            color="black",
            alpha=0.1,
            s=20,
            zorder=-100,
            edgecolor=None,
            linewidth=0,
        )

        Y_poly1deg = np.polyval(np.polyfit(Xs, Ys, 1), Xs)
        ax.plot(
            Xs, Y_poly1deg,
            color="#753",
            linestyle="-"
        )

        # set subplot attributes
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.tick_right()

        if domain == "general":
            ax.set_yticks([])
            ax.set_ylabel("Model:"+r"\textsc{" + model + "}", labelpad=1)

        if model == "base":
            ax.set_xticks([])
            ax.set_title("Test:"+domain.capitalize(), fontsize=8, pad=-5)

        ax.set_ylim(LIMIT, 1)
        ax.set_xlim(LIMIT, 1)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

plt.text(
    s="True score",
    x=0.48, y=0.01,
    ha="center",
    transform=fig.transFigure,
)
plt.text(
    s="Predicted",
    x=0.95, y=0.48,
    va="center",
    transform=fig.transFigure,
    rotation=90,
)
plt.tight_layout(pad=0.2, rect=[0, 0.03, 0.98, 1])
plt.subplots_adjust(wspace=0.1, hspace=0.1)
fig_utils.save(f"score_distribution")
plt.show()
