
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
# Connection between XLMR Perplexity and Tau.
#

import json
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import metrics_domain_adaptation.fig_utils as fig_utils
from metrics_domain_adaptation import utils
from matplotlib.path import Path
import argparse

args = argparse.ArgumentParser()
args.add_argument("-m", "--mode", default="src")
args = args.parse_args()

data = json.load(
    open(f"{utils.ROOT}/computed/prism_finetuning_precomputed.json", "r")
)

MODEL_TO_COLOR = {
    "base": "#ccc",
    "bio": "#ccc",
    "general": "#ccc",
}
ARCH_TO_SHAPE = {
    "OPUS": ".",
    "NLLB-600M": "^",
    "NLLB-1.3B": "s",
    "NLLB-3.3B": "p",
}
MODEL_TO_HATCH = {
    "base": "",
    "bio": "....",
    "general": "....",
}
MODEL_TO_NAME = {
    "base": "Base",
    "bio": "Finetuned",
    "general": "Finetuned",
}

plt.figure(figsize=(3, 1.7))
mpl.rcParams['hatch.linewidth'] = 0.6

xticks = set()

for arch in ["NLLB-600M", "NLLB-1.3B", "NLLB-3.3B"]:
    for eval_domain in ["bio"]:
        line_to_plot = []
        for ft_domain in ["base", "bio"]:
            if f"{arch}|{ft_domain}|{eval_domain}|bleu" not in data:
                continue
            if f"{arch}|{ft_domain}|{eval_domain}|prism2-{args.mode}" not in data:
                continue

            bleu = data[f"{arch}|{ft_domain}|{eval_domain}|bleu"]
            tau = data[f"{arch}|{ft_domain}|{eval_domain}|prism2-{args.mode}"]

            plt.scatter(
                [bleu], [tau], s=70 if ARCH_TO_SHAPE[arch] == "s" else 110,
                color=MODEL_TO_COLOR[ft_domain],
                hatch=MODEL_TO_HATCH[ft_domain],
                marker=ARCH_TO_SHAPE[arch],
                edgecolor="black",
            )

            # plot tau next to the marker
            plt.text(
                x=bleu+(
                    -0.95
                    if arch == "NLLB-1.3B" else
                    +0.95
                    if arch == "NLLB-3.3B" else
                    0
                ),
                y=tau+(
                    -0.003
                    if arch == "NLLB-600M" else
                    0
                ),
                s=f"{tau:.3f}",
                ha="center", va="center",
                color="#666", fontsize=7
            )

            line_to_plot.append((bleu, tau))
            xticks.add(bleu)

        if len(line_to_plot) < 2:
            # 3.3B variant
            plt.text(
                x=line_to_plot[0][0]+0.3,
                y=line_to_plot[0][1]+0.001,
                fontsize=6.5,
                s="3.3B"
            )
            continue

        dx = line_to_plot[0][0]-line_to_plot[1][0]
        dy = line_to_plot[0][1]-line_to_plot[1][1]
        plt.annotate(
            text="",
            xy=(line_to_plot[0]),
            xytext=(line_to_plot[1]),
            arrowprops=dict(
                arrowstyle="<|-",
                linestyle="-",
                color=fig_utils.COLORS[0] if dy < 0 else fig_utils.COLORS[1],
                shrinkA=6,
                shrinkB=6,
            )
        )
        plt.text(
            x=(line_to_plot[0][0]+line_to_plot[1][0])/2,
            y=(line_to_plot[0][1]+line_to_plot[1][1])/2,
            s="FT:Bio\n"+arch.replace("NLLB-", ""),
            ha="center", va="center",
            fontsize=7, linespacing=1.5,
            # somehow this angle didn't work np.rad2deg(np.arctan2(dy, dx))
            rotation=35 if arch == "NLLB-600M" else 48
        )


xticks = sorted(xticks)
plt.xticks(
    xticks,
    [f"{x:.0f}" for x in xticks]
)

plt.xlim(38, 49)
if args.mode == "src":
    plt.ylim(0.21, 0.234)
    plt.yticks(
        np.linspace(0.21, 0.234, 5),
        [0.21, "", "", "", 0.23]
    )
elif args.mode == "ref":
    plt.ylim(0.147, None)
    pass

# labels in subplots
plt.annotate(
    text="Test:Bio",
    xy=(0.05, 0.8) if args.mode == "src" else (0.05, 0.2),
    xycoords="axes fraction",
    ha="left",
    color="#666",
    fontsize=8
)

model_handles = {}
for model in ["base", "bio", "general"]:
    handle = plt.scatter(
        [0], [0], s=80,
        color=MODEL_TO_COLOR[model],
        hatch=MODEL_TO_HATCH[model],
        marker=Path([[-1, 1], [-1, -1], [1, -1]]),
        edgecolor="black",
        label=MODEL_TO_NAME[model],
        linewidth=0
    )
    model_handles[model] = handle

plt.ylabel(r"Kendall's $\tau$", labelpad=-10)
plt.suptitle(
    "BLEU",
    y=0.06, fontsize=9,
)


ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_position(('data', plt.ylim()[0]-0.0015))

plt.tight_layout(
    pad=0.1,
    rect=(0, 0.06, 1, 1.04)
)
plt.subplots_adjust(wspace=0.07, right=0.98)
fig_utils.save("prism_mts_"+args.mode)
plt.show()
