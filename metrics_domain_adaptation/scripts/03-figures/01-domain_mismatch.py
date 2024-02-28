
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
    for x in open(f"{utils.ROOT}/computed/metrics_base_rev_eng.jsonl", "r")
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
METRICS_N_PROMPT = ["gemba-dav003", "gemba-qe-dav003"]

METRICS = {
    # "String-Matching": METRICS_SM,
    "Surface-Form": METRICS_SM,
    "Pre-trained+Algorithm": METRICS_N_ALG,
    "Pre-tr.+Prompt": METRICS_N_PROMPT,
    "Pre-trained+Fine-tuned": METRICS_N_PROPER,
}
METRICS_FLAT = [x for l in METRICS.values() for x in l]

METRIC_NAMES = {
    "bleu": "BLEU",
    "chrf": "ChrF",
    "ter": "TER",

    "prism-ref": "\n{\\tiny REF}",
    "prism-src": "\n{\\tiny SRC}",
    "bertscore-xlmr": "BERTScore",

    "gemba-dav003": "\n{\\tiny DAV3}",
    "gemba-qe-dav003": "\n{\\tiny DAV3.QE}",
    "gemba-gpt4": "{\\tiny GPT4}",
    "gemba-qe-gpt4": "{\\tiny QE GPT4}",

    "comet": "\n{\\tiny MQM.21}",
    "comet-qe": "\n{\\tiny QE.21}",
    "comet-da": "\n{\\tiny DA.21}",
    "cometinho": "\n{\\tiny INHO.21}",
    "comet22-da": "\n{\\tiny DA.22}",
    # "unite-mup": "UniTE\n{\\tiny MUP}",
    "unite-mup": "UniTE",
    "bleurt": "\nBLEURT",

    # unused
    "comet-ours": "COMET {\\tiny W22$^*$}",
    "cometinho-da": "COMET{\\tiny{INHO}} {\\tiny E22-DA}",
    "bartscore": "BARTScore",
}


X_OFFSET = 0.35
SECTION_OFFSET_X = 0.5
BAR_KWARGS = dict(
    width=X_OFFSET,
    linewidth=1,
    edgecolor="black",
    bottom=0.072,
)

fig = plt.figure(figsize=(7, 2))
ax = plt.gca()
plt.ylim(0.07, 0.35)
plt.xlim(-0.7, 16)

ARROW_SPACING = 0.005
xticks = []
metric_i_global = 0
metric_i_global_prev = -SECTION_OFFSET_X


for metric_group_i, (metric_group_name, metric_group) in enumerate(METRICS.items()):
    for metric_i, metric in enumerate(metric_group):
        metric_i = metric_i_global
        xticks.append(metric_i)
        if (metric, "bio") not in data:
            continue

        tau_gen = data[(metric, "general")]
        tau_bio = data[(metric, "bio")]

        # precise values
        plt.text(
            x=metric_i-X_OFFSET*0.55, y=0.085,
            s=f"{tau_gen:.3f}".replace("0.", "."),
            ha="center", va="bottom", rotation=90,
            fontsize=7, color="white"
        )
        plt.text(
            x=metric_i+X_OFFSET*0.55, y=0.085 if tau_bio > 0.12 else 0.102,
            s=f"{tau_bio:.3f}".replace("0.", "."),
            ha="center", va="bottom", rotation=90,
            fontsize=7
        )

        dy = fig_utils.diff_to_arrow_dy(tau_bio-tau_gen)
        plt.annotate(
            text="",
            xy=(metric_i+X_OFFSET*0.7, 0.352+dy),
            xytext=(metric_i-X_OFFSET*0.7, 0.352-dy),
            color=fig_utils.COLORS[1] if dy < 0 else fig_utils.COLORS[0],
            clip_on=False,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-|>",
                linestyle="-",
                color=fig_utils.COLORS[1] if dy < 0 else fig_utils.COLORS[0],
            )
        )

        plt.bar(
            [metric_i-X_OFFSET/2],
            [tau_gen-BAR_KWARGS["bottom"]],
            color=fig_utils.COLORS_DOMAIN["general"],
            **BAR_KWARGS
        )
        plt.bar(
            [metric_i+X_OFFSET/2],
            [tau_bio-BAR_KWARGS["bottom"]],
            color=fig_utils.COLORS_DOMAIN["bio"],
            **BAR_KWARGS
        )
        # for each metric
        metric_i_global += 1

    # for each group
    group_x = (metric_i_global+metric_i_global_prev)/2-0.3
    plt.text(
        s=metric_group_name,
        x=group_x-2 if metric_group_i == 3 else group_x,
        y=0.369,
        fontsize=7, ha="center",
    )

    # bottom clouds
    clouds_x = np.arange(
        metric_i_global_prev+SECTION_OFFSET_X*0.5,
        metric_i_global-SECTION_OFFSET_X*0.9,
        0.38
    )
    plt.scatter(
        clouds_x,
        [plt.ylim()[0]-0.006]*len(clouds_x),
        s=150,
        marker="o",
        edgecolor="#000",
        color="white",
        linewidth=0.7,
        linestyle="-",
    )
    metric_i_global_prev = metric_i_global

    # skip last group for vertical lines
    if metric_group_i < len(METRICS) - 1:
        plt.axvline(
            x=metric_i_global-SECTION_OFFSET_X/2,
            color="#aaa",
            linewidth=1.2,
            zorder=-100,
            linestyle="-."
        )

    metric_i_global += SECTION_OFFSET_X


# plot "legend"
plt.text(
    x=0.85-0.06, y=1.15,
    s="Test:WMT",
    clip_on=False,
    transform=plt.gca().transAxes,
    fontsize=8,
    va="center", ha="left"
)
plt.text(
    x=0.85+0.08, y=1.15,
    s="Test:Bio",
    clip_on=False,
    transform=plt.gca().transAxes,
    fontsize=8,
    va="center", ha="left"
)
plt.scatter(
    x=[0.85-0.06-0.015], y=[1.15],
    marker="s",
    edgecolor="black",
    color=fig_utils.COLORS_DOMAIN["general"],
    s=50,
    clip_on=False,
    transform=plt.gca().transAxes,
)
plt.scatter(
    x=[0.85+0.08-0.015], y=[1.15],
    marker="s",
    edgecolor="black",
    color=fig_utils.COLORS_DOMAIN["bio"],
    s=50,
    clip_on=False,
    transform=plt.gca().transAxes,
)


# manual xticks because linespacing doesn't work with latex rendering
plt.xticks(xticks, [""]*len(xticks))
for x, m in zip(xticks, METRICS_FLAT):
    m = METRIC_NAMES[m] if m in METRIC_NAMES else m.upper()
    plt.text(
        x, y=0.048,
        s=m,
        ha="center",
        va="top",
        fontsize=8,
        linespacing=0.8,
    )

# additional manual xlabels
for text, x in [
    ("PRISM", 3+SECTION_OFFSET_X*2), ("GEMBA", 6 + SECTION_OFFSET_X*3),
    ("COMET", 8 + SECTION_OFFSET_X*4), ("COMET", 9.5 + + SECTION_OFFSET_X*4),
    ("COMET", 11 + + SECTION_OFFSET_X*4),
]:
    plt.text(
        # this is unfortunately both in data coordinates
        x=x, y=0.033,
        s=text,
        ha="center",
        fontsize=8,
    )


# make a gap at the bottom
ax.spines["bottom"].set_position(('data', plt.ylim()[0]-0.008))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

if args.langs is None:
    plt.ylabel(r"Kendall's $\tau$", labelpad=-10)
plt.yticks(
    np.linspace(0.1, 0.3, 5),
    [0.10, "", "", "", 0.30],
)

plt.tight_layout(pad=0.1)
fig_utils.save(
    "domain_mismatch" + (
        '_' + args.langs.replace('-', '')
        if args.langs is not None else ''
    )
)
plt.show()
