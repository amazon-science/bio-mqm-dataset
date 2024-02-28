
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

import matplotlib as mpl
import json
import collections
import numpy as np
import re
import scipy
import matplotlib.pyplot as plt
import metrics_domain_adaptation.fig_utils as fig_utils
from metrics_domain_adaptation import utils


def model_name_to_simple(txt):
    txt = re.sub(
        r"[\s;]+lr_scheduler.*",
        "", txt
    )
    txt = txt.split("/")[-1]
    txt = txt.replace("xlm-roberta-large", "no-ft")
    return txt


def normalize_domain_name(name):
    if name == "general":
        return "WMT"
    elif name == "bio":
        return "Bio"
    return name


data_ppl = [
    json.loads(x)
    for x in open(f"{utils.ROOT}/computed/xlmr_large_ppl.jsonl", "r")
]
data_ppl_dict = {}
FT_MODELS = set()
for line in data_ppl:
    line["model"] = model_name_to_simple(line["model"])
    data_ppl_dict[(
        line["model"],
        line["domain"],
        line["xlm_type"]
    )] = line["ppl"]
    FT_MODELS.add(line["model"])
FT_MODELS = sorted(list(FT_MODELS))

data_mqm_agg = collections.defaultdict(list)
data_mqm_bertscore = [
    json.loads(x)
    for x in open(f"{utils.ROOT}/computed/metrics_adaptlm_bertscore.jsonl", "r")
]
for line in data_mqm_bertscore:
    # skip ablation versions
    if "MLM" in line["args"]["model_path"] or "TLM" in line["args"]["model_path"]:
        continue
    line["model"] = model_name_to_simple(line["args"]["model_path"])
    data_mqm_agg[(
        line["model"], "bertscore", line["domain"]
    )].append(line["tau"])
data_mqm_comet = [
    json.loads(x)
    for x in open(f"{utils.ROOT}/computed/metrics_adaptlm_comet.jsonl", "r")
]
for line in data_mqm_comet:
    if "bioxlmr" in line["args"]["model_path"]:
        line["model"] = "bio"
    elif "generalxlmr" in line["args"]["model_path"]:
        line["model"] = "wmt"
    else:
        line["model"] = "no-ft"
    data_mqm_agg[(
        line["model"], "comet", line["domain"]
    )].append(line["tau"])

MODEL_TO_COLOR = {
    "no-ft": "#ccc",
    "bio": "#ccc",
    "wmt": "#ccc",
}
MODEL_TO_HATCH = {
    "no-ft": "",
    "bio": "....",
    "wmt": "....",
}
ARCH_TO_NAME = {
    "bertscore": "BERTScore",
    "comet": "COMET",
}
ARCH_TO_SHAPE = {
    "bertscore": "o",
    "comet": "s",
}

arch_all_points = collections.defaultdict(list)

plt.figure(figsize=(3.2, 2))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
mpl.rcParams['hatch.linewidth'] = 0.6

DOMAIN_TO_AX = {
    "bio": ax2,
    "general": ax1,
}


for domain in ["general", "bio"]:
    xticks = set()
    for arch in ["bertscore", "comet"]:
        line_to_plot = []
        for model in ["no-ft", "bio"]:
            ax = DOMAIN_TO_AX[domain]
            # average across languages
            if (model, arch, domain) not in data_mqm_agg:
                continue
            tau = np.average(data_mqm_agg[(model, arch, domain)])
            # average across mlm/tlm
            ppl = (
                data_ppl_dict[model, domain, "mlm"] +
                data_ppl_dict[model, domain, "tlm"]
            ) / 2

            arch_all_points[arch].append((ppl, tau))

            ax.scatter(
                [ppl], [tau], s=80,
                color=MODEL_TO_COLOR[model],
                hatch=MODEL_TO_HATCH[model],
                marker=ARCH_TO_SHAPE[arch],
                edgecolor="black",
            )
            line_to_plot.append((ppl, tau))

            # plot tau next to the marker
            ax.text(
                x=ppl,
                y=tau+(
                    -0.017
                    if arch == "comet" and domain == "general" else
                    0.013
                ),
                s=f"{tau:.3f}",
                ha="center", va="center",
                color="#666", fontsize=7
            )

            xticks.add(ppl)

        dx = line_to_plot[0][0]-line_to_plot[1][0]
        dy = line_to_plot[0][1]-line_to_plot[1][1]
        DOMAIN_TO_AX[domain].annotate(
            text="",
            xy=(line_to_plot[0]),
            xytext=(line_to_plot[1]),
            arrowprops=dict(
                arrowstyle="<|-",
                linestyle="-",
                color=fig_utils.COLORS[0] if dy < 0 else fig_utils.COLORS[1],
                shrinkA=4,
                shrinkB=5,
            )
        )
        DOMAIN_TO_AX[domain].text(
            x=(line_to_plot[0][0]+line_to_plot[1][0])/2,
            y=(line_to_plot[0][1]+line_to_plot[1][1])/2-0.001,
            s="FT:"+normalize_domain_name(domain)+"\n"+ARCH_TO_NAME[arch],
            ha="center", va="center",
            fontsize=7, linespacing=1.5,
            rotation=-(np.arctan2(dy, dx) / np.pi*2 * 360)
        )

    # xticks for each plot
    xticks = sorted(xticks)
    DOMAIN_TO_AX[domain].set_xticks(
        xticks,
        [f"{x:.1f}" for x in xticks]
    )

for arch in ["bertscore", "comet"]:
    pearson = scipy.stats.pearsonr(*list(zip(*arch_all_points[arch])))
    spearman = scipy.stats.spearmanr(*list(zip(*arch_all_points[arch])))
    print(arch, pearson, spearman)


# boundaries
ax1.set_yticks(np.linspace(0.21, 0.34, 5))
ax1.set_yticklabels([0.21, "", "", "", 0.34])
ax2.set_yticks(np.linspace(0.21, 0.34, 5))
ax2.set_yticklabels(["", "", "", "", ""])
ax2.set_xlim(2.55, 2.10)
ax1.set_xlim(2.82, 2.35)
ax2.set_ylim(0.21-0.005, 0.34+0.005)
ax1.set_ylim(0.21-0.005, 0.34+0.005)


ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_position(('data', plt.ylim()[0]-0.008))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_position(('data', plt.ylim()[0]-0.008))

# labels in subplots
ax2.annotate(
    text="Test:Bio",
    xy=(0.5, 1.06),
    xycoords="axes fraction",
    ha="center",
    color="#666",
)
ax1.annotate(
    text="Test:WMT",
    xy=(0.5, 1.06),
    xycoords="axes fraction",
    ha="center",
    color="#666",
)

ax1.set_ylabel(r"Kendall's $\tau$ {\tiny ($\uparrow$)}", labelpad=-10)
plt.suptitle(
    r"XLMR perplexity {\tiny ($\downarrow$)}",
    y=0.06, x=0.55, fontsize=9,
)
plt.tight_layout(pad=0.1, rect=(0, 0.05, 1, 1.05))
plt.subplots_adjust(wspace=0.07, right=0.98)
fig_utils.save("bertscore_ppl")
plt.show()
