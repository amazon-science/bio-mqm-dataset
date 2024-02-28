
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
# Aggregate results of 03/04a-.
# Not shown in the paper.
#

import matplotlib.pyplot as plt
import numpy as np
import metrics_domain_adaptation.fig_utils as fig_utils
from metrics_domain_adaptation import utils
import json

data = [
    json.loads(x)
    for x in open(f"{utils.ROOT}/computed/agg_scratch_datasize_ep1.jsonl", "r")
]
data = [x for x in data if x["count"] != 0]

MANUAL_DA = {
    'bio': (0.2823187469551273, None, None), 'general': (0.32378887984159843, None, None), 'k': -1, 'count': -1
}
MANUAL_DAMQM = {
    'bio': (0.2865356173486739, None, None), 'general': (0.3336455492540476, None, None), 'k': 0, 'count': 0
}

# add DA and DA+MQM manually
data = [MANUAL_DA, MANUAL_DAMQM] + data

# sort by count
data.sort(key=lambda x: x["count"])

plt.figure(figsize=(3.5, 2))

best_avg = 0

# connecting lines
line_to_plot = {"bio": [], "general": []}
xticks = []
for line_i, line in enumerate(data):
    tau_bio, inta_bio, intb_bio = line["bio"]
    tau_gen, inta_gen, intb_gen = line["general"]
    upsample = line["k"]

    # we'd be better off not using it
    if (
        line["k"] != -1 and
        tau_bio + tau_gen < MANUAL_DAMQM["bio"][0] + MANUAL_DAMQM["general"][0]
    ):
        tau_bio, inta_bio, intb_bio = MANUAL_DAMQM["bio"]
        tau_gen, inta_gen, intb_gen = MANUAL_DAMQM["general"]
        upsample = 0

    # separate DA and DA+MQM from the rest
    if line_i <= 1:
        left_side = True
        line_i -= 0.5
    else:
        left_side = False

        line_to_plot["bio"].append((line_i, tau_bio))
        line_to_plot["general"].append((line_i, tau_gen))

    xticks.append((line_i, line["count"]))

    plt.text(
        line_i, tau_gen, "G",
        **fig_utils.MARKER_CIRCLE,
        zorder=200,
    )
    plt.text(
        line_i, tau_bio, "B",
        **fig_utils.MARKER_CIRCLE,
        zorder=200,
    )
    if left_side:
        plt.text(
            line_i, tau_gen+fig_utils.EXPECTED_BOOST, "$*$",
            **fig_utils.MARKER_CIRCLE_DASH,
            zorder=-100,
        )
    # fake scatter to scale the plot
    plt.scatter([line_i], [tau_gen+fig_utils.EXPECTED_BOOST], s=0)
    plt.plot(
        [line_i, line_i],
        [tau_gen, tau_bio],
        color=fig_utils.COLORS[0] if tau_gen < tau_bio else fig_utils.COLORS[1],
    )

    plt.text(
        x=line_i, y=0.365,
        s=f"{tau_bio:.2f}\n{tau_gen:.2f}".replace("0.", "."),
        ha="center", va="bottom", fontsize=7
    )

    # confidence intervals
    if (
        inta_bio is not None and intb_bio is not None and
        inta_gen is not None and intb_gen is not None
    ):
        plt.plot(
            np.array([line_i]*2)+0.15,
            [inta_bio, intb_bio],
            zorder=-500,
            **fig_utils.MARKER_CONF_INT
        )
        plt.plot(
            np.array([line_i]*2)-0.15,
            [inta_gen, intb_gen],
            zorder=-500,
            **fig_utils.MARKER_CONF_INT
        )

    best_avg = max(best_avg, (tau_bio+tau_gen)/2)

print(f"best avg {best_avg:.3f}")

for domain in ["bio", "general"]:
    plt.plot(
        [x[0] for x in line_to_plot[domain]],
        [x[1] for x in line_to_plot[domain]],
        color="black", zorder=-100, linestyle="-", linewidth=1
    )

plt.text(
    x=-0.1, y=1,
    s="Bio\nGen.",
    clip_on=False,
    transform=plt.gca().transAxes,
    fontsize=8,
)
plt.ylabel("Kendal $\\tau$")
plt.xlabel("General + $k\,{\\times}$ Bio$_{:s}$")
# hack to move the label
plt.gca().xaxis.set_label_coords(0.6, -0.2)

plt.ylim(0.28, 0.365)

XTICK_OVERRIDE = {
    -1: "\\text{\\footnotesize DA}",
    0: "\\text{\\footnotesize DA+}" + "\n\\text{\\footnotesize MQM}",
}

plt.xticks(
    [i for i, x in xticks],
    [
        XTICK_OVERRIDE[x] if x in XTICK_OVERRIDE else
        str(x).replace("000", "k").replace("5500", "6k")
        for i, x in xticks
    ]
)
plt.yticks([0.28, 0.32, 0.36])

plt.tight_layout(pad=0.1)
fig_utils.save("adapt_mqm_scratch_datasize_aggregated")
plt.show()
