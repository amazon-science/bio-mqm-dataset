
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
# Histogram of loss distribution.
# Not shown in the paper.
#

import numpy as np
import matplotlib.pyplot as plt
from metrics_domain_adaptation import fig_utils, utils
import pickle

data_all = pickle.load(open(f"{utils.ROOT}/computed/scores_all_z.pkl", "rb"))


def plot_loss(ax, model, domain):
    data = [
        x for lang in ["en-de", "en-ru", "zh-en"]
        for x in data_all[(domain, lang)]
    ]
    scores_model = np.array([x["scores"][model] for x in data])
    scores_human = np.array([x["scores"]["human"] for x in data])
    losses = np.abs(scores_human-scores_model)

    losses = [x for x in losses if x <= 2]
    ax.hist(losses, bins=10, color="#777")
    ax.set_ylim(0, 15_000 if domain == "general" else 2_000)
    ax.set_ylabel(domain.capitalize() if model == "base" else "")

    ax.set_yticks([])
    if domain == "general":
        ax.set_xticks([])
        ax.set_title(
            r"\textsc{" + model + "}",
            y=0.8
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # use bottom spline to cover up half of the marker
    ax.spines['bottom'].set_color('#fff')


plt.figure(figsize=(3.2, 1.3))
plot_loss(plt.subplot(2, 2, 1), "base", "general")
plot_loss(plt.subplot(2, 2, 2), "ft", "general")
plot_loss(plt.subplot(2, 2, 3), "base", "bio")
plot_loss(plt.subplot(2, 2, 4), "ft", "bio")

plt.tight_layout(pad=0.1)
plt.subplots_adjust(hspace=0.1)

fig_utils.save("loss_hist")
