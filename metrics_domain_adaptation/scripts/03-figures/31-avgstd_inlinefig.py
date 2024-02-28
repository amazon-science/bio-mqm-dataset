
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
# Microplot of model output distribution.
#

import numpy as np
from metrics_domain_adaptation import utils, fig_utils
import matplotlib.pyplot as plt
import json

data_ft = [
    json.loads(x) for x in
    open(f"{utils.ROOT}/computed/scores_ft.jsonl")
]
data_base = [
    json.loads(x) for x in
    open(f"{utils.ROOT}/computed/scores_base.jsonl")
]
data_human_gen = [
    line["score"]
    for langs in ["en-de", "en-ru", "zh-en"]
    for split in ["train", "test"]
    for line in utils.load_data(kind="mqm", domain="general", langs=langs, split=split)
]
data_human_bio = [
    line["score"]
    for langs in ["en-de", "en-ru", "zh-en"]
    for split in ["dev", "test"]
    for line in utils.load_data(kind="mqm", domain="bio", langs=langs, split=split)
]


def plot_hit(name, data):
    print(f"{name} {np.average(data):.2f} {np.std(data):.2f}")
    data = [x for x in data if x >= -1 and x <= 1]
    plt.figure(figsize=(1.2, 0.4))
    plt.hist(data, bins=7, color="#777")

    ax = plt.gca()
    ax.axis('off')
    plt.xlim(-1, 1)
    plt.ylim(0, len(data)/2)

    plt.tight_layout(pad=0)
    fig_utils.save("avgstd_inline_" + name)


plot_hit("human_gen", data_human_gen)
plot_hit("human_bio", data_human_bio)
plot_hit(
    "base_gen",
    [x["model_score"] for x in data_base if x["domain"] == "general"]
)
plot_hit(
    "base_bio",
    [x["model_score"] for x in data_base if x["domain"] == "bio"]
)
plot_hit(
    "ft_gen",
    [x["model_score"] for x in data_ft if x["domain"] == "general"]
)
plot_hit(
    "ft_bio",
    [x["model_score"] for x in data_ft if x["domain"] == "bio"]
)
