
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
# Miscroplot of raw and zscored scores.
#

from metrics_domain_adaptation import utils, fig_utils
import matplotlib.pyplot as plt

data = [
    line
    for langs in ["en-de", "en-ru", "zh-en"]
    for split in ["train", "test"]
    for line in utils.load_data(kind="mqm", domain="general", langs=langs, split=split)
]

data_raw = [
    x["score_abs"] for x in data
]
data_raw = [
    x-100 if x > 0 else x
    for x in data_raw
]
data_raw = [
    x for x in data_raw
]
data_z = [
    x["score"] for x in data
]


def plot_hit(name, data, xlim, coef):
    data = [x for x in data if x >= xlim[0] and x <= xlim[1]]
    plt.figure(figsize=(1.4, 0.5))
    plt.hist(data, bins=10, color="#777")

    ax = plt.gca()
    ax.axis('off')
    plt.xlim(*xlim)
    plt.ylim(None, len(data)*coef)

    plt.tight_layout(pad=0)
    fig_utils.save("zscore_inlinefig_" + name)


plot_hit("raw", data_raw, (-20, 0), 0.79)
plot_hit("z", data_z, (-2, 2), 0.5)
