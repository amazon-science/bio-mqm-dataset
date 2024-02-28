
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

# load models
from comet import load_from_checkpoint
from metrics_domain_adaptation import utils
import numpy as np
import pickle
DEVICE = "cuda:0"

model_base = load_from_checkpoint(
    f"{utils.ROOT}/models/trained/mqm/main/checkpoints/epoch=0-step=5028-val_kendall=0.463.ckpt"
).to(DEVICE)
model_ft = load_from_checkpoint(
    f"{utils.ROOT}/models/trained/finetune_datasize/count5500_seed0/epoch=15-step=6288-val_kendall=0.367.ckpt"
).to(DEVICE)


def score_lines(model, lines):
    output = model.predict([
        {"src": line["src"], "mt": line["tgt"], "ref": line["ref"]}
        for line in lines
    ], batch_size=128, gpus=1, progress_bar=True)
    return output["scores"]


data_lines = {}
for domain in ["general", "bio"]:
    for langs in ["en-de", "en-ru", "zh-en"]:
        data_lines[(domain, langs)] = utils.load_data(
            "mqm", domain, langs,
            split="test"
        )

# do the inference (slow)
data_scores = {}
for (domain, langs), lines in data_lines.items():
    data_scores[(domain, langs, "base")] = score_lines(model_base, lines)
    data_scores[(domain, langs, "ft")] = score_lines(model_ft, lines)

# add human scores
for domain in ["general", "bio"]:
    for langs in ["en-de", "en-ru", "zh-en"]:
        data_scores[(domain, langs, "human")] = [
            x["score"]
            for x in data_lines[(domain, langs)]
        ]


def normalize_scores(model_name):
    all_scores = (
        data_scores[("general", "en-de", model_name)]+data_scores[("bio", "en-de", model_name)] +
        data_scores[("general", "en-ru", model_name)]+data_scores[("bio", "en-ru", model_name)] +
        data_scores[("general", "zh-en", model_name)] +
        data_scores[("bio", "zh-en", model_name)]
    )
    all_avg = np.average(all_scores)
    all_var = np.std(all_scores)
    print(model_name, all_avg, all_var)

    for domain in ["general", "bio"]:
        for langs in ["en-de", "en-ru", "zh-en"]:
            data_scores[(domain, langs, model_name)] = [
                (x-all_avg)/all_var for x in
                data_scores[(domain, langs, model_name)]
            ]


normalize_scores("base")
normalize_scores("ft")
normalize_scores("human")

# merge
data_all = {}
for (domain, langs), lines in data_lines.items():
    scores = [
        {"human": x[2], "base": x[0], "ft": x[1]}
        for x in zip(
            data_scores[(domain, langs, "base")],
            data_scores[(domain, langs, "ft")],
            data_scores[(domain, langs, "human")],
        )
    ]
    data_all[(domain, langs)] = [
        {"scores": y} | x
        for x, y in zip(lines, scores)
    ]

pickle.dump(data_all, open(f"{utils.ROOT}/computed/scores_all_z.pkl", "wb"))
