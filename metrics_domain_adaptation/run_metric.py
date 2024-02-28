
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
# Main wrapper around all metric evaluation.
# Individual metrics are in `metrics/`.
#

import argparse
import random
import numpy as np
import utils
import metrics
from scipy.stats import kendalltau
import json
import time

args = argparse.ArgumentParser()
args.add_argument("--langs", default="all")
args.add_argument("--metric", default="comet-da")
args.add_argument("--domain", default="all")
args.add_argument("--split", default="test")
args.add_argument("--save-scores-path", default=None)
args.add_argument("--count", type=int, default=None)
args, args_unknown = args.parse_known_args()

args_unknown = dict(zip(args_unknown[:-1:2], args_unknown[1::2]))
args_unknown = {
    k.lstrip("-").replace("-", "_"): v
    for k, v in args_unknown.items()
}

if args.langs == "all":
    langs = ["en-de", "en-ru", "zh-en"]
elif args.langs == "total":
    langs = utils.LANGS
else:
    langs = [args.langs]

if args.domain == "all":
    domains = ["bio", "general"]
else:
    domains = [args.domain]

line_for_export = []
for domain in domains:
    taus = []
    for lang in langs:
        data = utils.load_data(
            kind="mqm", domain=domain,
            langs=lang, split=args.split
        )

        if args.count:
            data = random.Random(0).sample(data, k=min(len(data), args.count))
        print(lang, domain, len(data))
        lang1, lang2 = lang.split("-")
        metric = metrics.get(
            args.metric,
            lang1=lang1, lang2=lang2, domain=domain,
            **args_unknown,
        )
        scores_true = [x["score"] for x in data]
        scores = metric.predict(*utils.transpose_keys(data))

        if args.save_scores_path:
            for line, score_new in zip(data, scores):
                line_new = {
                    "langs": lang, "metric": args.metric,
                    "domain": domain,
                    "args": args_unknown,
                    "model_score": score_new,
                } | line
                line_for_export.append(line_new)

        tau, _tau_p = kendalltau(scores_true, scores)
        print("JSON!" + json.dumps({
            "langs": lang, "metric": args.metric,
            "domain": domain, "tau": tau,
            "time": time.ctime(),
            "args": args_unknown,
        }))
        taus.append(tau)

    print(f"Average Tau for {domain}: {np.average(np.abs(taus)):.3f}")


if args.save_scores_path:
    with open(args.save_scores_path, "w") as f:
        for line in line_for_export:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
