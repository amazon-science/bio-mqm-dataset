
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
# Averages JSON results between domains with a specified metric.
#

import json
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument("jsonl_file")
args.add_argument("--metric", default=None)
args = args.parse_args()

data_raw = [
    json.loads(x)
    for x in open(args.jsonl_file, "r")
]
data_raw = [
    x for x in data_raw
    if args.metric is None or args.metric == x["metric"]
]

for domain in ["general", "bio"]:
    data_local = [x["tau"] for x in data_raw if x["domain"] == domain]
    print(f"{domain}: {np.average(data_local):.3f}")
