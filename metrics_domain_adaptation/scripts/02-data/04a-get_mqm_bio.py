
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
# Parse and zscore Bio MQM data.
#

import glob
import json
import collections
import re
import os
import csv
import numpy as np
import scipy
from metrics_domain_adaptation import utils

os.makedirs(f"{utils.ROOT}/data/mqm/bio", exist_ok=True)

re_langs = re.compile("[a-z]{2}2[a-z]{2}")

data = collections.defaultdict(list)
langs = collections.Counter()


def sanitize(text: str) -> str:
    return " ".join(x for x in text if x.isalnum())


print("NOTE: We are not collecting the reference translations (batch 2).")

references = collections.defaultdict(dict)
for f in glob.glob(f"{utils.ROOT}/data/raw/wmt21-biomed-mqm/target/main_phase/round_one/batch2/**/*.json", recursive=True):
    data_local = json.load(open(f, "r", encoding='utf-8-sig'))
    f = (f
         .replace("eslatam", "eses")
         .replace("zhcn", "zhzh")
         )
    langs_local = re_langs.search(f).group().replace("2", "-")
    for line in data_local:
        references[langs_local][
            line["SEG_ID"] + " ||| " + sanitize(line["source"])
        ] = line["target"]


def get_score(line):
    return -sum([
        utils.get_score_google(x["category"], x["severity"])
        for x in line["errors"]
    ])


CATEGORY_MAP = {
    "punctuation": "fluency",
    "character encoding": "fluency",
    "register": "fluency",
    "spelling": "fluency",
    "grammar": "fluency",
    "non-fluent": "fluency",

    "number format": "locale",
    "measurement format": "locale",
    "date format": "locale",
    "currency format": "locale",

    "unintelligible": "accuracy",
    "mistranslation": "accuracy",
    "addition": "accuracy",
    "untranslated": "untranslated",

    'inconsistent use of terminology': "terminology",
    "wrong term": "terminology",
}


def get_error(x):
    """
    Normalize error naming
    """
    category = x.pop("error_category").lower().strip()
    if category in CATEGORY_MAP:
        category = CATEGORY_MAP[category]
    severity = x["severity"].lower().strip()

    x["category"] = category
    x["severity"] = severity

    return x


for f in glob.glob(f"{utils.ROOT}/data/raw/wmt21-biomed-mqm/target/main_phase/round_one/batch1/**/*.json", recursive=True):
    data_local = json.load(open(f, "r", encoding='utf-8-sig'))
    langs_local = re_langs.search(f).group().replace("2", "-")

    ftsv = f.replace(".json", ".tsv")
    data_src_to_docid = csv.DictReader(open(ftsv, "r"), delimiter="\t")
    data_src_to_docid = {
        (
            line["Source"].replace('""', '"').strip(),
        ): line["Doc"]
        for line in data_src_to_docid
    }

    data_local_new = []
    for line in data_local:
        line["errors"] = [get_error(x) for x in line.pop("target_errors")]
        score = get_score(line)
        data_local_new.append({
            "annotator_id": line["Annotator_ID"],
            "src": line["source"],
            "tgt": line["target"],
            "ref": references[langs_local][line["SEG_ID"]+" ||| "+sanitize(line["source"])],
            "score": score,
            "score_abs": score,
            # save in a list to support multiple annotators
            "errors": [line["errors"]],
            "doc": data_src_to_docid[(line["source"].strip(),)],
        })
    langs[langs_local] += len(data_local_new)
    data[langs_local] += data_local_new

print("Total lines")
print(langs)

# zscoring within annotator, this should be propagated upstream because everything is
# reference-based (no pun intended)
for lang in data:
    data_annotator = collections.defaultdict(list)
    for line in data[lang]:
        data_annotator[line["annotator_id"]].append(line)
    data_annotator_new = collections.defaultdict(list)
    for annotator, annotator_lines in data_annotator.items():
        if len(annotator_lines) <= 20:
            data_annotator_new["small_merged"] += annotator_lines
        else:
            data_annotator_new[annotator] += annotator_lines

    for annotator, annotator_lines in data_annotator_new.items():
        is_constant = all([
            annotator_lines[0]["score"] == x["score"]
            for x in annotator_lines
        ])
        if not is_constant:
            scores = scipy.stats.zscore([x["score"] for x in annotator_lines])
            for line, score in zip(annotator_lines, scores):
                line["score"] = score

for lang in data:
    print("Saving", lang)
    with open(f"{utils.ROOT}/data/mqm/bio/{lang}.jsonl", "w") as f:
        f.write("\n".join([
            json.dumps(x, ensure_ascii=False)
            for x in data[lang]
        ]))


# flatten and get stats
data_flat = [line for lang in data for line in data[lang]]

stats_severities = collections.Counter([
    e["severity"]
    for line in data_flat
    for e in line["errors"][0]
])
_total = sum(stats_severities.values())
stats_severities = {k: f"{v/_total:.2%}" for k, v in stats_severities.items()}
stats_avg_error = np.average([
    len(line["errors"][0])
    for line in data_flat
    if len([s for s in line["errors"][0] if s["severity"] != "neutral"]) != 0
])
stats_error_free = np.average([
    len([s for s in line["errors"][0] if s["severity"] != "neutral"]) == 0
    for line in data_flat
])
stats_categories = collections.Counter([
    e["category"]
    for line in data_flat
    for e in line["errors"][0]
])
_total = sum(stats_categories.values())
stats_categories["accuracy"] += stats_categories.pop("untranslated")
stats_categories = {
    k: f"{v/_total:.2%}" for k,
    v in stats_categories.most_common()
}
stats_error_scores_abs = [
    line["score_abs"] for line in data_flat if
    len([s for s in line["errors"][0] if s["severity"] != "neutral"]) != 0
]

print(stats_categories)
print(stats_severities)
print(f"Avg. error count {stats_avg_error:.4f}")
print(f"Avg. abs error {np.average(stats_error_scores_abs):.4f}")
print(f"Proportion error-free {stats_error_free:.2%}")
