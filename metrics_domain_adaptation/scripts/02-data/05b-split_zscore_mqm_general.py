
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
# Parse and zscore General MQM data.
#

import random
import tqdm
import collections
import json
import os
import csv
import scipy
import csv
import scipy
import datasets
import numpy as np
from scipy import stats
from metrics_domain_adaptation import utils

os.makedirs(f"{utils.ROOT}/data/mqm/general/train", exist_ok=True)
os.makedirs(f"{utils.ROOT}/data/mqm/general/test", exist_ok=True)


def print_format_line(langs, len_data_test, len_data_train):
    len_data = len_data_test + len_data_train
    lang1, lang2 = langs.split("-")
    lang1 = lang1.capitalize()
    lang2 = lang2.capitalize()
    # for LaTeX
    print(
        f"{lang1}-{lang2} & "
        f"{len_data_test} ({{\\small {len_data_test/len_data*100:.0f}\\% }}) & "
        f"{len_data_train} & {len_data} & 0 \\\\"
    )


all_data_train = []
all_data_test = []
data = []


def build_reference_matching_key(text):
    """
    keep only alphanumeric characters
    """
    return "".join([x for x in text if x.isalnum()]).strip()


srcmt_to_ref = {}
dataset_ricardo = datasets.load_dataset(
    "RicardoRei/wmt-mqm-human-evaluation",
    split="train"
)
for line in dataset_ricardo:
    if line["src"] == None or line["mt"] == None:
        continue
    triplet_key = line["lp"], build_reference_matching_key(line["src"]), build_reference_matching_key(line["mt"])
    srcmt_to_ref[triplet_key] = line["ref"]
    # reference itself is a "system", so we also need to make sure self-matching key is there
    ref_key = line["lp"], build_reference_matching_key(line["src"]), build_reference_matching_key(line["ref"])
    if ref_key not in srcmt_to_ref:
        srcmt_to_ref[ref_key] = line["ref"]

data_train = []
data_test = []
data_train_langs = collections.defaultdict(list)
data_test_langs = collections.defaultdict(list)

# special treatment for EN-RU
data_enru = dataset_ricardo.filter(lambda example: example["lp"] == "en-ru")
data_enru_years = collections.defaultdict(list)
for line in data_enru:
    line_new = {
        "lp": "en-ru",
        "src": line["src"],
        "tgt": line["mt"],
        "ref": line["ref"],
        "score": line["score"],
        "system": line["system"],
    }
    data_enru_years[line["year"]].append(line_new)

# zscore within each year (only <2022 for train!)
# the scale goes from 0 to 100 instead of -inf to 0 but that's ok because we're z-norming it anyway
for year, year_v in data_enru_years.items():
    scores = scipy.stats.zscore([x["score"] for x in year_v])
    for line_new, score in zip(year_v, scores):
        line_new["score_abs"] = line_new["score"]
        line_new["score"] = score
        # no access to errors
        line_new["errors"] = None
        if year < 2022:
            data_train.append(line_new)
            data_train_langs["en-ru"].append(line_new)

CATEGORY_MAP = {
    "punctuation": "fluency",
    "whitespace": "fluency",
    "unintelligible": "fluency",
    "capitalization": "fluency",
    "markup_tag": "fluency",
    "non-fluent": "fluency",
    "unnatural_flow": "fluency",
    "lacks_creativity": "fluency",
    "style": "fluency",
    "word_order": "fluency",
    "agreement": "fluency",
    "grammar": "fluency",
    "spelling": "fluency",

    "measurement_format": "locale",
    "date_time_format": "locale",
    "number_format": "locale",
    "currency_format": "locale",
    "locale convention": "locale",

    "register": "accuracy",
    "addition": "accuracy",
    "omission": "accuracy",
    "non-translation!": "accuracy",
    "do_not_translate": "accuracy",
    "mt_hallucination": "accuracy",
    "mistranslation": "accuracy",
    "untranslated": "untranslated",

    'inconsistent use of terminology': "terminology",
    "wrong term": "terminology",
    "wrong_named_entity": "terminology",
    "inconsistency": "terminology",

    "no-error": "no error",
    "source error": "source",
    "source_issue": "source",
}

SEVERITY_MAP = {
    "no error": "neutral",
    "no-error": "neutral",
}

SEVERITIES_OK = {"neutral", "minor", "major", "critical"}


def get_error(x):
    """
    Normalize error naming
    """
    if x["category"] is None or x["severity"] is None:
        return None

    category = x["category"].lower().strip().split("/")[0]
    severity = x["severity"].lower().strip().split("/")[0]

    # swap hotfix
    if category in SEVERITIES_OK:
        category, severity = severity, category

    if category in CATEGORY_MAP:
        category = CATEGORY_MAP[category]
    if severity in SEVERITY_MAP:
        severity = SEVERITY_MAP[severity]

    if severity not in SEVERITIES_OK:
        return None

    x["category"] = category
    x["severity"] = severity

    return x


def read_tsv(csv_path):
    ret = []
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
        field_names = header.split('\t')

        for line in f:
            field_values = line.strip().split('\t')
            ret.append(dict(zip(field_names, field_values)))

    return ret


# other langs
for filepath, target_list, target_dict_list, minimum_annotators, langs_all in [
    (f"{utils.ROOT}/data/raw/mqm_newstest2020_{{LANGS}}.tsv",
        data_train, data_train_langs, 3, ["zh-en", "en-de"]),
    (f"{utils.ROOT}/data/raw/mqm_newstest2021_{{LANGS}}.tsv",
        data_train, data_train_langs, 1, ["zh-en", "en-de"]),
    (f"{utils.ROOT}/data/raw/mqm_ted_{{LANGS}}.tsv",
        data_train, data_train_langs, 1, ["zh-en", "en-de"]),
    (f"{utils.ROOT}/data/raw/mqm_general2022_{{LANGS}}.tsv",
        data_test, data_test_langs, 1, ["zh-en", "en-de", "en-ru"]),
]:
    for langs in langs_all:
        ref_not_found = 0
        data_annotator = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        filepath_new = filepath.replace("{LANGS}", langs.replace('-', ''))
        data_local = read_tsv(filepath_new)
        malformed_count = 0
        for line in tqdm.tqdm(data_local):
            # skip malformed lines
            # substitute current error object
            if (error_line := get_error(line)) is None:
                malformed_count += 1
                continue
            for k, v in error_line.items():
                line[k] = v

            line["target"] = (
                line["target"]
                .replace("<v>", "")
                .replace("</v>", "")
            )
            line["source"] = (
                line["source"]
                .replace("<v>", "")
                .replace("</v>", "")
            )
            data_annotator[line["rater"]][(
                line["system"], line["source"], line["target"]
            )].append(line)

        print(malformed_count, "malformed out of", len(data_local))

        data_annotator_clean = collections.defaultdict(dict)
        for rater, rater_v in data_annotator.items():
            for segment, error_lines in rater_v.items():
                # sum scores for each annotator-sentence
                data_annotator_clean[rater][segment] = (
                    -sum([
                        utils.get_score_google(
                            line["category"], line["severity"])
                        for line in error_lines
                    ]),
                    [
                        {
                            "severity": line["severity"],
                            "category": line["category"],
                        }
                        for line in error_lines
                    ]
                )

        # normalization
        data_sents = collections.defaultdict(list)
        for rater, rater_v in data_annotator_clean.items():
            lines = [line for line, _ in rater_v.items()]
            scores = [score for _, (score, error_line) in rater_v.items()]
            errors = [error_line for _, (score, error_line) in rater_v.items()]
            # z-score within annotator
            scores_z = scipy.stats.zscore(scores)
            for line, score_z, score_abs, error in zip(lines, scores_z, scores, errors):
                data_sents[line].append({
                    "z": score_z, "abs": score_abs, "errors": error
                })

        data_sents_clean = []
        for (line_system, line_src, line_tgt), scores in data_sents.items():
            # reference not found
            if not (langs, build_reference_matching_key(line_src), build_reference_matching_key(line_tgt)) in srcmt_to_ref:
                ref_not_found += 1
                continue
            # take only sentences with three annotations
            if len(scores) < minimum_annotators:
                continue
            data_sents_clean.append({
                "lp": langs,
                "src": line_src,
                "tgt": line_tgt,
                "ref": srcmt_to_ref[(langs, build_reference_matching_key(line_src), build_reference_matching_key(line_tgt))],
                # normalize between annotators
                "score": np.average([x["z"] for x in scores]),
                "score_abs": np.average([x["abs"] for x in scores]),
                # take all error annotations (mostly will be one)
                "errors": [x["errors"] for x in scores],
                "system": line_system,
            })

        target_list += data_sents_clean
        target_dict_list[langs] += data_sents_clean

        print(filepath_new, len(data_sents_clean), "new sentence pairs")
        print("train", {k: len(v) for k, v in data_train_langs.items()})
        print("test", {k: len(v) for k, v in data_test_langs.items()})
        print(f"Total number of annotations where references are not found {langs}:", ref_not_found)


for langs, langs_v in data_train_langs.items():
    print("train", langs, len(langs_v))
    with open(f"{utils.ROOT}/data/mqm/general/train/{langs}.jsonl", "w") as f:
        f.write("\n".join([
            json.dumps(line, ensure_ascii=False)
            for line in langs_v
        ]))

for langs, langs_v in data_test_langs.items():
    print("test", langs, len(langs_v))
    with open(f"{utils.ROOT}/data/mqm/general/test/{langs}.jsonl", "w") as f:
        f.write("\n".join([
            json.dumps(line, ensure_ascii=False)
            for line in langs_v
        ]))

with open(f"{utils.ROOT}/data/mqm/general/train/all.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,domain\n")
    f = csv.writer(f)
    f.writerows([
        (line["lp"], line["src"], line["tgt"], line["ref"],
         line["score"], line["system"], "general")
        for line in data_train
    ])

with open(f"{utils.ROOT}/data/mqm/general/test/all.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,domain\n")
    f = csv.writer(f)
    f.writerows([
        (line["lp"], line["src"], line["tgt"], line["ref"],
         line["score"], line["system"], "general")
        for line in data_test
    ])

# create dev split
random.Random(0).shuffle(data_train)
with open(f"{utils.ROOT}/data/mqm/general/train/pudding_train.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,annotators,domain\n")
    f = csv.writer(f)
    f.writerows([
        (
            "", line["src"], line["tgt"], line["ref"],
            line["score"], "", "", "general"
        )
        for line in data_train[4096:]
    ])
with open(f"{utils.ROOT}/data/mqm/general/train/pudding_eval.csv", "w") as f:
    f.write("lp,src,mt,ref,score,system,annotators,domain\n")
    f = csv.writer(f)
    f.writerows([
        (
            "", line["src"], line["tgt"], line["ref"],
            line["score"], "", "", "general"
        )
        for line in data_train[:4096]
    ])


# flatten and get stats
data_flat = [
    line for line in data_train + data_test
    if line["errors"] is not None
]

stats_severities = collections.Counter([
    e["severity"]
    for line in data_flat
    for e in line["errors"][0]
])
_total = sum(stats_severities.values())
stats_severities = {k: f"{v/_total:.2%}" for k, v in stats_severities.items()}
stats_avg_error = np.average([
    len([x for x in line["errors"][0]])
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
# remove no error from distribution
stats_categories.pop("no error")
_total = sum(stats_categories.values())
stats_categories["accuracy"] += stats_categories.pop("untranslated") if "untranslated" in stats_categories else 0
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
