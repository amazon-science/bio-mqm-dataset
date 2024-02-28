
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
# Essentially a duplicate of 02/04a- but does not write anything and only produces a table.
#


import glob
import json
import collections
import re
import numpy as np
import scipy
from metrics_domain_adaptation import utils

re_langs = re.compile("[a-z]{2}2[a-z]{2}")


def get_score(line):
    line["errors"] = [get_error(x) for x in line.pop("target_errors")]
    return -sum(
        utils.get_score_google(x["category"], x["severity"])
        for x in line["errors"]
    )


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


annotators = collections.defaultdict(list)
for f in glob.glob(f"{utils.ROOT}/data/raw/wmt21-biomed-mqm/target/main_phase/round_one/batch2/**/*.json", recursive=True):
    data_local = json.load(open(f, "r", encoding='utf-8-sig'))
    f = f.replace("eslatam", "eses").replace("zhcn", "zhzh")
    langs_local = re_langs.search(f).group().replace("2", "-")
    for line in data_local:
        line = {
            "score": get_score(line),
            "annotator_id": line["Annotator_ID"],
            "mt_engine": "ref",
            "langs": langs_local,
        }
        annotators[line["annotator_id"]].append(line)

for f in glob.glob(f"{utils.ROOT}/data/raw/wmt21-biomed-mqm/target/main_phase/round_one/batch1/**/*.json", recursive=True):
    data_local = json.load(open(f, "r", encoding='utf-8-sig'))
    langs_local = re_langs.search(f).group().replace("2", "-")

    data_local_new = []
    for line in data_local:
        line = {
            "score": get_score(line),
            "annotator_id": line["Annotator_ID"],
            "mt_engine": line["MT_Engine"],
            "langs": langs_local,
        }
        annotators[line["annotator_id"]].append(line)

# zscoring within annotator, this should be propagated upstream because everything is
# reference-based (no pun intended)
for annotator, annotator_lines in annotators.items():
    is_constant = all([
        annotator_lines[0]["score"] == x["score"]
        for x in annotator_lines])
    if not is_constant:
        scores = scipy.stats.zscore([x["score"] for x in annotator_lines])
        for line, score in zip(annotator_lines, scores):
            line["zscore"] = score
    else:
        for line in annotator_lines:
            line["zscore"] = line["score"]

systems = collections.defaultdict(list)
for annotator, annotator_lines in annotators.items():
    for line in annotator_lines:
        systems[line["mt_engine"]].append(line)

systems = list(systems.items())

for langs in utils.LANGS + [None]:
    print(f"|**{langs}** ({len(systems)})||||")
    table = list()
    for system, system_lines in systems:
        scores = [
            x["score"]
            for x in system_lines
            if x["langs"] == langs or langs is None
        ]
        zscores = [
            x["zscore"]
            for x in system_lines
            if x["langs"] == langs or langs is None
        ]
        if not scores:
            continue
        annotators = {
            x["annotator_id"]
            for x in system_lines if x["langs"] == langs or langs is None
        }
        table.append((
            system, np.average(scores),
            np.average(zscores),
            f"{len(scores)} ({len([x for x in scores if x != 0])})",
            len(annotators)
        ))

    # sort by avg score
    table.sort(key=lambda x: x[1], reverse=True)
    for system, scores, zscores, len_score, len_annotators in table:
        print(
            "", system,
            f"{scores:.2f}",
            f"{zscores:.2f}",
            f"{len_score}",
            f"{len_annotators}",
            "",
            sep=" | ",
        )
