
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

#
# Misc. utils used in this project.
# The imports are intentionally inside of the functions so that they don't get
# called whenever the utils are imported. (e.g. torch)
#

from typing import List, Dict
import os


# supress omnipresent TF warning for AVX2 FMA operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set wandb to offline by default
os.environ["WANDB_MODE"] = "offline"

if "ADAPTATION_ROOT" not in os.environ:
    raise Exception(
        "For all scripts to work properly, you need to set the `ADAPTATION_ROOT` environment variable. "
        "It is enough to set it to `.` though it may get messy because everything will be stored there."
    )
ROOT = os.environ["ADAPTATION_ROOT"]

LANGS = [
    "br-en", "de-en", "en-de", "en-es", "en-fr",
    "en-ru", "en-zh", "es-en", "fr-en", "ru-en",
    "zh-en",
]

LANG_TO_NLLB = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
}


def load_data(kind, domain, langs, split, count=None) -> List[Dict]:
    """
    Active loading of clean data
    """
    import json
    file = f"{ROOT}/data/{kind}/{domain}/{split}/{langs}.jsonl"
    data = []
    for line_i, line in enumerate(open(file, "r")):
        if line_i == count:
            break
        # hotfix for missing translation
        line = json.loads(line)
        if not line["tgt"]:
            continue
        data.append(line)
    return data


def load_data_lazy(kind, domain, langs):
    """
    Lazy loading of clean data
    """
    import json
    file = f"{ROOT}/data/{kind}/{domain}/{langs}.jsonl"
    for line in open(file, "r"):
        yield json.loads(line)


def transpose_keys(data, keys=["src", "tgt", "ref"]):
    return [[x[key] for x in data] for key in keys]


def transpose_dict(data, keys=[]):
    return {key: [x[key] for x in data] for key in keys}


def permissive_jsonl(line_raw):
    if "{" in line_raw:
        import json
        start_i = line_raw.index("{")
        line_raw = line_raw[start_i:]
        return json.loads(line_raw)
    else:
        return None


def get_mean_inta_intb(arr):
    import numpy as np
    import scipy.stats as st
    if len(arr) == 1:
        return np.mean(arr), None, None
    interval = st.t.interval(
        confidence=0.90, df=len(arr)-1,
        loc=np.mean(arr), scale=st.sem(arr),
    )
    return np.mean(arr), interval[0], interval[1]


def rev_langs(langs: str) -> str:
    lang1, lang2 = langs.split("-")
    return lang2 + "-" + lang1


def pretty_langs(langs: str) -> str:
    lang1, lang2 = langs.split("-")
    return lang1.capitalize() + "-" + lang2.capitalize()


def get_score_google(category, severity):
    """
    Taken from Table 2 in https://aclanthology.org/2021.wmt-1.73v3.pdf
    """
    if severity == "neutral":
        return 0.0
    elif severity == "minor":
        if category == "fluency":
            return 0.1
        else:
            return 1
    elif severity in {"major", "critical"}:
        if category == "untranslated":
            return 25
        else:
            return 5
    else:
        raise Exception(f"Unknown severity {severity} or category {category}.")
