
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

import argparse
import json
import requests
import numpy as np
import tqdm
import os
from metrics_domain_adaptation import utils

URL = "http://127.0.0.1:5000"

PROMPTS = {
    "ref": 'Score the following machine translation from {lang1} to {lang2} with respect to the human reference on a continuous scale from 0 to 100 that starts with "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{lang1} source: "{src}"\n{lang2} human reference: "{ref}"\n{lang2} machine translation: "{tgt}"\nScore (0-100):\n',

    "src": 'Score the following translation from {lang1} to {lang2} on a continuous scale from 0 to 100 that starts on "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{lang1} source: "{src}"\n{lang2} translation: "{tgt}"\nScore (0-100): ',

    "src-p3": ' Your task is to score machine translations from {lang1} to {lang2} on a continuous scale from 0 to 100 that starts with "No meaning preserved" up to "Perfect meaning and grammar".\n\nFor example, {lang1} source: "Geriatric rehabilitation care serves to maintain the self-determined participation of elderly people.", {lang2} translation: "Die geriatrische rehabilitationsversorgung dient der aufrechterhaltung der selbstbestimmten teilhabe älterer menschen.", Score: 80.\n\nAnother example, {lang1}: "Three cases of e-cigarette explosions which occurred between 2016 and 2019, are presented.", {lang2}: "Es werden drei Fälle von E-Zigaretten-Explosionen vorgestellt, die sich zwischen 2016 und 2019 ereignet haben.", Score: 100.\n\nYet another example, {lang1}: "The guidelines of the Advanced Trauma Life Support should be followed, signs of an inhalation trauma should be checked and litmus test should be performed prior to irrigation with aqueous solutions to prevent exothermic reactions with remaining metals.", {lang2}: "Die Richtlinien der Advanced Trauma Life Support sollten befolgt werden, Anzeichen eines Inhalationstraumas sollten überprüft und vor der Spülung mit wässrigen Lösungen ein Lackmustest durchgeführt werden, um exotherme Reaktionen mit restlichen Metallen zu verhindern.", Score: 95.\n\nNow given this source, {lang1}: "{src}" and translation, {lang2}: "{tgt}", provide the translation score: ',

    "src-p2": ' Your task is to score machine translations from {lang1} to {lang2} on a continuous scale from 0 to 100 that starts with "No meaning preserved" up to "Perfect meaning and grammar".\n\nFor example, {lang1} source: "Geriatric rehabilitation care serves to maintain the self-determined participation of elderly people.", {lang2} translation: "Die geriatrische rehabilitationsversorgung dient der aufrechterhaltung der selbstbestimmten teilhabe älterer menschen.", Score: 80.\n\nAnother example, {lang1}: "Three cases of e-cigarette explosions which occurred between 2016 and 2019, are presented.", {lang2}: "Es werden drei Fälle von E-Zigaretten-Explosionen vorgestellt, die sich zwischen 2016 und 2019 ereignet haben.", Score: 100.\n\nNow given this source, {lang1}: "{src}" and translation, {lang2}: "{tgt}", provide the translation score: ',

    "src-p0": ' Your task is to score machine translations from {lang1} to {lang2} on a continuous scale from 0 to 100 that starts with "No meaning preserved" up to "Perfect meaning and grammar".\n\nGiven this source, {lang1}: "{src}" and translation, {lang2}: "{tgt}", provide the translation score: '
}

LANGUAGE_CODES = {
    "en": "English",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
}


def attempt_parse(x):
    x = x.strip()
    if x.startswith("."):
        x = "0" + x
    if x.replace(".", "", 0).isnumeric():
        try:
            return float(x)
        except:
            return None
    else:
        return None


def generate(lang1, lang2, src, tgt, ref, mode) -> None:
    request_body = {
        "text": [
            PROMPTS[mode].format(
                src=src, tgt=tgt, lang1=lang1, lang2=lang2,
                **(dict(ref=ref) if mode == "ref" else {})
            ),
        ],
        "max_new_tokens": 1,
        "do_sample": True,
        "top_k": 100,
    }
    response = requests.post(
        url=URL+"/generate/",
        json=request_body, verify=False
    )
    response = response.json()["text"][0]
    return response


def get_blemba_score(lang1, lang2, src, tgt, ref, mode) -> float:
    outputs = []
    # make at most 10 attempts
    for _ in range(10):
        val = attempt_parse(generate(lang1, lang2, src, tgt, ref, mode))
        if val is not None:
            outputs.append(val)
        if len(outputs) == 1:
            break

    if outputs:
        return np.average(outputs)
    else:
        return 0


args = argparse.ArgumentParser()
args.add_argument("--domain", default="all")
args.add_argument("--langs", default="all")
args.add_argument("--mode", default="all")
args.add_argument("--prefix", default="")
args = args.parse_args()

if args.langs == "all":
    langs = ["en-de", "en-ru", "zh-en"]
else:
    langs = [args.langs]

if args.domain == "all":
    domains = ["bio", "general"]
else:
    domains = [args.domain]

if args.mode == "all":
    modes = ["src", "ref"]
else:
    modes = [args.mode]

for mode in modes:
    for domain in domains:
        os.makedirs(
            f"computed/blemba/{args.prefix}{mode}/{domain}/", exist_ok=True)
        for lang in langs:
            print("Running", mode, domain, lang)

            fname = f"computed/blemba/{args.prefix}{mode}/{domain}/{lang}.jsonl"
            lang1, lang2 = lang.split("-")
            lang1name = LANGUAGE_CODES[lang1]
            lang2name = LANGUAGE_CODES[lang2]

            if os.path.exists(fname):
                print("Skipping", mode, domain, lang)
                continue

            data = utils.load_data(
                kind="mqm", domain=domain,
                langs=lang, split="test"
            )

            for line_i, line in enumerate(tqdm.tqdm(data)):
                line["blemba_score"] = get_blemba_score(
                    lang1name, lang2name,
                    line["src"], line["tgt"], line["ref"],
                    mode=mode
                )
                if line_i % 100 == 0 or line_i == len(data) - 1:
                    with open(fname, "w") as f:
                        f.write("\n".join([
                            json.dumps(x, ensure_ascii=False) for x in data
                        ]))


# nohup ./metrics_domain_adaptation/scripts/11-bloom/06-inference.py --mode src-p3 --domain bio --langs en-de > logs/bio_src_p3.log &
# nohup ./metrics_domain_adaptation/scripts/11-bloom/06-inference.py --mode src-p2 --domain bio --langs en-de > logs/bio_src_p2.log &
# nohup ./metrics_domain_adaptation/scripts/11-bloom/06-inference.py --mode src-p0 --domain bio --langs en-de > logs/bio_src_p0.log &


# nohup ./metrics_domain_adaptation/scripts/11-bloom/06-inference.py --mode src-p3 --prefix "nomt/" --domain bio --langs en-de > logs/bio_src_p3.log &
# nohup ./metrics_domain_adaptation/scripts/11-bloom/06-inference.py --mode src-p2 --prefix "nomt/" --domain bio --langs en-de > logs/bio_src_p2.log &
# nohup ./metrics_domain_adaptation/scripts/11-bloom/06-inference.py --mode src-p0 --prefix "nomt/" --domain bio --langs en-de > logs/bio_src_p0.log &
# nohup ./metrics_domain_adaptation/scripts/11-bloom/06-inference.py --mode src --prefix "nomt/" --domain bio --langs en-de > logs/bio_src.log &
# nohup ./metrics_domain_adaptation/scripts/11-bloom/06-inference.py --mode ref --prefix "nomt/" --domain bio --langs en-de > logs/bio_ref.log &
