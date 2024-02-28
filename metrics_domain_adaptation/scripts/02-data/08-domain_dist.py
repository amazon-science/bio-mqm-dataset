
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
# Compute the domain distribution between Bio and General.
#

import random
import collections
import datasets
from metrics_domain_adaptation import utils
import tqdm
import numpy as np
import mosestokenizer
splitter = mosestokenizer.MosesSentenceSplitter(lang="zh")

datasets_all = collections.defaultdict(set)
dataset_general = datasets.load_dataset(
    "RicardoRei/wmt-mqm-human-evaluation",
    split="train"
)
for line_i, line in enumerate(tqdm.tqdm(dataset_general)):
    langs = line["lp"]
    if langs.startswith("en-"):
        datasets_all[line["domain"]].add(line["src"])
    elif langs.endswith("-en"):
        datasets_all[line["domain"]].add(line["ref"])
print(
    "Average src sent length General",
    "{:.2f}".format(np.average([
        len(line.split())
        for dataset_local in datasets_all.values()
        for line in dataset_local
    ]))
)

dataset_bio_raw = {
    langs: utils.load_data("mqm", "bio", langs, "test")
    for langs in utils.LANGS
}
dataset_bio = set()
for langs, data_local in tqdm.tqdm(list(dataset_bio_raw.items())):
    if langs.startswith("en-"):
        datasets_all["bio"] |= {x["src"] for x in data_local}
    elif langs.endswith("-en"):
        datasets_all["bio"] |= {x["ref"] for x in data_local}


for domain, data_local in datasets_all.items():
    sents = [sent for line in data_local for sent in splitter([line])]
    print(
        "Average src sent length", domain,
        "{:.2f}".format(np.average([
            len(line.split()) for line in sents
        ]))
    )

random.seed(0)

print([
    (domain, len(dataset_local))
    for domain, dataset_local in datasets_all.items()
])

datasets_all_vocab = {
    domain: {
        word
        for line in random.choices(list(data_local), k=697)
        for word in line.lower().split()
    }
    for domain, data_local in datasets_all.items()
}


def overlap_score(domain1, domain2):
    vocab1 = datasets_all_vocab[domain1]
    vocab2 = datasets_all_vocab[domain2]
    return 2*len(vocab1 & vocab2)/(len(vocab1) + len(vocab2))


overlap_bio_rest = np.average([
    overlap_score("bio", domain2)
    for domain2 in ["news", "ted", "social", "ecommerce", "conversation"]
])
overlap_rest_rest = np.average([
    overlap_score(domain1, domain2)
    for domain1 in ["news", "ted", "social", "ecommerce", "conversation"]
    for domain2 in ["news", "ted", "social", "ecommerce", "conversation"]
    if domain1 != domain2
])
print("Vocab overlap Bio-Rest", "{:.2%}".format(overlap_bio_rest))
print("Vocab overlap Bio-Rest", "{:.2%}".format(overlap_rest_rest))
