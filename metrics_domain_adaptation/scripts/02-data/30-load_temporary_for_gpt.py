
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

import os
import json
from metrics_domain_adaptation import utils
import glob
import re
import collections

re_langs = re.compile(
    r".*/([\w\s-]+)_(?:abstract|medline)_(\w{2})2(\w{2})_\w{2}(?:_corrected|).txt"
)

os.makedirs(f"{utils.ROOT}/data/mqm_interim/bio/", exist_ok=True)

data_src_ref = {}
data_all = collections.defaultdict(list)

# load references and sources
for f in glob.glob("data/shuoyang_ref/*.txt"):
    lang1, lang2 = re_langs.match(f).group(2, 3)
    data_src_ref[(lang1, lang2)] = {}
    for line_i, line in enumerate(open(f, "r")):
        # skip header
        if line_i == 0:
            continue
        doc_i, sent_i, src, ref = line.rstrip("\n").split("\t")
        data_src_ref[(lang1, lang2)][(doc_i, sent_i)] = (src, ref)

# load systems
for f in glob.glob("data/shuoyang_tgt/*.txt"):
    system, lang1, lang2 = re_langs.match(f).group(1, 2, 3)
    for line_i, line in enumerate(open(f, "r")):
        doc_i, sent_i, tgt = line.rstrip("\n").split("\t")
        src, ref = data_src_ref[(lang1, lang2)][(doc_i, sent_i)]
        data_all[(lang1, lang2)].append({
            "src": src,
            "tgt": tgt,
            "ref": ref,
            "doc": doc_i,
            "sent_i": sent_i,
            "system": system
        })

for (lang1, lang2), data in data_all.items():
    with open(f"{utils.ROOT}/data/mqm_interim/bio/{lang1}-{lang2}.jsonl", "w") as f:
        f.write("\n".join([
            json.dumps(line, ensure_ascii=False)
            for line in data
        ]))
