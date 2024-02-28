
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
# Split Bio paralel data.
# NOTE: Run after 06a-get_medline.py
#

import os
import glob
import mosestokenizer
import json
from tqdm import tqdm
from metrics_domain_adaptation import utils

os.makedirs(f"{utils.ROOT}/data/parallel/bio", exist_ok=True)

tokenizer_en = mosestokenizer.MosesSentenceSplitter(lang="en")
for langs in ["en-de", "en-ru", "en-zh"]:
    print("Processing", langs)
    lang1, lang2 = langs.split("-")
    tokenizer_xx = mosestokenizer.MosesSentenceSplitter(lang=lang2)
    f_out = open(f"{utils.ROOT}/data/parallel/bio/{langs}.jsonl", 'w')
    for article_en in tqdm(list(glob.glob(f"{utils.ROOT}/data/raw/wmt22_biomed/{langs}/*_{lang1}.txt"))):
        article_xx = article_en.replace("_en.txt", f"_{lang2}.txt")
        article_xx = article_xx.replace("zh.txt", "zh-cn.txt")
        if not os.path.exists(article_en) or not os.path.exists(article_xx):
            continue
        article_en = open(article_en, "r").read()
        article_xx = open(article_xx, "r").read()
        sents_en = tokenizer_en([article_en])
        if lang2 == "zh":
            # Mosestokenizer fails miserably on Chinese
            sents_xx = article_xx.strip("。").split("。")
        else:
            sents_xx = tokenizer_xx([article_xx])

        # There are many more sentences which we are skipping because they are not aligned
        if len(sents_en) == len(sents_xx):
            f_out.write('\n'.join([
                json.dumps(
                    {lang1: sent_en, lang2: sent_xx},
                    ensure_ascii=False
                )
                for sent_en, sent_xx in zip(sents_en, sents_xx)
            ])+'\n')
