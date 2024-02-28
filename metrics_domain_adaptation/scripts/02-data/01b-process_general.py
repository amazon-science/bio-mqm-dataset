
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
# Split News data into parallel and mono.
# NOTE: Run this after 01a-get_data_news.sh
#

import csv
import json
import os
from metrics_domain_adaptation import utils

os.makedirs(f"{utils.ROOT}/data/mono/general", exist_ok=True)
os.makedirs(f"{utils.ROOT}/data/parallel/general", exist_ok=True)

for lang in ["en", "ru", "zh", "de", "en-de", "en-ru", "en-zh"]:
    print("Processing", lang)
    if "-" in lang:
        f = open(f"{utils.ROOT}/data/parallel/general/{lang}.jsonl", "w")
        other_lang = lang.split("-")[1]
    else:
        f = open(f"{utils.ROOT}/data/mono/general/{lang}.jsonl", "w")
    for line in csv.reader(open(f"{utils.ROOT}/data/raw/news-commentary.{lang}.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE):
        if "-" in lang:
            assert len(line) == 2
            f.write(json.dumps(
                {"en": line[0], other_lang: line[1]}, ensure_ascii=False)+'\n')
        else:
            if not line:
                continue
            line = " ".join(line)
            f.write(json.dumps({lang: line}, ensure_ascii=False)+'\n')
