
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
from tqdm import tqdm
import mosestokenizer
import glob
from metrics_domain_adaptation import utils
import urllib.request
import zipfile

print("Downloading raw data")
urllib.request.urlretrieve(
    "https://www.openagrar.de/servlets/MCRFileNodeServlet/openagrar_derivate_00019621/nts-icd.zip",
    f"{utils.ROOT}/data/raw/nts-icd.zip"
)
with zipfile.ZipFile(f"{utils.ROOT}/data/raw/nts-icd.zip", 'r') as f:
    f.extractall(f"{utils.ROOT}/data/raw/nts-icd/")

os.makedirs(f"{utils.ROOT}/data/mono/bio", exist_ok=True)

splitter = mosestokenizer.MosesSentenceSplitter(lang="de")
dataset = []
for f in glob.glob(f"{utils.ROOT}/data/raw/nts-icd/docs-training/*.txt"):
    with open(f, "r") as f:
        dataset.append(f.read())

print(len(dataset), "docs")
data = []
for line in tqdm(dataset):
    data += splitter([line])

print("total", len(data))
data = set(data)
print("unique", len(data))


with open(f"{utils.ROOT}/data/mono/bio/de.jsonl", "w") as f:
    for line in data:
        f.write(json.dumps({"de": line}, ensure_ascii=False)+'\n')
