
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

#!/usr/bin/bash

# 
# Download mono and parallel News data.
# 

mkdir -p ${ADAPTATION_ROOT}/data/raw/

# parallel
for LANG2 in "zh" "ru"; do
    wget -q -O- "https://data.statmt.org/news-commentary/v18.1/training/news-commentary-v18.en-${LANG2}.tsv.gz" | \
        gzip -d > "${ADAPTATION_ROOT}/data/raw/news-commentary.en-${LANG2}.tsv"
done
# German is the only one which is de-en instead of en-de (lexically sorted aparently)
wget -q -O- "https://data.statmt.org/news-commentary/v18.1/training/news-commentary-v18.de-en.tsv.gz" | \
    gzip -d > "${ADAPTATION_ROOT}/data/raw/news-commentary.en-de.tsv"

# monolingual
for LANG in "en" "zh" "de" "ru"; do
    wget -q -O- "https://data.statmt.org/news-commentary/v18.1/training-monolingual/news-commentary-v18.${LANG}.gz" | \
        gzip -d > "${ADAPTATION_ROOT}/data/raw/news-commentary.${LANG}.tsv"
done

# print
wc -l ${ADAPTATION_ROOT}/data/raw/news-commentary.*.tsv

