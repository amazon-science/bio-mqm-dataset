
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
# Downloads General MQM data.
# 

wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/newstest2020/ende/mqm_newstest2020_ende.tsv -O ${ADAPTATION_ROOT}/data/raw/mqm_newstest2020_ende.tsv
wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/newstest2020/zhen/mqm_newstest2020_zhen.tsv -O ${ADAPTATION_ROOT}/data/raw/mqm_newstest2020_zhen.tsv
wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/newstest2021/ende/mqm_newstest2021_ende.tsv -O ${ADAPTATION_ROOT}/data/raw/mqm_newstest2021_ende.tsv
wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/newstest2021/zhen/mqm_newstest2021_zhen.tsv -O ${ADAPTATION_ROOT}/data/raw/mqm_newstest2021_zhen.tsv
wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/ted/ende/mqm_ted_ende.tsv -O ${ADAPTATION_ROOT}/data/raw/mqm_ted_ende.tsv
wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/ted/zhen/mqm_ted_zhen.tsv -O ${ADAPTATION_ROOT}/data/raw/mqm_ted_zhen.tsv
wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/generalMT2022/ende/mqm_generalMT2022_ende.tsv -O ${ADAPTATION_ROOT}/data/raw/mqm_general2022_ende.tsv
wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/generalMT2022/zhen/mqm_generalMT2022_zhen.tsv -O ${ADAPTATION_ROOT}/data/raw/mqm_general2022_zhen.tsv
wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/generalMT2022/enru/mqm_generalMT2022_enru.tsv -O ${ADAPTATION_ROOT}/data/raw/mqm_general2022_enru.tsv
