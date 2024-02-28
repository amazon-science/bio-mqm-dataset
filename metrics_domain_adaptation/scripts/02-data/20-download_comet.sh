
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
# Downloads all pretrained COMET models.
# 

mkdir -p ${ADAPTATION_ROOT}/models/comet/
mkdir -p ${ADAPTATION_ROOT}/models/cometinho/

wget -nc https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-qe-mqm.tar.gz -O ${ADAPTATION_ROOT}/models/comet/wmt21-qe-mqm.tar.gz
wget -nc https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-mqm.tar.gz -O ${ADAPTATION_ROOT}/models/comet/wmt21-mqm.tar.gz
wget -nc https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-cometinho-mqm.tar.gz -O ${ADAPTATION_ROOT}/models/cometinho/wmt21-mqm.tar.gz
wget -nc https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt22/eamt22-cometinho-da.tar.gz -O ${ADAPTATION_ROOT}/models/cometinho/eamt22-da.tar.gz
wget -nc https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-da.tar.gz -O ${ADAPTATION_ROOT}/models/comet/wmt21-da.tar.gz
wget -nc https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt22/eamt22-prune-comet-da.tar.gz -O ${ADAPTATION_ROOT}/models/comet/eamt22-da.tar.gz
tar --skip-old-files -xvzf ${ADAPTATION_ROOT}/models/comet/wmt21-qe-mqm.tar.gz -C ${ADAPTATION_ROOT}/models/comet/
tar --skip-old-files -xvzf ${ADAPTATION_ROOT}/models/comet/wmt21-mqm.tar.gz -C ${ADAPTATION_ROOT}/models/comet/
tar --skip-old-files -xvzf ${ADAPTATION_ROOT}/models/comet/wmt21-da.tar.gz -C ${ADAPTATION_ROOT}/models/comet/
tar --skip-old-files -xvzf ${ADAPTATION_ROOT}/models/cometinho/wmt21-mqm.tar.gz -C ${ADAPTATION_ROOT}/models/cometinho/
tar --skip-old-files -xvzf ${ADAPTATION_ROOT}/models/cometinho/eamt22-da.tar.gz -C ${ADAPTATION_ROOT}/models/cometinho/
tar --skip-old-files -xvzf ${ADAPTATION_ROOT}/models/comet/eamt22-da.tar.gz -C ${ADAPTATION_ROOT}/models/comet/
