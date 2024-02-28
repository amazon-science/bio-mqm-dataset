
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
# Find the upsampling factors for 05/10a- and 05/10b-.
#

ORIGINAL_GENERAL = 70_000
ORIGINAL_BIO = 5_500
ORIGINAL_Ks_EXTENDED = [1, 2, 4, 6, 8, 10, 12, 14, 16]
ORIGINAL_Ks_0 = [1, 2, 4, 8, 16]
ORIGINAL_Ks_1 = [0.5, 0.75, 1, 2, 4]
COMMAND = "CUDA_VISIBLE_DEVICES={i} nohup ./metrics_domain_adaptation/scripts/05-adapt_mqm/10b-scratch_datasize_1ep.sh {count} {k} 0 &"

# baseline
print(
    COMMAND
    .replace("{i}", str(0))
    .replace("{count}", str(0))
    .replace("{k}", str(0))
    + "\n"
)

for new_bio in [5_500, 4_000, 2_000, 1_000, 500, 100, 50]:
    original_ks = ORIGINAL_Ks_EXTENDED if new_bio == 5_500 else ORIGINAL_Ks_0 if new_bio >= 2_000 else ORIGINAL_Ks_1
    for k_i, k in enumerate(original_ks):
        k_new = int(k*ORIGINAL_BIO/new_bio)
        print(
            COMMAND
            .replace("{i}", str(k_i))
            .replace("{count}", str(new_bio))
            .replace("{k}", str(k_new))
        )
    print()
