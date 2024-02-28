
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

from .base import BaseMetric
from metrics_domain_adaptation import utils


class BLEMBAMetric(BaseMetric):
    def __init__(self, mode, **kwargs):
        super().__init__()
        import json
        import os

        fname = f"{utils.ROOT}/data/computed/blemba/{mode}/{kwargs['domain']}/{kwargs['lang1']}-{kwargs['lang2']}.jsonl"
        if not os.path.exists(fname):
            fname = f"computed/blemba/{mode}/{kwargs['domain']}/{kwargs['lang1']}-{kwargs['lang2']}.jsonl"

        self.data = [json.loads(x) for x in open(fname, "r")]
        self.data = {
            (x["src"], x["tgt"], x["ref"]): x["blemba_score"]
            # reverse so that accidental duplicity doesn't override the score with None
            for x in self.data[::-1]
            if "blemba_score" in x and x["blemba_score"] is not None
        }

    def _predict_single(self, src, tgt, ref):
        if (src, tgt, ref) not in self.data:
            return 0
            raise Exception(
                f"Encountered example which is not in the data: {src}\n{tgt}\n{ref}"
            )
        else:
            return self.data[(src, tgt, ref)]
