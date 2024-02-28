
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


class PRISMMetric(BaseMetric):
    def __init__(self, use_ref, **kwargs):
        super().__init__()

        self.use_ref = use_ref

        import sys
        import os

        if not os.path.exists("../prism"):
            raise Exception(
                "'../prism' should contain the old "
                "[PRISM code](https://github.com/thompsonb/prism)"
            )

        # make sure that prism is installed there
        sys.path.append("../prism")

        from prism import Prism
        self.model = Prism(model_dir="../prism/m39v1", lang=kwargs["lang2"])

    def _predict_single(self, src, tgt, ref):
        if self.use_ref:
            return self.model.score(cand=[tgt], ref=[ref], segment_scores=True)[0]
        else:
            return self.model.score(cand=[tgt], src=[src], segment_scores=True)[0]

    def _predict(self, src, tgt, ref):
        if self.use_ref:
            return self.model.score(cand=tgt, ref=ref, segment_scores=True)
        else:
            return self.model.score(cand=tgt, src=src, segment_scores=True)
