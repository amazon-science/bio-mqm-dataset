
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


class SEScore2Metric(BaseMetric):
    def __init__(self, use_ref, **kwargs):
        super().__init__()

        self.use_ref = use_ref

        import sys
        # make sure that prism is installed there
        sys.path.append("../SEScore2")

        from SEScore2 import SEScore2
        # to satisfy unpickler
        from train.regression import Regression_XLM_Roberta

        self.model = SEScore2(kwargs["lang2"])

    def _predict_single(self, src, tgt, ref):
        output = self.model.score([ref], [tgt], 1)[0]
        return output

    # def _predict(self, src, tgt, ref):
    #     output = self.model.score(ref, tgt, 1)
    #     return output
