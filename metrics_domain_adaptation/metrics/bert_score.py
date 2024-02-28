
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


class BERTScoreMetric(BaseMetric):
    def __init__(self, model, **kwargs):
        super().__init__()

        from bert_score import BERTScorer

        self.model_type = model
        self.lang2 = kwargs["lang2"]

        if "model_path" in kwargs:
            self.model_path = kwargs["model_path"]
        else:
            self.model_path = None

        self.scorer = BERTScorer(
            lang=self.lang2,
            model_type=self.model_type,
            model_path=self.model_path,
            batch_size=128,
        )

    def _predict_single(self, src, tgt, ref):
        output = self.scorer.score(
            [tgt], [ref],
        )[2][0]
        return output

    def _predict(self, src, tgt, ref):
        output = self.scorer.score(
            tgt, ref,
        )[2]
        return output
