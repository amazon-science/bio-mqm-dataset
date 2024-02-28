
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


class BLEURTMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__()

        from evaluate import load
        self.model = load("bleurt")

    def _predict_single(self, src, tgt, ref):
        return self.model.compute(predictions=[tgt], references=[ref])["scores"][0]

    def _predict(self, src, tgt, ref):
        return self.model.compute(predictions=tgt, references=ref)["scores"]
