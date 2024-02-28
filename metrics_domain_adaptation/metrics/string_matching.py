
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


class StringMatching(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__()
        import evaluate
        self.metric = evaluate.load(kwargs["name"])
        if "predict_kwargs" in kwargs:
            self.predict_kwargs = kwargs["predict_kwargs"]
        else:
            self.predict_kwargs = {}
        if "retrieve_key" in kwargs:
            self.retrieve_key = kwargs["retrieve_key"]
        else:
            self.retrieve_key = "score"

    def _predict_single(self, src, tgt, ref):
        output = self.metric.compute(
            predictions=[tgt], references=[[ref]],
            **self.predict_kwargs
        )
        return output[self.retrieve_key]


class StringMatchingSingle(StringMatching):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _predict_single(self, src, tgt, ref):
        try:
            output = self.metric.compute(
                # use single reference
                predictions=[tgt], references=[ref],
                **self.predict_kwargs
            )
            return output[self.retrieve_key]
        except:
            return 0
