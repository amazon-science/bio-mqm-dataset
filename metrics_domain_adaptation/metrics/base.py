
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

import tqdm


class BaseMetric():
    def predict(self, src, tgt, ref=None):
        """
        Wrapper for individual and batched queries.
        """
        if type(src) is str and type(tgt) is str:
            return self._predict([src], [tgt], [ref])
        else:
            return self._predict(src, tgt, ref)

    def _predict(self, src, tgt, ref):
        """
        Unless overriden, simply call _predict_single for each triplet
        """
        scores = []
        for src_line, tgt_line, ref_line in tqdm.tqdm(
            zip(src, tgt, ref+[None]*(len(tgt)-len(ref))),
            total=len(tgt)
        ):
            result = self._predict_single(src_line, tgt_line, ref_line)
            scores.append(result)

        return scores

    def _predict_single(self, src, tgt, ref):
        raise Exception("Method not implemented")
