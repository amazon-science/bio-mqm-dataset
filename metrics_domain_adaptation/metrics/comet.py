
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

import os
from .base import BaseMetric


class COMETMetric(BaseMetric):
    def __init__(self, model_path, **kwargs):
        super().__init__()

        from comet import load_from_checkpoint
        import torch

        # speed-up inference
        torch.set_float32_matmul_precision('medium')
        if not os.path.exists(model_path):
            from comet import download_model
            model_path = download_model(model_path)
        self.model = load_from_checkpoint(model_path)

    def _predict_single(self, src, tgt, ref):
        output = self.model.predict(
            {"src": src, "mt": tgt, "ref": ref},
            batch_size=64, gpus=1, progress_bar=False,
        )
        return output["scores"][0]

    def _predict(self, src, tgt, ref):
        output = self.model.predict([
            {"src": src, "mt": tgt, "ref": ref}
            for src, tgt, ref in zip(src, tgt, ref)
        ], batch_size=64, gpus=1, progress_bar=True,
        )
        return output["scores"]
