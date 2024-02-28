
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

#
# This is the main import module for all metrics.
# Do not use this directly -- use run_metric.py instead.
# Importantly none of the following imports should have any non-standard or heavy import themselves
# because this module is loaded often but not all metrics are.
#

from .string_matching import *
from .comet import *
from .hf_eval import *
from .prism import *
from .prism2 import *
from .gemba_wrap import *
from .blemba_wrap import *
from .bart_score import *
from .bert_score import *
from .sescore2 import *
from metrics_domain_adaptation import utils

_METRICS = {
    # string matching
    "bleu": lambda **kwargs: StringMatching(name="google_bleu", retrieve_key="google_bleu", **kwargs),
    "chrf": lambda **kwargs: StringMatching(name="chrf", **kwargs),
    "ter": lambda **kwargs: StringMatching(name="ter", **kwargs),
    "character": lambda **kwargs: StringMatching(name="character", retrieve_key="cer_score", **kwargs),
    "meteor": lambda **kwargs: StringMatching(name="meteor", retrieve_key="meteor", **kwargs),
    "rougel": lambda **kwargs: StringMatching(name="rouge", retrieve_key="rougeL", **kwargs),
    "nist_mt": lambda **kwargs: StringMatching(name="nist_mt", retrieve_key="nist_mt", **kwargs),
    # others
    "unite-mup": lambda **kwargs: COMETMetric(**kwargs, model_path=f"{utils.ROOT}/models/comet/unite-mup/checkpoints/model.ckpt"),
    # model path is in kwargs here
    "comet-ours": lambda **kwargs: COMETMetric(**kwargs),
    "comet": lambda **kwargs: COMETMetric(**kwargs, model_path=f"{utils.ROOT}/models/comet/wmt21-comet-mqm/checkpoints/model.ckpt"),
    "comet-da": lambda **kwargs: COMETMetric(**kwargs, model_path=f"{utils.ROOT}/models/comet/wmt21-comet-da/checkpoints/model.ckpt"),
    "comet22-da": lambda **kwargs: COMETMetric(**kwargs, model_path="Unbabel/wmt22-comet-da"),
    "comet-qe": lambda **kwargs: COMETMetric(**kwargs, model_path=f"{utils.ROOT}/models/comet/wmt21-comet-qe-mqm/checkpoints/model.ckpt"),
    "cometinho": lambda **kwargs: COMETMetric(**kwargs, model_path=f"{utils.ROOT}/models/cometinho/wmt21-cometinho-mqm/checkpoints/model.ckpt"),
    "cometinho-da": lambda **kwargs: COMETMetric(**kwargs, model_path=f"{utils.ROOT}/models/cometinho/eamt22-cometinho-da/checkpoints/model.ckpt"),
    "bertscore-xlmr-base": lambda **kwargs: BERTScoreMetric(model="xlm-roberta-base", **kwargs),
    "bertscore-xlmr": lambda **kwargs: BERTScoreMetric(model="xlm-roberta-large", **kwargs),
    "bertscore-roberta": lambda **kwargs: BERTScoreMetric(model="roberta-large", **kwargs),
    "bertscore-deberta": lambda **kwargs: BERTScoreMetric(model="microsoft/deberta-xlarge-mnli", **kwargs),
    "bertscore-infoxlm": lambda **kwargs: BERTScoreMetric(model="microsoft/infoxlm-large", **kwargs),
    "bertscore-mt5": lambda **kwargs: BERTScoreMetric(model="google/mt5-large", **kwargs),
    "bartscore": lambda **kwargs: BARTScoreMetric(**kwargs),
    "bleurt": lambda **kwargs: BLEURTMetric(**kwargs),
    "sescore2": lambda **kwargs: SEScore2Metric(use_ref=True, **kwargs),
    "prism-ref": lambda **kwargs: PRISMMetric(use_ref=True, **kwargs),
    "prism-src": lambda **kwargs: PRISMMetric(use_ref=False, **kwargs),
    "prism2-src": lambda **kwargs: PRISM2Metric(prism_mode="src", **kwargs),
    "prism2-ref": lambda **kwargs: PRISM2Metric(prism_mode="ref", **kwargs),
    "prism2-mix": lambda **kwargs: PRISM2Metric(prism_mode="mix", **kwargs),
    # prompt-based
    "blemba": lambda **kwargs: BLEMBAMetric(**kwargs),
    "gemba-dav003": lambda **kwargs: GEMBAMetric(signature="text-davinci-003---SQM_ref", **kwargs),
    "gemba-qe-dav003": lambda **kwargs: GEMBAMetric(signature="text-davinci-003---SQM", **kwargs),
    "gemba-gpt4": lambda **kwargs: GEMBAMetric(signature="gpt-4---SQM_ref", **kwargs),
    "gemba-qe-gpt4": lambda **kwargs: GEMBAMetric(signature="gpt-4---SQM", **kwargs),
}


def get(name, **kwargs):
    if name in _METRICS:
        return _METRICS[name](**kwargs)
    else:
        raise Exception(f"Unknown metric {name}")
