
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

from metrics_domain_adaptation import utils
from .base import BaseMetric


class PRISMModel:
    def __init__(self, lang1, lang2, device, model_name, model_type, nllb_finetuned=False):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device
        # based on this we might need to add an additional L2 token
        self.nllb_finetuned = nllb_finetuned
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # truncate to avoid OOM on long input (tested on V100 with 16Gb)
        self.max_src_len_tokens = 250
        self.max_tgt_len_tokens = 250

        self.lang1 = lang1
        self.lang2 = lang2

        # lower precision to not get OOM
        if "3.3b" in model_name.lower():
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)

        if model_type == "opus":
            self.score_w_src = self.score_w_src_opus
            self.score_w_ref = self.score_w_ref_opus
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif model_type == "nllb":
            self.score_w_src = self.score_w_src_nllb
            self.score_w_ref = self.score_w_ref_nllb

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, src_lang=utils.LANG_TO_NLLB[lang1], tgt_lang=utils.LANG_TO_NLLB[lang2]
            )
            self.bos_id = self.tokenizer.convert_tokens_to_ids('</s>')
            self.lang1nllb_id = self.tokenizer.convert_tokens_to_ids(
                utils.LANG_TO_NLLB[lang1]
            )
            self.lang2nllb_id = self.tokenizer.convert_tokens_to_ids(
                utils.LANG_TO_NLLB[lang2]
            )
        elif model_type == "m100":
            self.score_w_src = self.score_w_src_m100
            self.score_w_ref = self.score_w_ref_m100
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, tgt_lang=lang2
            )
        else:
            raise Exception("Unknown model type")

    def _score_1way_nllb(self, input_text, output_text, input_lang_id, input_lang):
        # need to import locally
        import torch
        import numpy as np

        # this shouldn't have an effect because we are not using tokenizer's special tokens but just for consistency
        self.tokenizer.src_lang = utils.LANG_TO_NLLB[input_lang]

        src_ids_list = [
            [input_lang_id, ] +
            self.tokenizer.encode(input_text, add_special_tokens=False)[:self.max_src_len_tokens] +
            [self.bos_id, ]
        ]
        src_ids = torch.tensor(src_ids_list, dtype=torch.int64)
        tgt_ids_list = [
            [self.bos_id, self.lang2nllb_id] +
            self.tokenizer.encode(output_text, add_special_tokens=False)[:self.max_tgt_len_tokens] +
            [self.bos_id, ]
        ]
        tgt_ids = torch.tensor(tgt_ids_list, dtype=torch.int64)
        logits = self.model.forward(
            input_ids=src_ids.to(self.device),
            decoder_input_ids=tgt_ids.to(self.device)
        )['logits']
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs2 = log_probs.squeeze().cpu().detach().numpy()

        # best for finetuned
        if self.nllb_finetuned:
            log_probs2 = log_probs2[0:-1, :]
            # lose nothing
            idxs = tgt_ids.flatten().detach().numpy()[1:]
        else:
            # lose the language code prediction and the garbage prediction at the end
            log_probs2 = log_probs2[1:-1, :]
            # lose </s> and language code (keep eos)
            idxs = tgt_ids.flatten().detach().numpy()[2:]
        scores = [log_probs2[ii, jj] for ii, jj in enumerate(idxs)]
        avg_score = np.mean(scores)

        return avg_score

    def _score_1way_opus(self, input_text, output_text):
        # need to import locally
        import torch
        import numpy as np

        tgt_ids_list = [
            self.tokenizer.encode(
                output_text, add_special_tokens=True
            )[:self.max_tgt_len_tokens]
        ]
        tgt_ids = torch.tensor(tgt_ids_list, dtype=torch.int64)
        src_ids_list = [
            self.tokenizer.encode(
                input_text, add_special_tokens=True
            )[:self.max_src_len_tokens]
        ]
        src_ids = torch.tensor(src_ids_list, dtype=torch.int64)
        logits = self.model.forward(
            input_ids=src_ids.to(self.device),
            decoder_input_ids=tgt_ids.to(self.device)
        )['logits']
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # [0] instead of squeeze to avoid undesired dimension collapse
        log_probs2 = log_probs[0].cpu().detach().numpy()
        idxs = tgt_ids.flatten().detach().numpy()
        scores = [log_probs2[ii, jj] for ii, jj in enumerate(idxs)]

        avg_score = np.mean(scores)

        return avg_score

    def _score_1way_m100(self, input_text, output_text):
        # need to import locally
        import torch
        import numpy as np

        tgt_ids_list = [
            self.tokenizer.encode(
                output_text, add_special_tokens=True
            )[:self.max_tgt_len_tokens]
        ]
        tgt_ids = torch.tensor(tgt_ids_list, dtype=torch.int64)
        src_ids_list = [
            self.tokenizer.encode(
                input_text, add_special_tokens=True
            )[:self.max_src_len_tokens]
        ]
        src_ids = torch.tensor(src_ids_list, dtype=torch.int64)
        logits = self.model.forward(
            input_ids=src_ids.to(self.device),
            decoder_input_ids=tgt_ids.to(self.device)
        )['logits']
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # [0] instead of squeeze to avoid undesired dimension collapse
        log_probs2 = log_probs[0].cpu().detach().numpy()
        idxs = tgt_ids.flatten().detach().numpy()
        scores = [log_probs2[ii, jj] for ii, jj in enumerate(idxs)]

        avg_score = np.mean(scores)

        return avg_score

    def score_w_src_nllb(self, src_sent, tgt_sent):
        return self._score_1way_nllb(
            input_text=src_sent, output_text=tgt_sent,
            input_lang_id=self.lang1nllb_id,
            input_lang=self.lang1,
        )

    def score_w_ref_nllb(self, ref_sent, tgt_sent):
        fwd = self._score_1way_nllb(
            input_text=ref_sent, output_text=tgt_sent,
            input_lang_id=self.lang2nllb_id,
            input_lang=self.lang2,
        )
        rev = self._score_1way_nllb(
            input_text=tgt_sent, output_text=ref_sent,
            input_lang_id=self.lang2nllb_id,
            input_lang=self.lang2,
        )
        return (fwd + rev)/2.0

    def score_w_src_m100(self, src_sent, tgt_sent):
        return self._score_1way_m100(src_sent, tgt_sent)

    def score_w_ref_m100(self, ref_sent, tgt_sent):
        fwd = self._score_1way_m100(ref_sent, tgt_sent)
        rev = self._score_1way_m100(tgt_sent, ref_sent)
        return (fwd + rev)/2.0

    def score_w_src_opus(self, src_sent, tgt_sent):
        return self._score_1way_opus(src_sent, tgt_sent)

    def score_w_ref_opus(self, ref_sent, tgt_sent):
        raise Exception("Not implemented")


class PRISM2Metric(BaseMetric):
    # lang codes from here https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

    def __init__(self, lang1, lang2, prism_mode, model_name='facebook/nllb-200-distilled-600M', **kwargs):
        super().__init__()

        # need to do imports locally
        import torch

        device = torch.device(
            'cuda'
            if torch.cuda.is_available() else
            'cpu'
        )

        assert prism_mode in {"src", "ref", "mix"}
        self.prism_mode = prism_mode

        if "opus" in model_name.lower():
            model_type = "opus"
        elif any(x in model_name.lower() for x in {"600m", "1.3b", "3.3b", "nllb"}):
            model_type = "nllb"
        elif "small100" in model_name.lower():
            model_type = "m100"
        else:
            raise Exception(
                "Unable to determine model type (opus/nllb/m100) from model name."
            )

        print(f"Using {model_type} as model type.")

        self.prism = PRISMModel(
            lang1=lang1, lang2=lang2,
            device=device,
            model_name=model_name,
            model_type=model_type,
            nllb_finetuned=not model_name.startswith("facebook/"),
        )

    def _predict_single(self, src, tgt, ref):
        if self.prism_mode == "ref":
            return self.prism.score_w_ref(ref_sent=ref, tgt_sent=tgt)
        elif self.prism_mode == "src":
            return self.prism.score_w_src(src_sent=src, tgt_sent=tgt)
        elif self.prism_mode == "mix":
            return (
                self.prism.score_w_src(src_sent=src, tgt_sent=tgt) +
                self.prism.score_w_ref(ref_sent=ref, tgt_sent=tgt)
            )
        else:
            return None
