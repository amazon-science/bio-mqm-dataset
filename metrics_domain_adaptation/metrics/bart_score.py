
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
from typing import List


class BARTScoreMetric(BaseMetric):
    import torch

    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        from transformers import BartTokenizer, BartForConditionalGeneration
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss used for scoring
        self.loss_fct = self.torch.nn.NLLLoss(
            reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = self.torch.nn.LogSoftmax(dim=1)

    def score(self, srcs, tgts, batch_size=8) -> List[int]:
        """ Score a batch of examples """

        import tqdm
        score_list = []
        for i in tqdm.tqdm(range(0, len(srcs), batch_size)):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            with self.torch.no_grad():
                encoded_src = self.tokenizer(
                    src_list,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                encoded_tgt = self.tokenizer(
                    tgt_list,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                src_tokens = encoded_src['input_ids'].to(self.device)
                src_mask = encoded_src['attention_mask'].to(self.device)

                tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                tgt_mask = encoded_tgt['attention_mask']
                tgt_len = tgt_mask.sum(dim=1).to(self.device)

                output = self.model(
                    input_ids=src_tokens,
                    attention_mask=src_mask,
                    labels=tgt_tokens
                )
                logits = output.logits.view(-1, self.model.config.vocab_size)
                loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                loss = loss.view(tgt_tokens.shape[0], -1)
                loss = loss.sum(dim=1) / tgt_len
                curr_score_list = [-x.item() for x in loss]
                score_list += curr_score_list
        return score_list

    def _predict_single(self, src, tgt, ref):
        return self.model.score(srcs=[tgt], tgts=[ref])[0]

    def _predict(self, src, tgt, ref):
        return self.model.score(srcs=tgt, tgts=ref)
