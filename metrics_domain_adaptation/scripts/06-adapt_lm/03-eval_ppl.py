
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

#!/usr/bin/env python3

#
# Evaluate perplexity of a finetuned LM.
#

import argparse
import json
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizerFast
from datasets import load_dataset
from metrics_domain_adaptation import utils
import random
import torch
import tensorflow as tf
import wandb

wandb.init(
    mode="disabled"
)

args = argparse.ArgumentParser()
args.add_argument("--model", default="xlm-roberta-large")
args = args.parse_args()

tokenizer = XLMRobertaTokenizerFast.from_pretrained(
    "xlm-roberta-large",
    max_len=512, truncation=True
)
model = XLMRobertaForMaskedLM.from_pretrained(args.model)


dataset = load_dataset(
    "text",
    data_files={
        "bio_mlm": f"{utils.ROOT}/data/experiments/lm/bio_mlm_test_mqm.txt",
        "bio_tlm": f"{utils.ROOT}/data/experiments/lm/bio_tlm_test_mqm.txt",
        "general_mlm": f"{utils.ROOT}/data/experiments/lm/general_mlm_test_mqm.txt",
        "general_tlm": f"{utils.ROOT}/data/experiments/lm/general_tlm_test_mqm.txt",
    }
)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, num_proc=8, remove_columns=["text"])

random.seed(0)
torch.random.manual_seed(0)
tf.random.set_seed(0)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="tmp",
        prediction_loss_only=True,
        fp16_full_eval=True,
        auto_find_batch_size=True,
    ),
    data_collator=data_collator,
)

for domain in ["bio", "general"]:
    for xlm_type in ["mlm", "tlm"]:
        loss = trainer.evaluate(
            eval_dataset=tokenized_dataset[domain + "_" + xlm_type]
        )
        print("JSON!" + json.dumps({
            "domain": domain, "xlm_type": xlm_type, "model": args.model,
            "loss": loss["eval_loss"], "ppl": 2**loss["eval_loss"],
        }))
