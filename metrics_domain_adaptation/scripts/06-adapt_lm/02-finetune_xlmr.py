
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
# Finetune XLMR.
# Adapted from https://huggingface.co/blog/how-to-train
#

import wandb
import torch
import random
from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from metrics_domain_adaptation import utils
import argparse
import tensorflow as tf

args = argparse.ArgumentParser()
args.add_argument("--domain", default="bio")
args.add_argument("--wandb-name", default="")
args.add_argument(
    "--file-train", default=f"{utils.ROOT}/data/experiments/lm/{{DOMAIN}}_xlm_train.txt"
)
args.add_argument(
    "--file-dev", default=f"{utils.ROOT}/data/experiments/lm/{{DOMAIN}}_xlm_test_mqm.txt"
)
args.add_argument(
    "--trainer-args", nargs="+",
    default=[
        "lr_scheduler_type='cosine'",
        "learning_rate=5e-7",
        "warmup_steps=10000"
    ]
)
args = args.parse_args()

config = XLMRobertaConfig.from_pretrained("xlm-roberta-large")
tokenizer = XLMRobertaTokenizerFast.from_pretrained(
    "xlm-roberta-large", max_len=512,
    truncation=True
)
model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-large")

dataset = load_dataset(
    "text",
    data_files={
        "train": args.file_train.replace("{DOMAIN}", args.domain),
        "dev": args.file_dev.replace("{DOMAIN}", args.domain),
    }
)

tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True),
    batched=True, num_proc=8, remove_columns=["text"],
)

# small helper that will help us batch different samples of the dataset together
# into an object that PyTorch knows how to perform backprop on.

# make sure we're on the same seed
random.seed(0)
torch.random.manual_seed(0)
tf.random.set_seed(0)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

args.wandb_name += f";{args.domain} " + ";".join(args.trainer_args)

# wandb.login()
wandb.init(
    project="xlmr-large",
    name=args.wandb_name,
)

training_args_manual = {}
for text_raw in args.trainer_args:
    key, val = text_raw.split("=")
    training_args_manual[key] = eval(val)

training_args = TrainingArguments(
    output_dir=f"{utils.ROOT}/models/trained/xlmr-large/{args.wandb_name}",
    overwrite_output_dir=True,
    num_train_epochs=2,
    auto_find_batch_size=True,
    prediction_loss_only=True,
    evaluation_strategy="steps",
    logging_steps=500,
    save_steps=20_000,
    eval_steps=10_000,
    # this is only for evaluation (I think)
    # does not matter that it makes the model worse, as long as it's consistent
    fp16_full_eval=True,
    logging_first_step=True,
    **training_args_manual,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    tokenizer=tokenizer,
)

# start with evaluation
# UPDATE: don't do this, causes troubles :(
# trainer.evaluate()
trainer.train()
