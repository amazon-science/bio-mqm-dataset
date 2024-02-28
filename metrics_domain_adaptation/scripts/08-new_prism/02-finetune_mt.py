
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
# Finetune NLLB using transformers.
#

import copy
from metrics_domain_adaptation import utils
import numpy as np
import json
from datasets import Dataset, DatasetDict
import evaluate
import transformers
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import NllbTokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import wandb

args = argparse.ArgumentParser()
args.add_argument("--domain", required=True)
args.add_argument("--langs", required=True)
# "facebook/nllb-200-distilled-600M"
# "facebook/nllb-200-1.3B"
# "facebook/nllb-200-3.3B"
args.add_argument("--mock", action="store_true")
args.add_argument("--batch-size", type=int, required=True)
args.add_argument("--learning-rate", type=float, default=1e-6)
args.add_argument("--model", required=True)
args.add_argument("--wandb-name", default=None)
args.add_argument("--gradient-accumulation-steps", type=int, default=1)
args.add_argument("--bf16", action="store_true")
args.add_argument(
    "--save-path",
    default='{ADAPTATION_ROOT}/models/trained/nllb/'
)
args = args.parse_args()

lang1, lang2 = args.langs.split("-")
langs12 = args.langs.replace("-", "")
lang1nllb = utils.LANG_TO_NLLB[lang1]
lang2nllb = utils.LANG_TO_NLLB[lang2]

model_name = (
    "600M" if "600m" in args.model.lower()
    else
    "1.3B" if "1.3b" in args.model.lower()
    else
    "3.3B" if "3.3b" in args.model.lower()
    else
    "opus" if "opus" in args.model.lower()
    else
    args.model.split("/")[-1]
)

wandb.init(
    project=f"nllb-finetuning",
    name=(
        f"{model_name}_{args.domain}_{langs12}" +
        f" [{args.wandb_name}]" if args.wandb_name else ""
    ),
    tags=[args.domain, langs12, model_name],
)

if model_name in {"600M", "1.3B", "3.3B"}:
    model_type = "nllb"
    tokenizer = NllbTokenizerFast.from_pretrained(
        args.model,
        src_lang=lang1nllb, tgt_lang=lang2nllb,
    )
elif model_name == "opus":
    model_type = "opus"
    # make sure we're using the correct OPUS model
    assert args.langs in args.model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
else:
    raise Exception(f"Unknown model {args.model}")

model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
if args.bf16:
    model = model.bfloat16()

dataset_train = [
    json.loads(x) for x
    in open(f'{utils.ROOT}/data/experiments/nllb/{args.domain}_{langs12}.jsonl', "r")
]
dataset_train = [
    {"src": x[lang1], "ref": x[lang2]}
    for x in dataset_train
]
dataset_train = utils.transpose_dict(dataset_train, keys=["src", "ref"])
dataset_train = Dataset.from_dict(dataset_train)

dataset_eval = utils.load_data(
    kind="mqm", domain=args.domain, langs=args.langs,
    split="test"
)[:2000 if not args.mock else 1]
dataset_eval = utils.transpose_dict(dataset_eval, keys=["src", "ref"])
dataset_eval = Dataset.from_dict(dataset_eval)

datasets_all = DatasetDict({"train": dataset_train, "test": dataset_eval})
datasets_all = datasets_all.map(
    lambda examples: tokenizer(
        text=[x for x in examples["src"]],
        text_target=[x for x in examples["ref"]],
        max_length=512,
        truncation=True,
        add_special_tokens=True,
    ),
    batched=True
)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels
    )

    result_bleu = metric_bleu.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    result_chrf = metric_chrf.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    result = {
        "bleu": result_bleu["score"],
        "chrf": result_chrf["score"],
    }

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id)
        for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


generation_config = copy.deepcopy(model.generation_config)
generation_config.forced_bos_token_id = tokenizer.lang_code_to_id[lang2nllb]
generation_config._from_model_config = False

training_args = Seq2SeqTrainingArguments(
    output_dir=(
        args.save_path.replace("{ADAPTATION_ROOT}", utils.ROOT) + "/" +
        f'{langs12}/' +
        f'{model_name}_{args.domain}' +
        '_' + args.wandb_name.replace(" ", "_").replace(",", "")
    ),
    **(
        dict(
            # only for generating phantom models
            save_strategy='steps',
            save_steps=1,
            max_steps=1,
            eval_steps=1,
        )
        if args.mock else
        dict(
            evaluation_strategy="epoch",
            save_strategy='epoch',
            logging_strategy='epoch',
        )
        if args.domain == "bio" else
        dict(
            evaluation_strategy="steps",
            save_strategy='steps',
            logging_strategy='steps',
            eval_steps=2_000,
        )
    ),

    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    auto_find_batch_size=True,
    num_train_epochs=100,
    predict_with_generate=True,
    lr_scheduler_type='cosine',
    learning_rate=args.learning_rate,
    warmup_steps=10_000,
    generation_config=generation_config,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    bf16=args.bf16,
    bf16_full_eval=args.bf16,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=datasets_all["train"],
    eval_dataset=datasets_all["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


class EvaluateFirstStepCallback(transformers.TrainerCallback):
    # hack to evaluate on the first epoch
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


trainer.add_callback(EvaluateFirstStepCallback())

trainer.train()
