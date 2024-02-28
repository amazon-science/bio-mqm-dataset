
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

from metrics_domain_adaptation import utils
import numpy as np
import json
from datasets import Dataset
import evaluate
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import wandb

args = argparse.ArgumentParser()
args.add_argument("--domain", required=True)
args.add_argument("--langs", required=True)
args.add_argument("--model", required=True)
args.add_argument("--batch-size", type=int, default=16)
args.add_argument("--fp16", action="store_true")
args = args.parse_args()

lang1, lang2 = args.langs.split("-")
langs12 = args.langs.replace("-", "")
lang1nllb = utils.LANG_TO_NLLB[lang1]
lang2nllb = utils.LANG_TO_NLLB[lang2]

if args.langs == "de-en":
    rev_loader = True
elif args.langs == "ru-en":
    rev_loader = True
elif args.langs == "en-zh":
    rev_loader = True
elif args.langs in {"en-de", "en-ru", "zh-en"}:
    rev_loader = False
else:
    raise Exception("Unknown language pair")

model_name = (
    "600M" if "600M" in args.model
    else
    "1.3B" if "1.3B" in args.model
    else
    "3.3B" if "3.3B" in args.model
    else
    "opus" if args.model.startswith("Helsinki-NLP")
    else
    "opus" if "opus" in args.model
    else
    args.model.split("/")[-1]
)

wandb.init(
    mode="disabled"
)

if model_name in {"600M", "1.3B", "3.3B"}:
    model_type = "nllb"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        src_lang=lang1nllb, tgt_lang=lang2nllb,
    )
elif model_name == "opus":
    model_type = "opus"
    # make sure we're using the correct OPUS model
    assert args.langs in args.model or langs12 in args.model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )
else:
    raise Exception(f"Unknown model type {args.model}")

model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

# half precision for the big model
if args.fp16:
    model = model.half()

dataset = utils.load_data(
    kind="mqm", domain=args.domain, langs=args.langs,
    split="test"
)
dataset = utils.transpose_dict(dataset, keys=["src", "ref"])
dataset = Dataset.from_dict(dataset)
dataset = dataset.map(
    lambda examples: tokenizer(
        text=examples["src"],
        text_target=examples["ref"],
        max_length=512,
        truncation=True,
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


training_args = Seq2SeqTrainingArguments(
    output_dir="tmp",
    auto_find_batch_size=True,
    per_device_eval_batch_size=args.batch_size,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if model_type == "nllb":
    output = trainer.evaluate(
        eval_dataset=dataset,
        # changing the forced bos token with generation_config didn't work
        forced_bos_token_id=tokenizer.lang_code_to_id[lang2nllb]
    )
else:
    output = trainer.evaluate(
        eval_dataset=dataset,
    )

output.pop("eval_samples_per_second")
output.pop("eval_steps_per_second")
output["langs"] = args.langs
output["domain"] = args.domain
output["model"] = args.model
print("JSON!" + json.dumps(output, ensure_ascii=False))
