
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
# Downloads Bio parallel data.
# Script adapted from https://github.com/biomedical-translation-corpora/corpora
#

import os
from langdetect import detect
from Bio import Entrez
import tqdm
from metrics_domain_adaptation import utils
import urllib.request
import zipfile


def get_abstract_text(record):
    all_abstracttexts = []
    try:
        texts = []
        texts.append(
            record["MedlineCitation"]["Article"]['Abstract']['AbstractText']
        )
        if 'OtherAbstract' in record["MedlineCitation"]:
            for item in record["MedlineCitation"]['OtherAbstract']:
                texts.append(item['AbstractText'])
        abstracttext = ""
        for text in texts:
            if len(text) > 1:
                abstracttext = ""
                for part in text:
                    part = part.replace('"', "'")
                    abstracttext += part + " "
            else:
                abstracttext = text[0]
                abstracttext = abstracttext.replace('"', "'")
            all_abstracttexts.append(abstracttext.strip())
    except:
        print(
            'PMID '+record["MedlineCitation"]["PMID"]+' - abstract not found!'
        )
    return all_abstracttexts


def build_article(record):
    articles = []
    langs = []
    all_abstracttexts = get_abstract_text(record)
    for index in range(0, len(all_abstracttexts)):
        article = {}
        article["pmid"] = record["MedlineCitation"]["PMID"]
        article["abstracttext"] = all_abstracttexts[index]
        # lang
        lang = detect(article["abstracttext"])
        article["lang"] = lang
        langs.append(lang)
        articles.append(article)
    return articles, langs


def fetch_pubmed_articles(ids):
    ids = ",".join(ids)
    try:
        handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
        records = Entrez.read(handle)
        set_articles = []
        set_langs = []
        for record in records["PubmedArticle"]:
            article, langs = build_article(record)
            set_articles.append(article)
            set_langs.append(langs)
        handle.close()
        return set_articles, set_langs
    except:
        print("Invalid request, skipping")
        return [], []


def fetch_multiple_articles(pmids, out_dir, lang1, lang2):
    set_articles, set_langs = fetch_pubmed_articles(pmids)
    for index in range(0, len(set_articles)):
        langs = set_langs[index]
        if len(langs) < 2 or lang1 not in langs or lang2 not in langs:
            continue
        article = set_articles[index]
        for item in article:
            lang = detect(item["abstracttext"])
            if lang != lang1 and lang != lang2:
                continue
            with open(os.path.join(out_dir, item["pmid"]+"_"+item["lang"]+".txt"), "w") as writer:
                writer.write(item["abstracttext"]+"\n")
            writer.close()


map_langs = {
    "eng": "en",
    "ita": "it",
    "chi": "zh-cn",
    "fre": "fr",
    "ger": "de",
    "por": "pt",
    "spa": "es",
    "rus": "ru"
}


def get_lang1_lang2(filename):
    lang1, lang2 = (
        filename
        .split(".")[-2]
        .split("/")[-1]
        .removeprefix("train22_")
        .split("_")
    )
    lang1 = map_langs[lang1]
    lang2 = map_langs[lang2]
    return lang1, lang2


def retrieve_abstracts(filename, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    lang1, lang2 = get_lang1_lang2(filename)
    pmids = []
    lines = list(open(os.path.join(filename), "r").readlines())
    for line in tqdm.tqdm(lines):
        pmid = line.strip()
        pmids.append(pmid)
        if len(pmids) < 100:
            continue
        fetch_multiple_articles(pmids, out_dir, lang1, lang2)
        pmids = []
    if len(pmids) > 0:
        fetch_multiple_articles(pmids, out_dir, lang1, lang2)


Entrez.email = ""

if not os.path.exists(f"{utils.ROOT}/data/raw/trainWmt22.zip"):
    print("Downloading document IDs")
    urllib.request.urlretrieve(
        "https://github.com/biomedical-translation-corpora/corpora/raw/master/trainWmt22.zip",
        f"{utils.ROOT}/data/raw/trainWmt22.zip"
    )
    with zipfile.ZipFile(f"{utils.ROOT}/data/raw/trainWmt22.zip", 'r') as f:
        f.extractall(f"{utils.ROOT}/data/raw/wmt22_biomed/")

retrieve_abstracts(
    f"{utils.ROOT}/data/raw/wmt22_biomed/train22_eng_chi.txt",
    f"{utils.ROOT}/data/raw/wmt22_biomed/en-zh"
)
retrieve_abstracts(
    f"{utils.ROOT}/data/raw/wmt22_biomed/train22_eng_ger.txt",
    f"{utils.ROOT}/data/raw/wmt22_biomed/en-de"
)
retrieve_abstracts(
    f"{utils.ROOT}/data/raw/wmt22_biomed/train22_eng_rus.txt",
    f"{utils.ROOT}/data/raw/wmt22_biomed/en-ru"
)
