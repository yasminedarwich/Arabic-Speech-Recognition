import streamlit as st
from functools import lru_cache

import json
import logging
import os
from functools import lru_cache
from typing import List
from urllib.parse import unquote

import more_itertools
import pandas as pd
import requests
import streamlit as st
import wikipedia
from codetiming import Timer
from fuzzysearch import find_near_matches
from googleapi import google
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline,
    set_seed,
)

from .modeling_gpt2 import GPT2LMHeadModel as GROVERLMHeadModel
from .preprocess import ArabertPreprocessor
from .sa_utils import *
from .utils import download_models, softmax

logger = logging.getLogger(__name__)
# Taken and Modified from https://huggingface.co/spaces/flax-community/chef-transformer/blob/main/app.py
class TextGeneration:
    def __init__(self):
        self.debug = False
        self.generation_pipline = {}
        self.preprocessor = ArabertPreprocessor(model_name="aragpt2-mega")
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "aubmindlab/aragpt2-mega", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.API_KEY = os.getenv("API_KEY")
        self.headers = {"Authorization": f"Bearer {self.API_KEY}"}
        # self.model_names_or_paths = {
        #     "aragpt2-medium": "D:/ML/Models/aragpt2-medium",
        #     "aragpt2-base": "D:/ML/Models/aragpt2-base",
        # }
        self.model_names_or_paths = {
            # "aragpt2-medium": "aubmindlab/aragpt2-medium",
            "aragpt2-base": "aubmindlab/aragpt2-base",
            # "aragpt2-large": "aubmindlab/aragpt2-large",
            "aragpt2-mega": "aubmindlab/aragpt2-mega",
        }
        set_seed(42)

    def load_pipeline(self):
        for model_name, model_path in self.model_names_or_paths.items():
            if "base" in model_name or "medium" in model_name:
                self.generation_pipline[model_name] = pipeline(
                    "text-generation",
                    model=GPT2LMHeadModel.from_pretrained(model_path),
                    tokenizer=self.tokenizer,
                    device=-1,
                )
            else:
                self.generation_pipline[model_name] = pipeline(
                    "text-generation",
                    model=GROVERLMHeadModel.from_pretrained(model_path),
                    tokenizer=self.tokenizer,
                    device=-1,
                )

    def load(self):
        if not self.debug:
            self.load_pipeline()

    def generate(
        self,
        model_name,
        prompt,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        do_sample: bool,
        num_beams: int,
    ):
        logger.info(f"Generating with {model_name}")
        prompt = self.preprocessor.preprocess(prompt)
        return_full_text = False
        return_text = True
        num_return_sequences = 1
        pad_token_id = 0
        eos_token_id = 0
        input_tok = self.tokenizer.tokenize(prompt)
        max_length = len(input_tok) + max_new_tokens
        if max_length > 1024:
            max_length = 1024
        if not self.debug:
            generated_text = self.generation_pipline[model_name.lower()](
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_full_text=return_full_text,
                return_text=return_text,
                do_sample=do_sample,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            )[0]["generated_text"]
        else:
            generated_text = self.generate_by_query(
                prompt,
                model_name,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_full_text=return_full_text,
                return_text=return_text,
                do_sample=do_sample,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            )
            # print(generated_text)
            if isinstance(generated_text, dict):
                if "error" in generated_text:
                    if "is currently loading" in generated_text["error"]:
                        return f"Model is currently loading, estimated time is {generated_text['estimated_time']}"
                    return generated_text["error"]
                else:
                    return "Something happened 🤷‍♂️!!"
            else:
                generated_text = generated_text[0]["generated_text"]

        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated text: {generated_text}")
        return self.preprocessor.unpreprocess(generated_text)

    def query(self, payload, model_name):
        data = json.dumps(payload)
        url = (
            "https://api-inference.huggingface.co/models/aubmindlab/"
            + model_name.lower()
        )
        response = requests.request("POST", url, headers=self.headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    def generate_by_query(
        self,
        prompt: str,
        model_name: str,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        pad_token_id: int,
        eos_token_id: int,
        return_full_text: int,
        return_text: int,
        do_sample: bool,
        num_beams: int,
        num_return_sequences: int,
    ):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length ": max_length,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "return_full_text": return_full_text,
                "return_text": return_text,
                "pad_token_id": pad_token_id,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "num_return_sequences": num_return_sequences,
            },
            "options": {
                "use_cache": True,
            },
        }
        return self.query(payload, model_name)


class SentimentAnalyzer:
    def __init__(self):
        self.sa_models = [
            "sa_trial5_1",
            # "sa_no_aoa_in_neutral",
            # "sa_cnnbert",
            # "sa_sarcasm",
            # "sar_trial10",
            # "sa_no_AOA",
        ]
        download_models(self.sa_models)
        # fmt: off
        self.processors = {
            "sa_trial5_1": Trial5ArabicPreprocessor(model_name='UBC-NLP/MARBERT'),
            # "sa_no_aoa_in_neutral": NewArabicPreprocessorBalanced(model_name='UBC-NLP/MARBERT'),
            # "sa_cnnbert": CNNMarbertArabicPreprocessor(model_name='UBC-NLP/MARBERT'),
            # "sa_sarcasm": SarcasmArabicPreprocessor(model_name='UBC-NLP/MARBERT'),
            # "sar_trial10": SarcasmArabicPreprocessor(model_name='UBC-NLP/MARBERT'),
            # "sa_no_AOA": NewArabicPreprocessorBalanced(model_name='UBC-NLP/MARBERT'),
        }

        self.pipelines = {
            "sa_trial5_1": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sa_trial5_1",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_trial5_1")],
            # "sa_no_aoa_in_neutral": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sa_no_aoa_in_neutral",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_no_aoa_in_neutral")],
            # "sa_cnnbert": [CNNTextClassificationPipeline("{}/train_{}/best_model".format("sa_cnnbert",i), device=-1, return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_cnnbert")],
            # "sa_sarcasm": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sa_sarcasm",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_sarcasm")],
            # "sar_trial10": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sar_trial10",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sar_trial10")],
            # "sa_no_AOA": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sa_no_AOA",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_no_AOA")],
        }
        # fmt: on

    def get_preds_from_sarcasm(self, texts):
        prep = self.processors["sar_trial10"]
        prep_texts = [prep.preprocess(x) for x in texts]

        preds_df = pd.DataFrame([])
        for i in range(0, 5):
            preds = []
            for s in more_itertools.chunked(list(prep_texts), 128):
                preds.extend(self.pipelines["sar_trial10"][i](s))
            preds_df[f"model_{i}"] = preds

        final_labels = []
        final_scores = []
        for id, row in preds_df.iterrows():
            pos_total = 0
            neu_total = 0
            for pred in row[:]:
                pos_total += pred[0]["score"]
                neu_total += pred[1]["score"]

            pos_avg = pos_total / len(row[:])
            neu_avg = neu_total / len(row[:])

            final_labels.append(
                self.pipelines["sar_trial10"][0].model.config.id2label[
                    np.argmax([pos_avg, neu_avg])
                ]
            )
            final_scores.append(np.max([pos_avg, neu_avg]))

        return final_labels, final_scores

    def get_preds_from_a_model(self, texts: List[str], model_name):
        try:
            prep = self.processors[model_name]

            prep_texts = [prep.preprocess(x) for x in texts]
            if model_name == "sa_sarcasm":
                sarcasm_label, _ = self.get_preds_from_sarcasm(texts)
                sarcastic_map = {"Not_Sarcastic": "غير ساخر", "Sarcastic": "ساخر"}
                labeled_prep_texts = []
                for t, l in zip(prep_texts, sarcasm_label):
                    labeled_prep_texts.append(sarcastic_map[l] + " [SEP] " + t)

            preds_df = pd.DataFrame([])
            for i in range(0, 5):
                preds = []
                for s in more_itertools.chunked(list(prep_texts), 128):
                    preds.extend(self.pipelines[model_name][i](s))
                preds_df[f"model_{i}"] = preds

            final_labels = []
            final_scores = []
            final_scores_list = []
            for id, row in preds_df.iterrows():
                pos_total = 0
                neg_total = 0
                neu_total = 0
                for pred in row[2:]:
                    pos_total += pred[0]["score"]
                    neu_total += pred[1]["score"]
                    neg_total += pred[2]["score"]

                pos_avg = pos_total / 5
                neu_avg = neu_total / 5
                neg_avg = neg_total / 5

                if model_name == "sa_no_aoa_in_neutral":
                    final_labels.append(
                        self.pipelines[model_name][0].model.config.id2label[
                            np.argmax([neu_avg, neg_avg, pos_avg])
                        ]
                    )
                else:
                    final_labels.append(
                        self.pipelines[model_name][0].model.config.id2label[
                            np.argmax([pos_avg, neu_avg, neg_avg])
                        ]
                    )
                final_scores.append(np.max([pos_avg, neu_avg, neg_avg]))
                final_scores_list.append((pos_avg, neu_avg, neg_avg))
        except RuntimeError as e:
            if model_name == "sa_cnnbert":
                return (
                    ["Neutral"] * len(texts),
                    [0.0] * len(texts),
                    [(0.0, 0.0, 0.0)] * len(texts),
                )
            else:
                raise RuntimeError(e)
        return final_labels, final_scores, final_scores_list

    def predict(self, texts: List[str]):
        logger.info(f"Predicting for: {texts}")
        # (
        #     new_balanced_label,
        #     new_balanced_score,
        #     new_balanced_score_list,
        # ) = self.get_preds_from_a_model(texts, "sa_no_aoa_in_neutral")
        # (
        #     cnn_marbert_label,
        #     cnn_marbert_score,
        #     cnn_marbert_score_list,
        # ) = self.get_preds_from_a_model(texts, "sa_cnnbert")
        trial5_label, trial5_score, trial5_score_list = self.get_preds_from_a_model(
            texts, "sa_trial5_1"
        )
        # no_aoa_label, no_aoa_score, no_aoa_score_list = self.get_preds_from_a_model(
        #     texts, "sa_no_AOA"
        # )
        # sarcasm_label, sarcasm_score, sarcasm_score_list = self.get_preds_from_a_model(
        #     texts, "sa_sarcasm"
        # )

        id_label_map = {0: "Positive", 1: "Neutral", 2: "Negative"}

        final_ensemble_prediction = []
        final_ensemble_score = []
        final_ensemble_all_score = []
        for entry in zip(
            # new_balanced_score_list,
            # cnn_marbert_score_list,
            trial5_score_list,
            # no_aoa_score_list,
            # sarcasm_score_list,
        ):
            pos_score = 0
            neu_score = 0
            neg_score = 0
            for s in entry:
                pos_score += s[0] * 1.57
                neu_score += s[1] * 0.98
                neg_score += s[2] * 0.93

                # weighted 2
                # pos_score += s[0]*1.67
                # neu_score += s[1]
                # neg_score += s[2]*0.95

            final_ensemble_prediction.append(
                id_label_map[np.argmax([pos_score, neu_score, neg_score])]
            )
            final_ensemble_score.append(np.max([pos_score, neu_score, neg_score]))
            final_ensemble_all_score.append(
                softmax(np.array([pos_score, neu_score, neg_score])).tolist()
            )

        logger.info(f"Result: {final_ensemble_prediction}")
        logger.info(f"Score: {final_ensemble_score}")
        logger.info(f"All Scores: {final_ensemble_all_score}")
        return final_ensemble_prediction, final_ensemble_score, final_ensemble_all_score


wikipedia.set_lang("ar")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

preprocessor = ArabertPreprocessor("wissamantoun/araelectra-base-artydiqa")
logger.info("Loading QA Pipeline...")
tokenizer = AutoTokenizer.from_pretrained("wissamantoun/araelectra-base-artydiqa")
qa_pipe = pipeline("question-answering", model="wissamantoun/araelectra-base-artydiqa")
logger.info("Finished loading QA Pipeline...")


@lru_cache(maxsize=100)
def get_qa_answers(question):
    logger.info("\n=================================================================")
    logger.info(f"Question: {question}")

    if "وسام أنطون" in question or "wissam antoun" in question.lower():
        return {
            "title": "Creator",
            "results": [
                {
                    "score": 1.0,
                    "new_start": 0,
                    "new_end": 12,
                    "new_answer": "My Creator 😜",
                    "original": "My Creator 😜",
                    "link": "https://github.com/WissamAntoun/",
                }
            ],
        }
    search_timer = Timer(
        "search and wiki", text="Search and Wikipedia Time: {:.2f}", logger=logging.info
    )
    try:
        search_timer.start()
        search_results = google.search(
            question + " site:ar.wikipedia.org", lang="ar", area="ar"
        )
        if len(search_results) == 0:
            return {}

        page_name = search_results[0].link.split("wiki/")[-1]
        wiki_page = wikipedia.page(unquote(page_name))
        wiki_page_content = wiki_page.content
        search_timer.stop()
    except:
        return {}

    sections = []
    for section in re.split("== .+ ==[^=]", wiki_page_content):
        if not section.isspace():
            prep_section = tokenizer.tokenize(preprocessor.preprocess(section))
            if len(prep_section) > 500:
                subsections = []
                for subsection in re.split("=== .+ ===", section):
                    if subsection.isspace():
                        continue
                    prep_subsection = tokenizer.tokenize(
                        preprocessor.preprocess(subsection)
                    )
                    subsections.append(subsection)
                    # logger.info(f"Subsection found with length: {len(prep_subsection)}")
                sections.extend(subsections)
            else:
                # logger.info(f"Regular Section with length: {len(prep_section)}")
                sections.append(section)

    full_len_sections = []
    temp_section = ""
    for section in sections:
        if (
            len(tokenizer.tokenize(preprocessor.preprocess(temp_section)))
            + len(tokenizer.tokenize(preprocessor.preprocess(section)))
            > 384
        ):
            if temp_section == "":
                temp_section = section
                continue
            full_len_sections.append(temp_section)
            # logger.info(
            #     f"full section length: {len(tokenizer.tokenize(preprocessor.preprocess(temp_section)))}"
            # )
            temp_section = ""
        else:
            temp_section += " " + section + " "
    if temp_section != "":
        full_len_sections.append(temp_section)

    reader_time = Timer("electra", text="Reader Time: {:.2f}", logger=logging.info)
    reader_time.start()
    results = qa_pipe(
        question=[preprocessor.preprocess(question)] * len(full_len_sections),
        context=[preprocessor.preprocess(x) for x in full_len_sections],
    )

    if not isinstance(results, list):
        results = [results]

    logger.info(f"Wiki Title: {unquote(page_name)}")
    logger.info(f"Total Sections: {len(sections)}")
    logger.info(f"Total Full Sections: {len(full_len_sections)}")

    for result, section in zip(results, full_len_sections):
        result["original"] = section
        answer_match = find_near_matches(
            " " + preprocessor.unpreprocess(result["answer"]) + " ",
            result["original"],
            max_l_dist=min(5, len(preprocessor.unpreprocess(result["answer"])) // 2),
            max_deletions=0,
        )
        try:
            result["new_start"] = answer_match[0].start
            result["new_end"] = answer_match[0].end
            result["new_answer"] = answer_match[0].matched
            result["link"] = (
                search_results[0].link + "#:~:text=" + result["new_answer"].strip()
            )
        except:
            result["new_start"] = result["start"]
            result["new_end"] = result["end"]
            result["new_answer"] = result["answer"]
            result["original"] = preprocessor.preprocess(result["original"])
            result["link"] = search_results[0].link
        logger.info(f"Answers: {preprocessor.preprocess(result['new_answer'])}")

    sorted_results = sorted(results, reverse=True, key=lambda x: x["score"])

    return_dict = {}
    return_dict["title"] = unquote(page_name)
    return_dict["results"] = sorted_results

    reader_time.stop()
    logger.info(f"Total time spent: {reader_time.last + search_timer.last}")
    return return_dict


import re
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from fuzzysearch import find_near_matches
from pyarabic import araby
from torch import nn
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, pipeline
from transformers.modeling_outputs import SequenceClassifierOutput

from .preprocess import ArabertPreprocessor, url_regexes, user_mention_regex

multiple_char_pattern = re.compile(r"(.)\1{2,}", re.DOTALL)

# ASAD-NEW_AraBERT_PREP-Balanced
class NewArabicPreprocessorBalanced(ArabertPreprocessor):
    def __init__(
        self,
        model_name: str,
        keep_emojis: bool = False,
        remove_html_markup: bool = True,
        replace_urls_emails_mentions: bool = True,
        strip_tashkeel: bool = True,
        strip_tatweel: bool = True,
        insert_white_spaces: bool = True,
        remove_non_digit_repetition: bool = True,
        replace_slash_with_dash: bool = None,
        map_hindi_numbers_to_arabic: bool = None,
        apply_farasa_segmentation: bool = None,
    ):
        if "UBC-NLP" in model_name or "CAMeL-Lab" in model_name:
            keep_emojis = True
            remove_non_digit_repetition = True
        super().__init__(
            model_name=model_name,
            keep_emojis=keep_emojis,
            remove_html_markup=remove_html_markup,
            replace_urls_emails_mentions=replace_urls_emails_mentions,
            strip_tashkeel=strip_tashkeel,
            strip_tatweel=strip_tatweel,
            insert_white_spaces=insert_white_spaces,
            remove_non_digit_repetition=remove_non_digit_repetition,
            replace_slash_with_dash=replace_slash_with_dash,
            map_hindi_numbers_to_arabic=map_hindi_numbers_to_arabic,
            apply_farasa_segmentation=apply_farasa_segmentation,
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = text.replace("\\r", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        text = re.sub("(URL\s*)+", " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        text = re.sub("(USER\s*)+", " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH ")
        text = text.replace("_", " ")
        text = " ".join(text.split())
        # text = re.sub("\B\\[Uu]\w+", "", text)
        text = text.replace("\\U0001f97a", "🥺")
        text = text.replace("\\U0001f928", "🤨")
        text = text.replace("\\U0001f9d8", "😀")
        text = text.replace("\\U0001f975", "😥")
        text = text.replace("\\U0001f92f", "😲")
        text = text.replace("\\U0001f92d", "🤭")
        text = text.replace("\\U0001f9d1", "😐")
        text = text.replace("\\U000e0067", "")
        text = text.replace("\\U000e006e", "")
        text = text.replace("\\U0001f90d", "♥")
        text = text.replace("\\U0001f973", "🎉")
        text = text.replace("\\U0001fa79", "")
        text = text.replace("\\U0001f92b", "🤐")
        text = text.replace("\\U0001f9da", "🦋")
        text = text.replace("\\U0001f90e", "♥")
        text = text.replace("\\U0001f9d0", "🧐")
        text = text.replace("\\U0001f9cf", "")
        text = text.replace("\\U0001f92c", "😠")
        text = text.replace("\\U0001f9f8", "😸")
        text = text.replace("\\U0001f9b6", "💩")
        text = text.replace("\\U0001f932", "🤲")
        text = text.replace("\\U0001f9e1", "🧡")
        text = text.replace("\\U0001f974", "☹")
        text = text.replace("\\U0001f91f", "")
        text = text.replace("\\U0001f9fb", "💩")
        text = text.replace("\\U0001f92a", "🤪")
        text = text.replace("\\U0001f9fc", "")
        text = text.replace("\\U000e0065", "")
        text = text.replace("\\U0001f92e", "💩")
        text = text.replace("\\U000e007f", "")
        text = text.replace("\\U0001f970", "🥰")
        text = text.replace("\\U0001f929", "🤩")
        text = text.replace("\\U0001f6f9", "")
        text = text.replace("🤍", "♥")
        text = text.replace("🦠", "😷")
        text = text.replace("🤢", "مقرف")
        text = text.replace("🤮", "مقرف")
        text = text.replace("🕠", "⌚")
        text = text.replace("🤬", "😠")
        text = text.replace("🤧", "😷")
        text = text.replace("🥳", "🎉")
        text = text.replace("🥵", "🔥")
        text = text.replace("🥴", "☹")
        text = text.replace("🤫", "🤐")
        text = text.replace("🤥", "كذاب")
        text = text.replace("\\u200d", " ")
        text = text.replace("u200d", " ")
        text = text.replace("\\u200c", " ")
        text = text.replace("u200c", " ")
        text = text.replace('"', "'")
        text = text.replace("\\xa0", "")
        text = text.replace("\\u2066", " ")
        text = re.sub("\B\\\[Uu]\w+", "", text)
        text = super(NewArabicPreprocessorBalanced, self).preprocess(text)

        text = " ".join(text.split())
        return text


"""CNNMarbertArabicPreprocessor"""
# ASAD-CNN_MARBERT
class CNNMarbertArabicPreprocessor(ArabertPreprocessor):
    def __init__(
        self,
        model_name,
        keep_emojis=False,
        remove_html_markup=True,
        replace_urls_emails_mentions=True,
        remove_elongations=True,
    ):
        if "UBC-NLP" in model_name or "CAMeL-Lab" in model_name:
            keep_emojis = True
            remove_elongations = False
        super().__init__(
            model_name,
            keep_emojis,
            remove_html_markup,
            replace_urls_emails_mentions,
            remove_elongations,
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        text = re.sub("(URL\s*)+", " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        text = re.sub("(USER\s*)+", " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH ")
        text = text.replace("_", " ")
        text = " ".join(text.split())
        text = super(CNNMarbertArabicPreprocessor, self).preprocess(text)
        text = text.replace("\u200d", " ")
        text = text.replace("u200d", " ")
        text = text.replace("\u200c", " ")
        text = text.replace("u200c", " ")
        text = text.replace('"', "'")
        # text = re.sub('[\d\.]+', ' NUM ', text)
        # text = re.sub('(NUM\s*)+', ' NUM ', text)
        text = multiple_char_pattern.sub(r"\1\1", text)
        text = " ".join(text.split())
        return text


"""Trial5ArabicPreprocessor"""


class Trial5ArabicPreprocessor(ArabertPreprocessor):
    def __init__(
        self,
        model_name,
        keep_emojis=False,
        remove_html_markup=True,
        replace_urls_emails_mentions=True,
    ):
        if "UBC-NLP" in model_name:
            keep_emojis = True
        super().__init__(
            model_name, keep_emojis, remove_html_markup, replace_urls_emails_mentions
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH TAG ")
        text = text.replace("_", " ")
        text = " ".join(text.split())
        text = super(Trial5ArabicPreprocessor, self).preprocess(text)
        # text = text.replace("السلام عليكم"," ")
        # text = text.replace(find_near_matches("السلام عليكم",text,max_deletions=3,max_l_dist=3)[0].matched," ")
        return text


"""SarcasmArabicPreprocessor"""


class SarcasmArabicPreprocessor(ArabertPreprocessor):
    def __init__(
        self,
        model_name,
        keep_emojis=False,
        remove_html_markup=True,
        replace_urls_emails_mentions=True,
    ):
        if "UBC-NLP" in model_name:
            keep_emojis = True
        super().__init__(
            model_name, keep_emojis, remove_html_markup, replace_urls_emails_mentions
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)
        else:
            return super(SarcasmArabicPreprocessor, self).preprocess(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH TAG ")
        text = text.replace("_", " ")
        text = text.replace('"', " ")
        text = " ".join(text.split())
        text = super(SarcasmArabicPreprocessor, self).preprocess(text)
        return text


"""NoAOAArabicPreprocessor"""


class NoAOAArabicPreprocessor(ArabertPreprocessor):
    def __init__(
        self,
        model_name,
        keep_emojis=False,
        remove_html_markup=True,
        replace_urls_emails_mentions=True,
    ):
        if "UBC-NLP" in model_name:
            keep_emojis = True
        super().__init__(
            model_name, keep_emojis, remove_html_markup, replace_urls_emails_mentions
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)
        else:
            return super(NoAOAArabicPreprocessor, self).preprocess(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH TAG ")
        text = text.replace("_", " ")
        text = " ".join(text.split())
        text = super(NoAOAArabicPreprocessor, self).preprocess(text)
        text = text.replace("السلام عليكم", " ")
        text = text.replace("ورحمة الله وبركاته", " ")
        matched = find_near_matches("السلام عليكم", text, max_deletions=3, max_l_dist=3)
        if len(matched) > 0:
            text = text.replace(matched[0].matched, " ")
        matched = find_near_matches(
            "ورحمة الله وبركاته", text, max_deletions=3, max_l_dist=3
        )
        if len(matched) > 0:
            text = text.replace(matched[0].matched, " ")
        return text


class CnnBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        filter_sizes = [1, 2, 3, 4, 5]
        num_filters = 32
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(4, num_filters, (K, config.hidden_size)) for K in filter_sizes]
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        x = outputs[2][-4:]

        x = torch.stack(x, dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=outputs.attentions,
        )


class CNNTextClassificationPipeline:
    def __init__(self, model_path, device, return_all_scores=False):
        self.model_path = model_path
        self.model = CnnBertForSequenceClassification.from_pretrained(self.model_path)
        # Special handling
        self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.return_all_scores = return_all_scores

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.
        Returns:
            Context manager
        Examples::
            # Explicitly ask for tensor allocation on CUDA device :0
            pipe = pipeline(..., device=0)
            with pipe.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = pipe(...)
        """

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.
        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.
        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {
            name: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in inputs.items()
        }

    def __call__(self, text):
        """
        Classify the text(s) given as inputs.
        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of prompts) to classify.
        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:
            - **label** (:obj:`str`) -- The label predicted.
            - **score** (:obj:`float`) -- The corresponding probability.
            If ``self.return_all_scores=True``, one such dictionary is returned per label.
        """
        # outputs = super().__call__(*args, **kwargs)
        inputs = self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
        )

        with torch.no_grad():
            inputs = self.ensure_tensor_on_device(**inputs)
            predictions = self.model(**inputs)[0].cpu()

        predictions = predictions.numpy()

        if self.model.config.num_labels == 1:
            scores = 1.0 / (1.0 + np.exp(-predictions))
        else:
            scores = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
        if self.return_all_scores:
            return [
                [
                    {"label": self.model.config.id2label[i], "score": score.item()}
                    for i, score in enumerate(item)
                ]
                for item in scores
            ]
        else:
            return [
                {"label": self.inv_label_map[item.argmax()], "score": item.max().item()}
                for item in scores
            ]

# @st.cache(allow_output_mutation=False, hash_funcs={Tokenizer: str})
@lru_cache(maxsize=1)
def load_text_generator():
    predictor = SentimentAnalyzer()
    return predictor


predictor = load_text_generator()


def write():
    st.markdown(
        """
        # Arabic Sentiment Analysis

        This is a simple sentiment analysis app that uses the prediction kernel from Wissam's (me) submission that won the [Arabic Senitment Analysis competition @ KAUST](https://www.kaggle.com/c/arabic-sentiment-analysis-2021-kaust)
        """
    )
    if st.checkbox("More info: "):
        st.markdown(
            """
            ### Submission Description:

            My submission is based on an ensemble of 5 models with varying preprocessing, and classifier design. All model variants are built over MARBERT [1], which is a BERT-based model pre-trained on 1B dialectal Arabic tweets.

            For preprocessing, all models shared the following steps:
            -   Replacing user mentions with “USER” and links with “URL”
            -   Replacing the “#” with “HASH”
            -   Removed the underscore character since it is missing the MARBERT vocabulary.
            -   Removed diacritics and elongations (tatweel)
            -   Spacing out emojis

            For classifier design, all models use a dense layer on top of MARBERT unless otherwise specified. Model training is done by hyperparameter grid-search with 5-fold cross-validation with the following search space:
            -   Learning rate: [2e-5,3e-5,4e-5]
            -   Batch size: 128
            -   Maximum sequence length: 64
            -   Epochs: 3 (we select the best epoch for the final prediction)
            -   Warmup ratio: [0,0.1]
            -   Seed: [1,25,42,123,666]

            Model I is a vanilla variant with only the preprocessing steps mention above applied. Model II enhances the emoji representation by replacing OOV emojis with ones that have similar meaning, for example 💊  😷.
            We noticed the repetitive use of “السلام عليكم” and “ورحمة الله وبركاته” in neutral tweets, especially when users were directing questions to business accounts. This could confuse the classifier, if it encountered these words in a for example a negative tweet, hence in Model III we removed variation of the phrase mentioned before using fuzzy matching algorithms.

            In Model IV, we tried to help the model by appending a sarcasm label to the input. We first trained a separate MARBERT on the ArSarcasm [2] dataset, and then used it to label the training and test sets.

            Model V uses the vanilla preprocessing approach, but instead of a dense layer built on top of MARBERT, we follow the approach detailed by Safaya et.al. [3] which uses a CNN-based classifier instead.

            For the final prediction, we first average the predictions of the 5 models from cross-validation (this is done for each model separately), we then average the results from the 5 model variants. We observed that the distribution of the predicted sentiment classes, doesn’t quite match the true distribution, this is due to the model preferring the neutral class over the positive class. To counter that, we apply what we call Label-Weighted average where during after the final averaging we rescale the score with the following weights 1.57,0.98 and 0.93 for positive, neutral, and negative (note that the weights were determined empirically).

            1- https://aclanthology.org/2021.acl-long.551/

            2- https://github.com/iabufarha/ArSarcasm

            3- https://github.com/alisafaya/OffensEval2020


            """
        )
    input_text = st.text_input(
        "Enter your text here:",
    )
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction, score, all_score = predictor.predict([input_text])
            st.write(f"Result: {prediction[0]}")
            detailed_score = {
                "Positive": all_score[0][0],
                "Neutral": all_score[0][1],
                "Negative": all_score[0][2],
            }
            st.write("All scores:")
            st.write(detailed_score)
