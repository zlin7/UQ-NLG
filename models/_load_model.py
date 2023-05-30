# This script exists just to load models faster
import functools
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)

from _settings import LLAMA_PATH


@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=torch.float16):
    if model_name.startswith('facebook/opt-'):
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    elif model_name == "microsoft/deberta-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")#, torch_dtype=torch_dtype)
    elif model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf':
        model = AutoModelForCausalLM.from_pretrained(os.path.join(LLAMA_PATH, model_name), cache_dir=None, torch_dtype=torch_dtype)
    elif model_name == 'roberta-large-mnli':
         model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")#, torch_dtype=torch_dtype)
    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name.startswith('facebook/opt-'):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    elif model_name == "microsoft/deberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    elif model_name == "roberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    elif model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf':
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(LLAMA_PATH, model_name), cache_dir=None, use_fast=use_fast)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer