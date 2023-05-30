from ._load_model import _load_pretrained_model, _load_pretrained_tokenizer
from .openai_models import openai_query


def load_model_and_tokenizer(model_name='opt-13b', device='cuda:2', **kwargs):
    if model_name in {'gpt-3.5-turbo'}:
        return None, None
    if model_name in {"opt-2.7b", "opt-1.3b", "opt-6.7b", 'opt-13b'}:
        return load_model_and_tokenizer(f"facebook/{model_name}", device, **kwargs)
    if model_name.startswith('facebook/opt-'):
        return _load_pretrained_model(model_name, device, **kwargs), _load_pretrained_tokenizer(model_name)
    return _load_pretrained_model(model_name, device, **kwargs), _load_pretrained_tokenizer(model_name)

def load_tokenizer(model_name='opt-13b', use_fast=False):
    if model_name in {'gpt-3.5-turbo'}:
        return None
    if model_name in {"opt-2.7b", "opt-1.3b", "opt-6.7b", 'opt-13b'}:
        return load_tokenizer(f"facebook/{model_name}", use_fast=use_fast)
    if model_name.startswith('facebook/opt-'):
        return _load_pretrained_tokenizer(model_name, use_fast=use_fast)
    return _load_pretrained_tokenizer(model_name, use_fast=use_fast)
