import argparse
import functools
import os
import pathlib
import pickle

# import config
import datasets
import ipdb
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import _settings


def sample_to_prompt(sample, **kwargs):
    if isinstance(sample['question'], list):
        return [sample_to_prompt({'question': _}, **kwargs) for _ in sample['question']]
    return f"""Answer these questions:
Q: In Scotland a bothy/bothie is a?
A: House
Q: {sample['question']}
A:"""


"""
https://arxiv.org/pdf/2302.13971.pdf

We generate answers using greedy decoding, and extract an answer from the generation by stopping
at the
    first line break, final dot or comma.
Generated answers are evaluated with the standard exact
match metric: a generated answer is considered correct if it matches any answer of the list of answers
after normalization. For this normalization step we lowercase generated answers and remove articles,
punctuation and duplicate whitespaces. Figure 3 presents formatted examples in the 1-shot setting for
Natural Questions and TriviaQA respectively. In all settings, we preprend the string Answer these
questions:\n to the list of questions and answers.
"""
def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
        #eos_token_id = [tokenizer(_)['input_ids'] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['\n', ',', '.']]
    else:
        raise NotImplementedError
    eos_token_id += [tokenizer.eos_token_id]
    #bad_words_ids = [tokenizer(_)['input_ids'][1] for _ in ['Q:']] # only "Q"
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']] # only "Q"
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)


def process_data_to_model_inputs(batch, tokenizer):
    # assert len(batch['answer']) == 1
    # tokenize the inputs and labels
    answers = [answer["value"] for answer in batch["answer"]]
    batch_with_prompt = sample_to_prompt(batch)
    inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
    outputs = tokenizer(answers, padding=False, truncation=False)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()
    batch['answer'] = answers
        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]
    batch['id'] = batch['question_id']
    batch['prompt'] = batch_with_prompt

    return batch


@functools.lru_cache()
def get_dataset(tokenizer, split='validation'):
    data = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split)
    id_mem = set()
    def remove_dups(batch):
        if batch['question_id'][0] in id_mem:
            return {_:[] for _ in batch.keys()}
        id_mem.add(batch['question_id'][0])
        return batch
    data = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    assert pd.Series([_['question_id'] for _ in data]).value_counts().max() == 1
    data = data.map(lambda _: process_data_to_model_inputs(_, tokenizer),
                            batched=True,
                            batch_size=10, # This does not matter
                            load_from_cache_file=False,
                            remove_columns=["search_results", "question_source", "entity_pages"])
    data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)
    return data

if __name__ == '__main__':
    import pandas as pd

    import models

    tokenizer = models.load_tokenizer('llama-7b-hf')
    #model, tokenizer = models.load_model_and_tokenizer('llama-7b-hf')
    data = get_dataset(tokenizer, split='validation')