import csv
import functools
import os
from collections import defaultdict
from importlib import reload

import evaluate
import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import torch
import tqdm
from pandarallel import pandarallel

import models
import models.nli as sc
import utils

# import wandb

pandarallel.initialize(progress_bar=True, nb_workers=16)

DEFAULT_DEVICE = 'cuda:3'

IGNORE_INDEX = -100

rouge = evaluate.load('rouge', keep_in_memory=True)
exact_match_metric = evaluate.load("exact_match")

def _compare_generated_text_to_answers(pred_txt, reference_answers):
    pred_txt = pred_txt.lstrip().lower()
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    sequence_dict = {_:0. for _ in ['exact_match'] + rouge_types}
    unique_reference_answers = set([_.lstrip().lower() for _ in reference_answers])
    for answer in unique_reference_answers:
        predictions = [pred_txt]
        references = [answer]
        results = exact_match_metric.compute(predictions=predictions,
                                             references=references,
                                             ignore_case=True,
                                             ignore_punctuation=True)
        sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
        rouge_results = rouge.compute(predictions=predictions, references=references)
        for rouge_type in rouge_types:
            sequence_dict[rouge_type] = max(rouge_results[rouge_type], sequence_dict[rouge_type])
    return sequence_dict

def _compare_generated_texts_to_answers(preds, reference_answers):
    pred_map = {pred: pred.lstrip().lower() for pred in preds}
    results = {_: _compare_generated_text_to_answers(_, reference_answers) for _ in pred_map.values()}
    return {pred: results[pred_map[pred]] for pred in preds}
    #return [results[pred_map[pred]] for pred in preds]

def _clean_sample(sample, tokenizer):
    # https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/clean_generated_strings.py
    def _clean_answer(old_text:str, old_token_ids, tokenizer):
        cleaned_text = old_text
        strings_to_filter_on = [
                    '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
                    'ANSWER:'
                ]
        for string in strings_to_filter_on:
            if string in cleaned_text:
                cleaned_text = cleaned_text.split(string)[0]
        if tokenizer is None:
            return dict(
                text_cleaned=cleaned_text,
                text=old_text,
            )
        token_ids = tokenizer.encode(cleaned_text, return_tensors='pt')[0]
        assert token_ids[0] == tokenizer.bos_token_id
        token_ids = token_ids[1:]
        return dict(text_cleaned=cleaned_text,
                    token_cleaned=token_ids.cpu(),
                    text=old_text,
                    token=old_token_ids.cpu(),
                    )
    ret = {k: sample[k] for k in ['prompt', 'id', 'question', 'answer', 'additional_answers']}
    ret['generations'] = [None] * len(sample['generations'])
    if tokenizer is None:
        for i, generation in enumerate(sample['generations']):
            ret['generations'][i] = _clean_answer(generation, None, tokenizer)
    else:
        for i, generation in enumerate(sample['generations_ids']):
            generation = generation[generation.ne(tokenizer.pad_token_id)]
            generation = generation[generation.ne(tokenizer.eos_token_id)]
            ret['generations'][i] = _clean_answer(sample['generations'][i], generation, tokenizer)
    ret['generations'] = {k: [v[k] for v in ret['generations']] for k in ret['generations'][0].keys()}
    return ret


def _old_syntactic_similarities(generated_texts):
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    syntactic_similarities = {rouge_type: None for rouge_type in rouge_types}
    if len(set(generated_texts)) == 1:
        return syntactic_similarities
    answer_list_1 = []
    answer_list_2 = []
    for i in generated_texts:
        for j in generated_texts:
            if i != j:
                answer_list_1.append(i)
                answer_list_2.append(j)
    results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
    for rouge_type in rouge_types:
        syntactic_similarities[rouge_type] = results[rouge_type]
    return syntactic_similarities

def _get_semantic_similarities_sample(sample, judge_model:sc.ClassifyWrapper, clean=False, logger=None):
    text_key = 'text_cleaned' if clean else 'text'
    _log_fn = lambda str: None if logger is None else logger.info(str)

    question = sample['question']
    has_semantically_different_answers = False
    all_ans = sample['generations'][text_key]
    unique_ans = sorted(list(set(all_ans)))
    semantic_set_ids = {ans: i for i, ans in enumerate(unique_ans)}
    _rev_mapping = semantic_set_ids.copy()
    sim_mat = torch.zeros((len(unique_ans), len(unique_ans),3))
    old_deberta_predictions = []

    _log_fn("Number of unique answers: " + str(len(unique_ans)))

    for i, ans_i in enumerate(unique_ans):
        for j, ans_j in enumerate(unique_ans[i+1:], i+1):
            sim_mat[i,j] = judge_model.pred_qa(question, ans_i, ans_j)[0]
            sim_mat[j,i] = judge_model.pred_qa(question, ans_j, ans_i)[0]

            # original logic
            deberta_prediction = torch.stack([sim_mat[i,j], sim_mat[j,i]], 0).argmax(1)
            _log_fn(f'Q: {question} || A1: {ans_i} || A2: {ans_j} || {deberta_prediction}')
            if deberta_prediction.min() == 0:
                has_semantically_different_answers = True
            else:
                semantic_set_ids[ans_j] = semantic_set_ids[ans_i]
            old_deberta_predictions.append([question, ans_i, ans_j, deberta_prediction.min().item()])
    return dict(
        id=sample['id'],
        mapping = [_rev_mapping[_] for _ in all_ans],
        sim_mat = sim_mat,
        old = {
        'has_semantically_different_answers': has_semantically_different_answers,
        'syntactic_similarities': _old_syntactic_similarities(sample['generations'][text_key])},
    ), old_deberta_predictions

@torch.no_grad()
def _get_semantic_similarities(samples, judge_model:sc.ClassifyWrapper, clean=False, logger=None):
    utils.seed_everything(10)
    result_dict, deberta_predictions = {}, []
    for sample in tqdm.tqdm(samples):
        result_dict[sample['id']], deberta_predictions_ = _get_semantic_similarities_sample(sample, judge_model, clean, logger)
        deberta_predictions.extend(deberta_predictions_)
    return result_dict, pd.DataFrame(deberta_predictions, columns=['question', 'ans1', 'ans2', 'deberta_prediction'])

# =======================loglikelihood=======================
def _compute_token_nll(model_output, prompt_len, generation):
    # log probabilities of the target words
    # Just in case the loss is not NLL for the model
    assert len(generation.shape) == 1
    _logits = model_output['logits'][0, prompt_len-1:-1]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
    assert generation[prompt_len:].ne(IGNORE_INDEX).all()
    loss = criterion(_logits, generation[prompt_len:])
    return loss

def _compute_token_entropy(model_output, prompt_len):
    # only the geenrated words
    _logits = model_output['logits'][0, prompt_len-1:-1] # NOTE: Can we include the last word here?
    _logp = torch.nn.functional.log_softmax(_logits, dim=-1)
    _p = torch.exp(_logp)
    _token_entropy = -torch.where(_p > 0, _p * _logp, 0).sum(1) # avoid -inf
    #higher -> more uncertain
    return _token_entropy

def _compute_token_mean(embedding, prompt_len):
    # only the geenrated words
    _embedding = embedding[0, prompt_len-1:-1] # NOTE: Can we include the last word here? If so, replace -1 with None
    return _embedding.mean(0)

def _create_output_prompt(model, tokenizer, prompt):
    prompt = prompt.to(model.device)
    assert 1 == len(prompt.shape) and prompt.ne(tokenizer.pad_token_id).all()
    model_output = model(prompt.unsqueeze(0), output_hidden_states=True,
                         labels=prompt.unsqueeze(0))
    token_nll = _compute_token_nll(model_output, 1, prompt)
    token_entropy = _compute_token_entropy(model_output, 1)
    sequence_embedding = _compute_token_mean(model_output['hidden_states'][-1], 1)
    return dict(
        neg_log_likelihood = token_nll.sum().item(),
        length = len(prompt),
        token_nll = token_nll.cpu(),
        token_entropy = token_entropy.cpu(),
        sequence_embedding = sequence_embedding.cpu(),
    )

@torch.no_grad()
def _create_output_from_generation(model, tokenizer, generation, prompt):
    prompt = prompt.to(model.device)
    generation = torch.concat([prompt, generation.to(model.device)])
    prompt_len = len(prompt)
    assert len(generation.shape) == 1 == len(prompt.shape)
    generation = generation[generation.ne(tokenizer.pad_token_id)]
    generation_only = generation.clone()[prompt_len - 1:] # with one token prefix
    generation = generation.clone()

    model_output = model(generation.unsqueeze(0), output_hidden_states=True)
    unconditioned_model_output = model(generation_only.unsqueeze(0), output_hidden_states=True,
                                       labels=generation_only.unsqueeze(0))

    token_nll = _compute_token_nll(model_output, prompt_len, generation)
    unconditioned_token_nll = _compute_token_nll(unconditioned_model_output, 1, generation_only)
    token_entropy = _compute_token_entropy(model_output, prompt_len)
    unconditioned_token_entropy = _compute_token_entropy(unconditioned_model_output, 1)

    # embedding
    sequence_embedding = _compute_token_mean(model_output['hidden_states'][-1], prompt_len)
    unconditioned_sequence_embedding = _compute_token_mean(unconditioned_model_output['hidden_states'][-1], 1)

    return dict(
        neg_log_likelihood = token_nll.sum().item(),
        unconditioned_neg_log_likelihood = unconditioned_token_nll.sum().item(),
        length = len(generation) - prompt_len,
        #
        token_nll = token_nll.cpu(),#.numpy(),
        unconditioned_token_nll = unconditioned_token_nll.cpu(),#.numpy(),
        token_entropy = token_entropy.cpu(),#.numpy(),
        unconditioned_token_entropy = unconditioned_token_entropy.cpu(),#.numpy(),
        # embeddings
        sequence_embedding = sequence_embedding.cpu(),#.numpy(),
        unconditioned_sequence_embedding=unconditioned_sequence_embedding.cpu(),#.numpy(),
    )

@torch.no_grad()
def _get_loglikelihoods(samples, model, tokenizer, clean:bool, logger=None):
    token_key = 'token_cleaned' if clean else 'token'
    ret = []
    for sample in tqdm.tqdm(samples):
        curr_summ = {'id': sample['id']}

        prompt = sample['prompt'].to(model.device)
        assert prompt.ne(tokenizer.pad_token_id).all() and len(prompt.shape) == 1
        curr_summ['prompt'] = _create_output_prompt(model, tokenizer, prompt)

        sampled_summ = [_create_output_from_generation(model, tokenizer, _, prompt) for _ in sample['generations'][token_key]]
        curr_summ['generations'] = {k: [_[k] for _ in sampled_summ] for k in sampled_summ[0].keys()}
        for _ in ['sequence_embedding', 'unconditioned_sequence_embedding']:
            curr_summ['generations'][_] = torch.stack(curr_summ['generations'][_])
        ret.append(curr_summ)
    return ret


def _get_self_eval_sample(row, text_key, dataset, model, tokenizer, logsm=False):
    import dataeval.coqa as coqa
    anss = [_.lstrip() for _ in row['generations'][text_key]]
    unique_answers = set(anss)
    few_shots = '\n'.join(list(unique_answers)[:10])
    story = (coqa.read_all_contexts()[row['id']] + '\n') if dataset == 'coqa' else ''
    A_tok = tokenizer.encode('(A')[-1]
    B_tok = tokenizer.encode('(B')[-1]

    ret = {}
    for _ans in anss:
        prompt = f"""{story}Question: {row['question']}
Here are some brainstormed ideas: {few_shots}
Possible Answer: {_ans}
Is the possible answer:
(A) True
(B) False
The possible answer is: ("""
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(model.device)
        res = model(input_ids, output_hidden_states=True)
        if logsm:
            logits = torch.nn.functional.log_softmax(res['logits'][0][-1], 0)
        else:
            logits = res['logits'][0][-1]
        ret[_ans] = logits[[A_tok, B_tok]].detach().cpu()
    return dict(id=row['id'], logits=pd.DataFrame(torch.stack([ret[_] for _ in anss]).cpu().numpy(), columns=['True', 'False']))

@torch.no_grad()
def _get_self_eval(samples, model, tokenizer, clean:bool, dataset:str, logger=None, logsm=False):
    text_key = 'text_cleaned' if clean else 'text'
    ret = [_get_self_eval_sample(_, text_key, dataset, model, tokenizer, logsm=logsm) for _ in tqdm.tqdm(samples)]
    return ret

def _get_gpt_eval_sample(row, text_key = None, ith=None, dataset:str=None, few_shots=None):
    if text_key is None:
        pred = row['pred']
    else:
        pred = row['generations'][text_key][ith]
    if few_shots is None:
        dataset = dataset or row['dataset']
        few_shots = {
            'triviaqa': [{'question': 'In Scotland a bothy/bothie is a?',
                        'reference': 'House','answer': 'House',
                            },
                        {'question': 'Where in England was Dame Judi Dench born?',
                        'reference':  'York', 'answer': 'London'
                            }],
            'nq_open': [{'question': 'who makes up the state council in russia',
                            'reference': 'governors and presidents', 'answer': 'governors and presidents'
                            },
                            {'question': 'when does real time with bill maher come back',
                                'reference': 'November 9, 2018', 'answer': 'September 8, 2000'
                            }],
            'coqa': [{'question': 'When was the Vat formally opened?',
                    'reference': 'It was formally established in 1475', 'answer': 'In 1475',
                    },
                    {'question': 'what is the library for?',
                    'reference': 'research', 'answer': 'tourism'}]

        }[dataset]
    prompt = f"""Rate the level of consistency between the answer to the question and the reference answer, from 0 to 100.
Question: {few_shots[0]['question']}
Reference: {few_shots[0]['reference']}
Answer: {few_shots[0]['answer']}
Rating: 100.

Question: {few_shots[1]['question']}
Reference: {few_shots[1]['reference']}
Answer: {few_shots[1]['answer']}
Rating: 0.

Question: {row['question']}
Reference: {row['answer']}
Answer: {pred.strip()}
Rating:"""
    return models.openai_query(prompt, model='gpt-3.5-turbo', attemptd_id=0, max_tries=50)

def _get_gpt_eval(samples, clean:bool, ith:int, dataset:str, logger=None, parallel=False):
    text_key = 'text_cleaned' if clean else 'text'
    df = pd.DataFrame({key: [sample[key] for sample in samples] for key in ['id', 'answer', 'question']})
    df['ith'] = ith
    df['text_key'] = text_key
    df['dataset'] = dataset
    df['pred'] = [sample['generations'][text_key][ith] for sample in samples]
    if parallel:
        ret = df.parallel_apply(_get_gpt_eval_sample, axis=1)
    else:
        ret = df.apply(_get_gpt_eval_sample, axis=1)
    return ret.values.tolist()

def _get_rouge_sample(row, text_key=None):
    _get_text = lambda x: x if text_key is None else x[text_key]
    all_ans = [row['answer']]
    if 'additional_answers' in row and row['additional_answers'] is not None:
        all_ans += row['additional_answers']
    all_preds = _get_text(row['generations'])
    all_results = _compare_generated_texts_to_answers(all_preds, all_ans)
    curr = {'id': row['id']}
    curr['generations'] = [all_results[_] for _ in _get_text(row['generations'])]
    return curr

def _get_rouge_parallel(samples, clean:bool, logger=None):
    text_key = 'text_cleaned' if clean else 'text'
    df = pd.DataFrame({key: [sample[key] for sample in samples] for key in ['id', 'answer', 'additional_answers']})
    df['text_key'] = text_key
    df['generations'] = [sample['generations'][text_key] for sample in samples]
    ret = df.parallel_apply(_get_rouge_sample, axis=1)
    return ret.values.tolist()

def _get_rouge(samples, clean:bool, logger=None):
    text_key = 'text_cleaned' if clean else 'text'
    ret = []
    for sample in tqdm.tqdm(samples):
        ret.append(_get_rouge_sample(sample, text_key))
    return ret



def _get_lexical_similarities_sample(sample):
    all_ans = sample['pred']
    unique_ans = sorted(list(set(all_ans)))
    ans2i = {ans: i for i, ans in enumerate(unique_ans)}
    sim_mat = np.eye(len(unique_ans))
    for i, ans_i in enumerate(unique_ans):
        for j, ans_j in enumerate(unique_ans[i+1:], i+1):
            sim_mat[i,j] = sim_mat[j,i] = rouge.compute(predictions=[ans_i], references=[ans_j], rouge_types=['rougeL'])['rougeL']
    return {'sim_mat': sim_mat, 'mapping': [ans2i[_] for _ in all_ans]}


def _get_lexical_similarities(samples, clean=False, logger=None, parallel=False):
    text_key = 'text_cleaned' if clean else 'text'
    df = pd.DataFrame({key: [sample[key] for sample in samples] for key in ['id', 'answer', 'question']})
    df['text_key'] = text_key
    df['pred'] = [sample['generations'][text_key] for sample in samples]
    if parallel:
        ret = df.parallel_apply(_get_lexical_similarities_sample, axis=1)
    else:
        ret = df.apply(_get_lexical_similarities_sample, axis=1)
    return ret.values.tolist()

if __name__ == '__main__':
    pass