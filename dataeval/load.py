import functools
import os
from typing import Dict

import persist_to_disk as ptd
import tqdm

import dataeval.load_worker as lw
import models
import models.nli as sc
import utils

DEFAULT_DEVICE = 'cuda:7'

IGNORE_INDEX = -100

def _get_model_name(path:str):
    base_fnames = os.path.basename(path).split("_")
    if base_fnames[-1] == 'generations.pkl':
        return base_fnames[0]
    else:
        return os.path.basename(os.path.dirname(path)).split("_")[0]

def get_key_from_generated_strings_path_new(path):
    run_id = os.path.basename(path).replace(".pkl", "")
    specs = os.path.basename(os.path.dirname(path))
    return f"{specs}_{run_id}"

@functools.lru_cache(maxsize=4)
def read_cleaned_outputs_new(path):
    # Re-organize the result and include the "cleaned" answers.
    # This is the same as the semantic entropy paper.
    # The post-processing is a bit ugly, but somewhat unavoidable because
    # differnt tokens could lead to the same output character (like \n),
    # so simply specifying the tokens in generation config is not enough.
    key = get_key_from_generated_strings_path_new(path)
    cleaned_sequences = ptd.manual_cache(key)
    if cleaned_sequences is None:
        sequences = utils.cached_read_pickle(path)
        tokenizer = models.load_tokenizer(_get_model_name(path))
        cleaned_sequences = [lw._clean_sample(sample, tokenizer) for sample in tqdm.tqdm(sequences)]
        ptd.manual_cache(key, obj=cleaned_sequences, write=True)
    return cleaned_sequences

@functools.lru_cache(maxsize=4)
def read_semantic_similarities_new(path:str, device=DEFAULT_DEVICE,
                               judge_model:str = "microsoft/deberta-large-mnli",
                               clean=True, debug=False) -> Dict:
    # read (save) the similarity scores between the generated strings
    key = get_key_from_generated_strings_path_new(path)
    assert judge_model == 'microsoft/deberta-large-mnli' # When changed, change the output csv
    key += f"_model={judge_model.replace('/', '#')}"
    if clean: key += '_cleaned'
    semantic_sims = ptd.manual_cache(key)
    if semantic_sims is None:
        cleaned_sequences = read_cleaned_outputs_new(path)[:5 if debug else None]
        sc_model = sc.ClassifyWrapper(judge_model, device=device)
        if False:
            #NOTE: We used this branch when writing the paper, but this is slow
            # use the batched version for faster computation
            name = f'read_semantic_similarities#{key}'
            logger = utils.get_logger(name, os.path.join(os.path.dirname(path), f'{name}.log'), propagate=False)

            semantic_sims, deberta_predictions = lw._get_semantic_similarities(cleaned_sequences, sc_model, clean, logger)
            if not debug:
                deberta_predictions.to_csv(os.path.join(os.path.dirname(path), f"{judge_model.replace('/', '#')}_{key}.csv"), index=False, escapechar='\\')
        else:
            text_key = 'text_cleaned' if clean else 'text'
            semantic_sims = {}
            for _ in tqdm.tqdm(cleaned_sequences, desc="computing similarities"):
                _tres = sc_model.create_sim_mat_batched(_['question'], _['generations'][text_key])
                _tres['id'] = _['id']
                semantic_sims[_['id']] = _tres
        ptd.manual_cache(key, semantic_sims, write=not debug)
    return semantic_sims

@functools.lru_cache(maxsize=4)
def read_gpt_eval(path:str, clean=True, debug=False, parallel=False, ith=0, read_only=False):
    # read the GPT-based automatic evaluation results, for the i-th generation
    key = get_key_from_generated_strings_path_new(path)
    key += f"_{ith}"
    if clean: key += '_cleaned'
    evals = None if debug else ptd.manual_cache(key)
    if evals is None:
        assert not read_only
        cleaned_sequences = read_cleaned_outputs_new(path)[:5 if debug else None]
        name = f'read_gpt_eval#{key}'
        logger = utils.get_logger(name, os.path.join(os.path.dirname(path), f'{name}.log'), propagate=False)
        dataset = 'triviaqa' if 'triviaqa' in path else ('nq_open' if 'nq_open' in path else 'coqa')
        evals = lw._get_gpt_eval(cleaned_sequences, clean, ith, dataset=dataset, logger=logger, parallel=parallel)
        evals = {_['id']: _eval for _, _eval in zip(cleaned_sequences, evals)}
        ptd.manual_cache(key, evals, write=not debug)
    evals = {k: {"id": k, "response": v.split(".")[0].split()[0]} for k, v in evals.items()}
    return evals

# ==============The following are optional or for baselines =================
@functools.lru_cache(maxsize=4)
def read_rouges_new(path:str, clean=True, debug=False, parallel=False):
    # alternative to GPT evaluation, using ROUGE
    key = get_key_from_generated_strings_path_new(path)
    if clean: key += '_cleaned'
    rouges = ptd.manual_cache(key)
    if rouges is None:
        cleaned_sequences = read_cleaned_outputs_new(path)[:5 if debug else None]
        name = f'read_rouge#{key}'
        logger = utils.get_logger(name, os.path.join(os.path.dirname(path), f'{name}.log'), propagate=False)
        if parallel:
            rouges = lw._get_rouge_parallel(cleaned_sequences, clean, logger)
        else:
            rouges = lw._get_rouge(cleaned_sequences, clean, logger)
        ptd.manual_cache(key, rouges, write=not debug)
    return rouges

@functools.lru_cache(maxsize=4)
def read_lexical_sim(path:str, clean=True, debug=False, parallel=False, read_only=False) -> Dict:
    # used in the lexical similarity baseline
    key = get_key_from_generated_strings_path_new(path)
    if clean: key += '_cleaned'
    lexical_similarities = ptd.manual_cache(key)
    if lexical_similarities is None:
        assert not read_only
        cleaned_sequences = read_cleaned_outputs_new(path)[:5 if debug else None]
        name = f'read_lexical_sim#{key}'
        logger = utils.get_logger(name, os.path.join(os.path.dirname(path), f'{name}.log'), propagate=False)
        lexical_similarities = lw._get_lexical_similarities(cleaned_sequences, clean, logger, parallel=parallel)
        lexical_similarities = {_['id']: _eval for _, _eval in zip(cleaned_sequences, lexical_similarities)}
        ptd.manual_cache(key, lexical_similarities, write=not debug)
    return lexical_similarities

@functools.lru_cache(maxsize=4)
def read_loglikelihoods_and_more_new(path:str, device=None, clean=True, debug=False):
    # used in the semantic entropy baseline
    if device is not None:
        device = utils.gpuid_to_device(device)
    key = get_key_from_generated_strings_path_new(path)
    if clean: key += '_cleaned'
    likelihoods = ptd.manual_cache(key)
    if device is None: return likelihoods
    if likelihoods is None:
        cleaned_sequences = read_cleaned_outputs_new(path)[:5 if debug else None]
        name = f'read_semantic_similarities#{key}'
        logger = utils.get_logger(name, os.path.join(os.path.dirname(path), f'{name}.log'), propagate=False)
        model, tokenizer = models.load_model_and_tokenizer(_get_model_name(path), device)
        likelihoods = lw._get_loglikelihoods(cleaned_sequences, model, tokenizer, clean=clean, logger=logger)
        ptd.manual_cache(key, likelihoods, write=not debug)
    return likelihoods


def read_self_eval(path:str, device=None, clean=True, debug=False):
    # used in the P(true) baseline
    if device is not None:
        device = utils.gpuid_to_device(device)
    key = get_key_from_generated_strings_path_new(path)
    if clean: key += '_cleaned'
    results = ptd.manual_cache(key) if not debug else None
    if device is None: return results
    if results is None:
        cleaned_sequences = read_cleaned_outputs_new(path)[:5 if debug else None]
        name = f'read_self_eval#{key}'
        logger = utils.get_logger(name, os.path.join(os.path.dirname(path), f'{name}.log'), propagate=False)
        model, tokenizer = models.load_model_and_tokenizer(_get_model_name(path), device)
        dataset = 'triviaqa' if 'triviaqa' in path else ('nq_open' if 'nq_open' in path else 'coqa')
        results = lw._get_self_eval(cleaned_sequences, model, tokenizer, clean=clean, dataset=dataset, logger=logger)
        ptd.manual_cache(key, results, write=not debug)
    return results

if __name__ == '__main__':
    import _settings
    device = 'cuda:1'
    for data, _ in _settings.GEN_PATHS.items():
        for model, path in _.items():
            print(path)
            # organize the results.
            read_cleaned_outputs_new(path)

            # compare each pair of generated responses - this could be sped up by using batches
            read_semantic_similarities_new(path, device=device)

            for ith in tqdm.tqdm(range(20)): # 20 generations in total
                read_gpt_eval(path, ith=ith) # evaluate the accuracy of the responses

            # the lexical similarity baseline
            read_lexical_sim(path, parallel=True)

            # for white-box method
            if model != 'gpt-3.5-turbo':
                read_loglikelihoods_and_more_new(path, device=device)
                read_self_eval(path, device=device)

            # For evaluation of the generated responses
            read_rouges_new(path, parallel=True) # compute the rougeL scores
