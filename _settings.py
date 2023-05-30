import getpass
import os
import sys

__USERNAME = getpass.getuser()

_BASE_DIR = f'/srv/local/data/{__USERNAME}/'

LLAMA_PATH = f'{_BASE_DIR}/LLM_weights/'

DATA_FOLDER = os.path.join(_BASE_DIR, 'NLGUQ')
GENERATION_FOLDER = os.path.join(DATA_FOLDER, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

# After running pipeline/generate.py, update the following paths to the generated files if necessary.
GEN_PATHS = {
    'coqa': {
        'llama-13b': f'{GENERATION_FOLDER}/llama-13b-hf_coqa_10/0.pkl',
        'opt-13b': f'{GENERATION_FOLDER}/opt-13b_coqa_10/0.pkl',
        'gpt-3.5-turbo': f'{GENERATION_FOLDER}/gpt-3.5-turbo_coqa_10/0.pkl',
    },
    'trivia': {
        'llama-13b': f'{GENERATION_FOLDER}/llama-13b-hf_triviaqa_10/0.pkl',
        'opt-13b': f'{GENERATION_FOLDER}/opt-13b_triviaqa_10/0.pkl',
        'gpt-3.5-turbo': f'{GENERATION_FOLDER}/gpt-3.5-turbo_triviaqa_10/0.pkl',
    },
    'nq_open':{
        'llama-13b': f'{GENERATION_FOLDER}/llama-13b-hf_nq_open_10/0.pkl',
        'opt-13b': f'{GENERATION_FOLDER}/opt-13b_nq_open_10/0.pkl',
        'gpt-3.5-turbo': f'{GENERATION_FOLDER}/gpt-3.5-turbo_nq_open_10/0.pkl',
    }
}