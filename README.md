

# Quick Start
We provided a simple evaluation in `notebook/demo.ipynb` using 500 samples and the corresponding responses.
Note that to get the automatic evaluation based on GPT, you would need to update `keys.json` with your API keys first.

# Replicate Our Experiments
First, set the corresponding paths in `_settings.py`.

## Generate the responses
Use the `llama-13b-hf`, `opt-13b` or `gpt-3.5-turbo` for model, and `coqa`, `triviaqa` and `nq_open` for the dataset  below. (You need to download the LLaMA weight first).
```
python -m pipeline.generate --model llama-13b-hf --dataset coqa
```
For `gpt-3.5-turbo` experiments, please update `keys.json` with your API keys first.

Update `GEN_PATHS` in `_settings.py` for next steps.

## Run UQ experiments
You can run `dataeval/load.py` to cache down results first.
Then, please refer to `notebook/main.ipynb` for an example.
