# Unlocking Non-Invasive Brain-to-Text
This repository contains the code for the paper "Unlocking Non-Invasive Brain-to-Text". Find the preprint on ArXiv [here](https://arxiv.org/abs/2505.13446). If you find this code helpful in your work, please cite the paper:
```
@article{jayalath2024unlocking,
  title={{Unlocking Non-Invasive Brain-to-Text}},
  author={Jayalath, Dulhan and Landau, Gilad and Jones, Oiwi Parker},
  journal={arXiv preprint arXiv:2505.13446},
  year={2025}
}
```

## Quick start
1. Install requirements with `pip install -r requirements.txt`.
2. Download all or some of the [LibriBrain](https://huggingface.co/datasets/pnpl/LibriBrain), [Armeni](https://data.ru.nl/collections/di/dccn/DSC_3011085.05_995), [Gwilliams](https://osf.io/ag3kj/), and [Broderick (Natural Speech)](https://datadryad.org/dataset/doi:10.5061/dryad.070jc) datasets.
3. Modify the paths in `data/dataset_configs.yaml` to point `root` to your dataset's BIDS root directory, and change `cache` to where you would like preprocessed data to be kept.
4. Make sure you have a [weights and biases](http://wandb.ai/) account and are signed in on your console.

## Training a model
All results during training and evaluation will be logged to a project called `word-to-sent` in your weights and biases account.

### Train a word predictor

Single dataset:
`python train.py --vocab_size 250 --dset libribrain`

Joint training: `python train.py --vocab_size 250 --dset gwilliams2022 --aux_dsets armeni2022 libribrain`

### Train a word predictor and evaluate with rescoring

`python train.py --vocab_size 250 --dset libribrain --post_proc --no_llm_api`

### Train a word predictor and evaluate with both rescoring and in-context methods
> If you want to use in-context LLM API methods, please register on the [Anthropic console](console.anthropic.com) and set the environment variable `ANTHROPIC_API_KEY` with your API key, ensuring that you have sufficient credits. We estimate that you will need around `$15-$20` to evaluate a dataset in full (less if you use `--limit_eval_samples`).

`python train.py --vocab_size 250 --dset libribrain --post_proc`

### Evaluate with random noise inputs to trained model

`python train.py --vocab_size 250 --dset libribrain --test_ckpt /path/to/trained/ckpt --random_noise_inputs`

### Usage tips
- Use `--limit_eval_samples <int>` to reduce the number of sentences you evaluate with to save API costs.
- Use a trained word predictor for evaluation by supplying `---test_ckpt /path/to/ckpt` (checkpoints are saved to `./checkpoints/*` automatically).
- Use `--name <run-name>` to change the name of the run logged to weights and biases.