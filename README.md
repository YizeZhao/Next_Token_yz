# Implicit Geometry of Next-token Prediction: From Language Sparsity Patterns to Model Representations

## Abstract

Next-token prediction (NTP) over large text corpora has become the go-to paradigm to train large language models. Yet, it remains unclear how NTP influences the mapping of linguistic patterns to geometric properties of the resulting model representations. We frame training of large language models as soft-label classification over sparse probabilistic label vectors, coupled with an analytical approximation that allows unrestricted generation of context embeddings. This approach links NTP training to rank-constrained, nuclear-norm regularized optimization in the logit domain, offering a framework for analyzing the geometry of word and context embeddings. We demonstrate that NTP implicitly favors learning logits with a sparse plus low-rank structure, which captures co-occurrence frequencies and underlying sparsity patterns. This results in a phenomenon we term **subspace-collapse**, where contexts followed by similar next-token sets align in a shared subspace. We validate our findings through synthetic and small-scale language datasets, highlighting fundamental principles shaping linguistic representations in large-scale models.

## ArXiv Link

For full details, see our paper on [arXiv](https://arxiv.org/abs/2408.15417).

## Issues and Contact

If you have any questions, please create an issue or contact **zhaoyize@ece.ubc.ca**.
Tokenizers are in \Tokenizer. 

## Tokenizer Training
To train the tokenizer, run the `Dataset.py` script. This script will handle the training of a custom sentencepiece tokenizer on your dataset.

### Steps:
1. Ensure your dataset is available in the `./data` directory.
2. Run the following command to train the tokenizer:
   ```bash
   python Dataset.py
   ```
   This will train the tokenizer and save the model in the `./tokenizer` directory.

## Model Training
To train the model, you can use the `train.py` script. This script will utilize the pre-tokenized data to train a language model.

### Steps:
1. Ensure your configuration file `config.json` is set up correctly.
2. Run the following command to start training:
   ```bash
   python train.py --config config.json
   ```
   This will start the training process using the specified configuration.

## Visualization
To create all visualizations, run the `paperallplot.py` script. This script will generate the necessary plots and save them in the `./plots` directory.

### Steps:
1. Ensure all necessary data files are available in the `./data` directory.
2. Run the following command to generate the visualizations:
   ```bash
   python paperallplot.py
   ```
   This will create the visualizations and save them in the `./plots` directory.

