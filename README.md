## Squad-Question Answering Project

This repository contains the code changes done by my team to SQuAD baseline model(Bidirectional Attention Flow model) to improve the performance by ~8% (58% to 66%). This project is using Pytorch implementation.

SQuAD is a question answering challenge to apply deep learning techniques to find the answer of a question from a given paragraph. The challenge information and current leaderboard positions is given in this link: https://rajpurkar.github.io/SQuAD-explorer/

## Approaches to improve performance

* On top of word embedding layer, we introduced Character Embedding to the model. The motivation behind this change was that the model will also be able to better recognize unknown words, i.e words not
present in the vocabulary as the vectors for the new words can be formed from the individual character vectors.
* We replaced dropouts with batch normalization and observed that scores improved by ~2% and training speed almost doubled. Possible reason for this improvement is that batch normalization has an additional effect of regularization which helps in generalization of model and avoid reduce over-fitting.

### Running the code
Refer to https://web.stanford.edu/class/cs224n/project/default-final-project-handout.pdf for setting up baseline model.
* Run ``conda env create -f environment.yml`` to install dependencies and create Conda environment called squad. After this, Run ``source activate squad``
* Please run ```setup.py``` to download the SQuAD dataset and GloVE Vectors
* To train the model, run ``train.py``
* The infer on test data, run ``test.py``
* Once you run the models, you will have a folder by the name save which will have the results from your code runs
* To start tensorboard, please run the following commands:
```
tensorboard --logdir save --port 5678
```
