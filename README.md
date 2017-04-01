# Sentiment Analysis using TensorFlow

## Overview
Sentiment Analysis using a simple LSTM network to classify short texts into 2 categories (positive and negative). The implemented LSTM network is structured as follows (note that the batch dimension is omitted in the explanation):
- **Embedding layer**: Transforms each input (a tensor of *k* words) into a tensor of *k* *N*-dimensional vectors (word embeddings), where *N* is the embedding size. Every word will be associated to a vector of weights that needs to be learnt during the training process. You can gain more insight into word embeddings at [Vector Representations of Words](https://www.tensorflow.org/versions/r0.12/tutorials/word2vec/index.html).
- **RNN layer**: It's made out of LSTM cells with a dropout wrapper. The intuition of LSTM networks is nicely described at [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). LSTM weights need to be learnt during the training process. The RNN layer is unrolled dynamically, taking *k* word embeddings as input and outputting *k* *M*-dimensional vectors, where *M* is the hidden size of LSTM cells. 
- **Softmax layer**: The RNN-layer output is averaged across *k* timesteps, obtaining a single tensor of size *M*. Finally, a softmax layer is used to compute classification probabilities.

Cross-entropy is used as the loss function and RMSProp is the optimizer that minimizes it.

TensorBoard provides a nice overview of the whole graph:
![TensorBoard graph](https://github.com/rvinas/sentiment_analysis_tensorflow/blob/master/graph_visualization.png)

## Prerequisites
- Python 3.5
- Pip 9.0.1

## Installation
1. Install TensorFlow. See [TensorFlow installation guide](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html)
2. Run `sudo pip install -r requirements.txt`

## Train
To train a model, run `python train.py`

Optional flags:
- `--data_dir`: Data directory containing `data.csv` (must have at least columns 'SentimentText' and 'Sentiment') Intermediate files will automatically be stored here. Default is `data/Kaggle` 
- `--stopwords_file`: Path to stopwords file. If `stopwords_file=None`, no stopwords will be used. Default is `data/stopwords.txt` 
- `--n_samples`: Number of samples to use from the dataset. Set `n_samples=None` to use the whole dataset. Default is `None`
- `--checkpoints_root`: Checkpoints directory root. Parameters will be saved there. Default is `checkpoints` 
- `--summaries_dir`: Directory where TensorFlow summaries will be stored. You can visualize learning using [TensorBoard](https://www.tensorflow.org/versions/r0.12/how_tos/summaries_and_tensorboard/index.html) by running `tensorboard --logdir=<summaries_dir>`. Default is `logs` 
- `--batch_size`: Batch size. Default is `100` 
- `--train_steps`: Number of training steps. Default is `300` 
- `--hidden_size`: Hidden size of LSTM layer. Default is `75` 
- `--embedding_size`: Size of embedding layer. Default is `75` 
- `--random_state`: Random state used for data splitting. Default is `0` 
- `--learning_rate`: RMSProp learning rate. Default is `0.01` 
- `--test_size`: Proportion of the dataset to be included in the test split (`0<test_size<1`). Default is `0.2` 
- `--dropout_keep_prob`: Dropout keep-probability (`0<dropout_keep_prob<=1`). Default is `0.5` 
- `--sequence_len`: Maximum sequence length. Let m be the maximum sequence length in the dataset. Then, it's required that `sequence_len >= m`. If `sequence_len=None`, then it'll be automatically assigned to `m`. Default is `None` 
- `--validate_every`: Step frequency in order to evaluate the model using a validation set. Default is `100` 

After training the model, the checkpoints directory will be printed out. For example: `Model saved in: checkpoints/1481294288`

## Predict
To make predictions using a previously trained model, run `python predict.py --checkpoints_dir <checkpoints directory>`
For example: `python predict.py --checkpoints_dir checkpoints/1481294288`
