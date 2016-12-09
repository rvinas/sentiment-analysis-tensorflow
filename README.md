# Tensorflow Sentiment Analysis
## Prerequisites
- Python 3.5
- Pip 9.0.1

## Installation
1. Install TensorFlow. See [TensorFlow installation guide](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html)
2. Run `sudo pip install -r requirements.txt`

## Train
To train a model run `python train.py`

Optional flags:
- `--data_dir`: Data directory containing `data.csv` (must have at least columns 'SentimentText' and 'Sentiment'). Intermediate files will automatically be stored here. Default is `data/Kaggle`.
- `--stopwords_file`: Path to stopwords file. If `stopwords_file=None`, no stopwords will be used. Default is `data/stopwords.txt`.
- `--n_samples`: Number of samples to use from the dataset. Set `n_samples=None` to use the whole dataset. Default is `None`.
- `--checkpoints_root`: Checkpoints directory root. Parameters will be saved there. Default is `checkpoints`.
- `--summaries_dir`: Directory where TensorFlow summaries will be stored. You can visualize learning using [TensorBoard](https://www.tensorflow.org/versions/r0.12/how_tos/summaries_and_tensorboard/index.html) by running `tensorboard --logdir=<summaries_dir>`. Default is `logs`.
- `--batch_size`: Batch size. Default is `100`.
- `--train_steps`: Number of training steps. Default is `300`.
- `--hidden_size`: Hidden size of LSTM layer. Default is `75`.
- `--embedding_size`: Size of embedding layer. Default is `75`.
- `--random_state`: Random state used for data splitting. Default is `0`.
- `--learning_rate`: RMSProp learning rate. Default is `0.01`.
- `--test_size`: Proportion of the dataset to be included in the test split (`0<test_size<1`). Default is `0.2`.
- `--dropout_keep_prob`: Dropout keep-probability (`0<dropout_keep_prob<=1`). Default is `0.5`.
- `--sequence_len`: Maximum sequence length. Let m be the maximum sequence length in the dataset. Then, it's required that `sequence_len >= m`. If `sequence_len=None`, then it'll be automatically assigned to `m`. Default is `None`.
- `--validate_every`: Step frequency in order to evaluate the model using a validation set. Default is `100`.

After training the model, the checkpoints directory will be printed out. For example: `Model saved in: checkpoints/1481294288`


## Predict
To make predictions using a previously trained model, run `python predict.py --checkpoints_dir <checkpoints directory>`.
For example: `python predict.py --checkpoints_dir checkpoints/1481294288`
