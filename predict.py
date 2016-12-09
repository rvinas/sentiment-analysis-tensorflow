# predict.py: Uses a previously trained TensorFlow model to make predictions on a test set
# Copyright 2016 Ramon Viñas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tsa.data_manager import DataManager
import pickle

tf.flags.DEFINE_string('checkpoints_dir', 'checkpoints',
                       'Checkpoints directory (example: checkpoints/1479670630). Must contain (at least):\n'
                       '- config.pkl: Contains parameters used to train the model \n'
                       '- model.ckpt: Contains the weights of the model \n'
                       '- model.ckpt.meta: Contains the TensorFlow graph definition \n')
FLAGS = tf.flags.FLAGS

if FLAGS.checkpoints_dir is None:
    raise ValueError('Please, a valid checkpoints directory is required (--checkpoints_dir <file name>)')

# Load configuration
with open('{}/config.pkl'.format(FLAGS.checkpoints_dir), 'rb') as f:
    config = pickle.load(f)

# Load data
dm = DataManager(data_dir=config['data_dir'],
                 stopwords_file=config['stopwords_file'],
                 sequence_len=config['sequence_len'],
                 n_samples=config['n_samples'],
                 test_size=config['test_size'],
                 val_samples=config['batch_size'],
                 random_state=config['random_state'],
                 ensure_preprocessed=True)

# Import graph and evaluate the model using test data
original_text, x_test, y_test, test_seq_len = dm.get_test_data(original_text=True)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    # Import graph and restore its weights
    print('Restoring graph ...')
    saver = tf.train.import_meta_graph("{}/model.ckpt.meta".format(FLAGS.checkpoints_dir))
    saver.restore(sess, ("{}/model.ckpt".format(FLAGS.checkpoints_dir)))

    # Recover input/output tensors
    input = graph.get_operation_by_name('input').outputs[0]
    target = graph.get_operation_by_name('target').outputs[0]
    seq_len = graph.get_operation_by_name('lengths').outputs[0]
    dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
    predict = graph.get_operation_by_name('final_layer/softmax/predictions').outputs[0]
    accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]

    # Perform prediction
    pred, acc = sess.run([predict, accuracy],
                         feed_dict={input: x_test,
                                    target: y_test,
                                    seq_len: test_seq_len,
                                    dropout_keep_prob: 1})

# Print results
print('\nAccuracy: {0:.4f}\n'.format(acc))
for i in range(100):
    print('Sample: {0}'.format(original_text[i]))
    print('Predicted sentiment: [{0:.4f}, {1:.4f}]'.format(pred[i, 0], pred[i, 1]))
    print('Real sentiment: {0}\n'.format(y_test[i]))
