import tensorflow as tf
import librosa as lr
import pickle
import glob
import time
import numpy as np
from utils.utils import array_to_sparse_tuple_1d, array_to_sparse_tuple
from utils.utils import pad_np_arrays
from utils.constants import Constants
from data.dataset import TIMITDataset
from keras.layers import Input, Bidirectional, LSTM
from keras.models import Sequential, Model
from keras.utils import to_categorical
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn, LSTMCell


LOAD_PICKLE = False
NUM_LAYERS = 5
NUM_HIDDEN = 500
BATCH_SIZE = 32
NUM_EPOCHS = 1000

def inference():
    pass

def loss():
    pass

def training():
    pass

def create_model(dataset):
    max_time_length = max([t for (t, f) in [x.shape for x in dataset.X_train]])
    num_features = dataset.X_train[0].shape[1]
    num_classes = max(TIMITDataset.phoneme_dict.values()) + 1
    num_examples = len(dataset.X_train)

    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=(None, None, num_features), name='input')
        targets = tf.sparse_placeholder(tf.int32, name='target')
        seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')

        lstm_cell_forward_list = []
        lstm_cell_backward_list = []
        for i in range(0, NUM_LAYERS):
            lstm_cell_forward_list.append(LSTMCell(NUM_HIDDEN))
            lstm_cell_backward_list.append(LSTMCell(NUM_HIDDEN))

        outputs, f_state, b_state = stack_bidirectional_dynamic_rnn(lstm_cell_forward_list, lstm_cell_backward_list,
                                        inputs, dtype=tf.float32, sequence_length=seq_length)
        
        # prepare the last fully-connected layer, which weights are shared throughout the time steps
        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
        W = tf.Variable(tf.truncated_normal([NUM_HIDDEN,
                                             num_classes],
                                            stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        fc_out = tf.matmul(outputs, W) + b
        fc_out = tf.reshape(fc_out, [BATCH_SIZE, -1, num_classes]) # Reshaping back to the original shape
        
        # time major
        fc_out = tf.transpose(fc_out, (1, 0, 2))

        loss = tf.nn.ctc_loss(targets, fc_out, seq_length, ignore_longer_outputs_than_inputs=True)
        cost = tf.reduce_mean(loss)

        optimizer = tf.train.MomentumOptimizer(0.0001,
                                               0.9).minimize(cost)

        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(fc_out, seq_length)
        
        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))
        
    with tf.Session(graph=graph) as session:
        # Initializate the weights and biases
        tf.global_variables_initializer().run()

        for curr_epoch in range(NUM_EPOCHS):
            train_cost = train_ler = 0
            start = time.time()

            num_batches_per_epoch = int(num_examples/BATCH_SIZE)

            for batch in range(num_batches_per_epoch):
                # prepare data and targets
                start_index = batch * BATCH_SIZE
                end_index = (batch + 1) * BATCH_SIZE 

                if end_index >= len(dataset.X_train) - 1:
                    end_index = len(dataset.X_train) - 1

                # get the data needed for the feed_dict this batch
                batch_seq_length = [time_length for time_length in dataset.train_timesteps[start_index:end_index]]
                batch_inputs = dataset.X_train[start_index:end_index]
                batch_targets = dataset.y_train[start_index:end_index]

                # pad both inputs and targets to max time length in the batch
                batch_inputs = TIMITDataset.pad_train_data(batch_inputs)
                batch_targets = pad_np_arrays(batch_targets)
                batch_dense_shape = np.array([x for x in np.array(batch_targets).shape])

                # get a sparse representation of the targets (tf.nn.ctc_loss needs it for some reason)
                batch_indices, batch_values = array_to_sparse_tuple(np.array(batch_targets))

                feed = {inputs: np.array(batch_inputs),
                        targets: (np.array(batch_indices), np.array(batch_values), batch_dense_shape),
                        seq_length: batch_seq_length}

                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost*BATCH_SIZE
                train_ler += session.run(ler, feed_dict=feed)*BATCH_SIZE

            train_cost /= num_examples
            train_ler /= num_examples

            log = "Epoch {:.0f}, train_cost = {:.3f}, train_ler = {:.3f} time = {:.3f}"
            print(log.format(curr_epoch+1, train_cost, train_ler,
                             time.time() - start))

        
            # decode a few examples each epoch to monitor progress
            # prepare data and targets
            start_index = 0
            end_index = BATCH_SIZE

            # get the data needed for the feed_dict this batch
            batch_seq_length = [time_length for time_length in dataset.train_timesteps[start_index:end_index]]
            batch_inputs = dataset.X_train[start_index:end_index]
            batch_targets = dataset.y_train[start_index:end_index]
            batch_dense_shape = np.array([x for x in np.array(batch_targets).shape])

            # pad both inputs and targets to max time length in the batch
            batch_inputs = TIMITDataset.pad_train_data(batch_inputs)
            batch_targets = pad_np_arrays(batch_targets)
            
            # get a sparse representation of the targets (tf.nn.ctc_loss needs it for some reason)
            batch_indices, batch_values = array_to_sparse_tuple(np.array(batch_targets))

            feed = {inputs: np.array(batch_inputs),
                    targets: (np.array(batch_indices), np.array(batch_values), batch_dense_shape),
                    seq_length: batch_seq_length}

            d = session.run(decoded[0], feed_dict=feed)
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)

            for i, seq in list(enumerate(dense_decoded))[:2]:
                seq = [s for s in seq if s != -1]
                print(seq)
                inverse_dict = {Constants.TIMIT_PHONEME_DICT[k] : k for k in Constants.TIMIT_PHONEME_DICT}
                original_phoneme_transcription = ' '.join([inverse_dict[k] for k in batch_targets[i]])
                estimated_phoneme_transcription = ' '.join([inverse_dict[k] for k in seq])
                print('Sequence %d' %i)
                print('Original \n%s' %original_phoneme_transcription)
                print('Estimated \n%s' %estimated_phoneme_transcription)

    return 

def create_model_keras():
    x = Input(shape=(NUM_FEATURES, None, None))
    y_pred = Bidirectional(LSTM(NUM_HIDDEN, return_sequences=True), merge_mode='sum')(x)
    #for i in range(0, NUM_LAYERS-2):
    #    y_pred = Bidirectional(LSTM(NUM_HIDDEN, return_sequences=True), merge_mode='sum')(y_pred)
    #y_pred = Bidirectional(LSTM(NUM_HIDDEN), merge_mode='sum')(y_pred)
    model = Model(inputs=x,outputs=y_pred)
    model.compile(loss=ctc_loss, optimizer='adam', metrics=['acc'])
    model.summary()
    return model

def evaluate_model():
    pass

if __name__ == "__main__":
    if LOAD_PICKLE == False:
        # create the dataset
        MyTimitDataset = TIMITDataset()
        MyTimitDataset.load()
        MyTimitDataset.to_file()
    else:
        # just load it from pickle
        filename = glob.glob("timit*.pickle")[0]
        with open(filename, "rb") as dataset_file:
            MyTimitDataset = pickle.load(dataset_file)
    create_model(MyTimitDataset)
    evaluate_model()
