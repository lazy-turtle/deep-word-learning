from datetime import date
import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from utils.constants import Constants
from utils.utils import softmax, get_plot_filename

class HebbianModel(object):

    def __init__(self, som_a, som_v, a_dim, v_dim, learning_rate=10,
                 n_presentations=1, n_classes=10, threshold=.6,
                 checkpoint_dir=None, a_threshold=.6, v_threshold=.6,
                 a_tau=.6, v_tau=.6):
        self.num_neurons_a = som_a._m * som_a._n
        self.num_neurons_v = som_v._m * som_v._n
        self._graph = tf.Graph()
        self.som_a = som_a
        self.som_v = som_v
        self.a_dim = a_dim
        self.v_dim = v_dim
        self.n_presentations = n_presentations
        self.n_classes = n_classes
        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = learning_rate
        self.threshold = threshold
        self._trained = False
        self.a_threshold = a_threshold
        self.v_threshold = v_threshold
        self.a_tau = a_tau
        self.v_tau = v_tau

        with self._graph.as_default():
            n = max(self.num_neurons_a, self.num_neurons_v)
            self.weights = tf.Variable(
                             tf.random_normal([self.num_neurons_a, self.num_neurons_v],
                             mean=1/n,
                             stddev=1/np.sqrt(1000*n)))

            self.activation_a = tf.placeholder(dtype=tf.float32, shape=[self.num_neurons_a])
            self.activation_v = tf.placeholder(dtype=tf.float32, shape=[self.num_neurons_v])
            self.assigned_weights = tf.placeholder(dtype=tf.float32, shape=[self.num_neurons_a, self.num_neurons_v])
            self.delta = 1 - tf.exp(-self.learning_rate * tf.matmul(tf.reshape(self.activation_a, (-1, 1)),
                                                                    tf.reshape(self.activation_v, (1, -1))))
            new_weights = tf.add(self.weights, self.delta)
            self.training = tf.assign(self.weights, new_weights)
            self.assign_op = tf.assign(self.weights, self.assigned_weights)
            self._sess  = tf.Session()
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def train(self, input_a, input_v, step=1):
        '''
        input_a: list containing a number of training examples equal to
                 self.n_presentations
        input_v : same as above

        This function will raise an AssertionError if:
        * len(input_a) != len(input_v)
        * len(input_a) != self.n_presentations * self.n_classes
        The first property assures we have a sane number of examples for both
        SOMs, while the second one ensures that the model has an example
        from each class to be trained on.
        '''
        assert len(input_a) == len(input_v) == self.n_presentations * self.n_classes, \
               'Number of training examples and number of desired presentations \
                is incoherent. len(input_a) = {}; len(input_v) = {}; \
                n_presentations = {}, n_classes = {}'.format(len(input_a), len(input_v),
                                                      self.n_presentations, self.n_classes)

        with self._sess:
            # present images to model
            for i in range(len(input_a)):
                # get activations from som
                activation_a, _ = self.som_a.get_activations(input_a[i])
                activation_v, _ = self.som_v.get_activations(input_v[i])

                # run training op
                _, d = self._sess.run([self.training, self.delta],
                                      feed_dict={self.activation_a: activation_a,
                                                 self.activation_v: activation_v})

            # normalize sum of weights to 1
            w = self._sess.run(self.weights)
            w = w.flatten()
            w_sum = np.sum(w)
            w_norm = [wi / w_sum for wi in w]
            w = np.reshape(w_norm, (self.num_neurons_a, self.num_neurons_v))

            self._sess.run(self.assign_op, feed_dict={self.assigned_weights: w})
            self._trained = True

            # save to checkpoint_dir
            if self.checkpoint_dir != None:
                saver = tf.train.Saver()
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                saver.save(self._sess,
                           os.path.join(self.checkpoint_dir,
                                       'model.ckpt'),
                           1)

            # convert weights to numpy arrays from tf tensors
            self.weights = self._sess.run(self.weights)

    def restore_trained(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            with self._sess:
              saver = tf.train.Saver()
              saver.restore(self._sess, ckpt.model_checkpoint_path)
              self.weights = self._sess.run(self.weights)
              print('RESTORED HEBBIAN MODEL')
              return True
        else:
            print('NO CHECKPOINT FOUND')
            return False

    def propagate_activation(self, source_activation, source_som='v'):
        source_activation = np.array(source_activation).reshape((-1, 1))
        if source_som == 'a':
            target_activation = np.matmul(self.weights.T, np.array(source_activation).reshape((-1, 1)))
            to_som = self.som_v
        else:
            target_activation = np.matmul(self.weights, np.array(source_activation).reshape((-1, 1)))
            to_som = self.som_a
        try:
            assert target_activation.shape[0] == (to_som._n * to_som._m)
        except AssertionError:
            print('Shapes do not match. target_activation: {};\
       som: {}'.format(target_activation.shape, to_som._n * to_som._m))
            sys.exit(1)
        return target_activation


    def get_bmus_propagate(self, x, source_som='v'):
        '''
        Get the best matching unit by propagating an input vector's activations
        to the other SOM. More specifically, we use the synapses connected to the
        source som's BMU to find a matching BMU in the target som.

        x: input vector. Must have a compatible size with the som described in
           'source_som' parameter
        source_som: a string representing the source som. If 'a', the activations
        of the audio som will be propagated to the visual one; if 'v', the opposite
        will happen.
        '''
        if source_som == 'v':
            from_som = self.som_v
            to_som = self.som_a
        elif source_som == 'a':
            from_som = self.som_a
            to_som = self.som_v
        else:
            raise ValueError('Wrong string for source_som parameter')
        source_activation, _ = from_som.get_activations(x)
        source_bmu_index = np.argmax(np.array(source_activation))
        source_activation = np.array(source_activation).reshape((-1, 1))
        target_activation = self.propagate_activation(source_activation, source_som=source_som)
        target_bmu_index = np.argmax(target_activation)
        return source_bmu_index, target_bmu_index

    def evaluate(self, X_a, X_v, y_a, y_v, source='v', img_path=None, prediction_alg='regular', k=4,
                 train=True):
        if source == 'v':
            X_source = X_v
            X_target = X_a
            y_source = y_v
            y_target = y_a
            source_som = self.som_v
            target_som = self.som_a
        elif source == 'a':
            X_source = X_a
            X_target = X_v
            y_source = y_a
            y_target = y_v
            source_som = self.som_a
            target_som = self.som_v
        else:
            raise ValueError('Wrong string for source parameter')

        y_pred = []
        img_n = 0

        for x, y in zip(X_source, y_source):
            if prediction_alg == 'regular':
                yi_pred = self.make_prediction(x, y, source_som, target_som, X_target, y_target, source)
            elif prediction_alg == 'knn':
                yi_pred = self.make_prediction_knn(x, y, k, source_som, target_som, source, train=train)
            elif prediction_alg == 'knn2':
                yi_pred = self.make_prediction_knn_weighted(x, y, k, source_som, target_som, source, train=train)
            elif prediction_alg == 'sorted':
                yi_pred = self.make_prediction_sort(x, source_som, target_som, source)
            else:
                raise ValueError('Unknown evaluation algorithm ' + str(prediction_alg))
            y_pred.append(yi_pred)
        correct = 0
        for yi, yj in zip(y_pred, y_source):
            if yi == yj:
                correct += 1
        print('source: {}, correct: {} out of {}' .format(source, correct, len(y_pred)))
        return correct/len(y_pred)

    def make_prediction(self, x, y, source_som, target_som, X_target, y_target, source):
        source_bmu, target_bmu = self.get_bmus_propagate(x, source_som=source)
        target_activations = []
        target_bmu_weights = np.reshape(target_som._weightages[target_bmu],
                                       (1, -1))
        for yi_target, xi_target in zip(y_target, X_target):
            xi_target = np.reshape(xi_target, (-1, 1))
            activation = np.dot(target_bmu_weights, xi_target)
            # alternative way to compute the activation. should be the same performance wise. (untested)
            #activation = np.absolute(xi_target - target_bmu_weights)
            #activation = np.exp(-(np.sum(activation)/len(activation))/self.tau)
            # save a correct example for later visualization, if necessary
            if yi_target == y:
                xi_true = xi_target
            # the float cast is so that 'activation' is not seen as an array but as an element
            target_activations.append(float(activation))
        yi_pred_idx = np.argmax(target_activations)
        yi_pred = y_target[yi_pred_idx]
        return yi_pred

    def make_plot(self, x_source, x_target, y_target, X_target_all, source, step, experiment_name, show=False):
        source_bmu, target_bmu = self.get_bmus_propagate(x_source, source_som=source)
        cmap = 'viridis' if source == 'a' else 'plasma'
        if source == 'a':
            hebbian_weights = self.weights[:][source_bmu]
            source_som = self.som_a
            target_som = self.som_v
        else:
            hebbian_weights = self.weights[target_bmu][:]
            source_som = self.som_v
            target_som = self.som_a

        source_activation, _ = source_som.get_activations(x_source)
        target_activation_true, _ = target_som.get_activations(x_target)

        # TODO: replicate some of the other make_prediction code to get this
        yi_pred = self.make_prediction_sort(x_source, source_som, target_som, source)
        target_activation_pred, _ = target_som.get_activations(X_target_all[yi_pred])
        propagated_activation = self.propagate_activation(source_activation, source_som=source)

        height, width = 3, 2
        fig, axis_arr = plt.subplots(height, width)
        fig.set_figheight(3 * height)
        fig.set_figwidth(3 * width)
        axis_arr[0, 0].matshow(np.array(source_activation).reshape((source_som._m, source_som._n)), cmap=cmap)
        axis_arr[0, 0].set_title('Source SOM activation')

        axis_arr[0, 1].matshow(propagated_activation.reshape((target_som._m, target_som._n)), cmap=cmap)
        axis_arr[0, 1].set_title('Propagation of activation to target')

        axis_arr[1, 0].matshow(np.array(target_activation_true).reshape((target_som._m, target_som._n)), cmap=cmap)
        axis_arr[1, 0].set_title('Target SOM activation true label ({})'.format(y_target))

        axis_arr[1, 1].matshow(np.array(target_activation_pred).reshape((target_som._m, target_som._n)), cmap=cmap)
        axis_arr[1, 1].set_title('Target SOM activation predicted label ({})'.format(yi_pred))

        mm = target_som._m if source == 'a' else source_som._m
        nn = target_som._n if source == 'a' else source_som._n
        axis_arr[2, 1].matshow(np.array(hebbian_weights).reshape((mm,nn)), cmap=cmap)
        axis_arr[2, 1].set_title('Hebbian weights of source BMU')

        axis_arr[2, 0].matshow(np.zeros((source_som._m, source_som._n)))
        plt.tight_layout()
        filename = 'stats_{}{}_ta{}tv{}th{}_'.format(step, source, self.som_a.tau, self.som_v.tau, self.som_a.threshold)
        filename += get_plot_filename(Constants.PLOT_FOLDER)
        if show:
            plt.show()
        else:
            save_loc = os.path.join(Constants.PLOT_FOLDER, 'hebbian', str(date.today()), experiment_name)
            if not os.path.exists(save_loc):
                os.makedirs(save_loc)
            plt.savefig(os.path.join(save_loc, filename))
        plt.clf()

    def get_bmu_k_closest(self, som, activations, pos_activations, k, train=True):
        '''
        Returns two lists containing respectively the level of activation
        and positions for the BMU and its closest k units. The length of these
        lists is therefore k+1, with the BMU information in the first position.
        '''
        if train:
            bmu_class_dict = som.train_bmu_class_dict
        else:
            bmu_class_dict = som.test_bmu_class_dict
        bmu_index = np.argmax(activations)
        bmu_position = pos_activations[bmu_index]
        distances_from_bmu = [np.linalg.norm(bmu_position - unit_position) for unit_position in pos_activations]
        sorted_indexes = list(range(len(distances_from_bmu)))
        sorted_indexes.sort(key=distances_from_bmu.__getitem__)
        sorted_activations = list(map(activations.__getitem__, sorted_indexes))
        sorted_positions = list(map(pos_activations.__getitem__, sorted_indexes))
        sorted_tuple = [(act, index) for act, index in zip(sorted_activations, sorted_indexes)
                          if bmu_class_dict[index] != []]
        sorted_activations = list(zip(*sorted_tuple))[0]
        sorted_indexes = list(zip(*sorted_tuple))[1]
        return sorted_activations[:k+1], sorted_indexes[:k+1]


    def make_prediction_knn(self, x, y, k, source_som, target_som, source, train=True):
        if train:
            bmu_class_dict = target_som.train_bmu_class_dict
        else:
            bmu_class_dict = target_som.test_bmu_class_dict
        source_activation, pos_source_activation = source_som.get_activations(x)
        source_activation = np.array(source_activation).reshape((-1, 1))
        target_activation = self.propagate_activation(source_activation, source_som=source)
        hebbian_bmu_index = np.argmax(target_activation)
        pos_activations = list(target_som._neuron_locations(target_som._m, target_som._n))
        closest_activations, closest_indexes = self.get_bmu_k_closest(target_som, target_activation,
                                                                      pos_activations, k, train=train)
        # perform a simple majority vote
        #class_count = [0 for i in set([c[0] for c in bmu_class_dict.values() if c != []])]
        #for i in range(len(closest_indexes)):
        #    bmu_class_list = bmu_class_dict[closest_indexes[i]]
        #    if bmu_class_list != []:
        #        class_count[bmu_class_list[0]] += 1

        class_count = [0 for i in range(source_som.num_classes)]
        for i in range(len(closest_indexes)):
            bmu_class_list = bmu_class_dict[closest_indexes[i]]
            for c in bmu_class_list:
                class_count[c] += 1
        return np.argmax(class_count)

    def make_prediction_knn_weighted(self, x, y, k, source_som, target_som, source,
                                     mode='none', train=True):
        if train:
            bmu_class_dict = target_som.train_bmu_class_dict
        else:
            bmu_class_dict = target_som.test_bmu_class_dict
        source_activation, pos_source_activation = source_som.get_activations(x)
        source_activation = np.array(source_activation).reshape((-1, 1))
        target_activation = self.propagate_activation(source_activation, source_som=source)
        # vote weighting alternatives
        if mode == 'softmax':
            # normalize using softmax. this brings some 0-valued votes to higher values
            vote_weights = softmax(target_activation)
        elif mode == 'none':
            # since hebbian weights and activations are normalized, the propagated activation's values
            # are already between 0 and 1
            vote_weights = target_activation
        elif mode == 'minmax':
            # minimum activation is mapped to 0 and maximum to 1
            min_ = min(target_activation)
            max_ = max(target_activation)
            vote_weights = (target_activation - min_) / float(max_- min_)
        hebbian_bmu_index = np.argmax(target_activation)
        pos_activations = list(target_som._neuron_locations(target_som._m, target_som._n))
        closest_activations, closest_indexes = self.get_bmu_k_closest(target_som, target_activation,
                                                                      pos_activations, k, train=train)
        # perform a weighted majority vote
        class_count = [0 for i in set([c[0] for c in bmu_class_dict.values() if c != []])]
        for i in range(len(closest_indexes)):
            #print(closest_indexes[i])
            bmu_class_list = bmu_class_dict[closest_indexes[i]]
            if bmu_class_list != []:
                class_count[bmu_class_list[0]] += 1 * vote_weights[closest_indexes[i]]
        #print(class_count)

        class_count = [0 for i in range(source_som.num_classes)]
        for i in range(len(closest_indexes)):
            bmu_class_list = bmu_class_dict[closest_indexes[i]]
            for c in bmu_class_list:
                class_count[c] += 1 * vote_weights[closest_indexes[i]]

        return np.argmax(class_count)

    def make_prediction_sort(self, x, source_som, target_som, source):
        source_activation, _ = source_som.get_activations(x)
        source_activation = np.array(source_activation).reshape((-1, 1))
        target_activation = self.propagate_activation(source_activation, source_som=source)
        for i in range(len(target_activation)):
            hebbian_bmu_index = np.argmax(target_activation)
            bmu_class_list = target_som.bmu_class_dict[hebbian_bmu_index]
            bmu_set = set(bmu_class_list)
            if len(bmu_set) == 1:
                return bmu_class_list[0]
            elif len(bmu_set) > 1:
                return self.bmu_max_count(bmu_class_list)
            if target_activation[hebbian_bmu_index] == -1:
                print('Error: make_prediction_sort failed.')
                return
            # discard this activation so that argmax does not find it again next
            # iteration
            target_activation[hebbian_bmu_index] = -1

    def bmu_max_count(self, bmu_class_dict):
        count_arr = np.array([0 for i in range(self.n_classes)])
        for c in bmu_class_dict:
            count_arr[c] += 1
        return np.argmax(count_arr)


    def threshold_activation(self, x):
        idx = x < self.threshold
        x[idx] = 0
        return x

# some test cases. do not use as an entry point for experiments!
if __name__ == '__main__':
    pass
