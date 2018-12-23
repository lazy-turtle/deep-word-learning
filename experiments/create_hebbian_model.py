from models.som.SOM import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_npy_visual_data, global_transform
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

soma_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'audio', 'audio_model_10classes')
somv_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'video', 'video_20x30_tau0.1_thrsh0.6_sigma10.0_batch128_alpha0.1_best_so_far')
hebbian_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'hebbian', 'hebbian_model')
audio_data_path = os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio_10classes_train.csv')
video_data_path = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_a.npy')

def create_folds(a_xs, v_xs, a_ys, v_ys, n_folds=1, n_classes=10):
    '''
    In this context, a fold is an array of data that has n_folds examples
    from each class.
    '''
    #assert len(a_xs) == len(v_xs) == len(a_ys) == len(v_ys)
    assert n_folds * n_classes <= len(a_xs)
    ind = a_ys.argsort()
    a_xs = a_xs[ind]
    a_ys = a_ys[ind]
    ind = v_ys.argsort()
    v_xs = v_xs[ind]
    v_ys = v_ys[ind]
    # note that a_xs_ is not a_xs
    a_xs_ = [a_x for a_x in a_xs]
    a_ys_ = [a_y for a_y in a_ys]
    v_xs_ = [v_x for v_x in v_xs]
    v_ys_ = [v_y for v_y in v_ys]
    a_xs_fold = []
    a_ys_fold = []
    v_xs_fold = []
    v_ys_fold = []
    for i in range(n_folds):
        for c in range(n_classes):
            a_idx = a_ys_.index(c)
            v_idx = v_ys_.index(c)
            a_xs_fold.append(a_xs_[a_idx])
            a_ys_fold.append(c)
            v_xs_fold.append(v_xs_[v_idx])
            v_ys_fold.append(c)
            # delete elements so that they are not found again
            # and put in other folds
            del a_xs_[a_idx]
            del a_ys_[a_idx]
            del v_xs_[v_idx]
            del v_ys_[v_idx]
    return a_xs_fold, v_xs_fold, a_ys_fold, v_ys_fold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Hebbian model.')
    parser.add_argument('--lr', metavar='lr', type=float, default=10, help='The model learning rate')
    parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Random generator seed')
    parser.add_argument('--algo', metavar='algo', type=str, default='sorted',
                        help='Algorithm choice')
    parser.add_argument('--source', metavar='source', type=str, default='v',
                        help='Source SOM')
    parser.add_argument('--train', action='store_true', default=False)
    args = parser.parse_args()
    exp_description = 'lr' + str(args.lr) + '_algo_' + args.algo + '_source_' + args.source

    a_xs, a_ys, _ = from_csv_with_filenames(audio_data_path)
    a_ys = [v - 1000 for v in a_ys]
    a_xs = np.array(a_xs)
    a_ys = np.array(a_ys)
    v_xs, v_ys, _ = from_npy_visual_data(video_data_path)
    # fix labels to 0-9 range

    # transform
    a_xs = MinMaxScaler().fit_transform(a_xs)
    v_xs,_ = global_transform(v_xs)
    a_dim = len(a_xs[0])
    v_dim = len(v_xs[0])
    print('Data loaded and transformed, building SOMs...')
    som_a = SOM(20, 30, a_dim, checkpoint_loc=soma_path, n_iterations=10000,
                tau=0.1, threshold=0.6)
    som_v = SOM(20, 30, v_dim, checkpoint_loc=somv_path, n_iterations=10000,
                tau=0.1, threshold=0.6)


    a_xs_train, a_xs_test, a_ys_train, a_ys_test = train_test_split(a_xs, a_ys, test_size=0.2)
    v_xs_train, v_xs_test, v_ys_train, v_ys_test = train_test_split(v_xs, v_ys, test_size=0.2)
    a_xs_train, a_xs_dev, a_ys_train, a_ys_dev = train_test_split(a_xs, a_ys, test_size=0.2)
    v_xs_train, v_xs_dev, v_ys_train, v_ys_dev = train_test_split(v_xs, v_ys, test_size=0.2)

    if args.train:
        som_a.train(a_xs_train, input_classes=a_ys_train, test_vects=a_xs_dev, test_classes=a_ys_dev)
        som_v.train(v_xs_train, input_classes=v_ys_train, test_vects=v_xs_dev, test_classes=v_ys_dev)
    else:
        som_a.restore_trained(soma_path)
        som_v.restore_trained(somv_path)

    acc_a_list = []
    acc_v_list = []
    print('Preparing Hebbian model...')
    for n in range(1, 15):
        hebbian_model = HebbianModel(som_a, som_v, a_dim=a_dim,
                                     v_dim=v_dim, n_presentations=n,
                                     checkpoint_dir=hebbian_path,
                                     learning_rate=args.lr)
        a_xs_fold, v_xs_fold, a_ys_fold, v_ys_fold = create_folds(a_xs_train, v_xs_train, a_ys_train, v_ys_train, n_folds=n)
        # prepare the soms for alternative matching strategies - this is not necessary
        # if prediction_alg='regular' in hebbian_model.evaluate(...) below
        som_a.memorize_examples_by_class(a_xs_train, a_ys_train)
        som_v.memorize_examples_by_class(v_xs_train, v_ys_train)
        print('Training...')
        hebbian_model.train(a_xs_fold, v_xs_fold)
        print('Evaluating...')
        accuracy_a = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='a',
                                          prediction_alg=args.algo)
        accuracy_v = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='v',
                                          prediction_alg=args.algo)
        print('n={}, accuracy_a={}, accuracy_v={}'.format(n, accuracy_a, accuracy_v))
        acc_a_list.append(accuracy_a)
        acc_v_list.append(accuracy_v)
        # make a plot - placeholder
        hebbian_model.make_plot(a_xs_test[0], v_xs_test[0], v_ys_test[0], v_xs_fold[0], source='a')
    plt.plot(acc_a_list, color='teal')
    plt.plot(acc_v_list, color='orange')
    plt.savefig('../data/plots/'+exp_description+'.png', transparent=False)
