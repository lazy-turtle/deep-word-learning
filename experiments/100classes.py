from models.som.SOM import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_csv_visual_100classes, from_csv, to_csv, create_folds, transform_data
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random_seed = 42 # use the same one if you want to avoid training SOMs all over again

soma_path = os.path.join(Constants.DATA_FOLDER, '100classes', 'audio_model_new', '')
somv_path = os.path.join(Constants.DATA_FOLDER, '100classes', 'visual_model_new', '')
hebbian_path = os.path.join(Constants.DATA_FOLDER, '100classes', 'hebbian_model_november', '')
audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '100classes',
                               'audio100classes.csv')
visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                '100classes',
                                'VisualInputTrainingSet.csv')

parser = argparse.ArgumentParser(description='Train a Hebbian model.')
parser.add_argument('apath', type=str, help='The path to the trained acoustic SOM model')
parser.add_argument('vpath', type=str, help='The path to the trained visual SOM model')
parser.add_argument('--lr', metavar='lr', type=float, default=100, help='The hebbian model learning rate')
parser.add_argument('--a-sigma', metavar='sigma', type=float, default=100, help='The SOM neighborhood value')
parser.add_argument('--a-alpha', metavar='alpha', type=float, default=100, help='The SOM initial learning rate')
parser.add_argument('--v-lr', metavar='lr', type=float, default=100, help='The model learning rate')
parser.add_argument('--v-sigma', metavar='sigma', type=float, default=100, help='The SOM neighborhood value')
parser.add_argument('--v-alpha', metavar='alpha', type=float, default=100, help='The SOM initial learning rate')
parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Random generator seed')
parser.add_argument('--algo', metavar='algo', type=str, default='sorted',
                    help='Algorithm choice')
parser.add_argument('--aneurons1', type=int, default=50,
                    help='Number of neurons for audio SOM, first dimension')
parser.add_argument('--aneurons2', type=int, default=50,
                    help='Number of neurons for audio SOM, second dimension')
parser.add_argument('--vneurons1', type=int, default=50,
                    help='Number of neurons for visual SOM, first dimension')
parser.add_argument('--vneurons2', type=int, default=50,
                    help='Number of examples in a batch for visual SOM')
parser.add_argument('--a-batch', type=int, default=128,
                    help='Number of examples in a batch for acoustic SOM')
parser.add_argument('--v-batch', type=int, default=128,
                    help='Number of neurons for visual SOM, second dimension')
parser.add_argument('--source', metavar='source', type=str, default='v',
                    help='Source SOM')
parser.add_argument('--subsample', action='store_true', default=False)
parser.add_argument('--rotation', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
args = parser.parse_args()
exp_description = 'lr' + str(args.lr) + '_algo_' + args.algo + '_source_' + args.source


if __name__ == '__main__':
    a_xs, a_ys, _ = from_csv_with_filenames(audio_data_path)
    v_xs, v_ys = from_csv_visual_100classes(visual_data_path)
    # scale data to 0-1 range
    #a_xs = MinMaxScaler().fit_transform(a_xs)
    #v_xs = MinMaxScaler().fit_transform(v_xs)
    a_dim = len(a_xs[0])
    v_dim = len(v_xs[0])
    som_a = SOM(args.aneurons1, args.aneurons2, a_dim, n_iterations=10000, alpha=args.a_alpha, checkpoint_loc=args.apath,
                 tau=0.1, threshold=0.6, batch_size=args.a_batch, data='audio', sigma=args.a_sigma)
    som_v = SOM(args.vneurons1, args.vneurons2, v_dim, n_iterations=10000, alpha=args.v_alpha, checkpoint_loc=args.vpath,
                 tau=0.1, threshold=0.6, batch_size=args.v_batch, data='video', sigma=args.v_sigma)

    v_ys = np.array(v_ys)
    v_xs = np.array(v_xs)
    a_xs = np.array(a_xs)
    a_ys = np.array(a_ys)

    if args.subsample:
        a_xs, _, a_ys, _ = train_test_split(a_xs, a_ys, test_size=0.8, stratify=a_ys)
        v_xs, _, v_ys, _ = train_test_split(v_xs, v_ys, test_size=0.8, stratify=v_ys)
        print('Audio: training on {} examples.'.format(len(a_xs)))
        print('Image: training on {} examples.'.format(len(v_xs)))


    a_xs_train, a_xs_test, a_ys_train, a_ys_test = train_test_split(a_xs, a_ys, test_size=0.2, stratify=a_ys,
                                                                    random_state=random_seed)
    a_xs_train, a_xs_val, a_ys_train, a_ys_val = train_test_split(a_xs_train, a_ys_train, test_size=0.5, stratify=a_ys_train,
                                                                    random_state=random_seed)

    v_xs_train, v_xs_test, v_ys_train, v_ys_test = train_test_split(v_xs, v_ys, test_size=0.2, stratify=v_ys,
                                                                    random_state=random_seed)
    v_xs_train, v_xs_val, v_ys_train, v_ys_val = train_test_split(v_xs_train, v_ys_train, test_size=0.5, stratify=v_ys_train,
                                                                    random_state=random_seed)

    a_xs_train, a_xs_val = transform_data(a_xs_train, a_xs_val, rotation=args.rotation)
    v_xs_train, v_xs_val = transform_data(v_xs_train, v_xs_val, rotation=args.rotation)


    if args.train:
        som_a.train(a_xs_train, input_classes=a_ys_train, test_vects=a_xs_test, test_classes=a_ys_test)
        som_v.train(v_xs_train, input_classes=v_ys_train, test_vects=v_xs_test, test_classes=v_ys_test)
    else:
        som_a.restore_trained(args.apath)
        som_v.restore_trained(args.vpath)

    acc_a_list = []
    acc_v_list = []
    for n in range(1, 15):
        hebbian_model = HebbianModel(som_a, som_v, a_dim=a_dim,
                                     v_dim=v_dim, n_presentations=n,
                                     checkpoint_dir=hebbian_path,
                                     learning_rate=args.lr,
                                     n_classes=100)

        a_xs_fold, v_xs_fold, a_ys_fold, v_ys_fold = create_folds(a_xs_train, v_xs_train, a_ys_train, v_ys_train, n_folds=n)
        print('Training...')
        hebbian_model.train(a_xs_fold, v_xs_fold)
        print('Memorizing...')
        # prepare the soms for alternative matching strategies - this is not necessary
        # if prediction_alg='regular' in hebbian_model.evaluate(...) below
        if args.algo != 'regular':
            if som_a.train_bmu_class_dict == None:
                som_a.memorize_examples_by_class(a_xs_train, a_ys_train)
            if som_v.train_bmu_class_dict == None:
                som_v.memorize_examples_by_class(v_xs_train, v_ys_train)
            if som_a.test_bmu_class_dict == None:
                som_a.memorize_examples_by_class(a_xs_train, a_ys_train, train=False)
            if som_v.test_bmu_class_dict == None:
                som_v.memorize_examples_by_class(v_xs_train, v_ys_train, train=False)
        print('Evaluating Train Set...')
        accuracy_a = hebbian_model.evaluate(a_xs_train, v_xs_train, a_ys_train, v_ys_train, source='a',
                                          prediction_alg=args.algo)
        accuracy_v = hebbian_model.evaluate(a_xs_train, v_xs_train, a_ys_train, v_ys_train, source='v',
                                          prediction_alg=args.algo)
        print('Training set: n={}, accuracy_a={}, accuracy_v={}'.format(n, accuracy_a, accuracy_v))
        print('Evaluating Val Set...')
        accuracy_a = hebbian_model.evaluate(a_xs_val, v_xs_val, a_ys_val, v_ys_val, source='a',
                                          prediction_alg=args.algo)
        accuracy_v = hebbian_model.evaluate(a_xs_val, v_xs_val, a_ys_val, v_ys_val, source='v',
                                          prediction_alg=args.algo)
        print('Evaluation set: n={}, accuracy_a={}, accuracy_v={}'.format(n, accuracy_a, accuracy_v))
        acc_a_list.append(accuracy_a)
        acc_v_list.append(accuracy_v)
        # make a plot - placeholder
    plt.plot(acc_a_list, color='teal')
    plt.plot(acc_v_list, color='orange')
    plt.savefig('./plots/'+exp_description+'.pdf', transparent=True)
