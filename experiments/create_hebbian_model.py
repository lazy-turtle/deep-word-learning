from models.som.SOM import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_npy_visual_data, global_transform, min_max_scale, \
    from_npy_audio_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from datetime import date
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


video_model_list = [
    'video_20x30_s10.0_b128_a0.1_group-a_seed42_1545208211_none',
    'video_20x20_s8.0_b128_a0.2_group-a_seed42_1545312173_global',
    'video_20x30_s15.0_b128_a0.2_group-a_seed42_1546936577_minmax',
    'video_20x30_s10.0_b128_a0.2_group-b_seed42_1545208672_global',
    'video_20x30_s12.0_b128_a0.1_group-b_seed33_1547237808_minmax',
    'video_20x30_s15.0_b128_a0.1_group-c1_seed42_1547303679_minmax',
    'video_20x30_s15.0_b128_a0.1_group-c2_seed42_1547388036_minmax',
    'video_20x30_s12.0_b128_a0.1_group-as_seed42_1547659811_minmax',
    'video_20x30_s10.0_b128_a0.1_group-as_seed42_1547662883_global',

    'video_20x30_s12.0_b64_a0.1_group-segm_seed42_1548697994_minmax',
    'video_20x30_s12.0_b128_a0.1_group-bbox_seed42_1548704755_global'
]

audio_model_list = [
    'audio_20x30_s10.0_b128_a0.1_group-x_seed42_1145208211_minmax',
    'audio_20x30_s10.0_b128_a0.1_group-s_seed10_1547394149_minmax',
    'audio_20x30_s10.0_b64_a0.1_group-s_seed42_1548662731_minmax'
]

video_model = video_model_list[-1]
audio_model = audio_model_list[-1]
soma_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'audio', audio_model)
somv_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'video', 'best', video_model)
hebbian_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'hebbian')
soma_data = os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio_10classes_synth.npy')

video_data_paths = {
    'a': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_a.npy'),
    'b': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_b.npy'),
    'z': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_z.npy'),
    'c1': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_c1.npy'),
    'c2': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_c2.npy'),
    'as':os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_as.npy'),
    'segm': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-10classes-segm.npy'),
    'bbox': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-10classes-bbox.npy')
}
num_presentations = 15


def extract_som_info(filename):
    info = dict()
    fields = filename.split('_')
    info['data'] = fields[0]
    info['dims'] = tuple([int(x) for x in fields[1].split('x')])
    info['sigma'] = float(fields[2][1:])
    info['alpha'] = float(fields[4][1:])
    info['batch'] = int(fields[3][1:])
    info['group'] = fields[5].replace('group-', '')
    info['seed'] = int(fields[6].replace('seed', ''))
    info['id'] = fields[-2]
    info['trsf'] = fields[-1]
    return info


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
    parser.add_argument('--lr', metavar='lr', type=float, default=1, help='The model learning rate')
    parser.add_argument('--taua', metavar='taua', type=float, default=1, help='Tau value audio som')
    parser.add_argument('--tauv', metavar='tauv', type=float, default=2, help='Tau value video som')
    parser.add_argument('--tha', metavar='tha', type=float, default=0.0, help='Threshold to cut values from (audio)')
    parser.add_argument('--thv', metavar='thv', type=float, default=0.0, help='Threshold to cut values from (video)')
    parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Random generator seed')
    parser.add_argument('--somv', metavar='somv', type=str, default=somv_path,
                        help='Video SOM model path')
    parser.add_argument('--algo', metavar='algo', type=str, default='regular',
                        help='Algorithm choice')
    parser.add_argument('--act', metavar='act', type=str, default='eucl',
                        help='Activation function choice')
    parser.add_argument('--source', metavar='source', type=str, default='v',
                        help='Source SOM')
    parser.add_argument('--train', action='store_true', default=False)

    args = parser.parse_args()
    somv_info = extract_som_info(os.path.basename(args.somv))
    somv_path = args.somv
    somv_data = video_data_paths[somv_info['group']]
    print('Training using video SOM: {}'.format(args.somv))
    print('---------Info-------')
    for k, v in somv_info.items():
        print('{:<20s}: {:<40s}'.format(k, str(v)))

    exp_description = 'lr{}_algo_{}_ta{:.1f}_tv{:.1f}_{}_som{}_'\
                          .format(args.lr, args.algo, args.taua, args.tauv, args.act, somv_info['id']) \
                      + str(int(time.time()))
    hebbian_path = os.path.join(hebbian_path, str(date.today()))
    if not os.path.exists(hebbian_path):
        os.makedirs(hebbian_path)
    model_path = os.path.join(hebbian_path, exp_description)

    #audio data
    if '.npy' not in soma_data:
        a_xs, a_ys, _ = from_csv_with_filenames(soma_data)
        a_ys = [v - 1000 for v in a_ys]
        a_xs = np.array(a_xs)
        a_ys = np.array(a_ys)
    else:
        a_xs, a_ys = from_npy_audio_data(soma_data)
    #video data
    v_xs, v_ys, _ = from_npy_visual_data(somv_data)

    # transform.
    a_xs = MinMaxScaler().fit_transform(a_xs)
    trasf = somv_info['trsf']
    if trasf == 'minmax':
        print('Using MinMaxScaler...')
        v_xs = MinMaxScaler().fit_transform(v_xs)
    elif trasf == 'global':
        print('Normalizing with global mean and std...')
        v_xs,_ = global_transform(v_xs)
    elif trasf != 'none':
        raise ValueError('Normalization not recognised, please check som filename.')

    #a_xs = np.array(a_xs)
    #a_ys = np.array(a_ys)

    a_dim = len(a_xs[0])
    v_dim = len(v_xs[0])
    print('Data loaded and transformed, building SOMs...')
    som_a = SOM(20, 30, a_dim, checkpoint_loc=soma_path, n_iterations=10000,
                tau=args.taua, threshold=args.tha)
    dims = somv_info['dims']
    som_v = SOM(dims[0], dims[1], v_dim, checkpoint_loc=somv_path, n_iterations=10000,
                tau=args.tauv, threshold=args.thv)


    a_xs_train, a_xs_test, a_ys_train, a_ys_test = train_test_split(a_xs, a_ys, test_size=0.2, random_state=args.seed)
    v_xs_train, v_xs_test, v_ys_train, v_ys_test = train_test_split(v_xs, v_ys, test_size=0.2, random_state=args.seed)
    a_xs_train, a_xs_dev, a_ys_train, a_ys_dev = train_test_split(a_xs, a_ys, test_size=0.2, random_state=args.seed)
    v_xs_train, v_xs_dev, v_ys_train, v_ys_dev = train_test_split(v_xs, v_ys, test_size=0.2, random_state=args.seed)

    if args.train:
        som_a.train(a_xs_train, input_classes=a_ys_train, test_vects=a_xs_dev, test_classes=a_ys_dev)
        som_v.train(v_xs_train, input_classes=v_ys_train, test_vects=v_xs_dev, test_classes=v_ys_dev)
    else:
        som_a.restore_trained(soma_path)
        som_v.restore_trained(somv_path)

    acc_a_list = []
    acc_v_list = []
    print('Preparing Hebbian model...')
    for n in range(1, num_presentations+1):
        print('Hebbian model with {} presentations.'.format(n))
        hebbian_model = HebbianModel(som_a, som_v, a_dim=a_dim,
                                     v_dim=v_dim, n_presentations=n,
                                     checkpoint_dir=model_path,
                                     learning_rate=args.lr)
        a_xs_fold, v_xs_fold, a_ys_fold, v_ys_fold = create_folds(a_xs_train, v_xs_train, a_ys_train, v_ys_train, n_folds=n)

        # # print(len(v_xs_fold))
        # for c in range(10):
        #     img_a = np.zeros((som_a._m, som_a._n))
        #     img_v = np.zeros((som_v._m, som_v._n))
        #     f, axarr = plt.subplots(1, 2)
        #
        #     for i in range(1):
        #         j = i*10 + c
        #         act_a,_ = som_a.get_activations(a_xs_fold[j], tau=1.0, threshold=0.5)
        #         act_v,_ = som_v.get_activations(v_xs_fold[j], tau=1.0, threshold=0.5)
        #         img_a += act_a.reshape((som_v._m, som_v._n))
        #         img_v += act_v.reshape((som_v._m, som_v._n))
        #     axarr[0].imshow(img_a, cmap='viridis', interpolation='nearest', origin='lower')
        #     axarr[1].imshow(img_v, cmap='plasma', interpolation='nearest', origin='lower')
        #     plt.show()
        # exit(0)

        # prepare the soms for alternative matching strategies - this is not necessary
        # if prediction_alg='regular' in hebbian_model.evaluate(...) below
        som_a.memorize_examples_by_class(a_xs_train, a_ys_train)
        som_v.memorize_examples_by_class(v_xs_train, v_ys_train)

        print('Training...')
        hebbian_model.train(a_xs_fold, v_xs_fold, step=n)
        print('Evaluating...')

        accuracy_a = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='a',
                                          prediction_alg=args.algo)
        accuracy_v = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='v',
                                          prediction_alg=args.algo)
        print('n={}, accuracy_a={}, accuracy_v={}'.format(n, accuracy_a, accuracy_v))
        acc_a_list.append(accuracy_a * 100)
        acc_v_list.append(accuracy_v * 100)
        # make a plot - placeholder
        hebbian_model.make_plot(a_xs_test[0], v_xs_test[0], v_ys_test[0], v_xs_fold[0], source='a', step=n)
        hebbian_model.make_plot(v_xs_test[0], a_xs_test[0], a_ys_test[0], a_xs_fold[0], source='v', step=n)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    xx = np.arange(1, num_presentations + 1)
    ax.plot(xx, acc_a_list, color='teal')
    ax.plot(xx, acc_v_list, color='orange')

    ax.set_xticks(xx)
    ax.set_yticks(np.arange(0, 101, 5))
    ax.yaxis.set_major_formatter(tkr.PercentFormatter())
    ax.grid(True, axis='y', linestyle=':')

    ax.set_xlabel('# presentations')
    ax.set_ylabel('Accuracy')
    plt.title('Model: {}'.format(os.path.basename(somv_path)))
    plot_path = os.path.join(Constants.PLOT_FOLDER, 'hebbian', str(date.today()))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plt.savefig(os.path.join(plot_path, '{}.png'.format(exp_description)), transparent=False)
