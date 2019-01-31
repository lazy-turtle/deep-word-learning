from models.som.SOM import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_npy_visual_data, global_transform, min_max_scale, \
    from_npy_audio_data, from_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from datetime import date
import pandas as pd

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
    'audio_20x30_s10.0_b64_a0.1_group-syn_seed42_syn_minmax',
    'audio_20x20_s8.0_b128_a0.01_group-last_seed42_2020_std',
    'audio_20x30_s8.0_b128_a0.3_group-20pca25t_seed42_pca_minmax',
    'audio_20x30_s10.0_b128_a0.1_group-old_seed42_old_minmax',
]

video_model = video_model_list[-2]
audio_model = audio_model_list[1]
soma_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'audio', audio_model)
somv_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'video', 'best', video_model)
hebbian_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'hebbian')

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

audio_data_paths = {
    'old': os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio10classes_old.csv'),
    'syn': os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio_10classes_synth.npy'),
    '20pca25t': os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio10classes20pca25t.csv'),
    'last': [os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio10classes-coco-imagenet_train.csv'),
             os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio10classes-coco-imagenet_test.csv')]
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


# loads three different types of audio data npy/single csv/double csv
def load_audio_data(data, seed):
    a_xs_train = a_ys_train = a_xs_test = a_ys_test = None
    if isinstance(data, list):
        a_xs_train, a_ys_train, = from_csv(data[0])
        a_xs_test, a_ys_test = from_csv(data[1])
        a_xs_train = np.array(a_xs_train)
        a_ys_train = np.array(a_ys_train).astype(int)
        a_xs_test = np.array(a_xs_test)
        a_ys_test = np.array(a_ys_test).astype(int)
    elif isinstance(data, str):
        if '.npy' not in data:
            a_xs, a_ys, _ = from_csv_with_filenames(data)
            a_xs = np.array(a_xs)
            a_ys = np.array(a_ys).astype(int)
        else:
            a_xs, a_ys = from_npy_audio_data(data)
        a_xs_train, a_xs_test, a_ys_train, a_ys_test = train_test_split(a_xs, a_ys, test_size=0.2,
                                                                   random_state=seed)
    else:
        raise ValueError("Unknown data type")
    return a_xs_train, a_xs_test, a_ys_train, a_ys_test


# loads video data and splits it
def load_video_data(data, seed):
    v_xs, v_ys, _ = from_npy_visual_data(data)
    v_xs_train, v_xs_test, v_ys_train, v_ys_test = train_test_split(v_xs, v_ys, test_size=0.2,
                                                                    random_state=seed)
    return v_xs_train, v_xs_test, v_ys_train, v_ys_test


# normalizes data following the specified transform type
def transform_data(train, test, transform_type):
    t_types = ['minmax', 'global', 'std']
    xs_train = train
    xs_test = test
    if transform_type == t_types[0]:
        scaler = MinMaxScaler()
        scaler.fit(train)
        xs_train = scaler.transform(train)
        xs_test = scaler.transform(test)
    elif transform_type == t_types[1]:
        xs_train, xs_test = global_transform(train, test)
    elif transform_type == t_types[2]:
        scaler = StandardScaler()
        scaler.fit(train)
        xs_train = scaler.transform(train)
        xs_test = scaler.transform(test)
    return xs_train, xs_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Hebbian model.')
    parser.add_argument('--lr', metavar='lr', type=float, default=5, help='The model learning rate')
    parser.add_argument('--taua', metavar='taua', type=float, default=0.2, help='Tau value audio som')
    parser.add_argument('--tauv', metavar='tauv', type=float, default=0.2, help='Tau value video som')
    parser.add_argument('--tha', metavar='tha', type=float, default=0.0, help='Threshold to cut values from (audio)')
    parser.add_argument('--thv', metavar='thv', type=float, default=0.0, help='Threshold to cut values from (video)')
    parser.add_argument('--seed', metavar='seed', type=int, default=33, help='Random generator seed')
    parser.add_argument('--somv', metavar='somv', type=str, default=somv_path,
                        help='Video SOM model path')
    parser.add_argument('--soma', metavar='soma', type=str, default=soma_path,
                        help='Audio SOM model path')
    parser.add_argument('--algo', metavar='algo', type=str, default='sorted',
                        help='Algorithm choice')
    parser.add_argument('--act', metavar='act', type=str, default='eucl',
                        help='Activation function choice')
    parser.add_argument('--source', metavar='source', type=str, default='v',
                        help='Source SOM')
    parser.add_argument('--train', action='store_true', default=False)

    args = parser.parse_args()
    somv_info = extract_som_info(os.path.basename(args.somv))
    soma_info = extract_som_info(os.path.basename(args.soma))

    # DATA
    video_data = video_data_paths[somv_info['group']]
    audio_data = audio_data_paths[soma_info['group']]

    a_xs_train, a_xs_test, a_ys_train, a_ys_test = load_audio_data(audio_data, seed=args.seed)
    v_xs_train, v_xs_test, v_ys_train, v_ys_test = load_video_data(video_data, seed=args.seed)
    a_xs_train, a_xs_test = transform_data(a_xs_train, a_xs_test, soma_info['trsf'])
    v_xs_train, v_xs_test = transform_data(v_xs_train, v_xs_test, somv_info['trsf'])

    # SOMs
    a_data_dim = len(a_xs_train[0])
    v_data_dim = len(v_xs_train[0])
    soma_dims = soma_info['dims']
    somv_dims = somv_info['dims']
    som_a = SOM(soma_dims[0], soma_dims[1], a_data_dim, checkpoint_loc=args.soma,
                tau=args.taua, threshold=args.tha)
    som_v = SOM(somv_dims[0], somv_dims[1], v_data_dim, checkpoint_loc=args.somv,
                tau=args.thv, threshold=args.tha)

    exp_description = 'lr{}_al-{}_ta{:.1f}_tv{:.1f}_tha{:.1f}_thv{:.1f}_{}_somv-{}_soma-{}'\
                          .format(args.lr, args.algo, args.taua, args.tauv,
                                  args.tha, args.thv, args.act, somv_info['id'], soma_info['id'])
    subfolder = str(date.today())
    main_path = os.path.join(hebbian_path, subfolder)
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, exp_description)

    a_data_dim = len(a_xs_train[0])
    v_data_dim = len(v_xs_train[0])
    soma_dims = soma_info['dims']
    somv_dims = somv_info['dims']
    som_a = SOM(soma_dims[0], soma_dims[1], a_data_dim, checkpoint_loc=soma_path, tau=args.taua, threshold=args.tha)
    som_v = SOM(somv_dims[0], somv_dims[1], v_data_dim, checkpoint_loc=somv_path, tau=args.tauv, threshold=args.thv)

    som_a.restore_trained(soma_path)
    som_v.restore_trained(somv_path)

    acc_a_list = []
    acc_v_list = []
    print('Preparing Hebbian model...')
    for n in range(1, num_presentations+1):
        print('Hebbian model with {} presentations.'.format(n))
        hebbian_model = HebbianModel(som_a, som_v, a_dim=a_data_dim,
                                     v_dim=v_data_dim, n_presentations=n,
                                     checkpoint_dir=model_path,
                                     learning_rate=args.lr)
        a_xs_fold, v_xs_fold, a_ys_fold, v_ys_fold = create_folds(a_xs_train, v_xs_train, a_ys_train, v_ys_train,
                                                                  n_folds=n)
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
        #hebbian_model.make_plot(a_xs_test[0], v_xs_test[0], v_ys_test[0], v_xs_fold[0],
        #                        source='a', step=n, experiment_name=exp_description)
        #hebbian_model.make_plot(v_xs_test[0], a_xs_test[0], a_ys_test[0], a_xs_fold[0],
        #                        source='v', step=n, experiment_name=exp_description)

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
    dest_path = os.path.join(Constants.PLOT_FOLDER, 'hebbian', subfolder, exp_description)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    results = pd.DataFrame({'acc_a': acc_a_list, 'acc_v': acc_v_list})
    results.to_csv(os.path.join(dest_path, 'results.csv'))
    plt.savefig(os.path.join(dest_path, '{}.png'.format(exp_description)), transparent=False)
