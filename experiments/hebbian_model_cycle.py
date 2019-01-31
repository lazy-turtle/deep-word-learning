import argparse
import sys

from models.som.som import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_npy_visual_data, global_transform, min_max_scale, \
    from_npy_audio_data, from_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from datetime import date
import pandas as pd
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

video_model_list = [
    'video_20x30_s12.0_b64_a0.1_group-segm_seed42_1548697994_minmax',
    'video_20x30_s12.0_b128_a0.1_group-bbox_seed42_1548704755_global',
]

audio_model_list = [
    'audio_20x20_s8.0_b128_a0.01_group-last_seed42_2020_std',
    'audio_20x30_s8.0_b128_a0.3_group-20pca25t_seed42_pca_minmax',
    'audio_20x30_s10.0_b128_a0.1_group-old_seed42_old_minmax',
]

hebbian_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'hebbian')

video_data_paths = {
    'segm': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-10classes-segm.npy'),
    'bbox': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-10classes-bbox.npy')
}

audio_data_paths = {
    'old': os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio-10classes-old.csv'),
    '20pca25t': os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio-10classes-20pca25t.csv'),
    'last': [os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio-10classes-coco-imagenet_train.csv'),
             os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio-10classes-coco-imagenet_test.csv')]
}

lr_values = [5, 10, 20]
tauv_values = [0.2, 0.5, 1.0]
taua_values = [0.2, 0.5, 1.0]
thv_values = [0.0, 0.2, 0.5]
tha_values = [0.0, 0.2, 0.5]


#################
# Code
#################

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


def print_info(somv_info, soma_info):
    print('---------| Video SOM Info |---------')
    keys = somv_info.keys()
    print('{:<10s}  {:<40s}: {:<40s}'.format("", "Video", "Audio"))
    for k in keys:
        v1 = str(somv_info[k])
        v2 = str(soma_info[k])
        print('{:<10s}: {:<40s} {:<40s}'.format(k, v1, v2))
    print("")


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
        a_xs, a_ys, _ = from_csv_with_filenames(data)
        a_xs = np.array(a_xs)
        a_ys = np.array(a_ys).astype(int)
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


# Main block: train a complete hebbian model with the given parameters
def iterate(path_som_video, path_som_audio, lr, taua, tauv, tha, thv, seed=42, it=-1, n_present=15, algo="sorted"):
    global hebbian_path
    path_som_video = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'video', 'best', path_som_video)
    path_som_audio = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'audio', path_som_audio)

    somv_info = extract_som_info(os.path.basename(path_som_video))
    soma_info = extract_som_info(os.path.basename(path_som_audio))
    print_info(somv_info, soma_info)

    # DATA
    video_data = video_data_paths[somv_info['group']]
    audio_data = audio_data_paths[soma_info['group']]

    a_xs_train, a_xs_test, a_ys_train, a_ys_test = load_audio_data(audio_data, seed=seed)
    v_xs_train, v_xs_test, v_ys_train, v_ys_test = load_video_data(video_data, seed= seed)
    a_xs_train, a_xs_test = transform_data(a_xs_train, a_xs_test, soma_info['trsf'])
    v_xs_train, v_xs_test = transform_data(v_xs_train, v_xs_test, somv_info['trsf'])

    #SOMs
    a_data_dim = len(a_xs_train[0])
    v_data_dim = len(v_xs_train[0])
    soma_dims = soma_info['dims']
    somv_dims = somv_info['dims']
    som_a = SOM(soma_dims[0], soma_dims[1], a_data_dim, checkpoint_loc=path_som_audio,
                tau=taua, threshold=tha)
    som_v = SOM(somv_dims[0], somv_dims[1], v_data_dim, checkpoint_loc=path_som_video,
                tau=tauv, threshold=thv)
    som_a.restore_trained(path_som_audio)
    som_v.restore_trained(path_som_video)

    # HEBBIAN MODEL
    exp_description = 'lr{}_al-{}_ta{:.1f}_tv{:.1f}_tha{:.1f}_thv{:.1f}_{}_somv-{}_soma-{}' \
                          .format(lr, algo, taua, tauv, tha, thv, "eucl", somv_info['id'], soma_info['id'])
    main_path = os.path.join(hebbian_path, str(date.today()))
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, exp_description)

    acc_a_list = []
    acc_v_list = []
    for n in range(1, n_present+1):
        print('Hebbian model with {} presentations.'.format(n))
        hebbian_model = HebbianModel(som_a, som_v, a_dim=a_data_dim,
                                     v_dim=v_data_dim, n_presentations=n,
                                     checkpoint_dir=model_path,
                                     learning_rate=lr)
        a_xs_fold, v_xs_fold, a_ys_fold, v_ys_fold = create_folds(a_xs_train, v_xs_train, a_ys_train, v_ys_train, n_folds=n)
        som_a.memorize_examples_by_class(a_xs_train, a_ys_train)
        som_v.memorize_examples_by_class(v_xs_train, v_ys_train)

        print('Training...')
        hebbian_model.train(a_xs_fold, v_xs_fold, step=n)
        print('Evaluating...')

        accuracy_a = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='a',
                                            prediction_alg=algo)
        accuracy_v = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='v',
                                            prediction_alg=algo)
        print('n={}, accuracy_a={}, accuracy_v={}'.format(n, accuracy_a, accuracy_v))
        acc_a_list.append(accuracy_a * 100)
        acc_v_list.append(accuracy_v * 100)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    xx = np.arange(1, n_present + 1)
    ax.plot(xx, acc_a_list, color='teal')
    ax.plot(xx, acc_v_list, color='orange')

    ax.set_xticks(xx)
    ax.set_yticks(np.arange(0, 101, 5))
    ax.yaxis.set_major_formatter(tkr.PercentFormatter())
    ax.grid(True, axis='y', linestyle=':')

    ax.set_xlabel('# presentations')
    ax.set_ylabel('Accuracy')
    plt.title('Model: {}'.format(exp_description))
    dest_path = os.path.join(Constants.PLOT_FOLDER, 'hebbian', str(date.today()), exp_description)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    results = pd.DataFrame({'acc_a': acc_a_list, 'acc_v': acc_v_list})
    suffix = str(it) if it >= 0 else ""
    results.to_csv(os.path.join(dest_path, 'results_{}.csv'.format(suffix)))
    plt.savefig(os.path.join(dest_path, '{}_{}.png'.format(exp_description, suffix)), transparent=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automation of the hebbian model training.")
    parser.add_argument('--combine', action='store_true', default=False, help='Generate all combinations or not.')
    parser.add_argument('--slices', metavar='slices', type=int, default=2, help='How many combination blocks to generate.')
    parser.add_argument('--index', metavar='index', type=int, default=1, help='Which slice to process.')
    parser.add_argument('--iter', metavar='iter', type=int, default=1, help='How many iterations for each combination.')
    parser.add_argument('--param-csv', metavar='param_csv', type=str, default=None, help='File containing parameters')
    args = parser.parse_args()



    if args.combine:
        print('Generating all possible combinations')
        iterator = itertools.product(video_model_list, audio_model_list,lr_values,
                                         taua_values, tauv_values,
                                         tha_values, thv_values)
        combinations = list(iterator)
    else:
        print('Loading combinations from {}'.format(args.param_csv))
        df = pd.read_csv(args.param_csv, index_col=0)
        combinations = []
        for row in df.values:
            combinations.append(tuple(row))

    print(combinations)
    print('Total combinations: {}'.format(len(combinations)))
    n_slices = args.slices
    slice_index = args.index
    slice_size = int(np.ceil(len(combinations) / float(n_slices)))
    print("# slice: {}, slice size: {}".format(slice_index, slice_size))
    start_index = slice_index * slice_size
    end_index = (slice_index + 1) * slice_size

    for i, t in enumerate(combinations[start_index:end_index]):
        for j in range(args.iter):
            iterate(*t, seed=j, it=j)
