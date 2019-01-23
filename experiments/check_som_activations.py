from models.som.SOM import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_npy_visual_data, global_transform, min_max_scale, \
    labels_dictionary, from_npy_audio_data
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
]

audio_model_list = [
    'audio_20x30_s10.0_b128_a0.1_group-x_seed42_1145208211_minmax',
    'audio_20x30_s10.0_b128_a0.1_group-s_seed10_1547394149_minmax'
]

video_model = video_model_list[-3]
audio_model = audio_model_list[0]

#uncomment the line needed, comment the other of course
som_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'audio', audio_model)
#som_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'video', 'best', video_model)

data_paths = {
    'a': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_a.npy'),
    'b': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_b.npy'),
    'c1': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_c1.npy'),
    'c2': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_c2.npy'),
    'z': os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_z.npy'),
    'as':os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_as.npy'),
    'x': os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio_10classes_train.csv'),
    's': os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio_10classes_synth.npy')
}

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check SOM activations.')
    parser.add_argument('--lr', metavar='lr', type=float, default=10, help='The model learning rate')
    parser.add_argument('--tau', metavar='tau', type=float, default=0.5, help='Tau value audio som')
    parser.add_argument('--th', metavar='th', type=float, default=0.5, help='Threshold to cut values from')
    parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Random generator seed')
    parser.add_argument('--som', metavar='som', type=str, default=som_path,
                        help='Video SOM model path')
    parser.add_argument('--algo', metavar='algo', type=str, default='sorted',
                        help='Algorithm choice')
    parser.add_argument('--act', metavar='act', type=str, default='abs',
                        help='Activation function choice')
    parser.add_argument('--source', metavar='source', type=str, default='v',
                        help='Source SOM')
    parser.add_argument('--train', action='store_true', default=False)

    num_classes = 10
    args = parser.parse_args()
    som_info = extract_som_info(os.path.basename(args.som))
    print('SOM info:')
    for k,v in som_info.items():
        print('{:<20s}: {:<40s}'.format(k, str(v)))

    som_path = args.som
    som_data = data_paths[som_info['group']]
    labels = None
    if som_info['data'] == 'audio':
        if som_info['group'] != 's':
            xs, ys, _ = from_csv_with_filenames(som_data)
            ys = [v - 1000 for v in ys]
            xs = np.array(xs)
            ys = np.array(ys)
        else:
            xs, ys = from_npy_audio_data(som_data)
    else:
        xs, ys, index_to_id = from_npy_visual_data(som_data)
        id_to_label = labels_dictionary(os.path.join(Constants.LABELS_FOLDER, 'coco-labels.json'))
        labels = [id_to_label[index_to_id[v]] for v in np.unique(ys)]

    # transform.
    # for video SOM uncomment the required transformation, if needed
    trasf = som_info['trsf']
    if trasf == 'minmax':
        print('Using min-max scaler...')
        xs = MinMaxScaler().fit_transform(xs)
    elif trasf == 'global':
        print('Using z-score with global stats...')
        xs,_ = global_transform(xs)
    elif trasf != 'none':
        raise ValueError('Normalization not recognised, please check som filename.')

    print('xs: {}, ys: {}'.format(xs.shape, ys.shape))
    print('Data loaded and transformed, building SOM...')
    dims = som_info['dims']
    som = SOM(dims[0], dims[1], xs.shape[1], checkpoint_loc=som_path, n_iterations=10000,
                tau=args.tau, threshold=args.th)

    # Load SOM and get the activations
    som.restore_trained(som_path)

    # get random samples for each class
    num_samples = 10
    for id in range(num_classes):
        indices = np.where(ys == id)[0]
        sampled_indices = np.random.choice(indices, size=num_samples, replace=False)
        j = id * num_samples
        xs_sampled = xs[sampled_indices]

        activation = np.zeros((som._m, som._n))
        for i in range(num_samples):
            a,_ = som.get_activations(xs_sampled[i])
            activation += a.reshape((som._m, som._n))

        plt.imshow(activation, cmap='plasma', origin='lower')
        plt.title('Class: {}'.format(id if labels is None else labels[id]))
        plt.show()