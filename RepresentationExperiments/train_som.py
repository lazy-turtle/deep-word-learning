from models.som.SOM import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_npy_visual_data, from_csv, to_csv
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.utils import create_folds, transform_data
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

soma_path = os.path.join(Constants.DATA_FOLDER, '100classes', 'audio_model_new', '')
audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '10classes',
                               'audio100classes.csv')
visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                '10classes',
                                'visual_10classes_train_b.npy')


def normalize(xs, xs_test=None):
    m = xs.min()
    M = xs.max()
    z = ((xs - m)/(M - m)) * 2 - 1
    z_test = None
    if xs_test is not None:
        z_test = ((xs_test - m) / (M - m)) * 2 - 1
    return z, z_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Hebbian model.')
    parser.add_argument('--sigma', metavar='sigma', type=float, default=30, help='The model neighborhood value')
    parser.add_argument('--alpha', metavar='alpha', type=float, default=0.3, help='The SOM initial learning rate')
    parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Random generator seed')
    parser.add_argument('--neurons1', type=int, default=20,
                        help='Number of neurons for audio SOM, first dimension')
    parser.add_argument('--neurons2', type=int, default=30,
                        help='Number of neurons for audio SOM, second dimension')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs the SOM will be trained for')
    parser.add_argument('--classes', type=int, default=10,
                        help='Number of classes the model will be trained on')
    parser.add_argument('--subsample', action='store_true', default=False)
    parser.add_argument('--data', metavar='data', type=str, default='video')
    parser.add_argument('--rotation', action='store_true', default=False)
    parser.add_argument('--logging', action='store_true', default=True)
    parser.add_argument('--batch', type=int, default=100)

    args = parser.parse_args()


    if args.data == 'audio':
        xs, ys, _ = from_csv_with_filenames(audio_data_path)
    elif args.data == 'video':
        print('Loading visual data...', end='')
        xs, ys, _ = from_npy_visual_data(visual_data_path)
        print('done. data: {} - labels: {}'.format(xs.shape, ys.shape))
        print("done")
    else:
        raise ValueError('--data argument not recognized')

    dim = xs.shape[1]
    som = SOM(args.neurons1, args.neurons2, dim, n_iterations=args.epochs, alpha=args.alpha,
                 tau=0.1, threshold=0.6, batch_size=args.batch, data=args.data, sigma=args.sigma,
                 num_classes=args.classes)

    # not necessary, already loading numpy arrays
    # ys = np.array(ys)
    # xs = np.array(xs)

    if args.subsample:
        xs, _, ys, _ = train_test_split(xs, ys, test_size=0.6, stratify=ys, random_state=args.seed)
    print('Training on {} examples.'.format(len(xs)))

    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2, stratify=ys, random_state=args.seed)
    xs_train, xs_val, ys_train, ys_val = train_test_split(xs_train, ys_train, test_size=0.5, stratify=ys_train, random_state=args.seed)
    xs_train, xs_test = transform_data(xs_train, xs_val, rotation=args.rotation)

    som.train(xs_train, input_classes=ys_train, test_vects=xs_val, test_classes=ys_val,
              logging=args.logging, save_every=1)
