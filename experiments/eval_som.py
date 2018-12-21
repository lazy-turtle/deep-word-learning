from models.som.SOM import SOM
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_csv_visual_100classes, from_csv, to_csv, transform_data
from sklearn.model_selection import train_test_split

import os
import argparse
import numpy as np

audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '100classes',
                               'audio100classes.csv')
visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                '100classes',
                                'VisualInputTrainingSet.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a SOM.')
    parser.add_argument('path', type=str, help='The path to the trained SOM model')
    parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Random generator seed')
    parser.add_argument('--data', metavar='data', type=str, default='audio')
    parser.add_argument('--neurons1', type=int, default=50,
                        help='Number of neurons for SOM, first dimension')
    parser.add_argument('--neurons2', type=int, default=50,
                        help='Number of neurons for SOM, second dimension')
    parser.add_argument('--subsample', action='store_true', default=False)
    parser.add_argument('--rotation', action='store_true', default=False)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--sigma', metavar='sigma', type=float, default=10, help='The model neighborhood value')
    parser.add_argument('--alpha', metavar='alpha', type=float, default=0.0001, help='The SOM initial learning rate')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs the SOM will be trained for')

    args = parser.parse_args()

    if args.data == 'audio':
        xs, ys, _ = from_csv_with_filenames(audio_data_path)
    elif args.data == 'video':
        xs, ys = from_csv_visual_100classes(visual_data_path)
    else:
        raise ValueError('--data argument not recognized')

    dim = len(xs[0])

    som = SOM(args.neurons1, args.neurons2, dim, n_iterations=args.epochs, alpha=args.alpha, checkpoint_loc=args.path,
                 tau=0.1, threshold=0.6, batch_size=args.batch, data=args.data, sigma=args.sigma)

    ys = np.array(ys)
    xs = np.array(xs)

    if args.subsample:
        xs, _, ys, _ = train_test_split(xs, ys, test_size=0.6, stratify=ys, random_state=args.seed)

    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2, stratify=ys,
                                                            random_state=args.seed)

    xs_train, xs_val, ys_train, ys_val = train_test_split(xs_train, ys_train, test_size=0.5, stratify=ys_train,
                                                            random_state=args.seed)

    xs_train, xs_test = transform_data(xs_train, xs_val, rotation=args.rotation)

    som.restore_trained(args.path)

    som.print_som_evaluation(xs_val, ys_val)
    
