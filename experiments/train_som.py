from models.som.SOM import SOM
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_npy_visual_data, from_csv_visual_10classes
from sklearn.model_selection import train_test_split
from utils.utils import transform_data, global_transform
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import argparse


audio_data_path = os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio100classes.csv')
visual_data_path_a = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_a.npy')
visual_data_path_b = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_b.npy')
visual_data_path_z = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_z.npy')
visual_data_path_as = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_as.npy')
visual_data_80classes = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_80classes_train.npy')
old_visual_path = os.path.join(Constants.VIDEO_DATA_FOLDER, 'VisualInputTrainingSet.csv')

TRANSFORMS = ['none', 'zscore', 'global', 'minmax']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Hebbian model.')
    parser.add_argument('--sigma', metavar='sigma', type=float, default=10, help='The model neighborhood value')
    parser.add_argument('--alpha', metavar='alpha', type=float, default=0.01, help='The SOM initial learning rate')
    parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Random generator seed')
    parser.add_argument('--neurons1', type=int, default=20,
                        help='Number of neurons for audio SOM, first dimension')
    parser.add_argument('--neurons2', type=int, default=30,
                        help='Number of neurons for audio SOM, second dimension')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs the SOM will be trained for')
    parser.add_argument('--classes', type=int, default=10,
                        help='Number of classes the model will be trained on')
    parser.add_argument('--subsample', action='store_true', default=False)
    parser.add_argument('--data', metavar='data', type=str, default='video')
    parser.add_argument('--group', metavar='group', type=str, default='a')
    parser.add_argument('--transform', metavar='transform', type=str, default='none')
    parser.add_argument('--logging', action='store_true', default=True)
    parser.add_argument('--use-gpu', action='store_true', default=True)
    parser.add_argument('--batch', type=int, default=128)

    args = parser.parse_args()


    if args.data == 'audio':
        xs, ys, _ = from_csv_with_filenames(audio_data_path)
        ys = np.array(ys)
        xs = np.array(xs)
    elif args.data == 'video':
        print('Loading visual data...', end='')
        if args.classes == 10:
            if args.group == 'a':
                path = visual_data_path_a
            elif args.group == 'b':
                path = visual_data_path_b
            elif args.group == 'z':
                path = visual_data_path_z
            elif args.group == 'as':
                path = visual_data_path_as
            else:
                raise ValueError('Data group not recognised')
        else:
            print('Training on {} classes, good luck!'.format(args.classes))
            path = visual_data_80classes

        xs, ys, _ = from_npy_visual_data(path, classes=args.classes)
        print('done. data: {} - labels: {}'.format(xs.shape, ys.shape))
    elif args.data == 'old':
        print('Loading old visual data...', end='')
        xs, ys = from_csv_visual_10classes(old_visual_path)
        ys = [v - 1000 for v in ys]
        ys = np.array(ys)
        xs = np.array(xs)
        print('done. data: {} - labels: {}'.format(xs.shape, ys.shape))
    else:
        raise ValueError('--data argument not recognized')

    if args.transform not in TRANSFORMS:
        raise ValueError('transformation not valid, choose one of the following: {}'.format(TRANSFORMS))
    min_val = np.min(xs)
    max_val = np.max(xs)
    dim = xs.shape[1]
    som = SOM(args.neurons1, args.neurons2, dim, n_iterations=args.epochs, alpha=args.alpha,
                 tau=0.1, threshold=0.6, batch_size=args.batch, data=args.data, sigma=args.sigma,
                 num_classes=args.classes, seed=args.seed, suffix='trsf_{}_group_{}'.format(args.transform, args.group))

    if args.subsample:
        xs, _, ys, _ = train_test_split(xs, ys, test_size=0.6, stratify=ys, random_state=args.seed)
    print('Training on {} examples.'.format(len(xs)))
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2, stratify=ys, random_state=args.seed)

    if args.transform == TRANSFORMS[1]:
        print('Centering data feature-wise...')
        xs_train, xs_test = transform_data(xs_train, xs_test, rotation=args.rotation)
    elif args.transform == TRANSFORMS[2]:
        print('Normalizing with global mean and std...')
        xs_train, xs_test = global_transform(xs_train, xs_test)
    elif args.transform == TRANSFORMS[3]:
        print('Normalizing with MinMaxScaler...')
        scaler = MinMaxScaler()
        scaler.fit(xs)
        xs_train = scaler.transform(xs_train)
        xs_test = scaler.transform(xs_test)

    xs_train, xs_val, ys_train, ys_val = train_test_split(xs_train, ys_train, test_size=0.5, stratify=ys_train, random_state=args.seed)


    som.init_toolbox(xs)
    som.train(xs_train, input_classes=ys_train, test_vects=xs_val, test_classes=ys_val,
              logging=args.logging, save_every=100, log_every=100)
