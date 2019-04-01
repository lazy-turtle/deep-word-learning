from models.som.som import SOM
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_npy_visual_data, from_csv_visual_10classes, from_npy_audio_data, \
    from_csv
from sklearn.model_selection import train_test_split
from utils.utils import transform_data, global_transform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import numpy as np
import argparse


audio_data_list =[
    os.path.join(Constants.AUDIO_DATA_FOLDER, 'new', 'audio10classes25pca20t.csv'),
    os.path.join(Constants.AUDIO_DATA_FOLDER, 'new', 'audio10classes20pca25t.csv'),
    os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio-10classes-coco-imagenet_train.csv'),
    os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio-10classes-coco-imagenet_test.csv'),
    os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio-80classes-synth.npy')
]
audio_data_path = audio_data_list[-1]

visual_data_path_a = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_a.npy')
visual_data_path_b = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_b.npy')
visual_data_path_c2 = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_c2.npy')
visual_data_path_z = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_10classes_train_z.npy')
visual_data_segm = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-10classes-segm.npy')
visual_data_bbox = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-10classes-bbox.npy')
visual_data_imagenet = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-10classes-imagenet.npy')
old_visual_path = os.path.join(Constants.VIDEO_DATA_FOLDER, 'VisualInputTrainingSet.csv')

big_visual_data_path = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-80classes-segm.npy')

TRANSFORMS = ['none', 'zscore', 'global', 'minmax', 'std']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Hebbian model.')
    parser.add_argument('--sigma', metavar='sigma', type=float, default=20, help='The model neighborhood value')
    parser.add_argument('--alpha', metavar='alpha', type=float, default=0.1, help='The SOM initial learning rate')
    parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Random generator seed')
    parser.add_argument('--neurons1', type=int, default=40,
                        help='Number of neurons for audio SOM, first dimension')
    parser.add_argument('--neurons2', type=int, default=40,
                        help='Number of neurons for audio SOM, second dimension')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs the SOM will be trained for')
    parser.add_argument('--classes', type=int, default=80,
                        help='Number of classes the model will be trained on')
    parser.add_argument('--subsample', action='store_true', default=False)
    parser.add_argument('--data', metavar='data', type=str, default='audio')
    parser.add_argument('--group', metavar='group', type=str, default='synth')
    parser.add_argument('--transform', metavar='transform', type=str, default='std')
    parser.add_argument('--logging', action='store_true', default=True)
    parser.add_argument('--use-gpu', action='store_true', default=True)
    parser.add_argument('--batch', type=int, default=64)

    args = parser.parse_args()


    if args.data == 'audio':
        print('Loading audio data...', end='')
        if 'csv' not in audio_data_path:
            xs, ys = from_npy_audio_data(audio_data_path, classes=10)
        else:
            print('Reading from csv...')
            xs_train, ys_train = from_csv(audio_data_path)
            xs_test, ys_test = from_csv(audio_data_list[-2])
            xs = np.array(xs_train + xs_test)
            ys = np.array(ys_train + ys_test, dtype=int)
            print(xs.shape, ys.shape)

        print('done, data: {} - labels: {}'.format(xs.shape, ys.shape))
    elif args.data == 'video':
        print('Loading visual data, group {}...'.format(args.group), end='')
        if args.classes == 10:
            if args.group == 'a':
                path = visual_data_path_a
            elif args.group == 'b':
                path = visual_data_path_b
            elif args.group == 'z':
                path = visual_data_path_z
            elif args.group == 'segm':
                path = visual_data_segm
            elif args.group == 'bbox':
                path = visual_data_bbox
            elif args.group == 'imagenet':
                path = visual_data_imagenet
            else:
                raise ValueError('Data group not recognised')
        else:
            print('Training on {} classes, good luck!'.format(args.classes))
            path = big_visual_data_path

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

    dim = xs.shape[1]
    som = SOM(args.neurons1, args.neurons2, dim, n_iterations=args.epochs, alpha=args.alpha,
                 tau=0.1, threshold=0.6, batch_size=args.batch, data=args.data, sigma=args.sigma,
                 num_classes=args.classes, seed=args.seed, suffix='trsf_{}_group-{}'.format(args.transform, args.group))

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
    elif args.transform == TRANSFORMS[4]:
        print('Normalizing with standard scaler')
        scaler = StandardScaler()
        scaler.fit(xs)
        xs_train = scaler.transform(xs_train)
        xs_test = scaler.transform(xs_test)

    xs_train, xs_val, ys_train, ys_val = train_test_split(xs_train, ys_train, test_size=0.4, stratify=ys_train, random_state=args.seed)

    som.init_toolbox(xs)
    som.train(xs_train, input_classes=ys_train, test_vects=xs_val, test_classes=ys_val,
              logging=args.logging, save_every=100, log_every=100)
