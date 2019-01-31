from models.som.som import SOM
from models.som.SOMTest import show_som
import numpy as np
from utils.constants import Constants
from utils.utils import from_npy_visual_data, from_csv, global_transform, \
    from_csv_with_filenames, from_csv_visual_10classes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import json
import argparse

DATA_TYPE = 'video'
visual_data_path = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-10classes-segm_test.npy')
#visual_data_path = os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio-10classes-coco-imagenet_train.csv')

#model_name = 'audio_20x20_s8.0_b128_a0.01_group-last_seed42_2020_std'
model_name = 'best/video_20x30_s12.0_b64_a0.1_group-segm_seed42_1548697994_minmax'
#model_name = 'best/video_20x30_s12.0_b128_a0.1_group-bbox_seed42_1548704755_global'
model_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, DATA_TYPE, model_name)
label_path = os.path.join(Constants.LABELS_FOLDER, 'coco-imagenet-10-labels.json')


def read_labels(path):
    with open(path) as f:
        labels = json.load(f)
    labels = {int(k): v for k, v in labels.items()}
    return labels

def extract_som_info(model_name):
    model_info = model_name.split('_')[1:-1]
    info = dict()
    shape = model_info[0].split('x')
    info['shape'] = tuple([int(shape[0]), int(shape[1])])
    info['sigma'] = float(model_info[1][1:])
    info['batch'] = int(model_info[2][1:])
    info['alpha'] = float(model_info[3][1:])
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise a Self Organising Map.')
    parser.add_argument('--model', metavar='model', type=str, default=model_path, help='The model neighborhood value')
    parser.add_argument('--subsample', action='store_true', default=False)
    args = parser.parse_args()

    labels_dict = read_labels(label_path)
    if 'csv' in visual_data_path:
        if DATA_TYPE == 'audio':
            xs, ys = from_csv(visual_data_path)
            xs = np.array(xs)
            ys = np.array(ys).astype(int)
        else:
            xs, ys = from_csv_visual_10classes(visual_data_path)
            ys = [v - 1000 for v in ys]
    else:
        id_to_label = json.load(open(label_path))
        id_to_label = {int(k): v for k,v in id_to_label.items()}
        xs, ys, ids_dict = from_npy_visual_data(visual_data_path, classes=10)
    labels = list(labels_dict.values())

    if args.subsample:
        np.random.seed(42)
        xs1 = []
        ys1 = []
        classes = np.arange(10)
        for i in classes:
            idx = np.where(ys == i)[0]
            idx = np.random.choice(idx, size=10)
            xs1.append(xs[idx])
            ys1.append(ys[idx])
        xs = np.array(xs1).reshape((100, 2048))
        ys = np.array(ys1).reshape(100)

    #xs, _ = global_transform(xs)
    #xs, _ = transform_data(xs)
    xs = MinMaxScaler().fit_transform(xs)
    #xs = StandardScaler().fit_transform(xs)
    dim = xs.shape[1]

    #info = extract_som_info(model_name)
    info = {'shape':[20,30], 'alpha':0.1, 'sigma':15.0, 'batch':128}
    som_shape = info['shape']
    som = SOM(som_shape[0], som_shape[1], dim, alpha=info['alpha'], sigma=info['sigma'],
              batch_size=info['batch'], checkpoint_loc=args.model, data=DATA_TYPE)
    som.restore_trained(args.model)

    show_som(som, xs, ys, labels, 'Video SOM (bounding boxes)',
             show=True, dark=False, scatter=True, legend=True, point_size=60, suffix='_segm_trsf_minmax')
