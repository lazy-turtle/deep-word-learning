from models.som.SOM import SOM
from models.som.SOMTest import show_som
import numpy as np
from utils.constants import Constants
from utils.utils import from_npy_visual_data, transform_data, from_csv_visual_10classes, global_transform
import os
import json
import argparse

visual_data_path = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual_80classes_train.npy')
model_name = 'video_60x60_s30.0_b256_a0.1__seed10_1545471476_final'
model_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, 'video', model_name)
label_path = os.path.join(Constants.LABELS_FOLDER, 'coco-labels.json')


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

    if 'csv' in visual_data_path:
        xs, ys = from_csv_visual_10classes(visual_data_path)
        ys = [v - 1000 for v in ys]
        xs = np.array(xs)
        ys = np.array(ys)
        labels = ys
    else:
        id_to_label = json.load(open(label_path))
        id_to_label = {int(k): v for k,v in id_to_label.items()}
        xs, ys, ids_dict = from_npy_visual_data(visual_data_path, classes=80)
        labels = np.array([id_to_label[ids_dict[x]] for x in ys])

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

    xs, _ = global_transform(xs)
    dim = xs.shape[1]

    #info = extract_som_info(model_name)
    info = {'shape':[60,60], 'alpha':0.1, 'sigma':20.0, 'batch':128}
    som_shape = info['shape']
    som = SOM(som_shape[0], som_shape[1], dim, alpha=info['alpha'], sigma=info['sigma'],
              batch_size=info['batch'], checkpoint_loc=args.model, data='video')
    som.restore_trained(args.model)

    show_som(som, xs, labels, 'Visual map', show=True, dark=True, suffix='')