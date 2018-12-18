from models.som.SOM import SOM
from models.som.SOMTest import show_som
import numpy as np
from utils.constants import Constants
from utils.utils import from_npy_visual_data, transform_data, from_csv_visual_10classes
import os
import json
import argparse

visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                'video',
                                'visual_10classes_train_a.npy')
model_name = 'video_20x30_tau0.1_thrsh0.6_sigma5.0_batch128_alpha0.1_final'
model_path = os.path.join(Constants.DATA_FOLDER,
                          'saved_models',
                           model_name)
label_path = os.path.join(Constants.DATA_FOLDER, 'labels', 'coco-labels.json')


def extract_som_info(model_name):
    model_info = model_name.split('_')[1:-1]
    info = dict()
    shape = model_info[0].split('x')
    info['shape'] = tuple([int(shape[0]), int(shape[1])])
    info['sigma'] = float(model_info[3][5:])
    info['batch'] = int(model_info[4][5:])
    info['alpha'] = float(model_info[5][5:])
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
        xs, ys, ids_dict = from_npy_visual_data(visual_data_path)
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

    #xs, _ = transform_data(xs)
    dim = xs.shape[1]

    info = extract_som_info(model_name)
    som_shape = info['shape']
    som = SOM(som_shape[0], som_shape[1], dim, alpha=info['alpha'], sigma=info['sigma'],
              batch_size=info['batch'], checkpoint_loc=args.model, data='video')
    som.restore_trained(args.model)

    show_som(som, xs, labels, 'Visual map', show=False, dark=True)
