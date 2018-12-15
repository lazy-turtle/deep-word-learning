from models.som.SOM import SOM
from models.som.SOMTest import showSom
import numpy as np
from utils.constants import Constants
from utils.utils import from_npy_visual_data, transform_data
from sklearn.preprocessing import MinMaxScaler
import os
import json
import argparse

visual_data_path = os.path.join(Constants.DATA_FOLDER, '10classes', 'visual_10classes_train_a.npy')
model_path = os.path.join(Constants.DATA_FOLDER, 'saved_models', 'video_20x30_tau0.1_thrsh0.6_sigma15.0_batch100_alpha0.01_final')
label_path = os.path.join(Constants.DATA_FOLDER, 'coco-labels.json')

N = 1000
dim = 2048

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise a Self Organising Map.')
    parser.add_argument('--model', metavar='model', type=str, default=model_path, help='The model neighborhood value')
    parser.add_argument('--subsample', action='store_true', default=False)
    args = parser.parse_args()

    id_to_label = json.load(open(label_path))
    id_to_label = {int(k): v for k,v in id_to_label.items()}
    xs, ys, ids_dict = from_npy_visual_data(visual_data_path)
    np.random.seed(42)

    if args.subsample:
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

    xs, _ = transform_data(xs)
    som_shape = (20, 30)

    som = SOM(som_shape[0], som_shape[1], dim, checkpoint_loc=args.model, data='video')
    som.restore_trained(args.model)

    labels = np.array([id_to_label[ids_dict[x]] for x in ys])
    showSom(som, xs, labels, 'Visual map', show=True)
