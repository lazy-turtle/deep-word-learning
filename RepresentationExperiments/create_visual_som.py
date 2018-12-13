from models.som.SOM import SOM
from models.som.SOMTest import showSom
import numpy as np
from utils.constants import Constants
from utils.utils import from_npy_visual_data, transform_data
from sklearn.preprocessing import MinMaxScaler
import os
import json
import logging

visual_data_path = os.path.join(Constants.DATA_FOLDER, '10classes', 'visual_10classes_train_a.npy')
model_path = os.path.join(Constants.DATA_FOLDER, 'saved_models', 'video_20x30_tau0.1_thrsh0.6_sigma10.0_batch100_alpha0.1_final')
label_path = os.path.join(Constants.DATA_FOLDER, 'coco-labels.json')

N = 1000
dim = 2048

if __name__ == '__main__':
    id_to_label = json.load(open(label_path))
    id_to_label = {int(k): v for k,v in id_to_label.items()}
    v_xs, v_ys, ids_dict = from_npy_visual_data(visual_data_path)
    np.random.seed(42)

    xs = []
    ys = []
    classes = np.arange(10)
    for i in classes:
        idx = np.where(v_ys == i)[0]
        idx = np.random.choice(idx, size=10)
        xs.append(v_xs[idx])
        ys.append(v_ys[idx])
    #v_xs = MinMaxScaler().fit_transform(v_xs)

    xs = np.array(xs).reshape((100, 2048))
    ys = np.array(ys).reshape(100)
    xs, _ = transform_data(xs)

    som = SOM(20, 30, dim, n_iterations=60, alpha=0.1, sigma=10.0, tau=0.1, batch_size=100, num_classes=10,
              checkpoint_loc=model_path,data='video')

    som.restore_trained(model_path)
    labels = np.array([id_to_label[ids_dict[x]] for x in ys])
    showSom(som, xs, labels, 'Visual map')
