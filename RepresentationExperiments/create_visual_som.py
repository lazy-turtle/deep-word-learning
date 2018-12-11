from models.som.SOM import SOM
from models.som.SOMTest import showSom
import numpy as np
from utils.constants import Constants
from utils.utils import from_npy_visual_data
from sklearn.preprocessing import MinMaxScaler
import os
import logging

visual_data_path = os.path.join(Constants.DATA_FOLDER, '10classes', 'visual_10classes_train.npy')
model_path = os.path.join(Constants.DATA_FOLDER, 'saved_models', 'video_20x30_tau0.1_thrsh0.6_sigma10.0_batch100_alpha0.1_final')

N = 1000
dim = 2048

if __name__ == '__main__':
    v_xs, v_ys, label_dict = from_npy_visual_data(visual_data_path)
    v_xs = MinMaxScaler().fit_transform(v_xs)

    som = SOM(20, 30, dim, n_iterations=60, alpha=0.1, sigma=10.0, tau=0.1, batch_size=100, num_classes=10,
              checkpoint_loc=model_path,data='video')

    som.restore_trained(model_path)

    showSom(som, v_xs, v_ys, 1, 'Visual map')
