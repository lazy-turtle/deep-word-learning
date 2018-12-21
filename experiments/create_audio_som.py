from models.som.SOM import SOM
from models.som.SOMTest import showSom
from utils.utils import load_from_pickle
from utils.utils import load_data
from utils.utils import from_csv_with_filenames
from utils.constants import Constants
from sklearn.externals import joblib
import os
import logging
import numpy as np

csv_path = os.path.join(Constants.DATA_FOLDER, '10classes', 'audio_data.csv')
LOAD = True

if __name__ == '__main__':
    logging.info('Loading data')
    xs, ys, filenames = from_csv_with_filenames(csv_path)
    ys = [int(y)-1000 for y in ys]
    vect_size = len(xs[0])
    audio_som = SOM(20, 30, vect_size, n_iterations=100,
        checkpoint_dir=os.path.join(Constants.DATA_FOLDER, '10classes', 'audio_model', ''))
    if not LOAD:
        audio_som.train(xs)
    else:
        logging.info('Training som')
        audio_som.restore_trained()
    #audio_som.plot_som(xs, ys, plot_name='audio_som.png')
    showSom(audio_som, xs, ys, 1, 'Audio Map', filenames=filenames)
