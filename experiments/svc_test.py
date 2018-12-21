import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from utils.constants import Constants
from utils.utils import from_csv_with_filenames
import os
import logging

OUTPUT_ERRORS = True

def fix_seq_length(xs, length=50):
    truncated = 0
    padded = 0
    print('xs[0]: {}'.format(str(xs[0].shape)))
    for i, x in enumerate(xs):
        if length < x.shape[0]:
            x = x[0:length][:]
            truncated += 1
        elif length > x.shape[0]:
            x = np.pad(x, ((0, length - x.shape[0]), (0, 0)), mode='constant', constant_values=(0))
            padded += 1
        xs[i] = x
    print('xs[0]: {}'.format(str(xs[0].shape)))
    print('Truncated {}; Padded {}'.format(truncated/len(xs), padded/len(xs)))
    return xs

def apply_loaded_pca(xs, path):
    pca = joblib.load(path)
    print('xs[0]: {}'.format(str(xs[0].shape)))
    xs = np.array([pca.transform(x) for x in xs])
    print('xs[0]: {}'.format(str(xs[0].shape)))
    return xs

def train_svc(xs, ys, k = 5):
    kf = KFold(n_splits=5)
    results_rbf = []
    results_linear = []
    flat_xs = np.array([x.ravel() for x in xs])
    ys = np.array(ys)
    for train_i, test_i in kf.split(flat_xs):
        rbf = SVC()
        linear = LinearSVC()
        print('Fitting RBF...')
        rbf.fit(flat_xs[train_i], ys[train_i])
        print('Fitting linear...')
        linear.fit(flat_xs[train_i], ys[train_i])
        pred = rbf.predict(flat_xs[test_i])
        results_rbf.append(np.average(pred == ys[test_i]))
        pred = linear.predict(flat_xs[test_i])
        results_linear.append(np.average(pred == ys[test_i]))
    print('SVC RBF: {}; SVC Linear: {}'.format(np.average(results_rbf), np.average(results_linear)))

def train_svc_report_errors(xs, ys, filenames, k=5):
    kf = KFold(n_splits=5)
    results_rbf = []
    results_linear = []
    flat_xs = np.array([x.ravel() for x in xs])
    ys = np.array(ys)
    j = 0
    for train_i, test_i in kf.split(flat_xs):
        rbf = SVC()
        linear = LinearSVC()
        print('Fitting RBF...')
        rbf.fit(flat_xs[train_i], ys[train_i])
        print('Fitting linear...')
        linear.fit(flat_xs[train_i], ys[train_i])
        pred = rbf.predict(flat_xs[test_i])
        results_rbf.append(np.average(pred == ys[test_i]))
        pred = linear.predict(flat_xs[test_i])
        results_linear.append(np.average(pred == ys[test_i]))
        print('Fold {}, wrong ones: '.format(j))
        for i, is_correct in enumerate(pred == ys[test_i]):
            if not is_correct:
                print('{}: was predicted {}'
                      .format(filenames[test_i[i]], pred[i]))
        j += 1
    print('SVC RBF: {}; SVC Linear: {}'.format(np.average(results_rbf), np.average(results_linear)))


if __name__ == '__main__':
    logging.info('Loading pickle')
    #xs, ys = load_data(os.path.join(Constants.DATA_FOLDER, 'activations.pkl'))
    xs, ys, filenames = from_csv_with_filenames(
                        os.path.join(Constants.DATA_FOLDER, '10classes', 'audio_data.csv')
                        )
    train_svc_report_errors(xs, ys, filenames)
