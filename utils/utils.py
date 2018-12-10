import numpy as np
import logging
import random
import pickle
import re
import csv
import sys
import json
import os
import datetime
from sklearn.decomposition import PCA
from utils.constants import Constants

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

def apply_pca(xs, n_components=25):
    pca = PCA(n_components=n_components)
    pca.fit([xi for x in xs for xi in x])
    print('xs[0]: {}'.format(str(xs[0].shape)))
    xs = [pca.transform(x) for x in xs]
    print('xs[0]: {}'.format(str(xs[0].shape)))
    return xs

def pad_np_arrays(X):
    ''' Pads a list of numpy arrays. The one with maximum length will force the others to its length.'''
    logging.debug('Shape of first example before padding: ' + str(X[0].shape))
    logging.debug('Shape of dataset before padding: ' + str(np.shape(X)))

    max_length = 0
    for x in X:
        if np.shape(x)[0] > max_length:
            max_length = np.shape(x)[0]
    X_padded = []
    for x in X:
        x_temp = np.lib.pad(x, (0, max_length - np.shape(x)[0]), 'constant', constant_values=(None, 0))
        X_padded.append(x_temp)
    logging.debug('Shape of first example after padding: ' + str(X_padded[0].shape))
    logging.debug('Shape of dataset after padding: ' + str(np.shape(X_padded)))
    return X_padded

def array_to_sparse_tuple(X):
    indices = []
    values = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            indices.append([i, j])
            values.append(X[i, j])
    return indices, values

def array_to_sparse_tuple_1d(X):
    indices = []
    values = []
    for i in range(X.shape[0]):
        indices.append(i)
        values.append(X[i])
    return indices, values

def get_next_batch_index(possible_list):
    i = random.randrange(0, len(possible_list))
    return possible_list[i]

def load_from_pickle(path):
    with open(path, 'rb') as f:
        activations = pickle.load(f)
    return activations

def extract_key(keyname):
    extract_key.re = re.compile(r".*/(\d+)")
    match = extract_key.re.match(keyname)
    if match == None:
        raise "Cannot match a label in key: %s" % (keyname)
    return match.group(1)

def load_data(filename):
    """
    Loads the data from filename and parses the keys inside
    it to retrieve the labels. Returns a pair xs,ys representing
    the data and the labels respectively
    """
    data = None
    with (open(filename, "rb")) as file:
        data = pickle.load(file)

    xs = []
    ys = []

    for key in data.keys():
        xs.append(data[key])
        #ys.append(extract_key(key))
        ys.append(key)

    return (xs,ys)

def to_csv(xs, ys, path, filename_list=None):
    np.set_printoptions(threshold=np.inf)
    with open(path, 'w') as csvfile:
        i = 0
        for x, y in zip(xs, ys):
            string = ""
            for xi in x:
                string = string + str(xi) + ','
            if filename_list != None:
                string = filename_list[i] + ',' + string
            string = string + str(y) + '\n'
            csvfile.write(string)
            i += 1
    np.set_printoptions(threshold=1000)

def from_csv(path):
    xs = []
    ys = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x = np.asfarray(row[:-1])
            y = row[-1]
            xs.append(x)
            ys.append(y)
    return xs, ys

def from_csv_with_filenames(path):
    xs = []
    ys = []
    filenames = []
    with open(path, newline='') as csvfile:
        for line in csvfile:
            l = line.split(',')
            xs.append(np.asfarray(l[1:-1]))
            ys.append(int(l[-1].strip('\n')))
            filenames.append(l[0])
    return xs, ys, filenames

def infer_label_10classes(label_string, labels_dict):
    label_string = label_string.split('/')[5]
    return int(labels_dict[label_string])

def from_csv_visual_10classes(path, labels='imagenet-labels.json'):
    f = open(path,'r')
    labels_dict_path = os.path.join(Constants.DATA_FOLDER, labels)
    labels_dict = json.load(open(labels_dict_path))
    labels_dict = {v: k for k, v in labels_dict.items()}
    xs = []
    ys = []
    for l in f:
        lSplit = l.split(',')
        xs.append(np.array(lSplit[1:]).astype(float))
        ys.append(infer_label_10classes(lSplit[0], labels_dict))
    f.close()
    return xs, ys

def coco_visual_data(path, classes=10):
    data = np.load(path)
    xs = data[:,:-1]
    ys = data[:,-1]
    assert np.unique(ys).size == classes
    return xs, ys

def synsets_txt_to_dict(txt_path):
    """
    Reads a file such as thesis_synsets_descr.txt, which is organized this way:

    [synset name] [description] \n
    [synset name] [description] \n
    ...

    And returns a dictionary in which the values are labels from 0 to 99 and the
    keys are the synset names.
    """
    f = open(txt_path, 'r')
    d = {}
    for i, l in enumerate(f):
        synset = l.split(" ")[0]
        d[synset] = i
    return d


def from_csv_visual_100classes(path):
    f = open(path,'r')
    labels_txt_path = os.path.join(Constants.DATA_FOLDER, '100classes',
                                   'thesis_synsets_fix.txt')
    synset_dict = synsets_txt_to_dict(labels_txt_path)
    xs = []
    ys = []
    for l in f:
        lSplit = l.split(',')
        data = lSplit[1:]
        img_path = lSplit[0]
        img_name = img_path.split('/')[6]
        synset = img_name.split('_')[0]
        assert len(img_name.split('_')) == 2
        xs.append(np.array(data).astype(float))
        ys.append(synset_dict[synset])
    f.close()
    return xs, ys

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def get_plot_filename(folder_path):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    i = 0
    while os.path.exists(os.path.join(folder_path, date_str + "_" + str(i)) + '.png'):
        i += 1
    return date_str + "_" + str(i) + '.png'

def create_folds(a_xs, v_xs, a_ys, v_ys, n_folds=1, n_classes=100):
    '''
    In this context, a fold is an array of data that has n_folds examples
    from each class.
    '''
    #assert len(a_xs) == len(v_xs) == len(a_ys) == len(v_ys)
    assert n_folds * n_classes <= len(a_xs)
    ind = a_ys.argsort()
    a_xs = a_xs[ind]
    a_ys = a_ys[ind]
    ind = v_ys.argsort()
    v_xs = v_xs[ind]
    v_ys = v_ys[ind]
    # note that a_xs_ is not a_xs
    a_xs_ = [a_x for a_x in a_xs]
    a_ys_ = [a_y for a_y in a_ys]
    v_xs_ = [v_x for v_x in v_xs]
    v_ys_ = [v_y for v_y in v_ys]
    a_xs_fold = []
    a_ys_fold = []
    v_xs_fold = []
    v_ys_fold = []
    for i in range(n_folds):
        for c in range(n_classes):
            a_idx = a_ys_.index(c)
            v_idx = v_ys_.index(c)
            a_xs_fold.append(a_xs_[a_idx])
            a_ys_fold.append(c)
            v_xs_fold.append(v_xs_[v_idx])
            v_ys_fold.append(c)
            # delete elements so that they are not found again
            # and put in other folds
            del a_xs_[a_idx]
            del a_ys_[a_idx]
            del v_xs_[v_idx]
            del v_ys_[v_idx]
    return a_xs_fold, v_xs_fold, a_ys_fold, v_ys_fold

def transform_data(xs, test_xs, rotation=True):
    '''
    Makes the column-wise mean of the input matrix xs 0; renders the variance 1;
    uses the eigenvector matrix $U^T$ to center the data around the direction
    of maximum variation in the data matrix xs.
    '''
    xs -= np.mean(xs, axis=0)
    xs /= np.std(xs, axis=0)
    if rotation:
        covariance_matrix = np.cov(xs, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(covariance_matrix)
        idx = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[idx]
        xs = np.dot(eig_vecs.T, xs.T).T

    test_xs -= np.mean(xs, axis=0)
    test_xs /= np.std(xs, axis=0)
    if rotation:
        test_xs = np.dot(eig_vecs.T, xs.T).T
    return xs, test_xs
