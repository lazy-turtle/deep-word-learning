### TODO
## qualità rappresentazioni:
# similarità al prototipo (già fatto ma da rivedere)
# clustering (compattezza ecc)
## qualità som:
# pag 82 pdf, mean compactness, compactness variance

import numpy as np
import argparse
from models.som.SOM import SOM
from utils.constants import Constants
from utils.utils import from_npy_visual_data, from_csv_visual_100classes, from_csv_with_filenames, labels_dictionary
import os


def my_compactness(som, xs, ys):
    """
    Computes the compactness value for each class.
    :param som: self organising map instance trained on xs data
    :param xs:  input data for the SOM, shape (n_examples, n_dim)
    :param ys:  labels for each xs value , shape (n_examples,_)
    :return:    list of class compactness values
    """
    classes = np.unique(ys)
    mapped = np.array(som.map_vects(xs))  #map inputs to their BMUs

    intra_class_dist = np.zeros(classes.size)
    inter_class_dist = 0.0

    #intra cluster distance
    # avg of ||bmu(i) - bmu(j)|| for each i,j in C
    for c in classes:
        class_bmus = mapped[np.where(ys == c)]
        n = class_bmus.shape[0]
        iter_count = ((n-1) * n) / 2  # gauss n(n+1)/2
        for i, x1 in enumerate(class_bmus):
            for x2 in class_bmus[i+1:]:
                intra_class_dist[c] += np.linalg.norm(x1 - x2)
        intra_class_dist[c] /= iter_count

    #inter cluster distance
    # avg of ||bmu(i) - bmu(j)|| for each i,j in every C
    iter_count = (len(mapped)-1) * len(mapped) / 2
    for i, x1 in enumerate(mapped):
        for x2 in mapped[i+1:]:
            inter_class_dist += np.linalg.norm(x1 - x2)

    inter_class_dist /= float(iter_count)
    return intra_class_dist / inter_class_dist

def class_compactness(som, xs, ys):
    """
    Computes the compactness value for each class.
    :param som: self organising map instance trained on xs data
    :param xs:  input data for the SOM, shape (n_examples, n_dim)
    :param ys:  labels for each xs value , shape (n_examples,_)
    :return:    list of class compactness values
    """
    class_belonging_dict = {y: [] for y in list(set(ys))}
    for i, y in enumerate(ys):
        class_belonging_dict[y].append(i)
    intra_class_distance = [0 for y in list(set(ys))]
    for y in set(ys):
        for index, j in enumerate(class_belonging_dict[y]):
            x1 = xs[j]
            for k in class_belonging_dict[y][index+1:]:
                x2 = xs[k]
                _, pos_x1 = som.get_BMU(x1)
                _, pos_x2 = som.get_BMU(x2)
                intra_class_distance[y] += np.linalg.norm(pos_x1-pos_x2)
    inter_class_distance = 0
    for i, x1 in enumerate(xs):
        for j, x2 in enumerate(xs[i+1:]):
            inter_class_distance += np.linalg.norm(x1-x2)
    inter_class_distance /= len(xs)
    class_compactness = intra_class_distance/inter_class_distance
    return class_compactness

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze a SOM and get some measures.')
    parser.add_argument('--csv-path', metavar='csv_path', type=str, help='The csv file with the test data.')
    parser.add_argument('--model-path', metavar='model_path', type=str, help='The folder containing the tf checkpoint file.')
    parser.add_argument('--classes100', action='store_true',
                        help='Specify whether you are analyzing \
                        a file with representations from 100 classes, as the loading functions are different.',
                        default=False)
    parser.add_argument('--is-audio', action='store_true', default=False,
                        help='Specify whether the csv contains audio representations, as the loading functions are different.')
    args = parser.parse_args()

    data_path = os.path.join(Constants.DATA_FOLDER,
                             'video', 'visual_10classes_train_a.npy')
    model_path = os.path.join(Constants.DATA_FOLDER,
                              'saved_models', 'video_20x30_tau0.1_thrsh0.6_sigma10.0_batch128_alpha0.1_best_so_far')
    id_dict = dict()
    if not args.classes100:
        num_classes = 10
        if not args.is_audio:
            xs, ys, id_dict = from_npy_visual_data(data_path)
        else:
            xs, ys, _ = from_csv_with_filenames(args.csv_path)
            ys = [int(y)-1000 for y in ys] # see comment in average_prototype_distance_matrix
    else:
        num_classes = 100
        if not args.is_audio:
            xs, ys = from_csv_visual_100classes(args.csv_path)
        else:
            xs, ys, _ =  from_csv_with_filenames(args.csv_path)

    som = SOM(20, 30, xs.shape[1], checkpoint_loc=model_path)
    som.restore_trained(model_path)
    #measure = class_compactness(som, xs, ys)
    measure = my_compactness(som, xs, ys)
    labels = labels_dictionary('../data/labels/coco-labels.json')
    cpt = {labels[id_dict[i]]: val for i, val in enumerate(measure)}
    print('Class Compactness: {}.'.format(cpt))
    print('Avg Compactness: {}\n Variance: {}'.format(np.mean(measure), np.var(measure)))
