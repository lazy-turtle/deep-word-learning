### TODO
## qualità rappresentazioni:
# similarità al prototipo (già fatto ma da rivedere)
# clustering (compattezza ecc)
## qualità som:
# pag 82 pdf, mean compactness, compactness variance

import numpy as np
import argparse

import sys

from models.som.som import SOM
from utils.constants import Constants
from utils.utils import from_npy_visual_data, from_csv_visual_100classes, from_csv_with_filenames, labels_dictionary, \
    global_transform, from_csv, from_csv_visual_10classes, from_npy_audio_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    parser.add_argument('--write', action='store_true', help='', default=False)
    parser.add_argument('--subsample', action='store_true', help='', default=True)
    parser.add_argument('--is-audio', action='store_true', default=True,
                        help='Specify whether the csv contains audio representations, as the loading functions are different.')
    args = parser.parse_args()

    data_type = 'video' if not args.is_audio else 'audio'
    data_group = 'visual-80classes-segm.npy' if not args.is_audio else "audio-80classes-synth.npy"
    video_model_name = 'best/video_60x60_s30.0_b256_a0.1_group-big_seed10_6060_minmax'
    audio_model_name = 'audio_10x8_s1.0_b64_a0.1_group-synth_seed42_fake_std'
    model_name = audio_model_name if args.is_audio else video_model_name
    data_path = os.path.join(Constants.DATA_FOLDER, data_type, data_group)
    model_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, data_type, model_name)
    out_path = os.path.join(Constants.OUTPUT_FOLDER, data_type, 'evaluate_som.txt')
    label_path = os.path.join(Constants.LABELS_FOLDER, 'coco-labels.json')
    labels_dict = labels_dictionary(label_path)

    id_dict = dict()
    if not args.classes100:
        num_classes = 80
        if not args.is_audio:
            xs, ys, id_dict = from_npy_visual_data(data_path, classes=num_classes)
        else:
            if data_type == 'audio':
                xs, ys = from_npy_audio_data(data_path)
            else:
                xs, ys = from_csv_visual_10classes(data_path)
                ys = [v - 1000 for v in ys]
            xs = np.array(xs)
            ys = np.array(ys).astype(int)
            id_dict = {v: v for v in np.unique(ys)}
    else:
        num_classes = 100
        if not args.is_audio:
            xs, ys = from_csv_visual_100classes(args.csv_path)
        else:
            xs, ys, _ =  from_csv_with_filenames(args.csv_path)

    #xs, _ = global_transform(xs)
    #xs = MinMaxScaler().fit_transform(xs)
    xs = StandardScaler().fit_transform(xs)

    if args.subsample:
        np.random.seed(42)
        xs1 = []
        ys1 = []
        classes = np.arange(num_classes)
        sample_size = 10
        for i in classes:
            idx = np.where(ys == i)[0]
            idx = np.random.choice(idx, size=sample_size)
            xs1.append(xs[idx])
            ys1.append(ys[idx])
        xs = np.array(xs1).reshape((sample_size*num_classes, xs.shape[1]))
        ys = np.array(ys1).reshape(sample_size*num_classes)

    som = SOM(10, 8, xs.shape[1], checkpoint_loc=model_path, data=data_type)
    som.restore_trained(model_path)
    #measure = class_compactness(som, xs, ys)
    measure = my_compactness(som, xs, ys)
    #labels = labels_dictionary(label_path)
    cpt = {labels_dict[id_dict[i]]: val for i, val in enumerate(measure)}

    f = open(out_path, 'a+') if args.write else sys.stdout
    f.write('-'*20 + '\n')
    f.write('MODEL: {} - DATA: {}\n'.format(model_name, data_group))
    f.write('Class Compactness:\n')
    for k, v in cpt.items():
        f.write('\t{:<20s}: {:.4f}\n'.format(k, v))
    f.write('\nAvg Comp: {:.4f}\n'.format(np.mean(measure)))
    f.write('Variance: {:.4f}\n'.format(np.var(measure)))
    f.close()
