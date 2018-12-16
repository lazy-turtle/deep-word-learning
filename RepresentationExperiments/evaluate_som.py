### TODO
## qualità rappresentazioni:
# similarità al prototipo (già fatto ma da rivedere)
# clustering (compattezza ecc)
## qualità som:
# pag 82 pdf, mean compactness, compactness variance

import numpy as np
import argparse
from models.som.SOM import SOM
from utils.utils import from_csv_with_filenames, from_csv_visual_10classes, from_csv_visual_100classes



def class_compactness(som, xs, ys):
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
    parser.add_argument('--csv-path', metavar='csv_path', type=str, required=True, help='The csv file with the test data.')
    parser.add_argument('--model-path', metavar='model_path', type=str, required=True, help='The folder containing the tf checkpoint file.')
    parser.add_argument('--classes100', action='store_true',
                        help='Specify whether you are analyzing \
                        a file with representations from 100 classes, as the loading functions are different.',
                        default=False)
    parser.add_argument('--is-audio', action='store_true', default=False,
                        help='Specify whether the csv contains audio representations, as the loading functions are different.')
    args = parser.parse_args()

    if not args.classes100:
        num_classes = 10
        if not args.is_audio:
            xs, ys = from_csv_visual_10classes(args.csv_path)
        else:
            xs, ys, _ = from_csv_with_filenames(args.csv_path)
        ys = [int(y)-1000 for y in ys] # see comment in average_prototype_distance_matrix
    else:
        num_classes = 100
        if not args.is_audio:
            xs, ys = from_csv_visual_100classes(args.csv_path)
        else:
            xs, ys, _ =  from_csv_with_filenames(args.csv_path)

    som = SOM(20, 30, len(xs[0]), checkpoint_dir=args.model_path)
    som.restore_trained()
    measure = class_compactness(som, xs, ys)
    print('Class Compactness: {}.'.format(measure))
    print('Avg Compactness: {}\n Variance: {}'.format(np.mean(measure), np.var(measure)))
