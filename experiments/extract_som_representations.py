import numpy as np
import argparse
import json
import os

class ExtractConfig(object):
    DATA_PATH = '/usr/home/studenti/sp160362/data/representations/representations_train_raw.npy'
    DEST_PATH = '../data/video/'
    RESULT_NAME ='visual_10classes_train_c2.npy'

    LABELS_DICT = '../data/labels/coco-labels.json'
    CLASSES_PATH = '../data/labels/coco_labels10classes_c.txt'

    SAMPLES = 100
    SAMPLES_SORT = 1000


def read_selected_classes(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def sqr_distance(v1, v2):
    return np.sum((v1 - v2)**2)


def avg_distance(x, others):
    distances = np.sum((others[:,:-1] - x[:-1])**2, axis=1)
    return np.mean(distances)

def main():
    cfg = ExtractConfig()
    parser = argparse.ArgumentParser(description='Generate the required data files for training the SOM.')
    parser.add_argument('--data', type=str, default=cfg.DATA_PATH, help='Raw data matrix location.')
    parser.add_argument('--dest', type=str, default=cfg.DEST_PATH, help='Destination folder.')
    parser.add_argument('--name', type=str, default=cfg.RESULT_NAME, help='Name of the resulting file.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for deterministic results.')
    parser.add_argument('--sort', action='store_true', default=False)
    args = parser.parse_args()

    cfg.DATA_PATH = args.data
    cfg.DEST_PATH = args.dest
    cfg.RESULT_NAME = args.name
    np.random.seed(args.seed)

    print("Loading data...")
    raw_data = np.load(cfg.DATA_PATH)
    labels = raw_data[:,-1]
    print("Data loaded, shape: {}".format(raw_data.shape))
    print("Distance based selection: {}".format(args.sort))

    print("Loading labels...")
    labels_dict = json.load(open(cfg.LABELS_DICT))
    name_to_id = {v: k for k, v in labels_dict.items()}

    class_names = read_selected_classes(cfg.CLASSES_PATH)
    ids = [int(name_to_id[n]) for n in class_names]

    selected = zip(ids, class_names)
    selected = sorted(selected, key = lambda t: t[0])
    num_classes = len(selected)
    print("Selected labels: ", selected)

    result = np.empty((num_classes * cfg.SAMPLES, raw_data.shape[1]))

    if args.sort:
        # sort by distance from the other instances, descending
        # and take only the most isolated ones
        print('Sub-sampling data...')
        #first select a subsample from each class
        data_subsample = np.empty((0, raw_data.shape[1]))
        for i, (id, label) in enumerate(selected):
            indices = np.where(labels == id)[0]
            samples = max(cfg.SAMPLES_SORT, len(indices))
            sampled_indices = np.random.choice(indices, size=samples, replace=False)
            data_subsample = np.concatenate((data_subsample, raw_data[sampled_indices]), axis=0)

        #then extract the n most distant ones for each class
        labels_subsample = data_subsample[:,-1]
        print('Subsampled data: {}'.format(data_subsample.shape))
        for i, (id, label_name) in enumerate(selected):
            print('Selecting the best {} samples from "{}..."'.format(cfg.SAMPLES, label_name))
            indices = np.where(labels_subsample == id)[0]
            xs_class = data_subsample[indices]

            mask = np.ones(labels_subsample.shape, dtype=bool)
            mask[indices] = False
            other_sampled_indices = np.random.choice(np.where(mask)[0], size=100 * num_classes, replace=False)
            xs_other = data_subsample[other_sampled_indices]

            print('class xs: {}, other xs: {} . Calculating avg distances...'.format(xs_class.shape, xs_other.shape))
            distances = np.zeros(len(xs_class))
            for k, x in enumerate(xs_class):
                print('{}/{}'.format(k, xs_class.shape[0]), end='\r')
            
            distances = np.array(distances).reshape(-1, 1)
            xs_sorted = np.concatenate((xs_class, distances), axis=1)
            print(distances.shape, xs_sorted.shape)

            print('Sorting...')
            xs_sorted = np.array(sorted(xs_sorted, key=lambda x: x[-1], reverse=True))
            best_samples = xs_sorted[:cfg.SAMPLES,:-1]

            print('Samples: {}'.format(best_samples.shape))
            j = i * cfg.SAMPLES
            result[j:j + cfg.SAMPLES] = best_samples
    else:
        #otherwise simply select n random samples without replacement
        print("Selecting random samples...")
        for i, (id, label_name) in enumerate(selected):
            indices = np.where(labels == id)[0]
            sampled_indices = np.random.choice(indices, size=cfg.SAMPLES, replace=False)
            j = i * cfg.SAMPLES
            result[j:j + cfg.SAMPLES] = raw_data[sampled_indices]

    print("Data selected, shape: {}".format(result.shape))
    print("Saving result to {}...".format(cfg.DEST_PATH))
    np.save(os.path.join(cfg.DEST_PATH, cfg.RESULT_NAME), result)
    print("Done!")


if __name__ == '__main__': main()


