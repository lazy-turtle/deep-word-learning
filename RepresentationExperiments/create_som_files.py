import numpy as np
import argparse
import json
import os

class ExtractConfig(object):
    DATA_PATH = '/usr/home/studenti/sp160362/data/representations/representations_train_raw.npy'
    DEST_PATH = '../data/10classes/'
    RESULT_NAME ='visual_10classes_train.npy'

    LABELS_DICT = '../data/coco-labels.json'
    CLASSES_PATH = '../data/labels10classes_coco.txt'

    SAMPLES = 100


def read_selected_classes(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def main():
    cfg = ExtractConfig()
    parser = argparse.ArgumentParser(description='Generate the required data files for training the SOM.')
    parser.add_argument('--data', type=str, default=cfg.DATA_PATH, help='Raw data matrix location.')
    parser.add_argument('--dest', type=str, default=cfg.DEST_PATH, help='Destination folder.')
    parser.add_argument('--name', type=str, default=cfg.RESULT_NAME, help='Name of the resulting file.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for deterministic results.')
    args = parser.parse_args()

    cfg.DATA_PATH = args.data
    cfg.DEST_PATH = args.dest
    cfg.RESULT_NAME = args.name
    np.random.seed(args.seed)

    print("Loading data...")
    raw_data = np.load(cfg.DATA_PATH)
    labels = raw_data[:,-1]
    print("Data loaded, shape: {}".format(raw_data.shape))

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


