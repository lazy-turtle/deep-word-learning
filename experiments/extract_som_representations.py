from utils.constants import Constants
from utils.utils import labels_dictionary
import numpy as np
import pandas as pd
import argparse
import os

class ExtractConfig(object):
    DATA_PATH = '/usr/home/studenti/sp160362/data/representations/rep-imagenet-10classes.npy'
    DEST_PATH = os.path.join(Constants.VIDEO_DATA_FOLDER)
    RESULT_NAME ='visual-10classes-imagenet.npy'
    FILE_LIST = '/usr/home/studenti/sp160362/data/representations/selection.csv'

    LABELS_PATH = os.path.join(Constants.LABELS_FOLDER, 'coco-imagenet-10-labels.json')
    COCO_LABELS = os.path.join(Constants.LABELS_FOLDER, 'coco-labels.json')

    SAMPLES = 100
    SAMPLES_SORT = 1000
    USE_TRUE_IDS = False
    SAVE_FILES = False


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
    parser.add_argument('--list', type=str, default=cfg.FILE_LIST, help='csv containing the original filenames.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for deterministic results.')
    parser.add_argument('--sort', action='store_true', default=False)

    args = parser.parse_args()

    cfg.DATA_PATH = args.data
    cfg.DEST_PATH = args.dest
    cfg.RESULT_NAME = args.name
    np.random.seed(args.seed)

    print("Loading data...")
    raw_data = np.load(cfg.DATA_PATH)
    labels = raw_data[:,-1].astype(int)
    selection = pd.read_csv(args.list, index_col=0)
    selection.sort_values('filenames', inplace=True)

    assert selection.shape[0] == raw_data.shape[0], "Data with different amount of rows!"

    filenames = selection['filenames'].values
    print("Data loaded, shape: {}".format(raw_data.shape))
    print("Distance based selection: {}".format(args.sort))

    print("Loading labels...")
    selected = labels_dictionary(cfg.LABELS_PATH)
    if cfg.USE_TRUE_IDS:
        coco_labels = labels_dictionary(cfg.COCO_LABELS)
        labels_ids = {v: k for k, v in coco_labels.items()}
        selected = {labels_ids[v]: v for v in selected.values()}

    num_classes = len(selected)
    print("Selected labels: ", selected)

    result = np.empty((num_classes * cfg.SAMPLES, raw_data.shape[1]))
    result_files = []

    if args.sort:
        # sort by distance from the other instances, descending
        # and take only the most isolated ones
        print('Sub-sampling data...')
        #first select a subsample from each class
        data_subsample = np.empty((0, raw_data.shape[1]))
        for i, (id, label) in enumerate(selected.items()):
            indices = np.where(labels == id)[0]
            samples = min(cfg.SAMPLES_SORT, len(indices))
            sampled_indices = np.random.choice(indices, size=samples, replace=False)
            data_subsample = np.concatenate((data_subsample, raw_data[sampled_indices]), axis=0)

        #then extract the n most closed ones for each class
        labels_subsample = data_subsample[:,-1]
        print('Subsampled data: {}'.format(data_subsample.shape))
        for i, (id, label_name) in enumerate(selected.items()):
            print('Selecting the closest {} samples from "{}..."'.format(cfg.SAMPLES, label_name))
            indices = np.where(labels_subsample == id)[0]
            num_samples = len(indices)
            xs_class = data_subsample[indices]

            if num_samples >= cfg.SAMPLES:
                print('class xs: {}, calculating prototype and distances...'.format(xs_class.shape))
                prototype = np.mean(xs_class, axis=0)
                distances = np.sum((xs_class - prototype)**2, axis=1)
                print(distances.shape)

                print('Sorting...')
                #sort and remove the worst examples (outliers), selecting only the first half
                best_samples = [x for x,_ in np.array(sorted(zip(xs_class, distances), key=lambda x: x[1]))]
                if (num_samples < cfg.SAMPLES * 2):
                    best_samples = np.array(best_samples[:cfg.SAMPLES])
                else:
                    best_samples = np.array(best_samples[:len(best_samples)//2])

                best_indices = np.random.choice(list(range(len(best_samples))), size=cfg.SAMPLES, replace=False)
                print('Chosen samples: {}'.format(best_indices))
                j = i * cfg.SAMPLES
                result[j:j + cfg.SAMPLES] = best_samples[best_indices]
            else:
                print("# samples < {}: selecting everything, with replacement.".format(cfg.SAMPLES))
                best_indices = np.random.choice(indices, size=cfg.SAMPLES, replace=True)
                j = i * cfg.SAMPLES
                result[j:j + cfg.SAMPLES] = xs_class[best_indices]
    else:
        #otherwise simply select n random samples without replacement
        print("Selecting random samples...")
        for i, (id, label_name) in enumerate(selected.items()):
            print('Processing class "{}",id: {}, index: {}'.format(label_name, id, i))
            indices = np.where(labels == id)[0]
            sampled_indices = np.random.choice(indices, size=cfg.SAMPLES, replace=False)
            j = i * cfg.SAMPLES
            result[j:j + cfg.SAMPLES] = raw_data[sampled_indices]
            result_files.extend(filenames[sampled_indices])

    print("Data selected, shape: {}".format(result.shape))
    print("Saving result to {}...".format(cfg.DEST_PATH))
    np.save(os.path.join(cfg.DEST_PATH, cfg.RESULT_NAME), result)
    if cfg.SAVE_FILES:
        print("Saving corresponding files to {}...".format(cfg.DEST_PATH))
        df = pd.DataFrame({'filenames': result_files})
        df.to_csv(os.path.join(cfg.DEST_PATH, '{}_files.csv'.format(cfg.RESULT_NAME.replace('.npy', ''))))
    print("Done!")


if __name__ == '__main__': main()


