from utils.utils import labels_dictionary
import numpy as np
import argparse
import os

class ExtractConfig(object):
    DATA_PATH = '/usr/home/studenti/sp160362/data/representations/rep-imagenet-10classes.npy'
    DEST_PATH = '../data/video/'
    RESULT_NAME ='visual-10classes-imagenet.npy'

    LABELS_PATH = '../data/labels/coco-imagenet-10-labels.json'

    SAMPLES = 100
    SAMPLES_SORT = 1000


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
    selected = labels_dictionary(cfg.LABELS_PATH)
    selected = sorted(selected.items(), key = lambda t: t[0])
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
            samples = min(cfg.SAMPLES_SORT, len(indices))
            sampled_indices = np.random.choice(indices, size=samples, replace=False)
            data_subsample = np.concatenate((data_subsample, raw_data[sampled_indices]), axis=0)

        #then extract the n most closed ones for each class
        labels_subsample = data_subsample[:,-1]
        print('Subsampled data: {}'.format(data_subsample.shape))
        for i, (id, label_name) in enumerate(selected):
            print('Selecting the closest {} samples from "{}..."'.format(cfg.SAMPLES, label_name))
            indices = np.where(labels_subsample == id)[0]
            xs_class = data_subsample[indices]

            print('class xs: {}, calculating prototype and distances...'.format(xs_class.shape))
            prototype = np.mean(xs_class, axis=0)
            distances = np.sum((xs_class - prototype)**2, axis=1)
            print(distances.shape)

            print('Sorting...')
            #sort and remove the worst examples (outliers), selecting only the first half
            best_samples = [x for x,_ in np.array(sorted(zip(xs_class, distances), key=lambda x: x[1]))]
            best_samples = np.array(best_samples[:len(best_samples)//2])

            best_indices = np.random.choice(list(range(len(best_samples))), size=cfg.SAMPLES, replace=False)
            print('Chosen samples: {}'.format(best_indices))
            j = i * cfg.SAMPLES
            result[j:j + cfg.SAMPLES] = best_samples[best_indices]
    else:
        #otherwise simply select n random samples without replacement
        print("Selecting random samples...")
        for i, (id, label_name) in enumerate(selected):
            indices = np.where(labels == id)[0]
            sampled_indices = np.random.choice(indices, size=cfg.SAMPLES, replace=False)
            print(sampled_indices)
            j = i * cfg.SAMPLES
            result[j:j + cfg.SAMPLES] = raw_data[sampled_indices]

    print("Data selected, shape: {}".format(result.shape))
    print("Saving result to {}...".format(cfg.DEST_PATH))
    np.save(os.path.join(cfg.DEST_PATH, cfg.RESULT_NAME), result)
    print("Done!")


if __name__ == '__main__': main()


