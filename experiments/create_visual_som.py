from models.som.som import SOM
from models.som.SOMTest import show_som, show_confusion
import numpy as np
from utils.constants import Constants
from utils.utils import from_npy_visual_data, from_csv, global_transform, \
    from_csv_with_filenames, from_csv_visual_10classes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import json
import argparse

DATA_TYPE = 'video'
visual_data_path = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-80classes-segm.npy')
#visual_data_path = os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio-80classes-synth.npy')

#model_name = 'audio_10x8_s5.0_b64_a0.05_trsf_std_group-synth_seed42_1554145245_0ep'
model_name = 'best/video_60x60_s30.0_b256_a0.1_group-big_seed10_6060_std'
#model_name = 'video_20x30_s12.0_b64_a0.1_trsf_minmax_group-bbox_seed42_1548698406_final'
#model_name = 'best/video_20x30_s12.0_b64_a0.1_group-segm_seed42_1548697994_minmax'
model_path = os.path.join(Constants.TRAINED_MODELS_FOLDER, DATA_TYPE, model_name)
label_path = os.path.join(Constants.LABELS_FOLDER, 'coco-labels.json')


def read_labels(path):
    with open(path) as f:
        labels = json.load(f)
    labels = {int(k): v for k, v in labels.items()}
    return labels

def extract_som_info(model_name):
    model_info = model_name.split('_')[1:-1]
    info = dict()
    shape = model_info[0].split('x')
    info['shape'] = tuple([int(shape[0]), int(shape[1])])
    info['sigma'] = float(model_info[1][1:])
    info['batch'] = int(model_info[2][1:])
    info['alpha'] = float(model_info[3][1:])
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise a Self Organising Map.')
    parser.add_argument('--model', metavar='model', type=str, default=model_path, help='The model neighborhood value')
    parser.add_argument('--subsample', action='store_true', default=True)
    args = parser.parse_args()

    labels_dict = read_labels(label_path)
    n_classes = len(labels_dict)

    if 'csv' in visual_data_path:
        if DATA_TYPE == 'audio':
            xs, ys, _ = from_csv_with_filenames(visual_data_path)
            xs = np.array(xs)
            ys = np.array(ys).astype(int)
        else:
            xs, ys = from_csv_visual_10classes(visual_data_path)
            ys = [v - 1000 for v in ys]
    else:
        id_to_label = json.load(open(label_path))
        id_to_label = {int(k): v for k,v in id_to_label.items()}
        xs, ys, ids_dict = from_npy_visual_data(visual_data_path, classes=n_classes)
        print(xs.shape)
    labels = list(labels_dict.values())

    if args.subsample:
        np.random.seed(42)
        xs1 = []
        ys1 = []
        classes = np.arange(n_classes)
        sample_size = 40
        for i in classes:
            idx = np.where(ys == i)[0]
            idx = np.random.choice(idx, size=sample_size)
            xs1.append(xs[idx])
            ys1.append(ys[idx])
        xs = np.array(xs1).reshape((sample_size*n_classes, xs.shape[1]))
        ys = np.array(ys1).reshape(sample_size*n_classes)

    #xs, _ = global_transform(xs)
    #xs, _ = transform_data(xs)
    #xs = MinMaxScaler().fit_transform(xs)
    xs = StandardScaler().fit_transform(xs)
    dim = xs.shape[1]

    #info = extract_som_info(model_name)
    info = {'shape':[60,60], 'alpha':0.2, 'sigma':5.0, 'batch':128}
    som_shape = info['shape']
    som = SOM(som_shape[0], som_shape[1], dim, alpha=info['alpha'], sigma=info['sigma'],
              batch_size=info['batch'], checkpoint_loc=args.model, data=DATA_TYPE)
    som.restore_trained(args.model)

    #show_som(som, xs, ys, labels, 'Video SOM (bounding boxes)', show=True, dark=False, scatter=True,
     #        legend=True, point_size=120, suffix='_segm_trsf_minmax')
    show_confusion(som, xs, ys, title="Video SOM confusion")