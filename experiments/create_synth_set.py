import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from utils.constants import Constants
import os

class SynthConfig(object):
    NUM_CLASSES = 10
    NUM_SAMPLES = 100
    MEAN_VAL = 1.0
    VARIANCE = 0.0
    SAVE = True
    DEST_PATH = os.path.join(Constants.AUDIO_DATA_FOLDER, 'audio_10classes_synth.npy')


def plot_samples(samples, cfg):
    data = samples[:,:-1]
    assert data.shape[1] <= 2
    colors = sb.color_palette('bright', n_colors=cfg.NUM_CLASSES)
    for c in range(cfg.NUM_CLASSES):
        j = c * cfg.NUM_SAMPLES
        xy = data[j:j + cfg.NUM_SAMPLES]
        plt.scatter(xy[:,0], xy[:,1], c=colors[c])
    plt.show()


def main():
    cfg = SynthConfig()
    result = np.empty((0, cfg.NUM_CLASSES + 1))
    cov = np.eye(cfg.NUM_CLASSES) * cfg.VARIANCE

    print('Generating random samples for {} classes.')
    print('Means: {} - Variances: {}'.format(cfg.MEAN_VAL, cfg.VARIANCE))
    for c in range(cfg.NUM_CLASSES):
        means = np.zeros(cfg.NUM_CLASSES, dtype=float)
        means[c] = cfg.MEAN_VAL
        xs = np.random.multivariate_normal(means, cov, size=cfg.NUM_SAMPLES)
        ys = np.repeat([c], cfg.NUM_SAMPLES).reshape((-1,1))
        data = np.concatenate((xs, ys), axis=1)
        result = np.concatenate((result, data), axis=0)

    print('Generated data: {}'.format(result.shape))
    print('Saving data to file...', end='')
    np.save(cfg.DEST_PATH, result)

if __name__ == '__main__': main()