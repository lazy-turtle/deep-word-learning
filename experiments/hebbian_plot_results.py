from matplotlib import pyplot as plt
import matplotlib.ticker as tkr

import seaborn as sb
import numpy as np
import pandas as pd
import os
import glob
from utils.constants import Constants

RESULTS_DIR = os.path.join(Constants.PLOT_FOLDER, 'hebbian', '2019-01-30',
                           'lr10_al-sorted_ta0.2_tv0.2_tha0.0_thv0.0_eucl_somv-1548704755_soma-syn')
PLOT_TAXONOMIC = False
SMOOTH = 0.5

def smooth(scalars, weight=0.0):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def main():
    dataframes = []
    for f in glob.glob(os.path.join(RESULTS_DIR, 'results_*.csv')):
        dataframes.append(pd.read_csv(f, index_col=0))

    audio_accuracies = np.empty((len(dataframes), 15)) #num trials x num presentations
    video_accuracies = np.empty((len(dataframes), 15)) #num trials x num presentations

    for i, df in enumerate(dataframes):
        audio_accuracies[i] = dataframes[i]['acc_a'].values
        video_accuracies[i] = dataframes[i]['acc_v'].values

    avg_acc_a = audio_accuracies.mean(axis=0)
    avg_acc_v = video_accuracies.mean(axis=0)
    taxonomic_factor = np.concatenate((avg_acc_a, avg_acc_v)).reshape((2,15))

    xx = np.arange(1,16)
    taxonomic_factor = np.mean(taxonomic_factor, axis=0)

    avg_acc_a = smooth(avg_acc_a, weight=SMOOTH)
    avg_acc_v = smooth(avg_acc_v, weight=SMOOTH)
    taxonomic_factor = smooth(taxonomic_factor, weight=SMOOTH)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(xx)
    ax.set_yticks(np.arange(0, 101, 5))
    ax.set_ylim([50,101])
    ax.yaxis.set_major_formatter(tkr.PercentFormatter())
    ax.grid(True, axis='y', linestyle=':')

    if PLOT_TAXONOMIC:
        cols = sb.color_palette(n_colors=1)
        ax.plot(xx, taxonomic_factor, color=cols[0], marker='o')
    else:
        cols = sb.color_palette(n_colors=2)
        ax.plot(xx, avg_acc_a, color=cols[0], marker='o', label='Audio source')
        ax.plot(xx, avg_acc_v, color=cols[1], marker='o', label='Video source')
        ax.legend()
    ax.set_xlabel('Presentations')
    ax.set_ylabel('Accuracy')
    plt.show()
if __name__ == '__main__': main()