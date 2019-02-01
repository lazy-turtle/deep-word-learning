import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as tkr
from matplotlib import lines

import seaborn as sb
import numpy as np
import pandas as pd
import os
import glob


RESULTS_DIR_SEGM = os.path.join('C:\\Users\\Edoardo\\Desktop\\segm')
RESULTS_DIR_BBOX = os.path.join('C:\\Users\\Edoardo\\Desktop\\bbox')
SMOOTH = 0.0


def smooth(scalars, weight=0.0):  # Weight between 0 and 1
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def main():
    dataframes_segm = []
    for f in glob.glob(os.path.join(RESULTS_DIR_SEGM, 'results_*.csv')):
        dataframes_segm.append(pd.read_csv(f, index_col=0))

    dataframes_bbox = []
    for f in glob.glob(os.path.join(RESULTS_DIR_BBOX, 'results_*.csv')):
        dataframes_bbox.append(pd.read_csv(f, index_col=0))

    audio_acc_segm = np.empty((len(dataframes_segm), 15)) #num trials x num presentations
    video_acc_segm = np.empty((len(dataframes_segm), 15))
    audio_acc_bbox = np.empty((len(dataframes_bbox), 15))
    video_acc_bbox = np.empty((len(dataframes_bbox), 15))

    for i, df in enumerate(dataframes_segm):
        audio_acc_segm[i] = dataframes_segm[i]['acc_a'].values
        video_acc_segm[i] = dataframes_segm[i]['acc_v'].values

    for i, df in enumerate(dataframes_bbox):
        audio_acc_bbox[i] = dataframes_bbox[i]['acc_a'].values
        video_acc_bbox[i] = dataframes_bbox[i]['acc_v'].values

    avg_audio_segm = audio_acc_segm.mean(axis=0)
    avg_video_segm = video_acc_segm.mean(axis=0)
    tax_fact_segm = np.concatenate((avg_audio_segm, avg_video_segm)).reshape((2,15))
    avg_audio_bbox = audio_acc_bbox.mean(axis=0)
    avg_video_bbox = video_acc_bbox.mean(axis=0)
    tax_fact_bbox = np.concatenate((avg_audio_bbox, avg_video_bbox)).reshape((2, 15))

    xx = np.arange(1,16)
    tax_fact_segm = np.mean(tax_fact_segm, axis=0)
    tax_fact_bbox = np.mean(tax_fact_bbox, axis=0)

    matplotlib.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(np.arange(1,16, 2))
    ax.set_yticks(np.arange(0, 101, 5))
    ax.set_ylim([50,101])
    ax.yaxis.set_major_formatter(tkr.PercentFormatter())
    ax.grid(True, axis='y', linestyle=':')

    cols = sb.color_palette('deep', n_colors=4)[2:]
    styles = [':','--']
    m = None
    ax.plot(xx, smooth(tax_fact_segm, weight=SMOOTH), color=cols[0], marker=m)
    ax.plot(xx, smooth(tax_fact_bbox, weight=SMOOTH), color=cols[1], marker=m)

    ax.plot(xx, smooth(avg_audio_segm, weight=SMOOTH), color=cols[0], marker=m, label='Audio segm', linestyle=styles[1])
    ax.plot(xx, smooth(avg_video_segm, weight=SMOOTH), color=cols[0], marker=m, label='Video segm', linestyle=styles[0])
    ax.plot(xx, smooth(avg_audio_bbox, weight=SMOOTH), color=cols[1], marker=m, label='Audio bbox', linestyle=styles[1])
    ax.plot(xx, smooth(avg_video_bbox, weight=SMOOTH), color=cols[1], marker=m, label='Video bbox', linestyle=styles[0])

    legend_items = [
        lines.Line2D([0], [0], color='k', lw=4, linestyle=':'),
        lines.Line2D([0], [0], color='k', lw=4, linestyle='--'),
        lines.Line2D([0], [0], color='k', lw=4, linestyle='-'),
        lines.Line2D([0], [0], color=cols[0], lw=4),
        lines.Line2D([0], [0], color=cols[1], lw=4)
    ]

    legend_labels = [
        'Production',
        'Comprehension',
        'Taxonomic factor',
        'Whole object',
        'Non whole object'
    ]
    ax.legend(legend_items, legend_labels, loc="lower right")
    ax.set_xlim([1,15])
    ax.set_xlabel('Number of joint presentations', fontsize=16)
    plt.show()


if __name__ == '__main__': main()