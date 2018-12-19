import numpy as np
from matplotlib import pyplot as plt
from utils.utils import transform_data
from sklearn.preprocessing import MinMaxScaler
import os

DATA_PATH = '../data/video/visual_10classes_train_b.npy'


def from_csv_visual_10classes(path, labels='imagenet-labels.json'):
    f = open(path,'r')
    xs = []
    ys = []

    for l in f:
        lSplit = l.split(',')
        xs.append(np.array(lSplit[1:]).astype(float))
        ys.append(lSplit[0])
    f.close()
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def load_data(type, path):
    if type == "new":
        data = np.load(path)
        data_x = data[:, :-1]
        data_y = data[:, -1].astype(np.int)
    else:
        data_x, data_y = from_csv_visual_10classes('../data/video/VisualInputTrainingSet.csv')
    return data_x, data_y

def smooth(scalars, weight=0.0):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

# Data
data_type = "old"
check = "mean"

data_x, data_y = load_data(data_type, DATA_PATH)

# normalize data somehow
#data_x, _ = transform_data(data_x)
#scaler = MinMaxScaler()
#data_x = scaler.fit_transform(data_x)
#m = data_x.min()
#M = data_x.max()
#data_x = ((data_x - m) / (M - m))
#print(data_x.max())

uniques = np.unique(data_y).tolist()
indices_dict = {val: i for i, val in enumerate(uniques)}
id_to_index = {label: index for index, label in enumerate(uniques)}
colors = ['red', 'pink', 'purple', 'blue', 'cyan', 'green', 'olive', 'orange', 'brown', 'gray', 'black']

smooth_val = 0.0
xs = np.arange(2048)
plt.figure(figsize=(12, 4))
for c in range(10):
    print("class {} - label: {} - col: {}".format(c, data_y[c*100], colors[c]))
    rows = data_x[c*100:c*100+100]
    if check == "mean":
        ys = rows.mean(axis = 0)
    elif check == "std":
        ys = rows.std(axis = 0)

    z_ind = np.argwhere(ys == 0)
    ys_zeros = ys[z_ind]
    ys = smooth(ys, weight=smooth_val)
    plt.plot(xs, ys, colors[c])
    plt.scatter(z_ind, ys_zeros, c='k', marker='x', zorder=100)

    for xy in zip(z_ind, ys_zeros):
        xyt = tuple([xy[0], xy[1] - 0.01])
        plt.gca().annotate('(%s, %s)' % xy, xy=xy, xytext=xyt, textcoords='data')

plt.xlim([-10,2058])
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
#plt.gca().set_ylim(bottom=-0.15)
plt.gca().yaxis.grid(True)
plt.show()
#plt.savefig("C:/Users/Edoardo/Desktop/representations/repr_{}_{}_smooth{}_b.png".format(data_type, check, smooth_val))
