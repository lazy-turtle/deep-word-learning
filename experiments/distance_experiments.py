import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler

from utils import utils
from utils.constants import Constants
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import seaborn as sb
from sklearn.cluster import KMeans, AgglomerativeClustering

random_seed = 42


def get_prototypes(xs, ys):
    prototype_dict = {unique_y: [] for unique_y in set(ys)}
    for i, x in enumerate(xs):
        prototype_dict[ys[i]].append(x)
    prototype_dict = {k: np.array(prototype) for k, prototype in prototype_dict.items()}
    result = dict()
    for y in set(sorted(ys)):
        result[y] = np.mean(prototype_dict[y], axis=0)
    return result


def average_prototype_distance_matrix(xs, ys):
    """
        Relies on all y in ys being in the interval [0, number of classes) - take care of
        handling that if importing it from another script.
        As a general rule, x in xs and y in ys are not ordered by class but by speaker
        That is why I am not using models.som.SOMTest.classPrototype, which
        has the opposite assumption.
    """
    prototype_distance_matrix = np.zeros((len(set(ys)), len(set(ys))))
    # compute prototypes in dictionary d
    prototype_dict = {unique_y: [] for unique_y in set(ys)}
    for i, x in enumerate(xs):
        prototype_dict[ys[i]].append(x)
    prototype_dict = {k: np.array(prototype) for k, prototype in prototype_dict.items()}
    for y in set(ys):
        prototype_dict[y] = np.mean(prototype_dict[y], axis=0)
    prototypes = np.asarray(list(prototype_dict.values())).T
    for i, x in enumerate(xs):
        prototype_distance_matrix[ys[i]][:] += np.mean(np.absolute(prototypes - x.reshape((-1, 1))), axis=0).T
    print(prototype_distance_matrix)
    fig, ax = plt.subplots()
    ax.matshow(prototype_distance_matrix, cmap=plt.get_cmap('Greys'))
    for (i, j), distance in np.ndenumerate(prototype_distance_matrix):
        ax.text(j, i, '{:0.2f}'.format(distance), ha='center', va='center')
    plt.show(True)


def examples_distance(xs, i1, i2):
    return np.linalg.norm(xs[i1]-xs[i2])


def cluster_compactness(xs, ys, num_classes):
    print('Fitting clustering model...')
    model = KMeans(num_classes, max_iter=10000, tol=1e-10, random_state=random_seed)
    #model = KMeans(n_clusters = num_classes,max_iter=100,n_init=50,algorithm='elkan') # giorgia
    cluster_ys = model.fit_predict(xs)
    print('Done. Computing occurrences...')
    # columns: clusters; rows: classes
    occurrences_matrix = np.zeros((num_classes, num_classes))
    cluster_belonging_dict = {y: [] for y in range(num_classes)}
    for i, c_y in enumerate(cluster_ys):
        occurrences_matrix[c_y][ys[i]] += 1
        cluster_belonging_dict[c_y].append(i)
    occurrences_matrix /= np.sum(occurrences_matrix, axis=0) # normalize column-wise
    occurrences_matrix = np.sort(occurrences_matrix, axis=0)
    #print(occurrences_matrix[:][-1])
    print('Computing intra-cluster distance...')
    intra_cluster_distance = [0 for i in range(num_classes)]
    inter_cluster_distance = 0
    for i in range(num_classes):
        intra_temp = 0
        for pos, j in enumerate(cluster_belonging_dict[i]):
            x1 = xs[j]
            for k in cluster_belonging_dict[i][pos+1:]:
                x2 = xs[k]
                intra_temp += np.linalg.norm(x1-x2)
        intra_cluster_distance[i] = intra_temp / len(cluster_belonging_dict[i])
    for i, x1 in enumerate(xs):
        for x2 in xs[i+1:]:
            inter_cluster_distance += np.linalg.norm(x1-x2)
    inter_cluster_distance /= len(xs)
    compactness = np.array(intra_cluster_distance)/inter_cluster_distance
    plt.ylim(0, 1)
    plt.bar(range(num_classes), compactness)
    plt.show()

    mean = np.mean(compactness)
    var = np.var(compactness)
    print('Cluster compactness')
    for c in compactness:
        print(c)
    print('Mean\n{};\nVariance\n{}'.format(mean, var))


def show_clustering(xs, ys, num_classes, labels, show_classes=False):
    print('Fitting clustering model...')
    #model = KMeans(num_classes, max_iter=10000, tol=1e-7, random_state=random_seed)
    model = AgglomerativeClustering(n_clusters=num_classes)
    cluster_ys = model.fit_predict(xs)
    print('Done. Computing occurrences...')

    print('Plotting bars...')
    values, counts = np.unique(cluster_ys, return_counts=True)

    plots = []
    width = 0.3
    classes = np.arange(num_classes)
    occurrences = np.zeros((num_classes, num_classes))

    for i in classes:
        class_dist = cluster_ys[np.where(ys == i)]
        clusters, counts = np.unique(class_dist, return_counts=True)
        for c, v in zip(clusters, counts):
            occurrences[i, c] += v

    occurrences = occurrences.T if show_classes else occurrences

    plt.figure(figsize=(15,10))

    cols = sb.color_palette('muted', num_classes)
    bot = np.zeros(num_classes)
    for j in classes:
        freqs = occurrences[j]
        p = plt.bar(classes, freqs, width, bottom=bot, color=cols[j])
        plots.append(p)
        bot += freqs

    cluster_labels = ['c{}'.format(i) for i in classes]
    classes_labels = [labels[i] for i in classes]

    plt.ylim([0,np.max(occurrences.sum(axis=0)) + 10])
    plt.ylabel('Cluster size')
    title = 'Classes colored by cluster subdivision' if show_classes else 'Clusters generated by k-means'
    plt.title(title)
    plt.xticks(classes, classes_labels if show_classes else  cluster_labels)
    plt.legend([p[0] for p in plots], cluster_labels if show_classes else classes_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def class_prototype(xs, ys, c):
    """
    Computes the class prototype, intended as the mean vector obtained by averaging
    all vectors in xs belonging to class c.
    :param xs:  numpy matrix containing data instances, shape (num_samples, dim)
    :param ys:  numpy array containing data labels, shape (num_samples,)
    :param c:   int indicating the class for the prototype computation
    :return: vector of size (dim) representing the prototype for class c
    """
    class_xs = xs[np.where(ys == c)[0]]
    return np.mean(class_xs, axis=0)


def prototype_similarity(xs, prototype):
    """
    Calculates the average norm-1 from each xs instance to the given prototype vector.
    Every vector in xs belongs to the same class, not necessarily the same one of the prototype.
    :param xs:          matrix of data vectors belonging to the same class, shape (n, dims)
    :param prototype:   prototype vector representing a single class, shape (dims,)
    :return:            float value indicating the similarity of the set xs to the prototype
    """
    distances = np.sum(np.abs(xs - prototype), axis=1) / float(prototype.size)
    return np.mean(distances)


def similarity_matrix(xs, ys):
    """
    Generates a square NxN matrix, where N is the number of unique classes and every
    element m_ij contains the similarity between class i and prototype j
    :param xs:
    :param ys:
    :return:
    """
    classes = np.unique(ys)
    sim_matrix = np.zeros((classes.size, classes.size))
    prototypes = [class_prototype(xs,ys,c) for c in classes]
    for i, c in enumerate(classes):
        class_xs = xs[np.where(ys == c)[0]]
        similarities = [prototype_similarity(class_xs, p) for p in prototypes]
        sim_matrix[i] = np.array(similarities)
    return sim_matrix


data_names = [
    'visual_10classes_train_a.npy',
    'visual_10classes_train_b.npy',
    'visual_10classes_train_c2.npy',
    'VisualInputTrainingSet.csv'
]
data_path = os.path.join(Constants.VIDEO_DATA_FOLDER, data_names[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze representation quality.')
    parser.add_argument('--csv-path', metavar='csv_path', type=str, required=False,
                        help='The csv file with the extracted representations.')
    parser.add_argument('--classes100', action='store_true',
                        help='Specify whether you are analyzing \
                        a file with representations from 100 classes, as the loading functions are different.',
                        default=False)
    parser.add_argument('--is-audio', action='store_true', default=False,
                        help='Specify whether the csv contains audio representations, as the loading functions are different.')
    parser.add_argument('--data', metavar='data', type=str, default='new', help='Use new data format or old')
    parser.add_argument('--path', metavar='path', type=str, default=data_path,
                        help='Specify the file containing data')
    parser.add_argument('--op', metavar='op', type=str, default='cluster',
                        help='Specify the operation to launch')

    args = parser.parse_args()

    if not args.classes100:
        num_classes = 10
        if not args.is_audio:
            print("Clustering visual data for 10 classes...")
            if args.data == 'new':
                print('Reading new data...')
                xs, ys, label_to_id = utils.from_npy_visual_data(args.path)
                xs = MinMaxScaler().fit_transform(xs)
                label_path = os.path.join(Constants.LABELS_FOLDER, 'coco-labels.json')
                id_names_dict = utils.labels_dictionary(label_path)
                vals = np.unique(ys)
                labels = {v: id_names_dict[label_to_id[v]] for v in vals}
            else:
                print("Reading old data...")
                xs, ys = utils.from_csv_visual_10classes(args.path)
                ys = [val - 1000 for val in ys]
                xs = np.array(xs)
                ys = np.array(ys)
                xs = MinMaxScaler().fit_transform(xs)
                labels = np.unique(ys)
        else:
            xs, ys, _ = utils.from_csv_with_filenames(args.csv_path)
    else:
        num_classes = 100
        if not args.is_audio:
            xs, ys = utils.from_csv_visual_100classes(args.csv_path)
        else:
            xs, ys, _ =  utils.from_csv_with_filenames(args.csv_path)
        labels = np.unique(ys)

    if args.op == 'avg':
        average_prototype_distance_matrix(xs, ys)
    elif args.op == 'cluster':
        #cluster_compactness(xs, ys, num_classes)
        show_clustering(xs, ys, num_classes, labels, show_classes=True)
    elif args.op == 'sim':
        m = similarity_matrix(xs, ys)

        #print in latex format because i'm lazy
        for row in m:
            best = np.argmin(row)
            for i, val in enumerate(row):
                if i == best:
                    print('\\textbf{{{:.4f}}}'.format(val), end='')
                else:
                    print('{:.4f}'.format(val), end='')
                last = ' & ' if i < (row.size - 1) else ' \\\\\n'
                print(last, end='')
        print(np.argmin(m, axis=1))
        print(labels)