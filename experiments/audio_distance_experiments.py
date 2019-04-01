import numpy as np
import os
import glob

from sklearn.preprocessing import MinMaxScaler

from utils import utils
from utils.constants import Constants
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import seaborn as sb
from sklearn.cluster import KMeans, AgglomerativeClustering

random_seed = 10


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


def cluster_compactness(xs, ys):
    """
    Computes the compactness value for each class.
    :param som: self organising map instance trained on xs data
    :param xs:  input data for the SOM, shape (n_examples, n_dim)
    :param ys:  labels for each xs value , shape (n_examples,_)
    :return:    list of class compactness values
    """
    classes = np.unique(ys)
    model = KMeans(num_classes, max_iter=10000, tol=1e-10, random_state=random_seed)
    mapped = model.fit_predict(xs)

    intra_class_dist = np.zeros(classes.size)
    inter_class_dist = 0.0

    #intra cluster distance
    # avg of ||bmu(i) - bmu(j)|| for each i,j in C
    for c in classes:
        class_xs = xs[np.where(mapped == c)]
        n = class_xs.shape[0]
        iter_count = ((n-1) * n) / 2  # gauss n(n+1)/2
        for i, x1 in enumerate(class_xs):
            for x2 in class_xs[i+1:]:
                intra_class_dist[c] += np.linalg.norm(x1 - x2)
        intra_class_dist[c] /= iter_count

    #inter cluster distance
    # avg of ||bmu(i) - bmu(j)|| for each i,j in every C
    iter_count = (len(mapped)-1) * len(mapped) / 2
    for i, x1 in enumerate(xs):
        for x2 in xs[i+1:]:
            inter_class_dist += np.linalg.norm(x1 - x2)

    inter_class_dist /= float(iter_count)
    return intra_class_dist / inter_class_dist


def show_compactness(xs, ys, num_classes):
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



DATA_PATH = os.path.join(Constants.AUDIO_DATA_FOLDER, "pca")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze representation quality.')
    parser.add_argument('--path', metavar='path', type=str, default=DATA_PATH,
                        help='Specify the file containing data')
    parser.add_argument('--op', metavar='op', type=str, default='cluster',
                        help='Specify the operation to launch')

    args = parser.parse_args()

    # # just to plot results
    # csv_path = os.path.join(args.path, "pca-results.csv")
    # data = []
    # filenames = []
    # with open(csv_path) as f:
    #     for line in f:
    #         l = line.split(',')
    #         data.append(np.asfarray(l[1:]))
    #         filenames.append(l[0])
    #
    # data = np.array(data)
    # xs = np.arange(0, len(data))
    # ys = data[:,0]
    #
    # sb.set()
    # plt.figure(figsize=(10, 5))
    # plt.plot(xs, ys, color="red")
    # plt.gca().set_ylim([0.6, 0.85])
    # plt.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], c="red")
    # plt.annotate("20pca25t", (xs[0], ys[0]), xytext=(xs[0], ys[0] - 0.02))
    # plt.annotate("no-pca100t", (xs[-1], ys[-1]), xytext=(xs[-1], ys[-1] + 0.01))
    # plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    #
    # plt.show()
    # exit(0)



    num_classes = 10
    files = list(glob.glob(os.path.join(args.path, '*.csv')))
    result = np.empty((len(files), num_classes))
    print('{} files found in {}'.format(len(files), args.path))

    for i, file in enumerate(files):
        xs, ys, _ = utils.from_csv_with_filenames(file)
        xs = np.array(xs)
        ys = np.array(ys)
        print(os.path.basename(file), xs.shape)
        continue
        labels = np.unique(ys)
        compactness = cluster_compactness(xs, ys)
        result[i] = compactness

    exit(0)
    means = result.mean(axis=-1)
    vars  = result.var(axis=-1)

    # merge results and sort based on mean values
    merged = zip(files, means, vars)
    merged = sorted(merged, key=lambda x: x[1])
    for f,m,v in merged:
        print('{},{},{}'.format(f,m,v))

    argmin = np.argmin(means)
    print('\n\nBest result: {} (mean = {:.5f}, var = {:.5f}'.format(files[int(argmin)], means[argmin], vars[argmin]))