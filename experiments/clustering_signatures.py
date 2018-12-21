
import data_utils
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut


# Importing data

xs,ys = data_utils.load_data("../activations-small.pkl")

# Creating a single array of time-steps descriptions by concatenating
# the descriptions of each example

clustering_xs = [t_description for x in xs for t_description in x]

# Performing k-means on the resulting dataset

N_CLUSTERS = 30
kmeans = KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(clustering_xs)

def signature_of(example, kmeans):
    """
    Returns the kmeans signature of the example.

    The kmean signature of the example is a vector with N_CLUSTERS
    dimensions where elem number k of the vector contains the number of
    objects in `example` assigned to cluster number k.
    """
    cluster_ids = kmeans.predict(example)

    result = np.zeros(N_CLUSTERS)
    for id in cluster_ids:
        result[id] += 1

    return result


signatures = np.array([signature_of(x, kmeans) for x in xs])
ys = np.array(ys)



# Fitting an SVC onto the built dataset

loo = LeaveOneOut()

results = []
for train_index, test_index in loo.split(signatures):
    print(test_index)

    # Probably we could initialize svc only once outside the loop,
    # but just to be safe we get a new one at each iteration
    svc = svm.SVC()
    svc.fit(signatures[train_index], ys[train_index])
    predicted_ys = svc.predict(signatures[test_index])
    results.append(np.average(predicted_ys == ys[test_index]))


print(np.average(results))
