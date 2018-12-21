"""
This script performs several experiments to test whether the concatenation
of the rnn states sufficies as a "good representation" of the word, i.e.,
if vectors obtained in such a way can be compared for similarity to conclude
that the corresponding "heared" words are similar.
"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import LeaveOneOut

import data_utils

# Importing data

xs,ys = data_utils.load_data("../activations-small.pkl")


# Truncating each example at time-step 27 and concatenating the features into a
# single 27 x 2048 (=55296) elements vector

MAX_LEN = 27
truncated_xs = np.array([x[:MAX_LEN,:].ravel() for x in xs])
ys = np.array(ys)

# Fitting an SVC onto the built dataset

loo = LeaveOneOut()

results = []
for train_index, test_index in loo.split(truncated_xs):
    print(test_index)

    # Probably we could initialize svc only once outside the loop,
    # but just to be safe we get a new one at each iteration
    svc = svm.SVC()
    svc.fit(truncated_xs[train_index], ys[train_index])
    predicted_ys = svc.predict(truncated_xs[test_index])
    results.append(np.average(predicted_ys == ys[test_index]))

print(np.average(results))
