import numpy as np
import json

print("Loading data...")
raw_data = np.load("tests/representations.npy")
print("Data loaded, shape: {}".format(raw_data.shape))

print("Loading labels...")
labels_dict = json.load(open("coco-labels.json"))
name_to_id = {v: k for k, v in labels_dict.items()}

class_names = ['bird', 'book', 'cat', 'dog', 'cup', 'dining table', 'apple', 'banana', 'umbrella', 'sports ball']
ids = [int(name_to_id[n]) for n in class_names]

selected = zip(ids, class_names)
selected = sorted(selected, key = lambda t: t[0])
print(selected)
print("Done!")

